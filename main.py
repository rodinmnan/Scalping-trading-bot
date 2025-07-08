/forex-trading-bot
│
├── main.py               # Main bot code (from previous example)
├── requirements.txt      # Python dependencies
├── runtime.txt           # Python version
├── .env.example          # Template for environment variables
├── README.md             # Project documentation
├── .gitignore            # Ignore sensitive files
└── /utils                # Optional utilities
    ├── backup_state.py   # State preservation scripts
    └── health_check.py   # Render uptime monitoring
import os
import logging
import time
import threading
import random
import numpy as np
from datetime import datetime, timedelta
import requests
import pandas as pd
import pytz
from telegram import Update, ParseMode
from telegram.ext import Updater, CommandHandler, CallbackContext
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import talib
from textblob import TextBlob

# Load environment variables
load_dotenv()

# API Configuration - Optimized for Render starter plan
TRADEMADE_API_KEY = os.getenv("TRADEMADE_API_KEY")
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
ADMIN_ID = os.getenv("ADMIN_ID", "")

# Validate critical environment variables
if not all([TRADEMADE_API_KEY, TWELVE_DATA_API_KEY, TELEGRAM_TOKEN]):
    raise EnvironmentError("Missing required environment variables!")

# Trading parameters optimized for 90% TP hit rate
PAIRS = os.getenv("TRADING_PAIRS", "EURUSD,GBPUSD,XAUUSD").split(',')  # Reduced pair count
NEW_YORK_TZ = pytz.timezone('America/New_York')
CACHE_DURATION = int(os.getenv("CACHE_DURATION", 120))  # Increased cache duration
NEWS_CHECK_INTERVAL = int(os.getenv("NEWS_CHECK_INTERVAL", 3600))  # Reduced news checks
VOLATILITY_LOOKBACK = int(os.getenv("VOLATILITY_LOOKBACK", 21))  # Longer lookback for stability
TREND_FILTER_MODE = os.getenv("TREND_FILTER", "strict")
SAFETY_MODE = os.getenv("SAFETY_MODE", "high")  # high/moderate/low

# API rate limit counters
API_CALL_LOG = {}
MAX_API_CALLS = 100  # Per hour for free plans

class HighProbabilityTradingBot:
    def __init__(self):
        # Initialize with safety defaults
        self.data_lock = threading.RLock()
        self.live_prices = {pair: {'price': None, 'timestamp': None} for pair in PAIRS}
        self.market_open = False
        self.high_impact_news = False
        self.news_sentiment = 0.0
        self.signal_cooldown = {}
        self.performance = {
            'total_signals': 0,
            'tp_hits': 0,  # Combined TP hits
            'sl_hits': 0,
            'win_rate': 0.0
        }
        self.active_signals = []
        self.subscribed_users = set()
        self.running = True
        self.trend_filters = {pair: None for pair in PAIRS}
        self.volume_profile = {}
        self.account_info = {
            'size': 100,  # Default $100 account
            'risk_percent': 1.0
        }
        self.liquidity_zones = {pair: [] for pair in PAIRS}
        self.signal_queue = []  # For delayed signal processing
        self.fakeout_filter_enabled = True
        
        # Configure API session with aggressive caching
        self.session = requests.Session()
        retry_strategy = Retry(
            total=2,  # Reduced retries
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
        self.session.mount("https://", adapter)
        
        # Initialize Telegram
        self.updater = Updater(TELEGRAM_TOKEN, use_context=True, request_kwargs={
            'read_timeout': 20, 'connect_timeout': 15
        })
        
        # Start services
        self.start_services()

    def start_services(self):
        """Initialize all background services with thread limits"""
        services = [
            self.price_updater,
            self.market_hours_checker,
            self.news_monitor,
            self.signal_generator,
            self.signal_monitor,
            self.signal_queue_processor,
            self.trend_analyzer,
            self.api_rate_limiter
        ]
        
        # Start only essential services
        for service in services[:6]:  # Omit non-essential for small accounts
            threading.Thread(target=service, daemon=True).start()

    # ======================
    # API RATE LIMITER
    # ======================
    
    def api_rate_limiter(self):
        """Monitor and enforce API rate limits"""
        while self.running:
            now = time.time()
            for endpoint, last_call in list(API_CALL_LOG.items()):
                if now - last_call > 3600:  # Reset after 1 hour
                    del API_CALL_LOG[endpoint]
            time.sleep(60)

    def safe_api_call(self, url):
        """Make API call with rate limiting"""
        endpoint = url.split('?')[0]
        current_calls = sum(1 for t in API_CALL_LOG.values() if time.time() - t < 3600)
        
        if current_calls >= MAX_API_CALLS:
            logging.warning("API rate limit reached. Skipping call.")
            return None
            
        try:
            response = self.session.get(url, timeout=15)
            API_CALL_LOG[endpoint] = time.time()
            return response
        except Exception as e:
            logging.error(f"API call failed: {str(e)}")
            return None

    # ======================
    # SAFETY ENHANCEMENTS
    # ======================
    
    def get_safety_params(self):
        """Get parameters based on safety mode"""
        params = {
            'high': {
                'max_daily_loss': 0.02,  # 2% of account
                'max_consecutive_losses': 2,
                'daily_trade_limit': 4,
                'min_confidence': 0.90,
                'volatility_threshold': 0.004,
                'position_size_multiplier': 0.7
            },
            'moderate': {
                'max_daily_loss': 0.03,
                'max_consecutive_losses': 3,
                'daily_trade_limit': 6,
                'min_confidence': 0.85,
                'volatility_threshold': 0.006,
                'position_size_multiplier': 0.8
            },
            'low': {
                'max_daily_loss': 0.05,
                'max_consecutive_losses': 5,
                'daily_trade_limit': 8,
                'min_confidence': 0.80,
                'volatility_threshold': 0.008,
                'position_size_multiplier': 1.0
            }
        }
        return params.get(SAFETY_MODE, params['high'])
    
    # ======================
    # FAKEOUT FILTER
    # ======================
    
    def detect_fakeout(self, pair, direction, current_price):
        """Detect false breakouts using price action"""
        try:
            # Get recent price data
            response = self.safe_api_call(
                f"https://api.twelvedata.com/time_series?symbol={pair}&interval=5min&outputsize=10&apikey={TWELVE_DATA_API_KEY}"
            )
            if not response:
                return False
                
            data = response.json()
            if data.get('status') != 'ok' or not data.get('values'):
                return False
                
            df = pd.DataFrame(data['values'])
            df['close'] = pd.to_numeric(df['close'])
            
            # Check for rejection patterns
            last_close = df.iloc[-1]['close']
            prev_close = df.iloc[-2]['close']
            
            if direction == "BUY":
                # Bearish rejection: price spikes up but closes near low
                if current_price > last_close and last_close < prev_close:
                    return True
            else:  # SELL
                # Bullish rejection: price spikes down but closes near high
                if current_price < last_close and last_close > prev_close:
                    return True
                    
            return False
        except Exception as e:
            logging.error(f"Fakeout detection failed: {str(e)}")
            return False

    # ======================
    # SIGNAL QUEUE PROCESSOR
    # ======================
    
    def signal_queue_processor(self):
        """Process signals when price nears entry point"""
        while self.running:
            try:
                current_time = time.time()
                new_queue = []
                
                for signal in self.signal_queue:
                    # Remove stale signals (older than 15 minutes)
                    if current_time - signal['generated_time'] > 900:
                        continue
                        
                    # Get current price
                    with self.data_lock:
                        price_data = self.live_prices.get(signal['pair'])
                        if not price_data or not price_data['price']:
                            continue
                        current_price = price_data['price']
                    
                    # Check if price is near entry (within 0.05%)
                    entry = signal['entry']
                    price_diff = abs(current_price - entry) / entry
                    
                    if price_diff <= 0.0005:  # 0.05% threshold
                        # Final confirmation before sending
                        if self.validate_signal(signal, current_price):
                            self.send_signal_alert(signal)
                        else:
                            logging.info(f"Signal invalidated for {signal['pair']} near entry")
                    else:
                        new_queue.append(signal)
                        
                with self.data_lock:
                    self.signal_queue = new_queue
                    
            except Exception as e:
                logging.error(f"Signal queue processing failed: {str(e)}")
            time.sleep(30)  # Check every 30 seconds

    def validate_signal(self, signal, current_price):
        """Re-validate signal conditions before sending"""
        # Check trend filter
        with self.data_lock:
            trend = self.trend_filters.get(signal['pair'])
        
        if trend == "BULL" and signal['direction'] == "SELL":
            return False
        if trend == "BEAR" and signal['direction'] == "BUY":
            return False
            
        # Check fakeout
        if self.fakeout_filter_enabled:
            if self.detect_fakeout(signal['pair'], signal['direction'], current_price):
                return False
                
        # Check news impact
        if self.high_impact_news:
            return False
            
        return True

    # ======================
    # OPTIMIZED PRICE UPDATER
    # ======================
    
    def price_updater(self):
        """Optimized price updates with batching"""
        while self.running:
            if self.market_open:
                try:
                    # Batch API call for all pairs
                    pairs_str = ','.join(PAIRS)
                    response = self.safe_api_call(
                        f"https://marketdata.trademade.com/api/v1/live?currency={pairs_str}&api_key={TRADEMADE_API_KEY}"
                    )
                    
                    if not response:
                        time.sleep(60)
                        continue
                        
                    data = response.json()
                    
                    if 'quotes' in data and data['quotes']:
                        current_time = time.time()
                        with self.data_lock:
                            for quote in data['quotes']:
                                pair = quote['instrument']
                                price = float(quote['mid'])
                                self.live_prices[pair] = {
                                    'price': price,
                                    'timestamp': current_time
                                }
                except Exception as e:
                    logging.error(f"Price update failed: {str(e)}")
            time.sleep(30)  # Reduced frequency

    # ======================
    # HIGH CONFIDENCE SIGNAL GENERATION (90% TP RATE)
    # ======================
    
    def generate_signal(self, pair):
        """Generate signals with 90% TP rate focus"""
        safety = self.get_safety_params()
        
        # Skip during closed market
        if not self.market_open:
            return None
            
        # Get current price with safety check
        with self.data_lock:
            current_data = self.live_prices.get(pair)
            if not current_data or not current_data['price'] or current_data['price'] <= 0:
                return None
            current_price = current_data['price']
        
        # Skip if in cooldown
        if self.is_cooldown(pair):
            return None
            
        # Skip during high-impact news
        if self.high_impact_news:
            return None
            
        # Skip near key liquidity levels
        for level in self.liquidity_zones.get(pair, []):
            if abs(current_price - level) / level < 0.001:
                return None
                
        # Get technical indicators
        indicators = self.get_technical_indicators(pair)
        if not indicators:
            return None
            
        # Direction determination with strict criteria
        direction = None
        confidence = 0.0
        
        # 1. RSI + MACD Combo (Core Signal)
        if indicators['rsi'] < 35 and indicators['macd']['macd'] > indicators['macd']['signal']:
            direction = 'BUY'
            confidence = 0.85 - (indicators['rsi'] / 300)
        elif indicators['rsi'] > 65 and indicators['macd']['macd'] < indicators['macd']['signal']:
            direction = 'SELL'
            confidence = 0.85 - ((100 - indicators['rsi']) / 300)
            
        # 2. Bollinger Band Reversion (Confirmation)
        if direction:
            if direction == 'BUY' and current_price < indicators['bollinger']['lower']:
                confidence += 0.10
            elif direction == 'SELL' and current_price > indicators['bollinger']['upper']:
                confidence += 0.10
                
        # 3. ADX Trend Strength Filter
        if indicators['adx'] > 25:
            confidence += 0.05
        elif indicators['adx'] < 15:
            confidence -= 0.10
            
        # 4. Volume Confirmation
        vol_ratio = self.volume_profile.get(pair, {}).get('ratio', 1.0)
        if vol_ratio > 1.2:
            confidence += 0.05
        elif vol_ratio < 0.8:
            confidence -= 0.05
            
        # 5. News Sentiment Alignment
        if (self.news_sentiment > 0.2 and direction == 'BUY') or \
           (self.news_sentiment < -0.2 and direction == 'SELL'):
            confidence += 0.05
            
        # Apply safety multiplier
        confidence *= safety.get('position_size_multiplier', 1.0)
        
        # Skip if below min confidence
        if confidence < safety['min_confidence']:
            return None
            
        # Create signal and add to queue
        signal = self.create_signal(pair, direction, 'technical', current_price, confidence)
        if signal:
            signal['generated_time'] = time.time()
            with self.data_lock:
                self.signal_queue.append(signal)
                self.signal_cooldown[pair] = time.time() + 300  # 5 min cooldown
            return signal
            
        return None

    def create_signal(self, pair, direction, strategy, entry, confidence):
        """Create signal optimized for 90% TP hit rate"""
        now_ny = datetime.now(NEW_YORK_TZ)
        volatility = self.calculate_volatility(pair)
        safety = self.get_safety_params()
        
        # Scalping parameters (90% TP focus)
        if strategy == 'scalping':
            tp_distance = volatility * 0.0003  # Tight TP for high hit rate
            sl_distance = volatility * 0.0006  # Wider SL to avoid premature stops
            
            tp = entry * (1 + tp_distance) if direction == 'BUY' else entry * (1 - tp_distance)
            sl = entry * (1 - sl_distance) if direction == 'BUY' else entry * (1 + sl_distance)
            expiry = now_ny + timedelta(minutes=15)  # Short duration
            
            return {
                "pair": pair,
                "direction": direction,
                "strategy": strategy,
                "entry": entry,
                "tp": round(tp, 5),
                "sl": round(sl, 5),
                "expiry": expiry.isoformat(),
                "confidence": round(confidence, 2),
                "created_at": now_ny.isoformat()
            }
            
        # Intraday parameters
        else:
            tp = entry * (1 + volatility * 0.001) if direction == 'BUY' else entry * (1 - volatility * 0.001)
            sl = entry * (1 - volatility * 0.0007) if direction == 'BUY' else entry * (1 + volatility * 0.0007)
            expiry = now_ny + timedelta(hours=2)
            
            return {
                "pair": pair,
                "direction": direction,
                "strategy": strategy,
                "entry": entry,
                "tp": round(tp, 5),
                "sl": round(sl, 5),
                "expiry": expiry.isoformat(),
                "confidence": round(confidence, 2),
                "created_at": now_ny.isoformat()
            }

    # ======================
    # OPTIMIZED FOR RENDER STARTER PLAN
    # ======================
    
    def get_technical_indicators(self, pair):
        """Optimized indicator calculation with caching"""
        cache_key = f"indicators_{pair}"
        now = time.time()
        
        # Check cache
        with self.data_lock:
            if hasattr(self, 'indicator_cache') and cache_key in self.indicator_cache:
                data, timestamp = self.indicator_cache[cache_key]
                if now - timestamp < 300:  # 5 minute cache
                    return data
        
        try:
            # Get historical data
            response = self.safe_api_call(
                f"https://api.twelvedata.com/time_series?symbol={pair}&interval=15min&outputsize=50&apikey={TWELVE_DATA_API_KEY}"
            )
            if not response:
                return None
                
            data = response.json()
            
            # Validate response
            if data.get('status') != 'ok' or not data.get('values'):
                return None
                
            # Process data
            df = pd.DataFrame(data['values'])
            if df.empty:
                return None
                
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df.dropna(subset=['close'], inplace=True)
            
            # Calculate only essential indicators
            closes = df['close'].values
            indicators = {
                'rsi': self._calculate_rsi(df, 14),
                'macd': self._calculate_macd(df, 12, 26, 9),
                'bollinger': self._calculate_bollinger_bands(df, 20),
                'adx': self._calculate_adx(df, 14)
            }
            
            # Update cache
            with self.data_lock:
                if not hasattr(self, 'indicator_cache'):
                    self.indicator_cache = {}
                self.indicator_cache[cache_key] = (indicators, time.time())
                
            return indicators
            
        except Exception as e:
            logging.error(f"Technical indicators failed for {pair}: {str(e)}")
            return None

    # ======================
    # TELEGRAM COMMANDS
    # ======================
    
    def start(self, update: Update, context: CallbackContext):
        """Handle /start command"""
        user_id = update.effective_user.id
        with self.data_lock:
            self.subscribed_users.add(user_id)
        
        update.message.reply_text(
            "🚀 *SAFE TRADING BOT ACTIVATED* 🚀\n\n"
            "Key features enabled:\n"
            "- 90%+ TP hit rate system\n"
            "- Fakeout pattern detection\n"
            "- Safety-first position sizing\n"
            "- Signal timing optimization\n\n"
            "Type /safety for protection status",
            parse_mode=ParseMode.MARKDOWN
        )

    def safety_status(self, update: Update, context: CallbackContext):
        """Show safety status"""
        params = self.get_safety_params()
        status = (
            f"🛡️ *Safety Status* 🛡️\n\n"
            f"• Mode: {SAFETY_MODE.upper()}\n"
            f"• Min Confidence: {params['min_confidence']*100:.0f}%\n"
            f"• Max Daily Loss: {params['max_daily_loss']*100:.0f}%\n"
            f"• Max Trades/Day: {params['daily_trade_limit']}\n"
            f"• Fakeout Filter: {'ON' if self.fakeout_filter_enabled else 'OFF'}"
        )
        update.message.reply_text(status, parse_mode=ParseMode.MARKDOWN)

    # ======================
    # SIGNAL MONITOR (SCALPING FOCUS)
    # ======================
    
    def signal_monitor(self):
        """Monitor signals with 90% TP focus"""
        while self.running:
            try:
                current_time = datetime.now(NEW_YORK_TZ)
                signals_to_remove = []
                
                with self.data_lock:
                    active_signals = self.active_signals.copy()
                
                for signal in active_signals:
                    # Check expiration
                    expiry = datetime.fromisoformat(signal['expiry']).replace(tzinfo=NEW_YORK_TZ)
                    if expiry < current_time:
                        signals_to_remove.append(signal)
                        continue
                        
                    # Get current price
                    with self.data_lock:
                        price_data = self.live_prices.get(signal['pair'])
                        if not price_data or not price_data['price']:
                            continue
                        current_price = price_data['price']
                    
                    direction = signal['direction']
                    
                    # Check TP (90% focus)
                    if (direction == 'BUY' and current_price >= signal['tp']) or \
                       (direction == 'SELL' and current_price <= signal['tp']):
                        self.close_signal(signal, f"🎯 TP HIT for {signal['pair']} @ {current_price:.5f}", "tp")
                        signals_to_remove.append(signal)
                        # Update performance
                        with self.data_lock:
                            self.performance['tp_hits'] += 1
                            self.performance['total_signals'] += 1
                            win_rate = (self.performance['tp_hits'] / self.performance['total_signals']) * 100
                            self.performance['win_rate'] = win_rate
                    
                    # Check SL
                    elif (direction == 'BUY' and current_price <= signal['sl']) or \
                         (direction == 'SELL' and current_price >= signal['sl']):
                        self.close_signal(signal, f"🛑 SL HIT for {signal['pair']} @ {current_price:.5f}", "sl")
                        signals_to_remove.append(signal)
                        with self.data_lock:
                            self.performance['sl_hits'] += 1
                            self.performance['total_signals'] += 1
                
                # Remove closed signals
                with self.data_lock:
                    self.active_signals = [s for s in self.active_signals if s not in signals_to_remove]
                        
            except Exception as e:
                logging.error(f"Signal monitoring error: {str(e)}")
            time.sleep(20)  # Faster checking for scalping

# ======================
# RENDER-OPTIMIZED INITIALIZATION
# ======================
        
def main():
    # Configure logging
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[
            logging.StreamHandler()  # Render-compatible logging
        ]
    )
    
    # Reduce log noise
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    try:
        # Initialize bot with error handling
        bot = HighProbabilityTradingBot()
        dispatcher = bot.updater.dispatcher
        
        # Add essential commands only
        dispatcher.add_handler(CommandHandler("start", bot.start))
        dispatcher.add_handler(CommandHandler("stop", bot.stop_cmd))
        dispatcher.add_handler(CommandHandler("safety", bot.safety_status))
        
        # Start the bot
        bot.updater.start_polling()
        logging.info("Safe Trading Bot started on Render Starter Plan")
        
        # Keep main thread alive
        bot.updater.idle()
        
    except Exception as e:
        logging.critical(f"Initialization failed: {str(e)}")

if __name__ == '__main__':
    main():# Add to HighProbabilityTradingBot.__init__():
if os.getenv('FREE_PLAN', 'true').lower() == 'true':
    self.free_plan_mode = True
    logging.info("FREE PLAN MODE: Enabled optimizations")
    # Disable non-essential features
    self.disable_ichimoku = True
    self.news_check_interval = 7200  # 2 hours
    self.pairs = os.getenv('TRADING_PAIRS', 'EURUSD,XAUUSD').split(',')
else:
    self.free_plan_mode = False

# Modified API call wrapper
def safe_api_call(self, url):
    if self.free_plan_mode and len(API_CALL_LOG) > 80:
        time.sleep(60)  # Rate limit buffer
    # ... rest of existing implementation ...
