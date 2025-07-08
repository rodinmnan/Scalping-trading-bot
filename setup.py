from setuptools import setup

setup(
    name="scalping_bot",
    version="0.1",
    install_requires=[
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
        "python-telegram-bot>=20.3",
        "pytz>=2023.3",
        "numpy>=1.24.3",
        "talib-binary>=0.4.24"
    ],
)
