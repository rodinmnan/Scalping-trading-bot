from setuptools import setup

setup(
    name="scalping-bot",
    version="0.1",
    install_requires=[
        line.strip() for line in open("requirements.txt") 
        if line.strip() and not line.startswith("#")
    ],
)
