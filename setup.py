"""
AI-Driven XAUUSD Trading System: Maximum Profitability Framework

A comprehensive reinforcement learning trading system for XAUUSD (Gold vs US Dollar)
featuring ensemble AI models, market regime detection, and advanced risk management.
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
def read_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="ai-xauusd-trading",
    version="1.0.0",
    author="JonusNattapong / Zombitx64",
    author_email="jonusnattapong@zombitx64.com",
    description="AI-Driven XAUUSD Trading System with Ensemble RL and Market Regime Adaptation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JonusNattapong/AI-XAUUSD-Trading",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements('requirements.txt'),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "sphinx>=7.0.0",
        ],
        "gpu": [
            "torch>=2.1.0",  # GPU version
            "tensorflow[and-cuda]>=2.15.0",
        ],
        "optimization": [
            "optuna>=3.4.0",
            "jax>=0.4.20",
        ],
    },
    entry_points={
        "console_scripts": [
            "ai-trading-demo=advanced_trading_demo:run_advanced_trading_demo",
            "ai-live-trading=live_ensemble_trading:start_live_trading",
            "ai-regime-analysis=market_regime_detector:analyze_current_regime",
        ],
    },
    include_package_data=True,
    package_data={
        "ai_xauusd_trading": [
            "models/*",
            "config/*",
            "data/*",
        ],
    },
    keywords=[
        "trading",
        "reinforcement-learning",
        "machine-learning",
        "finance",
        "forex",
        "gold",
        "xauusd",
        "algorithmic-trading",
        "ensemble-learning",
        "market-regime",
        "risk-management",
    ],
    project_urls={
        "Bug Reports": "https://github.com/JonusNattapong/AI-XAUUSD-Trading/issues",
        "Source": "https://github.com/JonusNattapong/AI-XAUUSD-Trading",
        "Documentation": "https://jonusnattapong.github.io/AI-XAUUSD-Trading",
        "Hugging Face": "https://huggingface.co/JonusNattapong/AI-XAUUSD-Trading",
        "White Paper": "https://github.com/JonusNattapong/AI-XAUUSD-Trading/blob/main/AI_XAUUSD_Trading_White_Paper.pdf",
    },
)