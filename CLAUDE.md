# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Setup
```bash
pip install -r requirements.txt
# Or with dev tools:
pip install -e ".[dev]"
```

### Data
```bash
python data_fetch.py          # Fetch XAUUSD daily data → xauusd_data.csv (GC=F via yfinance)
python download_models.py     # Download pre-trained models from Hugging Face
```

### Training
```bash
python train_model.py --model ensemble --timesteps 10000   # Train PPO+TD3+SAC ensemble
python train_model.py --model ppo --timesteps 10000        # Train single PPO model
python train_model.py --model transformer --timesteps 10000 # Train PPO with Transformer policy
python curriculum_training.py                              # Train with confidence-based reward callback
```

### Backtesting & Demos
```bash
python backtest.py                      # Backtest transformer model on test split
python ensemble_backtest.py             # Backtest full ensemble
python advanced_trading_demo.py         # Comprehensive demo with performance charts
python regime_adaptive_trading_demo.py  # Demo showing regime-adaptive behavior
python quick_demo.py                    # Fast smoke test
```

### Live Trading
```bash
python live_ensemble_trading.py         # Start live ensemble trading (real capital)
```

### Tests & Quality
```bash
pytest tests/                           # Run all tests
pytest tests/ -m "not slow"            # Skip slow tests
pytest tests/ -m unit                  # Unit tests only
python test_env.py                      # Smoke test the gym environment
black .                                 # Format code
flake8 .                                # Lint
mypy .                                  # Type check
```

### Analysis & Reporting
```bash
python results_analysis.py             # Analyze backtest results
python trading_performance_analysis.py # Generate performance charts
python capital_calculator.py           # Position sizing calculator
python maximum_profitability_summary.py # Print profitability summary
```

### TensorBoard
```bash
tensorboard --logdir ./tensorboard/    # Monitor training (logs written during train_model.py)
```

## Architecture

### Data Flow
```
data_fetch.py (yfinance GC=F) → xauusd_data.csv
    → TradingEnv / OptimalTimingTradingEnv (gym environments)
        → train_model.py (PPO/TD3/SAC via stable-baselines3)
            → ensemble_models/{ppo,td3,sac}_model.zip
                → EnsembleTrader (confidence-weighted voting)
                    → LiveEnsembleTrader (real-time via yfinance)
```

### Core Modules

**[trading_env.py](trading_env.py)** — Primary gym environment (`TradingEnv`). Observation: 10-bar price history + RSI + MACD + MACD signal + position + balance (15 features). Action: continuous [-1, 1]. Implements scaled profit-taking at [1%, 2%, 5%, 10%], 2.5% trailing stop, and breakeven protection at 1.5% profit. Depends on `market_regime_detector.py`.

**[optimal_timing_env.py](optimal_timing_env.py)** — Alternative gym environment with 3D action space (position size, entry threshold, exit threshold), 20-bar lookback, and TA-Lib indicators. Used for curriculum training focused on entry/exit timing.

**[ensemble_trader.py](ensemble_trader.py)** — `EnsembleTrader` trains/loads PPO, TD3, and SAC models and combines their predictions with confidence weighting. Saved to `./ensemble_models/`.

**[transformer_policy.py](transformer_policy.py)** — Custom SB3 policy using a 3-layer Transformer encoder (8 heads, d_model=64) as the feature extractor, wired into PPO via `TransformerTradingPolicy`.

**[market_regime_detector.py](market_regime_detector.py)** — Classifies market into 7 regimes (`MarketRegime` enum: STRONG_BULL, BULL_TREND, BEAR_TREND, STRONG_BEAR, RANGING, HIGH_VOLATILITY, LOW_VOLATILITY). `RegimeParameters` maps each regime to different profit targets, position size multipliers, and holding time limits.

**[live_ensemble_trading.py](live_ensemble_trading.py)** — `LiveEnsembleTrader` pulls live data via yfinance, runs the ensemble, applies regime-adaptive parameters, and executes orders. Logs to `ensemble_trading.log`.

**[curriculum_training.py](curriculum_training.py)** — SB3 `BaseCallback` subclass (`ConfidenceRewardCallback`) that augments rewards based on action magnitude as a confidence proxy during training.

### Pre-trained Models
- `ppo_trading_model.zip` — Single PPO model (root)
- `ppo_continuous_trading_model.zip` — PPO with continuous actions
- `dqn_trading_model.zip` — DQN model
- `ensemble_models/` — PPO, TD3, SAC models for the ensemble

### Key Parameters (defined in TradingEnv / LiveEnsembleTrader)
- Leverage: 50x default
- Stop loss: 2% of position
- Trailing stop: 2.5%
- Breakeven trigger: 1.5% profit
- Profit targets: [1%, 2%, 5%, 10%] (scaled partial exits)
- Max holding period: 24 hours

### External Dependencies
- `stable-baselines3` (PPO, TD3, SAC) + `gymnasium`/`gym`
- `yfinance` for both historical data (`GC=F`) and live price feeds
- `ta` / `pandas-ta` for technical indicators; `talib` used in `optimal_timing_env.py` (requires separate system install)
- `torch` for Transformer policy

### Training Data
`train_model.py` expects `xauusd_data.csv` in the project root (generated by `data_fetch.py`). The 80/20 train/test split is applied by index position, not by date.
