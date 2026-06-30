#!/usr/bin/env python3
"""
Train scalping ensemble (PPO + SAC + TD3) on M5 XAUUSD data.

Usage:
  python train_scalp_model.py                          # default: 500k steps, last 200k bars
  python train_scalp_model.py --timesteps 1000000      # longer training
  python train_scalp_model.py --bars 500000            # use more history
  python train_scalp_model.py --bars 0                 # use ALL 1.4M bars (slow)
  python train_scalp_model.py --model ppo              # single PPO only

Output: scalp_models/{ppo,sac,td3}_model.zip + scalp_models/scalp_config.json
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from trading_env_scalp import ScalpTradingEnv

SAVE_PATH = './scalp_models/'
DATA_FILE = 'xauusd_m5_data.csv'

# Hyperparameters tuned for M5 scalping
PPO_PARAMS = dict(
    learning_rate=1e-4,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.95,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1,
)

SAC_PARAMS = dict(
    learning_rate=3e-4,
    buffer_size=500_000,
    learning_starts=5_000,
    batch_size=256,
    gamma=0.95,
    tau=0.005,
    ent_coef='auto',
    verbose=1,
)

TD3_PARAMS = dict(
    learning_rate=3e-4,
    buffer_size=500_000,
    learning_starts=5_000,
    batch_size=256,
    gamma=0.95,
    tau=0.005,
    verbose=1,
)


# ── Progress callback ─────────────────────────────────────────────────────────

class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps: int, log_every: int = 10_000):
        super().__init__()
        self.total = total_timesteps
        self.log_every = log_every
        self.start = time.time()

    def _on_step(self) -> bool:
        if self.n_calls % self.log_every == 0:
            elapsed = time.time() - self.start
            pct = self.n_calls / self.total * 100
            eta = (elapsed / max(self.n_calls, 1)) * (self.total - self.n_calls)
            print(f"  [{pct:5.1f}%] {self.n_calls:>8,} steps | "
                  f"elapsed {elapsed/60:.1f}m | ETA {eta/60:.1f}m")
        return True


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(csv_path: str, max_bars: int) -> pd.DataFrame:
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)

    if max_bars > 0 and len(df) > max_bars:
        # Use the most recent bars — most relevant market conditions
        df = df.iloc[-max_bars:].reset_index(drop=True)
        print(f"Using last {max_bars:,} bars: {df['date'].iloc[0]} → {df['date'].iloc[-1]}")
    else:
        print(f"Using all {len(df):,} bars: {df['date'].iloc[0]} → {df['date'].iloc[-1]}")

    return df


def split_data(df: pd.DataFrame, train_ratio: float = 0.8):
    n = len(df)
    split = int(n * train_ratio)
    train_df = df.iloc[:split].reset_index(drop=True)
    test_df  = df.iloc[split:].reset_index(drop=True)
    print(f"Train: {len(train_df):,} bars | Test: {len(test_df):,} bars")
    return train_df, test_df


# ── Environment factory ───────────────────────────────────────────────────────

def make_env_fn(df: pd.DataFrame):
    def _init():
        return Monitor(ScalpTradingEnv(df))
    return _init


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(name: str, algo_class, params: dict,
                train_df: pd.DataFrame, test_df: pd.DataFrame,
                timesteps: int, save_path: str) -> object:
    print(f"\n{'='*55}")
    print(f"  Training {name.upper()} scalp model ({timesteps:,} steps)")
    print(f"{'='*55}")

    train_env = DummyVecEnv([make_env_fn(train_df)])
    model = algo_class('MlpPolicy', train_env, **params)

    callback = ProgressCallback(timesteps)
    model.learn(total_timesteps=timesteps, callback=callback, reset_num_timesteps=True)

    model_path = os.path.join(save_path, f'{name}_model')
    model.save(model_path)
    print(f"  Saved → {model_path}.zip")

    return model


def quick_eval(model, test_df: pd.DataFrame, name: str):
    """Run one episode on test data and print results."""
    env = ScalpTradingEnv(test_df)
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    trades = pd.DataFrame(env.trades)
    closed = trades[trades['action'] == 'close'] if not trades.empty else pd.DataFrame()

    print(f"\n  {name.upper()} eval on test set:")
    print(f"    Final balance : ${env.balance:.2f}")
    print(f"    Total profit  : ${env.total_profit:.2f}")
    if not closed.empty:
        wins = closed[closed['pnl'] > 0]
        print(f"    Trades        : {len(closed)}")
        print(f"    Win rate      : {len(wins)/len(closed)*100:.1f}%")
        print(f"    Avg PnL       : ${closed['pnl'].mean():.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train scalping ensemble on M5 data')
    parser.add_argument('--model', choices=['ppo', 'sac', 'td3', 'ensemble'],
                        default='ensemble', help='Model(s) to train')
    parser.add_argument('--timesteps', type=int, default=500_000,
                        help='Training timesteps per model (default: 500000)')
    parser.add_argument('--bars', type=int, default=200_000,
                        help='Max M5 bars to use for training. 0 = all 1.4M (default: 200000)')
    parser.add_argument('--csv', default=DATA_FILE,
                        help=f'M5 data CSV (default: {DATA_FILE})')
    args = parser.parse_args()

    os.makedirs(SAVE_PATH, exist_ok=True)

    df = load_data(args.csv, args.bars)
    train_df, test_df = split_data(df)

    models_to_train = {
        'ppo': (PPO, PPO_PARAMS),
        'sac': (SAC, SAC_PARAMS),
        'td3': (TD3, TD3_PARAMS),
    }

    if args.model != 'ensemble':
        models_to_train = {args.model: models_to_train[args.model]}

    trained = {}
    start_all = time.time()

    for name, (algo_class, params) in models_to_train.items():
        model = train_model(name, algo_class, params, train_df, test_df,
                            args.timesteps, SAVE_PATH)
        trained[name] = model
        quick_eval(model, test_df, name)

    # Save ensemble config
    config = {
        'models': list(trained.keys()),
        'weights': {name: 1.0 for name in trained},
        'config': {
            name: {'algorithm': name.upper(), 'policy': 'MlpPolicy', 'timesteps': args.timesteps}
            for name in trained
        },
        'timeframe': 'M5',
        'style': 'scalp',
        'bars_trained': args.bars or 'all',
    }
    # Save as ensemble_config.json so EnsembleTrader.load_ensemble() can find it
    config_path = os.path.join(SAVE_PATH, 'ensemble_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig saved → {config_path}")

    total_min = (time.time() - start_all) / 60
    print(f"\n{'='*55}")
    print(f"  Training complete in {total_min:.1f} minutes")
    print(f"  Models saved to: {SAVE_PATH}")
    print(f"\n  To run scalp live trading:")
    print(f"    python mt5_trading.py --mode live --style scalp")
    print(f"{'='*55}")


if __name__ == '__main__':
    main()
