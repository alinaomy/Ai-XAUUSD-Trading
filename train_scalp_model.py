#!/usr/bin/env python3
"""
Train scalping ensemble (PPO + SAC + TD3) on M5 or M1 XAUUSD data.
Optimised for Apple Silicon (MPS) — Mac Mini M4 / MacBook Pro M-series.

Usage:
  python train_scalp_model.py --style scalp    --timesteps 5000000 --bars 0
  python train_scalp_model.py --style hf_scalp --timesteps 5000000 --bars 0
  python train_scalp_model.py --model td3      --timesteps 5000000 --bars 0   # resume single
  python train_scalp_model.py --device cpu                                     # force CPU

Output:
  scalp_models/    {ppo,sac,td3}_model.zip + ensemble_config.json
  hf_scalp_models/ {ppo,sac,td3}_model.zip + ensemble_config.json
"""

import argparse
import json
import os
import time

# Must be set before torch import — enables CPU fallback for MPS ops not yet implemented
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from trading_env_scalp import ScalpTradingEnv

STYLE_PATHS = {
    'scalp':    './scalp_models/',
    'hf_scalp': './hf_scalp_models/',
}
STYLE_DATA = {
    'scalp':    'xauusd_m5_data.csv',
    'hf_scalp': 'xauusd_m1_data.csv',
}
STYLE_TIMEFRAME = {
    'scalp':    'M5',
    'hf_scalp': 'M1',
}


def _detect_device(override: str = None) -> str:
    if override:
        return override
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def _print_device_info(device: str):
    print(f"\n  Device: {device.upper()}", end='')
    if device == 'mps':
        print(f"  (Apple Silicon GPU — MPS fallback enabled)")
    elif device == 'cuda':
        print(f"  ({torch.cuda.get_device_name(0)})")
    else:
        print(f"  (CPU)")


# ── Hyperparameters — tuned for Apple M4 (24 GB unified memory) ──────────────
#
# MPS has no CPU↔GPU copy overhead (unified memory), so large batch sizes are
# efficient. batch_size=1024 gives ~2× throughput vs 512 on MPS.
# n_steps=4096 with batch_size=1024 → 4 minibatches per PPO update (valid: 4096 % 1024 == 0).
#
# Each model gets its OWN policy_kwargs dict — SB3 mutates policy_kwargs in-place
# during SAC init (adds use_sde=True), which crashes TD3 if they share the same dict.

def _build_params(device: str) -> tuple:
    ppo = dict(
        learning_rate=3e-4,
        n_steps=4096,
        batch_size=1024,       # 2× vs default; MPS handles large batches efficiently
        n_epochs=10,
        gamma=0.95,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        policy_kwargs={'net_arch': [256, 256, 128]},
        device=device,
        verbose=1,
    )
    sac = dict(
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=10_000,
        batch_size=1024,
        gamma=0.95,
        tau=0.005,
        ent_coef='auto',
        policy_kwargs={'net_arch': [256, 256, 128]},
        device=device,
        verbose=1,
    )
    td3 = dict(
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=10_000,
        batch_size=1024,
        gamma=0.95,
        tau=0.005,
        policy_kwargs={'net_arch': [256, 256, 128]},
        device=device,
        verbose=1,
    )
    return ppo, sac, td3


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


# ── MTF feature generation ────────────────────────────────────────────────────

def add_mtf_features(df: pd.DataFrame, base_tf_minutes: int = 5) -> pd.DataFrame:
    """Resample base OHLCV to M15 and M30, compute HTF indicators, forward-fill to base.

    Adds 7 columns: m30_ema_cross, m30_rsi, m30_trend, m15_ema_cross, m15_rsi, m15_bos, m15_fvg
    These match ScalpTradingEnv.MTF_FEATURES (indices 28-34).
    """
    df = df.copy()
    df.index = pd.to_datetime(df['date'])

    def _resample(rule: str) -> pd.DataFrame:
        r = df[['Open', 'High', 'Low', 'Close', 'Volume']].resample(
            rule, label='right', closed='right'
        )
        htf = r.agg({'Open': 'first', 'High': 'max', 'Low': 'min',
                     'Close': 'last', 'Volume': 'sum'})
        return htf.dropna(subset=['Close'])

    def _compute_indicators(htf: pd.DataFrame) -> pd.DataFrame:
        htf = htf.copy()
        htf['ema8']  = htf['Close'].ewm(span=8,  adjust=False).mean()
        htf['ema21'] = htf['Close'].ewm(span=21, adjust=False).mean()
        delta = htf['Close'].diff()
        gain  = delta.where(delta > 0, 0).rolling(7).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(7).mean()
        htf['rsi7']  = 100 - (100 / (1 + gain / (loss + 1e-9)))
        htf['trend'] = np.sign(htf['ema21'].diff())
        # BOS: close breaks above rolling 5-bar swing high or below swing low
        swing_high = htf['High'].shift(1).rolling(5).max()
        swing_low  = htf['Low'].shift(1).rolling(5).min()
        htf['bos'] = np.where(htf['Close'] > swing_high, 1.0,
                     np.where(htf['Close'] < swing_low, -1.0, 0.0))
        # FVG: 3-candle imbalance (candle[i].low > candle[i-2].high = bullish gap)
        htf['fvg'] = np.where(htf['Low']  > htf['High'].shift(2),  1.0,
                     np.where(htf['High'] < htf['Low'].shift(2),  -1.0, 0.0))
        return htf

    m30 = _compute_indicators(_resample('30min'))
    m15 = _compute_indicators(_resample('15min'))

    def _merge(col_series: pd.Series, prefix: str) -> pd.Series:
        return col_series.reindex(df.index, method='ffill')

    df['m30_ema_cross'] = _merge((m30['ema8'] - m30['ema21']) / (m30['Close'] + 1e-9), 'm30')
    df['m30_rsi']       = _merge(m30['rsi7'] / 100.0, 'm30')
    df['m30_trend']     = _merge(m30['trend'], 'm30')
    df['m15_ema_cross'] = _merge((m15['ema8'] - m15['ema21']) / (m15['Close'] + 1e-9), 'm15')
    df['m15_rsi']       = _merge(m15['rsi7'] / 100.0, 'm15')
    df['m15_bos']       = _merge(m15['bos'], 'm15')
    df['m15_fvg']       = _merge(m15['fvg'], 'm15')

    df = df.fillna(0).reset_index(drop=True)
    print(f"  MTF features added (M30 + M15 → {len(df):,} base bars)")
    return df


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(csv_path: str, max_bars: int) -> pd.DataFrame:
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)

    if max_bars > 0 and len(df) > max_bars:
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

def make_env_fn(df: pd.DataFrame, use_mtf: bool = False):
    def _init():
        return Monitor(ScalpTradingEnv(df, use_mtf=use_mtf))
    return _init


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(name: str, algo_class, params: dict,
                train_df: pd.DataFrame, test_df: pd.DataFrame,
                timesteps: int, save_path: str, use_mtf: bool = False) -> object:
    print(f"\n{'='*55}")
    print(f"  Training {name.upper()} scalp model ({timesteps:,} steps)"
          + (" [+MTF]" if use_mtf else ""))
    print(f"{'='*55}")

    train_env = DummyVecEnv([make_env_fn(train_df, use_mtf=use_mtf)])
    model = algo_class('MlpPolicy', train_env, **params)

    callback = ProgressCallback(timesteps)
    model.learn(total_timesteps=timesteps, callback=callback, reset_num_timesteps=True)

    model_path = os.path.join(save_path, f'{name}_model')
    model.save(model_path)
    print(f"  Saved → {model_path}.zip")

    return model


def quick_eval(model, test_df: pd.DataFrame, name: str, use_mtf: bool = False):
    """Run one episode on test data and print results."""
    env = ScalpTradingEnv(test_df, use_mtf=use_mtf)
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
    parser = argparse.ArgumentParser(description='Train scalping ensemble on M5 or M1 data')
    parser.add_argument('--style', choices=['scalp', 'hf_scalp'], default='scalp',
                        help='scalp=M5 → scalp_models/  |  hf_scalp=M1 → hf_scalp_models/')
    parser.add_argument('--model', choices=['ppo', 'sac', 'td3', 'ensemble'],
                        default='ensemble', help='Model(s) to train (default: ensemble = all 3)')
    parser.add_argument('--timesteps', type=int, default=500_000,
                        help='Training timesteps per model (default: 500000)')
    parser.add_argument('--bars', type=int, default=200_000,
                        help='Max bars to use. 0 = all available (default: 200000)')
    parser.add_argument('--csv', default=None,
                        help='Override data CSV (default: auto by --style)')
    parser.add_argument('--device', default=None,
                        help='Force device: mps | cuda | cpu (default: auto-detect)')
    args = parser.parse_args()

    SAVE_PATH = STYLE_PATHS[args.style]
    DATA_FILE = args.csv or STYLE_DATA[args.style]
    TIMEFRAME = STYLE_TIMEFRAME[args.style]

    device = _detect_device(args.device)
    _print_device_info(device)

    PPO_PARAMS, SAC_PARAMS, TD3_PARAMS = _build_params(device)

    os.makedirs(SAVE_PATH, exist_ok=True)

    print(f"\n  Style    : {args.style.upper()}")
    print(f"  Data     : {DATA_FILE}")
    print(f"  Output   : {SAVE_PATH}")
    print(f"  Timesteps: {args.timesteps:,} per model")

    df = load_data(DATA_FILE, args.bars)

    # hf_scalp uses MTF confluence in training: resample M1 → M15/M30, add 7 columns
    use_mtf = (args.style == 'hf_scalp')
    if use_mtf:
        base_tf = 1   # M1
        print(f"\n  Computing MTF features (M15, M30) from {base_tf}-min base ...")
        df = add_mtf_features(df, base_tf_minutes=base_tf)

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
                            args.timesteps, SAVE_PATH, use_mtf=use_mtf)
        trained[name] = model
        quick_eval(model, test_df, name, use_mtf=use_mtf)

    # Save ensemble config (timeframe, style and use_mtf are dynamic, not hardcoded)
    config = {
        'models': list(trained.keys()),
        'weights': {name: 1.0 for name in trained},
        'config': {
            name: {'algorithm': name.upper(), 'policy': 'MlpPolicy', 'timesteps': args.timesteps}
            for name in trained
        },
        'timeframe': TIMEFRAME,
        'style': args.style,
        'use_mtf': use_mtf,
        'obs_features': 35 if use_mtf else 28,
        'bars_trained': args.bars if args.bars > 0 else 'all',
    }
    config_path = os.path.join(SAVE_PATH, 'ensemble_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig saved → {config_path}")

    total_min = (time.time() - start_all) / 60
    print(f"\n{'='*55}")
    print(f"  Training complete in {total_min:.1f} minutes")
    print(f"  Models saved to: {SAVE_PATH}")
    print(f"\n  To run live trading:")
    print(f"    python mt5_trading.py --mode live --style {args.style}")
    print(f"{'='*55}")


if __name__ == '__main__':
    main()
