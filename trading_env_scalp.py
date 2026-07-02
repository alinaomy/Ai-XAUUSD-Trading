#!/usr/bin/env python3
"""
Scalping Trading Environment for M5 XAUUSD data.

Key differences from swing TradingEnv:
  - Profit targets: 0.1%, 0.2%, 0.3%, 0.5%  (vs 1-10% swing)
  - Stop loss:      0.15%                      (vs 2% swing)
  - Trailing stop:  0.1%                       (vs 2.5% swing)
  - Max hold:       6 bars = 30 min on M5      (vs 24 bars swing)
  - Transaction cost applied on every entry     (spread simulation)
  - Faster indicators: EMA8/21, RSI(7), ATR(14), Stoch(5,3)
  - Observation: 20-bar lookback (28 features total)
"""

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

import numpy as np
import pandas as pd


class ScalpTradingEnv(gym.Env):

    # Scalping-specific parameters
    PROFIT_TARGETS    = [0.001, 0.002, 0.003, 0.005]  # 0.1%, 0.2%, 0.3%, 0.5%
    STOP_LOSS_PCT     = 0.0015   # 0.15%
    TRAILING_STOP_PCT = 0.001    # 0.1%
    BREAKEVEN_TRIGGER = 0.0008   # 0.08% — move to breakeven fast
    MAX_HOLD_BARS     = 6        # 6 × 5 min = 30 minutes
    TRANSACTION_COST  = 0.0002   # 2 pip spread per side
    LOOKBACK          = 20       # bars in observation window

    # MTF features appended when use_mtf=True:
    #   m30_ema_cross, m30_rsi, m30_trend,          ← 3 × M30 trend bias
    #   m15_ema_cross, m15_rsi, m15_bos, m15_fvg    ← 4 × M15 setup
    MTF_FEATURES = 7

    def __init__(self, df: pd.DataFrame, initial_balance: float = 1000.0,
                 leverage: int = 50, use_mtf: bool = False):
        super().__init__()

        self.df      = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.use_mtf  = use_mtf

        self._calculate_indicators()

        # Action: continuous [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation: 20 closes + 8 base indicators [+ 7 MTF features if use_mtf]
        n_features = self.LOOKBACK + 8 + (self.MTF_FEATURES if use_mtf else 0)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32
        )

        self.trades = []
        self.reset()

    # ── Indicators ───────────────────────────────────────────────────────────

    def _calculate_indicators(self):
        df = self.df

        # Fast EMAs for crossover signal
        df['ema8']  = df['Close'].ewm(span=8,  adjust=False).mean()
        df['ema21'] = df['Close'].ewm(span=21, adjust=False).mean()

        # RSI(7) — faster than standard 14 for scalping
        delta = df['Close'].diff()
        gain  = delta.where(delta > 0, 0).rolling(7).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(7).mean()
        df['rsi7'] = 100 - (100 / (1 + gain / loss))

        # ATR(14) for volatility normalisation
        hl  = df['High'] - df['Low']
        hpc = (df['High'] - df['Close'].shift()).abs()
        lpc = (df['Low']  - df['Close'].shift()).abs()
        df['atr14'] = pd.concat([hl, hpc, lpc], axis=1).max(axis=1).rolling(14).mean()

        # MACD(12,26,9)
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd']     = ema12 - ema26
        df['macd_sig'] = df['macd'].ewm(span=9, adjust=False).mean()

        # Stochastic(5,3) — fast stochastic for scalping
        low5  = df['Low'].rolling(5).min()
        high5 = df['High'].rolling(5).max()
        df['stoch_k'] = 100 * (df['Close'] - low5) / (high5 - low5 + 1e-9)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()

        df.fillna(0, inplace=True)

    # ── Reset / Observation ──────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step    = self.LOOKBACK
        self.balance         = self.initial_balance
        self.position        = 0.0    # lot size (positive=long, negative=short)
        self.entry_price     = 0.0
        self.entry_step      = 0
        self.trailing_stop   = 0.0
        self.highest_extreme = 0.0    # highest for long, lowest for short
        self.breakeven_on    = False
        self.total_profit    = 0.0
        self.done            = False
        self.trades          = []
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        i = self.current_step
        row = self.df.iloc[i]
        close = row['Close']

        # Normalise last 20 closes by current close so prices are scale-free
        prices = self.df['Close'].iloc[i - self.LOOKBACK:i].values / close - 1.0

        indicators = np.array([
            (row['ema8'] - row['ema21']) / (close + 1e-9),
            row['rsi7'] / 100.0,
            row['atr14'] / (close + 1e-9),
            row['macd']     / (close + 1e-9),
            row['macd_sig'] / (close + 1e-9),
            row['stoch_k']  / 100.0,
            np.sign(self.position),
            (self.current_step - self.entry_step) / self.MAX_HOLD_BARS,
        ], dtype=np.float32)

        obs = np.concatenate([prices.astype(np.float32), indicators])

        if self.use_mtf:
            mtf = np.array([
                float(row.get('m30_ema_cross', 0.0)),  # M30 EMA cross → trend direction
                float(row.get('m30_rsi',       0.5)),  # M30 RSI
                float(row.get('m30_trend',     0.0)),  # M30 EMA slope sign
                float(row.get('m15_ema_cross', 0.0)),  # M15 EMA cross → setup bias
                float(row.get('m15_rsi',       0.5)),  # M15 RSI
                float(row.get('m15_bos',       0.0)),  # M15 break of structure
                float(row.get('m15_fvg',       0.0)),  # M15 fair value gap
            ], dtype=np.float32)
            obs = np.concatenate([obs, mtf])

        return obs

    # ATR spike threshold: if current ATR > this multiple of rolling mean → news bar
    NEWS_ATR_MULTIPLIER = 2.5

    def _is_news_bar(self) -> bool:
        """Detect high-impact news bars using ATR spike detection."""
        i = self.current_step
        atr_now = self.df['atr14'].iloc[i]
        # Rolling mean of ATR over past 50 bars (excluding current)
        atr_mean = self.df['atr14'].iloc[max(0, i - 50):i].mean()
        if atr_mean == 0:
            return False
        return (atr_now / atr_mean) > self.NEWS_ATR_MULTIPLIER

    # ── Step ─────────────────────────────────────────────────────────────────

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = float(action[0])

        current_price = float(self.df.iloc[self.current_step]['Close'])
        reward = 0.0
        news_bar = self._is_news_bar()

        # 1 — Check exits first (always exit even during news)
        exit_reason = self._check_exits(current_price)
        if exit_reason:
            reward = self._close_position(current_price, exit_reason)

        # 2 — Entry if flat AND not a news bar
        elif self.position == 0:
            if news_bar:
                reward = -0.0005   # small penalty for wanting to trade during news
            elif action > 0.15:
                reward = self._open_position(action, current_price, direction=1)
            elif action < -0.15:
                reward = self._open_position(action, current_price, direction=-1)

        # 3 — Manage open position
        else:
            self._update_trailing_stop(current_price)

        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            # Force-close any open position at episode end
            if self.position != 0:
                self._close_position(current_price, 'episode_end')
            self.done = True

        # gymnasium API: (obs, reward, terminated, truncated, info)
        return self._obs(), reward, self.done, False, {}

    # ── Position management ──────────────────────────────────────────────────

    def _open_position(self, action: float, price: float, direction: int) -> float:
        # Spread cost on entry
        cost = price * self.TRANSACTION_COST
        self.balance -= cost

        size = (self.balance * self.leverage * min(abs(action), 1.0)) / price
        self.position        = size * direction
        self.entry_price     = price
        self.entry_step      = self.current_step
        self.breakeven_on    = False
        self.highest_extreme = price

        if direction == 1:   # long
            self.trailing_stop = price * (1 - self.TRAILING_STOP_PCT)
        else:                 # short
            self.trailing_stop = price * (1 + self.TRAILING_STOP_PCT)

        self.trades.append({
            'step': self.current_step, 'action': 'open',
            'direction': 'long' if direction == 1 else 'short',
            'price': price, 'size': abs(self.position)
        })
        return 0.0

    def _close_position(self, price: float, reason: str) -> float:
        if self.position == 0:
            return 0.0

        # Spread cost on exit
        cost = price * self.TRANSACTION_COST
        direction = 1 if self.position > 0 else -1
        raw_pnl = (price - self.entry_price) * direction * abs(self.position)
        net_pnl = raw_pnl - cost

        self.balance      += net_pnl
        self.total_profit += net_pnl
        profit_pct         = net_pnl / self.initial_balance

        # Reward shaping
        if reason == 'take_profit':
            reward = profit_pct * 200    # Strong positive
        elif reason == 'trailing_stop' and net_pnl > 0:
            reward = profit_pct * 100
        elif reason == 'stop_loss':
            reward = profit_pct * 50     # Penalise but less than random loss
        elif reason == 'max_hold':
            reward = profit_pct * 20 - 0.001   # Time penalty
        else:
            reward = profit_pct * 50

        self.trades.append({
            'step': self.current_step, 'action': 'close', 'reason': reason,
            'price': price, 'pnl': net_pnl, 'profit_pct': profit_pct
        })

        # Reset
        self.position        = 0.0
        self.entry_price     = 0.0
        self.entry_step      = 0
        self.trailing_stop   = 0.0
        self.highest_extreme = 0.0
        self.breakeven_on    = False

        return float(reward)

    def _check_exits(self, price: float):
        if self.position == 0:
            return None

        direction = 1 if self.position > 0 else -1
        profit_pct = (price - self.entry_price) * direction / self.entry_price

        # Breakeven activation
        if not self.breakeven_on and profit_pct >= self.BREAKEVEN_TRIGGER:
            self.breakeven_on = True
            buf = self.entry_price * 0.0002   # 0.02% buffer
            self.trailing_stop = (self.entry_price + buf) if direction == 1 else (self.entry_price - buf)

        # Profit targets (exit fully at each level for scalping — no partials)
        for target in sorted(self.PROFIT_TARGETS, reverse=True):
            if profit_pct >= target:
                return 'take_profit'

        # Trailing stop
        if direction == 1 and price <= self.trailing_stop:
            return 'trailing_stop'
        if direction == -1 and price >= self.trailing_stop:
            return 'trailing_stop'

        # Hard stop loss
        if profit_pct <= -self.STOP_LOSS_PCT:
            return 'stop_loss'

        # Max hold time
        if (self.current_step - self.entry_step) >= self.MAX_HOLD_BARS:
            return 'max_hold'

        return None

    def _update_trailing_stop(self, price: float):
        if self.position > 0:   # Long
            if price > self.highest_extreme:
                self.highest_extreme = price
                new_stop = price * (1 - self.TRAILING_STOP_PCT)
                self.trailing_stop = max(self.trailing_stop, new_stop)
        else:                    # Short
            if price < self.highest_extreme:
                self.highest_extreme = price
                new_stop = price * (1 + self.TRAILING_STOP_PCT)
                self.trailing_stop = min(self.trailing_stop, new_stop)

    def render(self, mode='human'):
        print(f"Step {self.current_step} | Balance ${self.balance:.2f} | "
              f"Pos {self.position:.4f} | Profit ${self.total_profit:.2f}")
