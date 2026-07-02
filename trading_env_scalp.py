#!/usr/bin/env python3
"""
Scalping Trading Environment for M5 XAUUSD data.

RR-based take-profit system (3 partial exits):
  - TP1 at RR 0.5 → close 1/3, move SL to breakeven
  - TP2 at RR 1.0 → close 1/3, tighten trailing stop
  - TP3 at RR 2.0 → close final 1/3

  - SL:      0.15% hard stop
  - Trail:   0.1%  (tightens to 0.05% after TP2)
  - Max hold: 6 bars = 30 min on M5
  - Transaction cost: 2 pip spread per side

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

    # Risk parameters
    STOP_LOSS_PCT     = 0.0015   # 0.15% hard SL
    TRAILING_STOP_PCT = 0.001    # 0.1% initial trailing (tightens to 0.05% after TP2)
    BREAKEVEN_BUF     = 0.0002   # 0.02% buffer above entry when moving SL to BE
    MAX_HOLD_BARS     = 6        # 6 × 5 min = 30 minutes max
    TRANSACTION_COST  = 0.0002   # 2 pip spread per side
    LOOKBACK          = 20       # bars in observation window

    # 3 partial TPs at RR 0.5 / 1.0 / 2.0 — equal thirds each
    RR_TARGETS   = [0.5, 1.0, 2.0]
    TP_FRACTIONS = [1/3, 1/3, 1/3]

    # ATR spike threshold for news bar detection in training
    NEWS_ATR_MULTIPLIER = 2.5

    def __init__(self, df: pd.DataFrame, initial_balance: float = 1000.0, leverage: int = 50):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.leverage = leverage

        self._calculate_indicators()

        # Action: continuous [-1, 1]  negative=short, positive=long, ~0=hold
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation: 20 closes + 8 indicator/state features = 28
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.LOOKBACK + 8,), dtype=np.float32
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

        # ATR(14) for volatility normalisation and news detection
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
        self.current_step       = self.LOOKBACK
        self.balance            = self.initial_balance
        self.position           = 0.0    # remaining signed lots
        self.original_size      = 0.0    # signed lots at open (for partial sizing)
        self.remaining_fraction = 0.0    # fraction of original still open
        self.entry_price        = 0.0
        self.entry_step         = 0
        self.sl_price           = 0.0    # dynamic SL (moves to BE after TP1)
        self.tp_prices          = [0.0, 0.0, 0.0]
        self.tp_hit             = [False, False, False]
        self.trailing_stop      = 0.0
        self.highest_extreme    = 0.0
        self.breakeven_on       = False
        self.total_profit       = 0.0
        self.done               = False
        self.trades             = []
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        i = self.current_step
        row = self.df.iloc[i]
        close = row['Close']

        # Normalise last 20 closes by current close
        prices = self.df['Close'].iloc[i - self.LOOKBACK:i].values / close - 1.0

        # Signed remaining fraction: +1.0=full long, -1.0=full short, 0=flat
        # After TP1: ±0.67, after TP2: ±0.33
        direction = 1 if self.original_size > 0 else (-1 if self.original_size < 0 else 0)
        signed_remaining = self.remaining_fraction * direction

        # TP progress: 0 → 0.33 → 0.67 → 1.0 as each TP is hit
        tp_progress = sum(self.tp_hit) / 3.0

        indicators = np.array([
            (row['ema8'] - row['ema21']) / (close + 1e-9),
            row['rsi7'] / 100.0,
            row['atr14'] / (close + 1e-9),
            row['macd']     / (close + 1e-9),
            row['macd_sig'] / (close + 1e-9),
            row['stoch_k']  / 100.0,
            signed_remaining,   # position direction + size fraction
            tp_progress,        # how far through the TP cascade
        ], dtype=np.float32)

        return np.concatenate([prices.astype(np.float32), indicators])

    # ── News filter ──────────────────────────────────────────────────────────

    def _is_news_bar(self) -> bool:
        """Detect high-impact news via ATR spike — penalise entries during these bars."""
        i = self.current_step
        atr_now  = self.df['atr14'].iloc[i]
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

        if self.position != 0:
            # 1 — Partial TP exits (process all newly-reached TPs this bar)
            reward += self._check_partial_tps(current_price)

            # 2 — Update trailing stop on remaining position
            if self.position != 0:
                self._update_trailing_stop(current_price)

            # 3 — Full exit: SL / trailing stop / max hold
            if self.position != 0:
                exit_reason = self._check_full_exit(current_price)
                if exit_reason:
                    reward += self._close_remaining(current_price, exit_reason)

        else:
            # 4 — Entry if flat AND not a news bar
            if news_bar:
                reward = -0.0005
            elif action > 0.15:
                reward = self._open_position(action, current_price, direction=1)
            elif action < -0.15:
                reward = self._open_position(action, current_price, direction=-1)

        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            if self.position != 0:
                self._close_remaining(current_price, 'episode_end')
            self.done = True

        # gymnasium new API: (obs, reward, terminated, truncated, info)
        return self._obs(), reward, self.done, False, {}

    # ── Position management ──────────────────────────────────────────────────

    def _open_position(self, action: float, price: float, direction: int) -> float:
        cost = price * self.TRANSACTION_COST
        self.balance -= cost

        size = (self.balance * self.leverage * min(abs(action), 1.0)) / price
        signed_size = size * direction

        self.position           = signed_size
        self.original_size      = signed_size
        self.remaining_fraction = 1.0
        self.entry_price        = price
        self.entry_step         = self.current_step
        self.breakeven_on       = False
        self.highest_extreme    = price
        self.tp_hit             = [False, False, False]

        sl_dist = price * self.STOP_LOSS_PCT

        if direction == 1:   # long
            self.sl_price      = price - sl_dist
            self.tp_prices     = [price + sl_dist * rr for rr in self.RR_TARGETS]
            self.trailing_stop = price * (1 - self.TRAILING_STOP_PCT)
        else:                 # short
            self.sl_price      = price + sl_dist
            self.tp_prices     = [price - sl_dist * rr for rr in self.RR_TARGETS]
            self.trailing_stop = price * (1 + self.TRAILING_STOP_PCT)

        self.trades.append({
            'step': self.current_step, 'action': 'open',
            'direction': 'long' if direction == 1 else 'short',
            'price': price, 'size': size,
            'sl': self.sl_price,
            'tp1': self.tp_prices[0], 'tp2': self.tp_prices[1], 'tp3': self.tp_prices[2],
        })
        return 0.0

    def _check_partial_tps(self, price: float) -> float:
        """Process all newly-hit TP levels this bar. Returns total reward."""
        if self.position == 0:
            return 0.0

        direction    = 1 if self.position > 0 else -1
        total_reward = 0.0

        for i, (tp_price, fraction) in enumerate(zip(self.tp_prices, self.TP_FRACTIONS)):
            if self.tp_hit[i]:
                continue

            tp_reached = (direction == 1 and price >= tp_price) or \
                         (direction == -1 and price <= tp_price)
            if not tp_reached:
                continue

            self.tp_hit[i] = True
            close_size = abs(self.original_size) * fraction
            cost       = price * self.TRANSACTION_COST
            raw_pnl    = (price - self.entry_price) * direction * close_size
            net_pnl    = raw_pnl - cost

            self.balance      += net_pnl
            self.total_profit += net_pnl
            profit_pct         = net_pnl / self.initial_balance

            self.position -= direction * close_size
            self.remaining_fraction = (
                abs(self.position) / abs(self.original_size)
                if self.original_size != 0 else 0.0
            )

            # Reward scales with RR: TP1→×75, TP2→×100, TP3→×150
            rr = self.RR_TARGETS[i]
            total_reward += profit_pct * (50 + 50 * rr)

            self.trades.append({
                'step': self.current_step, 'action': 'partial_close',
                'tp_level': i + 1, 'rr': rr,
                'price': price, 'pnl': net_pnl, 'profit_pct': profit_pct,
            })

            # TP1 hit → move SL to breakeven (tiny buffer above/below entry)
            if i == 0:
                buf = self.entry_price * self.BREAKEVEN_BUF
                self.sl_price = (
                    self.entry_price + buf if direction == 1
                    else self.entry_price - buf
                )
                self.breakeven_on = True

            # TP2 hit → tighten trailing stop by half
            if i == 1:
                tight = self.TRAILING_STOP_PCT * 0.5
                if direction == 1:
                    self.trailing_stop = max(self.trailing_stop, price * (1 - tight))
                else:
                    self.trailing_stop = min(self.trailing_stop, price * (1 + tight))

        return total_reward

    def _check_full_exit(self, price: float):
        """Check SL, trailing stop, and max hold for remaining position."""
        if self.position == 0:
            return None

        direction = 1 if self.position > 0 else -1

        if direction == 1 and price <= self.trailing_stop:
            return 'trailing_stop'
        if direction == -1 and price >= self.trailing_stop:
            return 'trailing_stop'

        if direction == 1 and price <= self.sl_price:
            return 'stop_loss'
        if direction == -1 and price >= self.sl_price:
            return 'stop_loss'

        if (self.current_step - self.entry_step) >= self.MAX_HOLD_BARS:
            return 'max_hold'

        return None

    def _close_remaining(self, price: float, reason: str) -> float:
        """Close whatever fraction of the position is still open."""
        if self.position == 0:
            return 0.0

        direction  = 1 if self.position > 0 else -1
        close_size = abs(self.position)
        cost       = price * self.TRANSACTION_COST
        raw_pnl    = (price - self.entry_price) * direction * close_size
        net_pnl    = raw_pnl - cost

        self.balance      += net_pnl
        self.total_profit += net_pnl
        profit_pct         = net_pnl / self.initial_balance
        tps_hit            = sum(self.tp_hit)

        # Penalise full SL hit harder than SL after already banking TP profits
        if reason == 'stop_loss' and tps_hit == 0:
            reward = profit_pct * 50
        elif reason == 'stop_loss':
            reward = profit_pct * 20
        elif reason == 'trailing_stop':
            reward = profit_pct * (80 if net_pnl > 0 else 30)
        elif reason == 'max_hold':
            reward = profit_pct * 20 - 0.001
        else:
            reward = profit_pct * 50

        self.trades.append({
            'step': self.current_step, 'action': 'close', 'reason': reason,
            'price': price, 'pnl': net_pnl, 'profit_pct': profit_pct,
        })

        # Reset position state
        self.position           = 0.0
        self.original_size      = 0.0
        self.remaining_fraction = 0.0
        self.entry_price        = 0.0
        self.entry_step         = 0
        self.sl_price           = 0.0
        self.tp_prices          = [0.0, 0.0, 0.0]
        self.tp_hit             = [False, False, False]
        self.trailing_stop      = 0.0
        self.highest_extreme    = 0.0
        self.breakeven_on       = False

        return float(reward)

    def _update_trailing_stop(self, price: float):
        if self.position > 0:
            if price > self.highest_extreme:
                self.highest_extreme = price
                new_stop = price * (1 - self.TRAILING_STOP_PCT)
                self.trailing_stop = max(self.trailing_stop, new_stop)
        elif self.position < 0:
            if price < self.highest_extreme:
                self.highest_extreme = price
                new_stop = price * (1 + self.TRAILING_STOP_PCT)
                self.trailing_stop = min(self.trailing_stop, new_stop)

    def render(self, mode='human'):
        tps = sum(self.tp_hit)
        print(
            f"Step {self.current_step} | Balance ${self.balance:.2f} | "
            f"Pos {self.position:.4f} ({self.remaining_fraction:.0%} remaining) | "
            f"TPs {tps}/3 | Profit ${self.total_profit:.2f}"
        )
