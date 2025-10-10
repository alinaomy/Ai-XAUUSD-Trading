#!/usr/bin/env python3
"""
Enhanced Trading Environment for Optimal Entry/Exit Timing
Focuses on teaching models to identify the best buy and sell points
"""

import gym
import numpy as np
import pandas as pd
from gym import spaces
import talib

class OptimalTimingTradingEnv(gym.Env):
    """
    Enhanced trading environment that teaches optimal entry/exit timing
    """

    def __init__(self, df, initial_balance=1000, leverage=50, transaction_cost=0.0002):
        super(OptimalTimingTradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.transaction_cost = transaction_cost

        # Enhanced technical indicators for timing
        self._add_advanced_indicators()

        # Action space: [position_size, entry_threshold, exit_threshold]
        # position_size: -1 to 1 (negative = short, positive = long)
        # entry_threshold: 0 to 1 (confidence threshold for entry)
        # exit_threshold: 0 to 1 (confidence threshold for exit)
        self.action_space = spaces.Box(
            low=np.array([-1, 0, 0]),
            high=np.array([1, 1, 1]),
            dtype=np.float32
        )

        # Enhanced observation space
        self.lookback = 20  # More historical data
        obs_size = self.lookback * 6 + 15  # prices + volumes + 13 indicators + position info
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        self.reset()

    def _add_advanced_indicators(self):
        """Add comprehensive technical indicators for timing analysis"""
        close = self.df['Close'].values
        high = self.df['High'].values
        low = self.df['Low'].values
        volume = self.df['Volume'].values

        # Trend indicators
        self.df['SMA_20'] = talib.SMA(close, timeperiod=20)
        self.df['SMA_50'] = talib.SMA(close, timeperiod=50)
        self.df['EMA_12'] = talib.EMA(close, timeperiod=12)
        self.df['EMA_26'] = talib.EMA(close, timeperiod=26)

        # Momentum indicators
        self.df['RSI'] = talib.RSI(close, timeperiod=14)
        self.df['MACD'], self.df['MACD_SIGNAL'], self.df['MACD_HIST'] = talib.MACD(close)
        self.df['STOCH_K'], self.df['STOCH_D'] = talib.STOCH(high, low, close)
        self.df['WILLR'] = talib.WILLR(high, low, close)
        self.df['CCI'] = talib.CCI(high, low, close, timeperiod=14)

        # Volatility indicators
        self.df['ATR'] = talib.ATR(high, low, close, timeperiod=14)
        self.df['BB_UPPER'], self.df['BB_MIDDLE'], self.df['BB_LOWER'] = talib.BBANDS(close)
        self.df['NATR'] = talib.NATR(high, low, close, timeperiod=14)

        # Volume indicators
        self.df['OBV'] = talib.OBV(close, volume)
        self.df['AD'] = talib.AD(high, low, close, volume)
        self.df['ADOSC'] = talib.ADOSC(high, low, close, volume)

        # Support/Resistance levels (simplified)
        self.df['PIVOT'] = (high + low + close) / 3
        self.df['R1'] = 2 * self.df['PIVOT'] - low
        self.df['S1'] = 2 * self.df['PIVOT'] - high

        # Fill NaN values
        self.df.fillna(method='bfill', inplace=True)
        self.df.fillna(0, inplace=True)

    def reset(self):
        self.current_step = self.lookback
        self.balance = self.initial_balance
        self.position = 0  # -1 (short), 0 (neutral), 1 (long)
        self.position_size = 0
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0
        self.total_pnl = 0
        self.trades = []
        self.holding_time = 0

        return self._get_observation()

    def _get_observation(self):
        """Create comprehensive observation for timing decisions"""
        start_idx = self.current_step - self.lookback
        end_idx = self.current_step

        # Price data (OHLCV)
        prices = self.df.iloc[start_idx:end_idx][['Open', 'High', 'Low', 'Close', 'Volume']].values.flatten()

        # Technical indicators
        indicators = self.df.iloc[self.current_step][[
            'RSI', 'MACD', 'MACD_SIGNAL', 'MACD_HIST',
            'STOCH_K', 'STOCH_D', 'WILLR', 'CCI',
            'ATR', 'BB_UPPER', 'BB_MIDDLE', 'BB_LOWER',
            'OBV', 'AD', 'PIVOT'
        ]].values

        # Position information
        position_info = np.array([
            self.position,  # Current position (-1, 0, 1)
            self.position_size,  # Position size (0-1)
            self.holding_time,  # Bars held
            (self.df.iloc[self.current_step]['Close'] - self.entry_price) / max(self.entry_price, 0.01) if self.entry_price > 0 else 0,  # Unrealized P&L %
            self.balance / self.initial_balance  # Normalized balance
        ])

        return np.concatenate([prices, indicators, position_info])

    def _calculate_timing_reward(self, action, current_price, next_price):
        """Calculate reward based on timing quality"""
        position_size, entry_threshold, exit_threshold = action

        reward = 0

        # Entry timing reward
        if self.position == 0 and abs(position_size) > entry_threshold:
            # Evaluate if this is a good entry point
            future_returns = []
            for i in range(1, min(11, len(self.df) - self.current_step)):  # Look ahead 10 bars
                future_price = self.df.iloc[self.current_step + i]['Close']
                ret = (future_price - current_price) / current_price
                future_returns.append(ret)

            if future_returns:
                avg_future_return = np.mean(future_returns[:5])  # 5-bar average
                if position_size > 0:  # Long position
                    reward += avg_future_return * 10  # Reward for buying before price increase
                else:  # Short position
                    reward -= avg_future_return * 10  # Reward for selling before price decrease

        # Exit timing reward
        elif self.position != 0 and abs(position_size) < exit_threshold:
            # Evaluate if this is a good exit point
            entry_return = (current_price - self.entry_price) / self.entry_price
            if self.position > 0:  # Was long
                reward += entry_return * 5  # Reward based on profit captured
            else:  # Was short
                reward -= entry_return * 5

        # Holding penalty (encourage active trading)
        if self.position != 0:
            reward -= 0.01  # Small penalty for holding

        return reward

    def step(self, action):
        position_size, entry_threshold, exit_threshold = action
        current_price = self.df.iloc[self.current_step]['Close']
        next_price = self.df.iloc[min(self.current_step + 1, len(self.df) - 1)]['Close']

        reward = 0
        done = False

        # Calculate timing-based reward
        timing_reward = self._calculate_timing_reward(action, current_price, next_price)
        reward += timing_reward

        # Execute trading logic
        if self.position == 0:  # No position
            if position_size > entry_threshold:  # Buy signal
                self.position = 1
                self.position_size = min(position_size, 1.0)
                self.entry_price = current_price
                self.holding_time = 0

                # Set stop loss and take profit based on ATR
                atr = self.df.iloc[self.current_step]['ATR']
                self.stop_loss = current_price - (atr * 2)
                self.take_profit = current_price + (atr * 3)

                reward += 0.1  # Small reward for entering position

            elif position_size < -entry_threshold:  # Sell signal
                self.position = -1
                self.position_size = min(abs(position_size), 1.0)
                self.entry_price = current_price
                self.holding_time = 0

                # Set stop loss and take profit
                atr = self.df.iloc[self.current_step]['ATR']
                self.stop_loss = current_price + (atr * 2)
                self.take_profit = current_price - (atr * 3)

                reward += 0.1  # Small reward for entering position

        else:  # Have position - check exit conditions
            self.holding_time += 1

            # Check stop loss / take profit
            if self.position == 1:  # Long
                if current_price <= self.stop_loss:
                    reward -= 1.0  # Penalty for hitting stop loss
                    self._close_position(current_price, "stop_loss")
                elif current_price >= self.take_profit:
                    reward += 2.0  # Reward for hitting take profit
                    self._close_position(current_price, "take_profit")
                elif position_size < -exit_threshold:  # Exit signal
                    pnl_pct = (current_price - self.entry_price) / self.entry_price
                    reward += pnl_pct * 5  # Reward based on profit
                    self._close_position(current_price, "signal_exit")

            else:  # Short
                if current_price >= self.stop_loss:
                    reward -= 1.0  # Penalty for hitting stop loss
                    self._close_position(current_price, "stop_loss")
                elif current_price <= self.take_profit:
                    reward += 2.0  # Reward for hitting take profit
                    self._close_position(current_price, "take_profit")
                elif position_size > exit_threshold:  # Exit signal
                    pnl_pct = (self.entry_price - current_price) / self.entry_price
                    reward += pnl_pct * 5  # Reward based on profit
                    self._close_position(current_price, "signal_exit")

            # Max holding time
            if self.holding_time >= 50:  # Max 50 bars
                pnl_pct = (current_price - self.entry_price) / self.entry_price * self.position
                reward += pnl_pct * 2  # Reward/penalty based on final P&L
                self._close_position(current_price, "max_time")

        # Move to next step
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True
            # Final position closure
            if self.position != 0:
                final_pnl = (next_price - self.entry_price) / self.entry_price * self.position
                reward += final_pnl * 3
                self._close_position(next_price, "end_episode")

        return self._get_observation(), reward, done, {}

    def _close_position(self, price, reason):
        """Close current position and record trade"""
        if self.position == 0:
            return

        pnl = (price - self.entry_price) * self.position * self.position_size * self.leverage
        pnl -= abs(pnl) * self.transaction_cost  # Transaction cost

        self.balance += pnl
        self.total_pnl += pnl

        trade = {
            'entry_price': self.entry_price,
            'exit_price': price,
            'position': 'LONG' if self.position == 1 else 'SHORT',
            'pnl': pnl,
            'holding_time': self.holding_time,
            'exit_reason': reason
        }
        self.trades.append(trade)

        # Reset position
        self.position = 0
        self.position_size = 0
        self.entry_price = 0
        self.holding_time = 0

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Balance: ${self.balance:.2f}, Position: {self.position}, Total P&L: ${self.total_pnl:.2f}")