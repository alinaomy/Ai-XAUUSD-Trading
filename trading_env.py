try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces
import numpy as np
import pandas as pd
from market_regime_detector import MarketRegimeDetector, MarketRegime

class TradingEnv(gym.Env):
    """
    Custom Gym environment for XAUUSD trading with trailing stops and dynamic exits.
    """
    def __init__(self, df, initial_balance=1000, transaction_cost=0, leverage=50, stop_loss_pct=0.02):
        super(TradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.leverage = leverage
        self.stop_loss_pct = stop_loss_pct

        # Trailing stop parameters - TIGHTENED for better profit capture
        self.trailing_stop_pct = 0.025  # Reduced from 5% to 2.5% trailing stop
        self.trailing_stop_distance = 0  # Current trailing stop level
        self.highest_price_since_entry = 0  # Track highest price for trailing stops

        # Profit taking parameters - MULTIPLE SCALED TARGETS
        self.profit_targets = [0.01, 0.02, 0.05, 0.10]  # 1%, 2%, 5%, 10% profit targets
        self.take_profit_pct = 0.10  # 10% take profit target (final target)
        self.partial_take_profit_pct = 0.02  # Take partial profits at 2% (reduced from 5%)

        # Breakeven stop parameters
        self.breakeven_trigger_pct = 0.015  # Move to breakeven after 1.5% profit
        self.breakeven_activated = False

        # Dynamic exit parameters
        self.max_holding_period = 24  # Max hours to hold position
        self.entry_time = None

        # Calculate technical indicators
        self._calculate_indicators()

        # Action space: continuous action between -1 and 1
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Observation space: price history (last 10 closes), RSI, MACD, MACD_signal, position, balance
        self.lookback = 10
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.lookback + 5,), dtype=np.float32  # 10 prices + RSI + MACD + MACD_signal + position + balance
        )

        self.trades = []  # Log all trades

        self.reset()

    def _calculate_indicators(self):
        """Calculate technical indicators for the dataset"""
        # RSI
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = self.df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = self.df['Close'].ewm(span=26, adjust=False).mean()
        self.df['MACD'] = ema12 - ema26
        self.df['MACD_signal'] = self.df['MACD'].ewm(span=9, adjust=False).mean()

        self.df.fillna(0, inplace=True)  # Fill NaN with 0
        self.trailing_stop_distance = 0  # Current trailing stop level
        self.highest_price_since_entry = 0  # Track highest price for trailing stops

        # Initialize market regime detector
        self.regime_detector = MarketRegimeDetector()

        # Dynamic regime-adaptive parameters (will be updated based on current regime)
        self._update_regime_parameters()

        # Add technical indicators

        # Add technical indicators
        self._calculate_indicators()

        # Action space: continuous action between -1 and 1
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Observation space: price history (last 10 closes), RSI, MACD, MACD_signal, position, balance
        self.lookback = 10
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.lookback + 5,), dtype=np.float32  # 10 prices + RSI + MACD + MACD_signal + position + balance
        )

        self.trades = []  # Log all trades

        self.reset()

    def reset(self):
        self.current_step = self.lookback
        self.balance = self.initial_balance
        self.position = 0  # 0: no position, positive: long amount
        self.entry_price = 0
        self.total_profit = 0
        self.done = False
        self.trades = []  # Reset trades

        # Reset trailing stop variables
        self.trailing_stop_distance = 0
        self.highest_price_since_entry = 0
        self.entry_time = None

        return self._get_observation()

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Balance: {self.balance}, Position: {self.position}, Total Profit: {self.total_profit}")

    def _update_regime_parameters(self):
        """Update trading parameters based on current market regime"""
        # Detect current regime
        current_regime, regime_params = self.regime_detector.detect_regime(self.df, self.current_step)

        # Update trading parameters based on regime
        self.profit_targets = regime_params['profit_targets']
        self.trailing_stop_pct = regime_params['trailing_stop_pct']
        self.take_profit_pct = regime_params['profit_targets'][-1]  # Last target is final take profit
        self.partial_take_profit_pct = regime_params['profit_targets'][1]  # Second target for partial profit
        self.breakeven_trigger_pct = regime_params['breakeven_trigger']
        self.max_holding_period = regime_params['max_holding_time']

        # Store current regime for logging
        self.current_regime = current_regime

    def _get_observation(self):
        prices = self.df.loc[self.current_step - self.lookback:self.current_step - 1, 'Close'].values
        rsi = self.df.loc[self.current_step - 1, 'RSI']
        macd = self.df.loc[self.current_step - 1, 'MACD']
        macd_signal = self.df.loc[self.current_step - 1, 'MACD_signal']
        return np.concatenate([prices, [rsi, macd, macd_signal, self.position, self.balance]])

    def step(self, action, confidence=1.0):
        """
        Execute action with confidence-based position sizing and dynamic exits
        confidence: float between 0 and 1, higher = more confident
        """
        # Update regime parameters dynamically
        self._update_regime_parameters()

        current_price = self.df.loc[self.current_step, 'Close']
        reward = 0

        # Apply confidence to action magnitude
        if isinstance(action, np.ndarray):
            action = action[0]

        # Scale action by confidence (minimum confidence threshold)
        min_confidence = 0.3  # Minimum confidence to trade
        if confidence < min_confidence:
            action = 0  # Force hold if not confident enough

        effective_action = action * confidence

        # Check for dynamic exits first (trailing stops, profit taking, time limits)
        exit_reason = self._check_dynamic_exits(current_price)
        if exit_reason:
            profit = (current_price - self.entry_price) * self.position
            self.balance += profit
            self.total_profit += profit

            # Reward based on exit reason
            if exit_reason == 'trailing_stop':
                reward = profit / self.initial_balance * 30  # Moderate reward for trailing stop
            elif exit_reason == 'take_profit':
                reward = profit / self.initial_balance * 100  # High reward for profit taking
            elif exit_reason == 'partial_profit':
                reward = profit / self.initial_balance * 50   # Good reward for partial profits
            elif exit_reason == 'max_time':
                reward = profit / self.initial_balance * 10   # Low reward for time-based exit
            else:
                reward = profit / self.initial_balance * 20   # Default reward

            self.trades.append({
                'step': self.current_step,
                'action': 'exit',
                'reason': exit_reason,
                'price': current_price,
                'profit': profit,
                'confidence': confidence
            })

            # Reset position
            self.position = 0
            self.entry_price = 0
            self.trailing_stop_distance = 0
            self.highest_price_since_entry = 0
            self.entry_time = None

        # Execute new trades if no position
        elif self.position == 0:
            if effective_action > 0.1:  # Buy signal
                # Position size based on action magnitude and confidence
                position_multiplier = min(effective_action, 1.0)
                confidence_multiplier = confidence ** 0.5
                self.position = self.balance * self.leverage * position_multiplier * confidence_multiplier / current_price
                self.entry_price = current_price
                self.entry_time = self.current_step
                self.highest_price_since_entry = current_price
                self.trailing_stop_distance = current_price * (1 - self.trailing_stop_pct)
                self.breakeven_activated = False  # Reset breakeven flag

                self.trades.append({
                    'step': self.current_step,
                    'action': 'buy',
                    'price': current_price,
                    'position': self.position,
                    'confidence': confidence
                })

            elif effective_action < -0.1:  # Sell signal (short)
                # Position size based on action magnitude and confidence
                position_multiplier = min(abs(effective_action), 1.0)
                confidence_multiplier = confidence ** 0.5
                self.position = -self.balance * self.leverage * position_multiplier * confidence_multiplier / current_price
                self.entry_price = current_price
                self.entry_time = self.current_step
                self.highest_price_since_entry = current_price  # For short positions, track lowest
                self.trailing_stop_distance = current_price * (1 + self.trailing_stop_pct)
                self.breakeven_activated = False  # Reset breakeven flag

                self.trades.append({
                    'step': self.current_step,
                    'action': 'sell_short',
                    'price': current_price,
                    'position': self.position,
                    'confidence': confidence
                })

        # Update trailing stops for existing positions
        else:
            self._update_trailing_stops(current_price)

        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            self.done = True

        next_obs = self._get_observation()
        return next_obs, reward, self.done, {}

    def _check_dynamic_exits(self, current_price):
        """Check for various exit conditions with improved profit-taking"""
        if self.position == 0:
            return None

        # Calculate current profit percentage
        if self.position > 0:  # Long position
            profit_pct = (current_price - self.entry_price) / self.entry_price
        else:  # Short position
            profit_pct = (self.entry_price - current_price) / self.entry_price

        # Breakeven stop activation
        if not self.breakeven_activated and profit_pct >= self.breakeven_trigger_pct:
            self.breakeven_activated = True
            # Move trailing stop to breakeven + small buffer
            buffer_pct = 0.005  # 0.5% buffer above breakeven
            if self.position > 0:
                self.trailing_stop_distance = self.entry_price * (1 + buffer_pct)
            else:
                self.trailing_stop_distance = self.entry_price * (1 - buffer_pct)

        # Scaled profit taking - take partial profits at multiple levels
        for target_pct in sorted(self.profit_targets, reverse=True):
            if profit_pct >= target_pct:
                # Calculate how much profit to take at this level
                if target_pct <= 0.02:  # Small profits (1-2%) - take 25% of position
                    profit_portion = 0.25
                    exit_reason = f'profit_{int(target_pct*100)}pct_partial'
                elif target_pct <= 0.05:  # Medium profits (5%) - take 50% of position
                    profit_portion = 0.50
                    exit_reason = f'profit_{int(target_pct*100)}pct_partial'
                else:  # Large profits (10%) - take full position
                    profit_portion = 1.0
                    exit_reason = 'take_profit'

                if profit_portion < 1.0:
                    # Partial exit
                    self.position *= (1 - profit_portion)
                    # Don't reset trailing stops for partial exits
                else:
                    # Full exit
                    self.position = 0
                    self._reset_position_state()

                return exit_reason

        # Trailing stop check (only if breakeven not activated or profit is positive)
        if self.position > 0:  # Long position
            if current_price <= self.trailing_stop_distance:
                self.position = 0
                self._reset_position_state()
                return 'trailing_stop'
        else:  # Short position
            if current_price >= self.trailing_stop_distance:
                self.position = 0
                self._reset_position_state()
                return 'trailing_stop'

        # Maximum holding time - more aggressive for losing positions
        if self.entry_time and (self.current_step - self.entry_time) >= self.max_holding_period:
            self.position = 0
            self._reset_position_state()
            return 'max_time'

        # Early exit for significant losses (stop loss)
        if profit_pct <= -0.03:  # 3% stop loss
            self.position = 0
            self._reset_position_state()
            return 'stop_loss'

        return None

    def _reset_position_state(self):
        """Reset position-related state variables"""
        self.trailing_stop_distance = 0
        self.highest_price_since_entry = 0
        self.entry_price = 0
        self.entry_time = None
        self.breakeven_activated = False

    def _update_trailing_stops(self, current_price):
        """Update trailing stop levels based on current price"""
        if self.position > 0:  # Long position
            if current_price > self.highest_price_since_entry:
                self.highest_price_since_entry = current_price
                self.trailing_stop_distance = current_price * (1 - self.trailing_stop_pct)
        else:  # Short position
            if current_price < self.highest_price_since_entry:
                self.highest_price_since_entry = current_price
                self.trailing_stop_distance = current_price * (1 + self.trailing_stop_pct)

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Balance: {self.balance}, Position: {self.position}, Total Profit: {self.total_profit}")