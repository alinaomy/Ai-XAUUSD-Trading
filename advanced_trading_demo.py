#!/usr/bin/env python3
"""
Advanced Trading System Demo
Showcases trailing stops, profit taking, and dynamic exit strategies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
from market_regime_detector import MarketRegimeDetector

class AdvancedTradingSystem:
    """
    Advanced trading system with trailing stops, profit taking, and dynamic exits
    """

    def __init__(self, capital=1000, leverage=50):
        self.capital = capital
        self.leverage = leverage

        # Trailing stop parameters - TIGHTENED for better profit capture
        self.trailing_stop_pct = 0.025  # Reduced from 5% to 2.5% trailing stop
        self.trailing_stop_distance = 0

        # Profit taking parameters - MULTIPLE SCALED TARGETS
        self.profit_targets = [0.01, 0.02, 0.05, 0.10]  # 1%, 2%, 5%, 10% profit targets
        self.take_profit_pct = 0.10  # 10% take profit
        self.partial_take_profit_pct = 0.02  # Take partial profits at 2% (reduced from 5%)

        # Breakeven stop parameters
        self.breakeven_trigger_pct = 0.015  # Move to breakeven after 1.5% profit
        self.breakeven_activated = False

        # Initialize market regime detector
        self.regime_detector = MarketRegimeDetector()

        # Risk management
        self.max_holding_time = 24  # Max hours
        self.min_confidence = 0.3

        # Performance tracking
        self.trades = []
        self.portfolio_values = []

        # Current position
        self.position = 0
        self.entry_price = 0
        self.entry_time = None
        self.highest_price_since_entry = 0

    def calculate_confidence(self, signal_strength, market_condition):
        """Calculate trading confidence"""
        base_confidence = min(abs(signal_strength), 1.0)
        market_multiplier = market_condition  # 0.5-1.5
        confidence = base_confidence * market_multiplier
        return max(self.min_confidence, min(1.0, confidence))

    def execute_trade(self, signal, confidence, price, timestamp):
        """Execute trade with advanced exit strategies"""
        # Update regime parameters dynamically
        self._update_regime_parameters(price, timestamp)

        pnl = 0
        exit_reason = None

        # Check for exits first
        if self.position != 0:
            exit_reason = self._check_exits(price, timestamp, signal)

            if exit_reason:
                # Calculate P&L
                if self.position > 0:  # Long
                    pnl = (price - self.entry_price) * self.position
                else:  # Short
                    pnl = (self.entry_price - price) * abs(self.position)

                self.capital += pnl
                self._record_trade(timestamp, price, pnl, exit_reason, confidence)

                # Reset position
                self.position = 0
                self.entry_price = 0
                self.entry_time = None
                self.trailing_stop_distance = 0
                self.breakeven_activated = False  # Reset breakeven flag
                self.highest_price_since_entry = 0

        # Enter new positions
        elif abs(signal) > 0.1 and confidence > 0.2:  # Lower thresholds
            position_size = self._calculate_position_size(signal, confidence, price)

            if signal > 0:  # Buy
                self.position = position_size
                self.entry_price = price
                self.entry_time = timestamp
                self.highest_price_since_entry = price
                self.trailing_stop_distance = price * (1 - self.trailing_stop_pct)
                self.breakeven_activated = False  # Reset breakeven flag
                self._record_trade(timestamp, price, 0, 'entry', confidence, 'BUY')

            elif signal < 0:  # Sell
                self.position = -position_size
                self.entry_price = price
                self.entry_time = timestamp
                self.highest_price_since_entry = price
                self.trailing_stop_distance = price * (1 + self.trailing_stop_pct)
                self.breakeven_activated = False  # Reset breakeven flag
                self._record_trade(timestamp, price, 0, 'entry', confidence, 'SELL')

        # Update trailing stops
        if self.position != 0:
            self._update_trailing_stops(price)

        self.portfolio_values.append(self.capital)
        return pnl, exit_reason

    def _check_exits(self, price, timestamp, current_signal):
        """Check various exit conditions with improved profit-taking"""
        # Calculate current profit percentage
        if self.position > 0:
            profit_pct = (price - self.entry_price) / self.entry_price
        else:
            profit_pct = (self.entry_price - price) / self.entry_price

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
                    return exit_reason
                else:
                    # Full exit
                    return exit_reason

        # Trailing stop check (only if breakeven not activated or profit is positive)
        if self.position > 0 and price <= self.trailing_stop_distance:
            return 'trailing_stop'
        elif self.position < 0 and price >= self.trailing_stop_distance:
            return 'trailing_stop'

        # Maximum holding time - more aggressive for losing positions
        if self.entry_time and (timestamp - self.entry_time).total_seconds() > self.max_holding_time * 3600:
            return 'max_time'

        # Early exit for significant losses (stop loss)
        if profit_pct <= -0.03:  # 3% stop loss
            return 'stop_loss'

        # Opposite signal (exit on strong reversal)
        if (self.position > 0 and current_signal < -0.3) or (self.position < 0 and current_signal > 0.3):
            return 'opposite_signal'

        return None

    def _update_regime_parameters(self, current_price, timestamp):
        """Update trading parameters based on current market regime"""
        # Create a simple DataFrame with recent price history for regime detection
        # In a real system, this would use more comprehensive market data

        # For demo purposes, create synthetic OHLC data around current price
        recent_prices = []
        base_price = current_price

        # Generate last 100 periods of synthetic data
        for i in range(100):
            # Add some realistic price movement
            change = np.random.normal(0, 0.005)  # 0.5% volatility
            price = base_price * (1 + change)
            recent_prices.append({
                'Close': price,
                'High': price * (1 + abs(np.random.normal(0, 0.002))),
                'Low': price * (1 - abs(np.random.normal(0, 0.002))),
                'Volume': np.random.randint(1000, 10000)
            })
            base_price = price

        # Create DataFrame for regime detection
        price_df = pd.DataFrame(recent_prices)

        try:
            # Detect current regime
            current_regime, regime_params = self.regime_detector.detect_regime(price_df, -1)

            # Update trading parameters based on regime
            self.profit_targets = regime_params['profit_targets']
            self.trailing_stop_pct = regime_params['trailing_stop_pct']
            self.take_profit_pct = regime_params['profit_targets'][-1]
            self.partial_take_profit_pct = regime_params['profit_targets'][1]
            self.breakeven_trigger_pct = regime_params['breakeven_trigger']
            self.max_holding_time = regime_params['max_holding_time']

            # Store current regime
            self.current_regime = current_regime

        except Exception as e:
            # If regime detection fails, use default parameters
            self._set_default_parameters()

    def _set_default_parameters(self):
        """Set default trading parameters"""
        self.profit_targets = [0.01, 0.02, 0.05, 0.10]
        self.trailing_stop_pct = 0.025
        self.take_profit_pct = 0.10
        self.partial_take_profit_pct = 0.02
        self.breakeven_trigger_pct = 0.015
        self.max_holding_time = 24

    def _update_trailing_stops(self, price):
        """Update trailing stop levels"""
        if self.position > 0:  # Long position
            if price > self.highest_price_since_entry:
                self.highest_price_since_entry = price
                self.trailing_stop_distance = price * (1 - self.trailing_stop_pct)
        else:  # Short position
            if price < self.highest_price_since_entry:
                self.highest_price_since_entry = price
                self.trailing_stop_distance = price * (1 + self.trailing_stop_pct)

    def _calculate_position_size(self, signal, confidence, price):
        """Calculate position size based on confidence"""
        base_size = self.capital * self.leverage * min(abs(signal), 1.0) / price
        confidence_multiplier = confidence ** 0.5  # Square root scaling
        return base_size * confidence_multiplier

    def _record_trade(self, timestamp, price, pnl, reason, confidence, action_type='EXIT'):
        """Record trade details"""
        trade = {
            'timestamp': timestamp,
            'price': price,
            'pnl': pnl,
            'reason': reason,
            'confidence': confidence,
            'action': action_type,
            'position_size': abs(self.position) if self.position != 0 else 0
        }
        self.trades.append(trade)

def generate_realistic_price_data(n_points=1000):
    """Generate realistic price data with trends and volatility"""
    base_price = 2000
    prices = [base_price]
    timestamps = [datetime.now() + timedelta(hours=i) for i in range(n_points)]

    # Generate price movements
    for i in range(1, n_points):
        # Trend component
        trend = 0.001 * np.sin(i * 0.01)  # Slow trend

        # Volatility component
        volatility = np.random.normal(0, 0.02)

        # Momentum component
        momentum = 0.1 * (prices[-1] - prices[-2]) / prices[-2] if len(prices) > 1 else 0

        # Combine components
        price_change = trend + volatility + momentum
        new_price = prices[-1] * (1 + price_change)

        # Ensure reasonable bounds
        new_price = max(1800, min(2200, new_price))
        prices.append(new_price)

    return pd.DataFrame({
        'timestamp': timestamps,
        'price': prices
    })

def generate_trading_signals(price_data):
    """Generate realistic trading signals"""
    signals = []
    confidences = []

    for i, row in price_data.iterrows():
        price = row['price']

        # Simple trend signal
        if i > 20:
            short_ma = price_data['price'].iloc[i-10:i].mean()
            long_ma = price_data['price'].iloc[i-20:i].mean()
            trend_signal = 2 * (short_ma - long_ma) / long_ma  # Amplified signal
        else:
            trend_signal = 0

        # Volatility-based confidence
        if i > 10:
            returns = price_data['price'].iloc[i-10:i].pct_change()
            volatility = returns.std()
            market_condition = max(0.5, min(1.5, 1 / (1 + volatility * 10)))
        else:
            market_condition = 1.0

        # Add some noise to signals
        noise = np.random.normal(0, 0.2)  # Increased noise
        signal = trend_signal + noise

        confidence = min(1.0, abs(signal) * market_condition)

        signals.append(signal)
        confidences.append(confidence)

    price_data['signal'] = signals
    price_data['confidence'] = confidences

    return price_data

def run_advanced_trading_demo():
    """Run comprehensive demo of advanced trading features"""
    print("🚀 Advanced Trading System Demo")
    print("=" * 50)
    print("Features:")
    print("• Confidence-based position sizing")
    print("• Trailing stops (5%)")
    print("• Profit taking (5% partial, 10% full)")
    print("• Maximum holding time (24 hours)")
    print("• Opposite signal exits")
    print()

    # Generate data
    price_data = generate_realistic_price_data(500)
    trading_data = generate_trading_signals(price_data)

    # Initialize trading system
    trader = AdvancedTradingSystem(capital=1000, leverage=50)

    # Execute trades
    print("Executing trades...")
    for i, row in trading_data.iterrows():
        pnl, exit_reason = trader.execute_trade(
            row['signal'],
            row['confidence'],
            row['price'],
            row['timestamp']
        )

        if i % 100 == 0:
            print(f"Processed {i}/{len(trading_data)} data points...")

    print("\n✅ Trading simulation completed!")
    print(f"Final Portfolio Value: ${trader.capital:.2f}")

    # Analyze results
    analyze_results(trader)

    # Plot results
    plot_advanced_results(trader, trading_data)

def analyze_results(trader):
    """Analyze trading performance"""
    if not trader.trades:
        print("No trades executed")
        return

    df_trades = pd.DataFrame(trader.trades)

    # Separate entries and exits
    entries = df_trades[df_trades['action'].isin(['BUY', 'SELL'])]
    exits = df_trades[df_trades['action'] == 'EXIT']

    print("\n📊 Performance Analysis:")
    print(f"Total Trades: {len(entries)}")
    print(f"Total Exits: {len(exits)}")

    if len(exits) > 0:
        profitable_trades = len(exits[exits['pnl'] > 0])
        win_rate = profitable_trades / len(exits) * 100
        avg_win = exits[exits['pnl'] > 0]['pnl'].mean() if profitable_trades > 0 else 0
        avg_loss = exits[exits['pnl'] < 0]['pnl'].mean() if len(exits[exits['pnl'] < 0]) > 0 else 0

        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Average Win: ${avg_win:.2f}")
        print(f"Average Loss: ${avg_loss:.2f}")

        # Exit reason analysis
        exit_reasons = exits['reason'].value_counts()
        print("\nExit Reasons:")
        for reason, count in exit_reasons.items():
            pct = count / len(exits) * 100
            print(f"  {reason}: {count} ({pct:.1f}%)")
def plot_advanced_results(trader, price_data):
    """Plot comprehensive results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Price chart with trades
    ax1.plot(price_data['timestamp'], price_data['price'], alpha=0.7)

    # Plot trades
    if trader.trades:
        df_trades = pd.DataFrame(trader.trades)
        entries = df_trades[df_trades['action'].isin(['BUY', 'SELL'])]

        buy_signals = entries[entries['action'] == 'BUY']
        sell_signals = entries[entries['action'] == 'SELL']

        ax1.scatter(buy_signals['timestamp'], buy_signals['price'],
                   color='green', marker='^', s=100, label='Buy')
        ax1.scatter(sell_signals['timestamp'], sell_signals['price'],
                   color='red', marker='v', s=100, label='Sell')

    ax1.set_title('Price Chart with Trade Signals')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)

    # Portfolio value over time
    ax2.plot(trader.portfolio_values)
    ax2.set_title('Portfolio Value Over Time')
    ax2.set_ylabel('Portfolio Value ($)')
    ax2.grid(True)

    # P&L distribution by exit reason
    if trader.trades:
        df_trades = pd.DataFrame(trader.trades)
        exits = df_trades[df_trades['action'] != 'entry']

        if len(exits) > 0:
            exit_reasons = exits['reason'].unique()
            pnl_by_reason = [exits[exits['reason'] == reason]['pnl'] for reason in exit_reasons]

            ax3.boxplot(pnl_by_reason, tick_labels=exit_reasons)
            ax3.set_title('P&L Distribution by Exit Reason')
            ax3.set_ylabel('P&L ($)')
            ax3.grid(True)

    # Confidence vs P&L
    if trader.trades:
        df_trades = pd.DataFrame(trader.trades)
        exits = df_trades[df_trades['action'] != 'entry']

        if len(exits) > 0:
            colors = ['green' if x > 0 else 'red' for x in exits['pnl']]
            ax4.scatter(exits['confidence'], exits['pnl'], c=colors, alpha=0.6)
            ax4.set_title('Confidence vs P&L')
            ax4.set_xlabel('Confidence')
            ax4.set_ylabel('P&L ($)')
            ax4.grid(True)

    plt.tight_layout()
    plt.savefig('advanced_trading_demo.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    run_advanced_trading_demo()