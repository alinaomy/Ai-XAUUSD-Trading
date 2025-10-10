#!/usr/bin/env python3
"""
Regime-Adaptive Trading System Demo
Showcases how the trading system adapts to different market conditions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
from market_regime_detector import MarketRegimeDetector, MarketRegime

class RegimeAdaptiveTradingSystem:
    """
    Trading system that adapts parameters based on market regime
    """

    def __init__(self, capital=1000, leverage=50):
        self.capital = capital
        self.leverage = leverage

        # Initialize market regime detector
        self.regime_detector = MarketRegimeDetector()

        # Dynamic parameters (updated based on regime)
        self._set_default_parameters()

        # Performance tracking
        self.trades = []
        self.portfolio_values = []
        self.regime_history = []

        # Current position
        self.position = 0
        self.entry_price = 0
        self.entry_time = None
        self.highest_price_since_entry = 0
        self.breakeven_activated = False

    def _set_default_parameters(self):
        """Set default trading parameters"""
        self.profit_targets = [0.01, 0.02, 0.05, 0.10]
        self.trailing_stop_pct = 0.025
        self.take_profit_pct = 0.10
        self.partial_take_profit_pct = 0.02
        self.breakeven_trigger_pct = 0.015
        self.max_holding_time = 24
        self.min_confidence = 0.3

    def _update_regime_parameters(self, price_data):
        """Update trading parameters based on current market regime"""
        try:
            current_regime, regime_params = self.regime_detector.detect_regime(price_data, -1)

            # Update all trading parameters
            self.profit_targets = regime_params['profit_targets']
            self.trailing_stop_pct = regime_params['trailing_stop_pct']
            self.take_profit_pct = regime_params['profit_targets'][-1]
            self.partial_take_profit_pct = regime_params['profit_targets'][1]
            self.breakeven_trigger_pct = regime_params['breakeven_trigger']
            self.max_holding_time = regime_params['max_holding_time']
            self.min_confidence = regime_params['min_confidence_threshold']

            # Store regime info
            self.current_regime = current_regime
            self.regime_history.append({
                'regime': current_regime.value,
                'profit_targets': self.profit_targets,
                'trailing_stop_pct': self.trailing_stop_pct,
                'position_size_multiplier': regime_params['position_size_multiplier']
            })

        except Exception as e:
            # Use defaults if regime detection fails
            self._set_default_parameters()
            self.current_regime = MarketRegime.RANGING

    def execute_trade(self, signal, confidence, price, timestamp, price_data):
        """Execute trade with regime-adaptive parameters"""
        # Update regime parameters based on current market conditions
        self._update_regime_parameters(price_data)

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
                self.breakeven_activated = False
                self.highest_price_since_entry = 0

        # Enter new positions
        elif abs(signal) > 0.1 and confidence > self.min_confidence:
            position_size = self._calculate_position_size(signal, confidence, price)

            if signal > 0:  # Buy
                self.position = position_size
                self.entry_price = price
                self.entry_time = timestamp
                self.highest_price_since_entry = price
                self.trailing_stop_distance = price * (1 - self.trailing_stop_pct)
                self._record_trade(timestamp, price, 0, 'entry', confidence, 'BUY')

            elif signal < 0:  # Sell
                self.position = -position_size
                self.entry_price = price
                self.entry_time = timestamp
                self.highest_price_since_entry = price
                self.trailing_stop_distance = price * (1 + self.trailing_stop_pct)
                self._record_trade(timestamp, price, 0, 'entry', confidence, 'SELL')

        # Update trailing stops
        if self.position != 0:
            self._update_trailing_stops(price)

        self.portfolio_values.append(self.capital)
        return pnl, exit_reason

    def _check_exits(self, price, timestamp, current_signal):
        """Check various exit conditions with regime-adaptive parameters"""
        # Calculate current profit percentage
        if self.position > 0:
            profit_pct = (price - self.entry_price) / self.entry_price
        else:
            profit_pct = (self.entry_price - price) / self.entry_price

        # Breakeven stop activation
        if not self.breakeven_activated and profit_pct >= self.breakeven_trigger_pct:
            self.breakeven_activated = True
            buffer_pct = 0.005
            if self.position > 0:
                self.trailing_stop_distance = self.entry_price * (1 + buffer_pct)
            else:
                self.trailing_stop_distance = self.entry_price * (1 - buffer_pct)

        # Scaled profit taking - use regime-specific targets
        for target_pct in sorted(self.profit_targets, reverse=True):
            if profit_pct >= target_pct:
                if target_pct <= 0.02:  # Small profits - take 25% position
                    profit_portion = 0.25
                    exit_reason = f'profit_{int(target_pct*100)}pct_partial'
                elif target_pct <= 0.05:  # Medium profits - take 50% position
                    profit_portion = 0.50
                    exit_reason = f'profit_{int(target_pct*100)}pct_partial'
                else:  # Large profits - take full position
                    profit_portion = 1.0
                    exit_reason = 'take_profit'

                if profit_portion < 1.0:
                    self.position *= (1 - profit_portion)
                    return exit_reason
                else:
                    return exit_reason

        # Trailing stop check
        if self.position > 0 and price <= self.trailing_stop_distance:
            return 'trailing_stop'
        elif self.position < 0 and price >= self.trailing_stop_distance:
            return 'trailing_stop'

        # Maximum holding time
        if self.entry_time and (timestamp - self.entry_time).total_seconds() > self.max_holding_time * 3600:
            return 'max_time'

        # Early exit for significant losses
        if profit_pct <= -0.03:
            return 'stop_loss'

        return None

    def _update_trailing_stops(self, price):
        """Update trailing stop levels"""
        if self.position > 0:
            if price > self.highest_price_since_entry:
                self.highest_price_since_entry = price
                self.trailing_stop_distance = price * (1 - self.trailing_stop_pct)
        else:
            if price < self.highest_price_since_entry:
                self.highest_price_since_entry = price
                self.trailing_stop_distance = price * (1 + self.trailing_stop_pct)

    def _calculate_position_size(self, signal, confidence, price):
        """Calculate position size based on confidence and regime"""
        base_size = self.capital * self.leverage * min(abs(signal), 1.0) / price
        confidence_multiplier = confidence ** 0.5

        # Get position size multiplier from current regime
        regime_multiplier = 1.0
        if hasattr(self, 'regime_history') and self.regime_history:
            regime_multiplier = self.regime_history[-1]['position_size_multiplier']

        return base_size * confidence_multiplier * regime_multiplier

    def _record_trade(self, timestamp, price, pnl, reason, confidence, action_type='EXIT'):
        """Record trade details"""
        trade = {
            'timestamp': timestamp,
            'price': price,
            'pnl': pnl,
            'reason': reason,
            'confidence': confidence,
            'action': action_type,
            'position_size': abs(self.position) if self.position != 0 else 0,
            'regime': self.current_regime.value if hasattr(self, 'current_regime') else 'unknown'
        }
        self.trades.append(trade)

def generate_regime_based_price_data(n_points=1000):
    """Generate price data with different market regimes"""
    base_price = 2000
    prices = [base_price]
    timestamps = [datetime.now() + timedelta(hours=i) for i in range(n_points)]

    # Define regime periods
    regime_periods = [
        (0, 200, 'ranging'),      # Low volatility ranging
        (200, 400, 'high_vol'),    # High volatility
        (400, 600, 'bull_trend'),  # Strong bull trend
        (600, 800, 'bear_trend'),  # Bear trend
        (800, 1000, 'ranging')     # Back to ranging
    ]

    for i in range(1, n_points):
        # Determine current regime
        current_regime = 'ranging'
        for start, end, regime in regime_periods:
            if start <= i < end:
                current_regime = regime
                break

        # Set volatility and trend based on regime
        if current_regime == 'ranging':
            volatility = 0.005
            trend = 0.0001
        elif current_regime == 'high_vol':
            volatility = 0.02
            trend = 0.0002
        elif current_regime == 'bull_trend':
            volatility = 0.015
            trend = 0.001
        elif current_regime == 'bear_trend':
            volatility = 0.012
            trend = -0.0008

        # Add momentum component
        momentum = 0.1 * (prices[-1] - prices[-2]) / prices[-2] if len(prices) > 1 else 0

        # Generate price movement
        price_change = trend + np.random.normal(0, volatility) + momentum
        new_price = prices[-1] * (1 + price_change)

        # Ensure reasonable bounds
        new_price = max(1800, min(2200, new_price))
        prices.append(new_price)

    # Create DataFrame with OHLC
    data = []
    for i, price in enumerate(prices):
        data.append({
            'timestamp': timestamps[i],
            'Close': price,
            'High': price * (1 + abs(np.random.normal(0, 0.003))),
            'Low': price * (1 - abs(np.random.normal(0, 0.003))),
            'Volume': np.random.randint(1000, 10000)
        })

    return pd.DataFrame(data)

def generate_adaptive_signals(price_data):
    """Generate trading signals that adapt to market conditions"""
    signals = []
    confidences = []

    for i, row in price_data.iterrows():
        price = row['Close']

        # Simple trend signal
        if i > 20:
            short_ma = price_data['Close'].iloc[i-10:i].mean()
            long_ma = price_data['Close'].iloc[i-20:i].mean()
            trend_signal = (short_ma - long_ma) / long_ma
        else:
            trend_signal = 0

        # Volatility-based confidence
        if i > 10:
            returns = price_data['Close'].iloc[i-10:i].pct_change().dropna()
            volatility = returns.std()
            market_condition = max(0.5, min(1.5, 1 / (1 + volatility * 10)))
        else:
            market_condition = 1.0

        # Add some noise to signals
        noise = np.random.normal(0, 0.15)  # More noise for realism
        signal = trend_signal + noise

        confidence = min(1.0, abs(signal) * market_condition)

        signals.append(signal)
        confidences.append(confidence)

    price_data['signal'] = signals
    price_data['confidence'] = confidences

    return price_data

def run_regime_adaptive_demo():
    """Run comprehensive demo of regime-adaptive trading"""
    print("🧠 REGIME-ADAPTIVE TRADING SYSTEM DEMO")
    print("=" * 60)
    print("Features:")
    print("• Market regime detection (ranging, trending, volatile)")
    print("• Dynamic parameter adjustment based on regime")
    print("• Regime-specific profit targets and risk management")
    print("• Adaptive position sizing")
    print()

    # Generate regime-based price data
    price_data = generate_regime_based_price_data(800)
    trading_data = generate_adaptive_signals(price_data)

    # Initialize regime-adaptive trading system
    trader = RegimeAdaptiveTradingSystem(capital=1000, leverage=50)

    # Execute trades
    print("Executing trades with regime adaptation...")
    for i, row in trading_data.iterrows():
        # Create rolling window for regime detection
        window_data = trading_data.iloc[max(0, i-100):i+1].copy() if i >= 50 else None

        if window_data is not None and len(window_data) > 50:
            pnl, exit_reason = trader.execute_trade(
                row['signal'],
                row['confidence'],
                row['Close'],
                row['timestamp'],
                window_data
            )
        else:
            # Use default parameters for early trades
            pnl, exit_reason = trader.execute_trade(
                row['signal'],
                row['confidence'],
                row['Close'],
                row['timestamp'],
                None
            )

        if i % 100 == 0:
            print(f"Processed {i}/{len(trading_data)} data points...")

    print("\n✅ Regime-adaptive trading simulation completed!")
    print(f"Final Portfolio Value: ${trader.capital:.2f}")

    # Analyze results
    analyze_regime_performance(trader)

    # Create visualizations
    plot_regime_adaptive_results(trader, trading_data)

def analyze_regime_performance(trader):
    """Analyze performance across different market regimes"""
    print("\n📊 REGIME PERFORMANCE ANALYSIS:")
    if not trader.trades:
        print("No trades executed")
        return

    df_trades = pd.DataFrame(trader.trades)

    # Overall performance
    total_trades = len(df_trades)
    winning_trades = len(df_trades[df_trades['pnl'] > 0])
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

    if winning_trades > 0:
        avg_win = df_trades[df_trades['pnl'] > 0]['pnl'].mean()
    else:
        avg_win = 0

    if len(df_trades[df_trades['pnl'] < 0]) > 0:
        avg_loss = df_trades[df_trades['pnl'] < 0]['pnl'].mean()
    else:
        avg_loss = 0

    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Average Win: ${avg_win:.2f}")
    print(f"Average Loss: ${avg_loss:.2f}")

    # Regime-specific performance
    if 'regime' in df_trades.columns:
        print("\nRegime-Specific Performance:")
        regime_performance = df_trades.groupby('regime')['pnl'].agg(['count', 'mean', 'sum'])
        for regime, data in regime_performance.iterrows():
            count = data['count']
            avg_pnl = data['mean']
            total_pnl = data['sum']
            print(f"  {regime:15s}: {count:3.0f} trades, Avg P&L: ${avg_pnl:7.2f}")

    # Exit reason analysis
    if 'reason' in df_trades.columns:
        exit_reasons = df_trades['reason'].value_counts()
        print("\nExit Reasons:")
        for reason, count in exit_reasons.items():
            pct = count / len(df_trades) * 100
            print(f"  {reason}: {count} ({pct:.1f}%)")
def plot_regime_adaptive_results(trader, price_data):
    """Create comprehensive visualizations for regime-adaptive trading"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Price chart with regime overlays
    ax1.plot(price_data['timestamp'], price_data['Close'], alpha=0.7, color='blue')

    # Add regime background colors
    regime_colors = {
        'ranging': 'lightgray',
        'high_volatility': 'lightcoral',
        'bull_trend': 'lightgreen',
        'bear_trend': 'lightsalmon',
        'strong_bull': 'darkgreen',
        'strong_bear': 'darkred'
    }

    if hasattr(trader, 'regime_history') and trader.regime_history:
        # This would overlay regime periods - simplified for demo
        pass

    ax1.set_title('Price Chart with Regime-Adaptive Trading')
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.3)

    # Portfolio value over time
    ax2.plot(trader.portfolio_values, linewidth=2, color='green')
    ax2.fill_between(range(len(trader.portfolio_values)),
                     trader.portfolio_values, alpha=0.3, color='green')
    ax2.set_title('Portfolio Value (Regime-Adaptive)')
    ax2.set_ylabel('Portfolio Value ($)')
    ax2.grid(True, alpha=0.3)

    # P&L distribution by regime
    if trader.trades:
        df_trades = pd.DataFrame(trader.trades)
        if 'regime' in df_trades.columns:
            regimes = df_trades['regime'].unique()
            pnl_by_regime = [df_trades[df_trades['regime'] == regime]['pnl'] for regime in regimes]

            if pnl_by_regime and all(len(pnl) > 0 for pnl in pnl_by_regime):
                ax3.boxplot(pnl_by_regime, tick_labels=regimes)
                ax3.set_title('P&L Distribution by Market Regime')
                ax3.set_ylabel('P&L ($)')
                ax3.tick_params(axis='x', rotation=45)
                ax3.grid(True, alpha=0.3)

    # Regime parameter adaptation
    if hasattr(trader, 'regime_history') and trader.regime_history:
        regimes = [r['regime'] for r in trader.regime_history]
        trailing_stops = [r['trailing_stop_pct'] for r in trader.regime_history]

        ax4.plot(trailing_stops, marker='o', linestyle='-', color='orange')
        ax4.set_title('Trailing Stop Adaptation by Regime')
        ax4.set_xlabel('Trade Number')
        ax4.set_ylabel('Trailing Stop %')
        ax4.grid(True, alpha=0.3)

        # Add regime labels
        unique_regimes = list(set(regimes))
        for i, regime in enumerate(regimes):
            if i % 10 == 0:  # Label every 10th point to avoid clutter
                ax4.annotate(regime.replace('_', ' ').title(),
                           (i, trailing_stops[i]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)

    plt.tight_layout()
    plt.savefig('regime_adaptive_trading_demo.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    run_regime_adaptive_demo()