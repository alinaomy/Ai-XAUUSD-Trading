#!/usr/bin/env python3
"""
Confidence-Based Position Sizing Demo
Demonstrates how confidence scores affect position sizing for better entry/exit timing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import random

class ConfidenceBasedSizer:
    """
    Demonstrates confidence-based position sizing for optimal entry/exit timing
    """

    def __init__(self, capital=1000, leverage=50, min_confidence=0.3):
        self.capital = capital
        self.leverage = leverage
        self.min_confidence = min_confidence
        self.trades = []

    def calculate_confidence(self, signal_strength, market_volatility, trend_strength):
        """
        Calculate confidence score based on multiple factors
        """
        # Normalize inputs to 0-1 scale
        signal_norm = min(abs(signal_strength), 1.0)
        volatility_norm = 1 - min(market_volatility, 1.0)  # Lower volatility = higher confidence
        trend_norm = min(trend_strength, 1.0)

        # Weighted combination
        confidence = (signal_norm * 0.5 + volatility_norm * 0.3 + trend_norm * 0.2)

        return max(0, min(1, confidence))  # Clamp to 0-1

    def size_position(self, confidence, signal_strength, current_price):
        """
        Size position based on confidence score
        """
        if confidence < self.min_confidence:
            return 0  # No position if not confident enough

        # Base position size
        base_size = self.capital * self.leverage * min(abs(signal_strength), 1.0) / current_price

        # Apply confidence multiplier (square root for smoother scaling)
        confidence_multiplier = confidence ** 0.5

        position_size = base_size * confidence_multiplier

        return position_size

    def simulate_trading(self, signals_df):
        """
        Simulate trading with confidence-based sizing
        """
        print("🎯 Simulating Confidence-Based Trading...")

        for i, row in signals_df.iterrows():
            signal = row['signal']
            confidence = self.calculate_confidence(
                signal,
                row['volatility'],
                row['trend_strength']
            )

            position_size = self.size_position(confidence, signal, row['price'])

            if position_size > 0:
                # Execute trade
                entry_price = row['price']
                exit_price = entry_price * (1 + np.random.normal(0, 0.02))  # Simulate exit

                pnl = (exit_price - entry_price) * position_size

                trade = {
                    'timestamp': row['timestamp'],
                    'signal': signal,
                    'confidence': confidence,
                    'position_size': position_size,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl
                }

                self.trades.append(trade)
                self.capital += pnl

        print(f"✅ Simulation completed with {len(self.trades)} trades")

    def analyze_performance(self):
        """
        Analyze trading performance
        """
        if not self.trades:
            return {}

        df_trades = pd.DataFrame(self.trades)

        # Group by confidence levels
        df_trades['confidence_bucket'] = pd.cut(df_trades['confidence'],
                                               bins=[0, 0.3, 0.5, 0.7, 1.0],
                                               labels=['Low', 'Medium', 'High', 'Very High'])

        performance = {}

        for bucket in ['Low', 'Medium', 'High', 'Very High']:
            bucket_trades = df_trades[df_trades['confidence_bucket'] == bucket]

            if len(bucket_trades) > 0:
                win_rate = (bucket_trades['pnl'] > 0).mean() * 100
                avg_pnl = bucket_trades['pnl'].mean()
                total_pnl = bucket_trades['pnl'].sum()
                avg_confidence = bucket_trades['confidence'].mean()

                performance[bucket] = {
                    'trades': len(bucket_trades),
                    'win_rate': win_rate,
                    'avg_pnl': avg_pnl,
                    'total_pnl': total_pnl,
                    'avg_confidence': avg_confidence
                }

        return performance

    def plot_results(self):
        """
        Plot confidence-based sizing results
        """
        if not self.trades:
            return

        df_trades = pd.DataFrame(self.trades)
        performance = self.analyze_performance()

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Confidence vs Position Size
        ax1.scatter(df_trades['confidence'], df_trades['position_size'], alpha=0.6)
        ax1.set_title('Confidence vs Position Size')
        ax1.set_xlabel('Confidence Score')
        ax1.set_ylabel('Position Size')
        ax1.grid(True)

        # Confidence vs P&L
        colors = ['red' if x < 0 else 'green' for x in df_trades['pnl']]
        ax2.scatter(df_trades['confidence'], df_trades['pnl'], c=colors, alpha=0.6)
        ax2.set_title('Confidence vs P&L')
        ax2.set_xlabel('Confidence Score')
        ax2.set_ylabel('P&L ($)')
        ax2.grid(True)

        # Performance by confidence bucket
        if performance:
            buckets = list(performance.keys())
            win_rates = [performance[b]['win_rate'] for b in buckets]
            avg_pnls = [performance[b]['avg_pnl'] for b in buckets]

            ax3.bar(buckets, win_rates, alpha=0.7, color='blue')
            ax3.set_title('Win Rate by Confidence Level')
            ax3.set_ylabel('Win Rate (%)')
            ax3.grid(True, axis='y')

            ax4.bar(buckets, avg_pnls, alpha=0.7, color='green')
            ax4.set_title('Average P&L by Confidence Level')
            ax4.set_ylabel('Average P&L ($)')
            ax4.grid(True, axis='y')

        plt.tight_layout()
        plt.savefig('confidence_sizing_demo.png', dpi=300, bbox_inches='tight')
        plt.show()

def generate_sample_signals(n_signals=1000):
    """
    Generate sample trading signals with varying confidence factors
    """
    signals = []

    base_time = datetime.now()

    for i in range(n_signals):
        timestamp = base_time + timedelta(hours=i)

        # Generate realistic market conditions
        trend_strength = random.uniform(0, 1)
        volatility = random.uniform(0.1, 0.5)
        signal_strength = random.uniform(-1, 1)

        # Price simulation (simplified)
        price = 2000 + np.sin(i * 0.1) * 100 + random.gauss(0, 10)

        signals.append({
            'timestamp': timestamp,
            'signal': signal_strength,
            'volatility': volatility,
            'trend_strength': trend_strength,
            'price': price
        })

    return pd.DataFrame(signals)

def main():
    """Main demo function"""
    print("🎯 Confidence-Based Position Sizing Demo")
    print("=" * 50)

    # Generate sample signals
    signals_df = generate_sample_signals(500)

    # Create confidence-based sizer
    sizer = ConfidenceBasedSizer(capital=1000, leverage=50)

    # Simulate trading
    sizer.simulate_trading(signals_df)

    # Analyze performance
    performance = sizer.analyze_performance()

    print("\n📊 Performance Analysis by Confidence Level:")
    print("-" * 50)

    for level, stats in performance.items():
        print(f"{level} Confidence:")
        print(f"  Trades: {stats['trades']}")
        print(f"  Win Rate: {stats['win_rate']:.1f}%")
        print(f"  Avg P&L: ${stats['avg_pnl']:.2f}")
        print(f"  Total P&L: ${stats['total_pnl']:.2f}")
        print(f"  Avg Confidence: {stats['avg_confidence']:.2f}")
        print()

    print(f"Final Capital: ${sizer.capital:.2f}")
    # Plot results
    sizer.plot_results()

    print("✅ Demo completed! Check 'confidence_sizing_demo.png' for visualizations")

if __name__ == "__main__":
    main()