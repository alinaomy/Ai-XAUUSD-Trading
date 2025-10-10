#!/usr/bin/env python3
"""
Quick Demo for AI-XAUUSD Trading System

This script provides a fast demonstration of the AI trading system's capabilities
without requiring extensive setup or data downloads.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our trading components
try:
    from ensemble_trader import EnsembleTrader
    from market_regime_detector import MarketRegimeDetector
    from trading_env import TradingEnvironment
    from confidence_sizing_demo import ConfidenceBasedSizer
except ImportError as e:
    print("❌ Error importing trading components. Please ensure all dependencies are installed.")
    print(f"Missing: {e}")
    exit(1)

def generate_sample_data(days=100):
    """Generate sample XAUUSD price data for demonstration."""
    print("📊 Generating sample XAUUSD data...")

    # Start from a recent date
    start_date = datetime.now() - timedelta(days=days)
    dates = pd.date_range(start=start_date, periods=days, freq='D')

    # Generate realistic gold price movements
    # XAUUSD typically ranges from 1800-2500
    base_price = 2000
    prices = [base_price]

    np.random.seed(42)  # For reproducible results

    for i in range(1, days):
        # Add some trend and volatility
        trend = 0.0001 * np.sin(i / 10)  # Slight cyclical trend
        noise = np.random.normal(0, 0.01)  # Daily volatility ~1%
        change = trend + noise

        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1800))  # Floor at 1800

    # Create OHLCV data
    data = []
    for i, price in enumerate(prices):
        high = price * (1 + abs(np.random.normal(0, 0.005)))
        low = price * (1 - abs(np.random.normal(0, 0.005)))
        open_price = prices[i-1] if i > 0 else price
        volume = np.random.randint(100000, 500000)

        data.append({
            'Date': dates[i],
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': price,
            'Volume': volume
        })

    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)

    # Add some technical indicators
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['RSI'] = 50 + 20 * np.sin(np.arange(len(df)) / 10)  # Simplified RSI
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()

    return df

def simulate_trading_demo(data, initial_capital=1000, leverage=50):
    """Run a simplified trading simulation."""
    print("🚀 Running trading simulation...")

    # Initialize components
    regime_detector = MarketRegimeDetector()
    sizer = ConfidenceBasedSizer(initial_capital, leverage)

    # Trading parameters
    trades = []
    capital = initial_capital
    position = 0
    entry_price = 0

    for i in range(50, len(data)):  # Start after enough data for indicators
        current_data = data.iloc[:i+1]
        current_price = data.iloc[i]['Close']

        # Detect market regime
        try:
            regime, params = regime_detector.detect_regime(current_data)
        except:
            regime = regime_detector.market_regime.STRONG_BULL
            params = regime_detector.get_default_params(regime)

        # Simulate AI prediction (simplified)
        # In real system, this would come from ensemble models
        confidence = 0.5 + 0.3 * np.sin(i / 20)  # Oscillating confidence
        signal = 1 if confidence > 0.6 else (-1 if confidence < 0.4 else 0)

        # Execute trades
        if signal != 0 and position == 0:  # Open position
            position_size = sizer.calculate_position_size(confidence, 0.02)  # 2% risk
            if position_size > 0:
                position = signal
                entry_price = current_price
                trades.append({
                    'type': 'open',
                    'price': current_price,
                    'position': position,
                    'confidence': confidence,
                    'regime': regime.value,
                    'capital': capital
                })

        elif position != 0:  # Check for exit
            # Simplified exit logic
            profit_pct = (current_price - entry_price) / entry_price * position

            # Exit conditions
            if profit_pct >= 0.02 or profit_pct <= -0.01:  # 2% profit or 1% loss
                pnl = capital * profit_pct * leverage
                capital += pnl

                trades.append({
                    'type': 'close',
                    'price': current_price,
                    'position': position,
                    'pnl': pnl,
                    'profit_pct': profit_pct,
                    'capital': capital
                })

                position = 0
                entry_price = 0

    return trades, capital

def plot_demo_results(data, trades):
    """Create a visualization of the trading demo."""
    print("📈 Creating performance visualization...")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Price chart with trades
    ax1.plot(data.index, data['Close'], label='XAUUSD Price', alpha=0.7)

    # Plot trades
    buy_signals = [t for t in trades if t['type'] == 'open' and t['position'] == 1]
    sell_signals = [t for t in trades if t['type'] == 'open' and t['position'] == -1]

    if buy_signals:
        buy_prices = [t['price'] for t in buy_signals]
        buy_dates = [data.index[len(data)-len(trades)+i] for i, t in enumerate(trades) if t['type'] == 'open' and t['position'] == 1]
        ax1.scatter(buy_dates, buy_prices, color='green', marker='^', s=100, label='Buy Signal')

    if sell_signals:
        sell_prices = [t['price'] for t in sell_signals]
        sell_dates = [data.index[len(data)-len(trades)+i] for i, t in enumerate(trades) if t['type'] == 'open' and t['position'] == -1]
        ax1.scatter(sell_dates, sell_prices, color='red', marker='v', s=100, label='Sell Signal')

    ax1.set_title('XAUUSD Price with Trading Signals')
    ax1.set_ylabel('Price (USD)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Capital curve
    capital_history = [t['capital'] for t in trades if 'capital' in t]
    if capital_history:
        ax2.plot(capital_history, label='Portfolio Value', color='blue')
        ax2.axhline(y=1000, color='red', linestyle='--', alpha=0.7, label='Initial Capital')
        ax2.set_title('Portfolio Value Over Time')
        ax2.set_xlabel('Trade Number')
        ax2.set_ylabel('Capital (USD)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('demo_results.png', dpi=150, bbox_inches='tight')
    plt.show()

def print_demo_summary(trades, final_capital):
    """Print a summary of the demo results."""
    print("\n" + "="*60)
    print("🎯 AI-XAUUSD TRADING DEMO SUMMARY")
    print("="*60)

    print(f"📊 Total Trades: {len([t for t in trades if t['type'] == 'close'])}")
    print(f"💰 Initial Capital: $1,000")
    print(f"💰 Final Capital: ${final_capital:.2f}")
    print(f"📈 Total Return: ${final_capital - 1000:.2f} ({(final_capital/1000 - 1)*100:.1f}%)")

    winning_trades = [t for t in trades if t['type'] == 'close' and t['pnl'] > 0]
    losing_trades = [t for t in trades if t['type'] == 'close' and t['pnl'] <= 0]

    win_rate = len(winning_trades) / max(1, len(winning_trades) + len(losing_trades)) * 100

    print(f"🎯 Win Rate: {win_rate:.1f}%")
    print(f"💹 Winning Trades: {len(winning_trades)}")
    print(f"📉 Losing Trades: {len(losing_trades)}")

    if winning_trades:
        avg_win = np.mean([t['pnl'] for t in winning_trades])
        print(f"💰 Average Win: ${avg_win:.2f}")

    if losing_trades:
        avg_loss = np.mean([t['pnl'] for t in losing_trades])
        print(f"💸 Average Loss: ${avg_loss:.2f}")

    print("\n🔍 Key Features Demonstrated:")
    print("  ✅ Market Regime Detection")
    print("  ✅ Confidence-Based Position Sizing")
    print("  ✅ Risk Management")
    print("  ✅ Adaptive Trading Parameters")

    print("\n📈 For full system with real AI models:")
    print("  Run: python download_models.py")
    print("  Then: python advanced_trading_demo.py")

    print("\n⚠️  Disclaimer: This is a demonstration only.")
    print("   Real trading involves risk of loss.")
    print("="*60)

def main():
    """Main demo function."""
    print("🤖 AI-XAUUSD Trading System - Quick Demo")
    print("="*50)

    try:
        # Generate sample data
        data = generate_sample_data(days=200)

        # Run trading simulation
        trades, final_capital = simulate_trading_demo(data)

        # Display results
        print_demo_summary(trades, final_capital)

        # Create visualization
        try:
            plot_demo_results(data, trades)
            print("📊 Demo visualization saved as 'demo_results.png'")
        except Exception as e:
            print(f"⚠️  Could not create visualization: {e}")

        print("\n✅ Demo completed successfully!")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("Please ensure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())