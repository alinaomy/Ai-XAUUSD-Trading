#!/usr/bin/env python3
"""
Capital Scaling Calculator for Trading Model
Shows how profits scale with different capital amounts
"""

def calculate_scaled_profits(base_capital=1000, base_daily_profit=16.27, target_capital=300):
    """
    Calculate scaled profits for different capital amounts.
    Profits scale linearly with capital in leveraged trading.
    """
    scaling_factor = target_capital / base_capital
    scaled_daily_profit = base_daily_profit * scaling_factor
    scaled_total_profit = scaled_daily_profit * 100  # For 100 days

    return {
        'base_capital': base_capital,
        'base_daily_profit': base_daily_profit,
        'target_capital': target_capital,
        'scaling_factor': scaling_factor,
        'scaled_daily_profit': scaled_daily_profit,
        'scaled_total_profit_100_days': scaled_total_profit,
        'final_balance_100_days': target_capital + scaled_total_profit
    }

# Test different capital amounts
capitals = [100, 200, 300, 400, 500]

print("🚀 Trading Model Capital Scaling Calculator")
print("=" * 50)
print(f"Base Performance: $1,000 capital → $16.27/day profit")
print()

for capital in capitals:
    result = calculate_scaled_profits(target_capital=capital)
    print(f"💰 ${capital:3d} USD Capital:")
    print(f"  Daily Profit: ${result['scaled_daily_profit']:.2f}")
    print(f"  100-Day Profit: ${result['scaled_total_profit_100_days']:.2f}")
    print(f"  Final Balance: ${result['final_balance_100_days']:.2f}")
    print()

print("📊 Scaling Explanation:")
print("- Profits scale linearly with capital in leveraged trading")
print("- Higher capital = higher position sizes = higher profits")
print("- Risk also scales proportionally - use appropriate risk management")
print("- Model uses 50x leverage with 2% stop-loss for balanced risk/reward")