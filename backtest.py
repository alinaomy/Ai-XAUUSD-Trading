import pandas as pd
from stable_baselines3 import PPO
from trading_env import TradingEnv
from transformer_policy import TransformerTradingPolicy
import numpy as np

# Load data
df = pd.read_csv('xauusd_data.csv', parse_dates=['date'], index_col='date')

# Split data
train_end = int(len(df) * 0.8)
test_df = df.iloc[train_end:]

# Load model
model = PPO.load('transformer_trading_model')

# Create test environment
test_env = TradingEnv(test_df)

# Run backtest
obs = test_env.reset()
done = False
total_reward = 0
actions = []
balances = [test_env.balance]

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = test_env.step(action)
    total_reward += reward
    actions.append(action)
    balances.append(test_env.balance)

print(f"Backtest Results:")
print(f"Total Reward: {total_reward}")
print(f"Final Balance: {test_env.balance}")
print(f"Total Profit: {test_env.total_profit}")
print(f"Initial Balance: {test_env.initial_balance}")

# Save results
results = {
    'total_reward': total_reward,
    'final_balance': test_env.balance,
    'total_profit': test_env.total_profit,
    'initial_balance': test_env.initial_balance
}

import json
with open('backtest_results.json', 'w') as f:
    json.dump(results, f)

print("Backtest results saved to backtest_results.json")

# Save trades log
import pandas as pd
trades_df = pd.DataFrame(test_env.trades)
trades_df.to_csv('trades_log.csv', index=False)
print("Trades log saved to trades_log.csv")