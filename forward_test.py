import pandas as pd
from stable_baselines3 import PPO
from trading_env import TradingEnv
from transformer_policy import TransformerTradingPolicy
import numpy as np
import json

# Load data
df = pd.read_csv('xauusd_data.csv', parse_dates=['date'], index_col='date')

# Use last 100 days for forward test
forward_df = df.tail(100)

# Load model
model = PPO.load('ppo_continuous_trading_model')

# Create test environment
forward_env = TradingEnv(forward_df)

# Run forward test
obs = forward_env.reset()
done = False
total_reward = 0
actions = []
balances = [forward_env.balance]

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = forward_env.step(action)
    total_reward += reward
    actions.append(action)
    balances.append(forward_env.balance)

print(f"Forward Test Results (last 100 days):")
print(f"Total Reward: {total_reward}")
print(f"Final Balance: {forward_env.balance}")
print(f"Total Profit: {forward_env.total_profit}")
print(f"Initial Balance: {forward_env.initial_balance}")
print(f"Profit per day: {forward_env.total_profit / 100}")

# Save results
results = {
    'total_reward': total_reward,
    'final_balance': forward_env.balance,
    'total_profit': forward_env.total_profit,
    'initial_balance': forward_env.initial_balance,
    'profit_per_day': forward_env.total_profit / 100
}

with open('forward_test_results.json', 'w') as f:
    json.dump(results, f)

print("Forward test results saved to forward_test_results.json")