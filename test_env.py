import numpy as np
from trading_env import TradingEnv
import pandas as pd

# Load data
df = pd.read_csv('xauusd_data.csv', parse_dates=['date'], index_col='date')

# Create environment with small data
env = TradingEnv(df.head(100))

obs = env.reset()
print("Initial obs shape:", obs.shape)
print("Action space:", env.action_space)

# Test a few steps
for i in range(5):
    action = env.action_space.sample()
    print(f"Step {i}, Action: {action}")
    obs, reward, done, info = env.step(action)
    print(f"Reward: {reward}, Done: {done}")
    if done:
        break

print("Trades:", env.trades)