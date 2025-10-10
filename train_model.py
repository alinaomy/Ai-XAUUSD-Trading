import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from trading_env import TradingEnv
from transformer_policy import TransformerTradingPolicy
from ensemble_trader import EnsembleTrader
import os
import argparse

def train_single_model(df, model_type='ppo', timesteps=10000):
    """Train a single model"""
    def make_env(df):
        def _init():
            return TradingEnv(df)
        return _init

    train_env = DummyVecEnv([make_env(df)])

    if model_type == 'transformer':
        model = PPO(TransformerTradingPolicy, train_env, verbose=1, tensorboard_log='./tensorboard/',
                    learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10)
    else:
        model = PPO('MlpPolicy', train_env, verbose=1, tensorboard_log='./tensorboard/',
                    learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10)

    model.learn(total_timesteps=timesteps)
    return model

def main():
    parser = argparse.ArgumentParser(description='Train trading models')
    parser.add_argument('--model', type=str, default='ensemble',
                       choices=['ppo', 'transformer', 'ensemble'],
                       help='Model type to train')
    parser.add_argument('--timesteps', type=int, default=10000,
                       help='Training timesteps')
    args = parser.parse_args()

    # Load data
    df = pd.read_csv('xauusd_data.csv', parse_dates=['date'], index_col='date')

    # Split data: train on past, test on recent
    train_end = int(len(df) * 0.8)
    train_df = df.iloc[:train_end]

    if args.model == 'ensemble':
        print("🎯 Training Ensemble Model...")
        ensemble = EnsembleTrader()
        ensemble.train_ensemble(train_df)
        print("✅ Ensemble training completed!")

    elif args.model == 'transformer':
        print("🧠 Training Transformer Model...")
        model = train_single_model(train_df, 'transformer', args.timesteps)
        model.save('transformer_trading_model')
        print("✅ Transformer model trained and saved!")

    else:  # PPO
        print("🤖 Training PPO Model...")
        model = train_single_model(train_df, 'ppo', args.timesteps)
        model.save('ppo_trading_model')
        print("✅ PPO model trained and saved!")

if __name__ == "__main__":
    main()