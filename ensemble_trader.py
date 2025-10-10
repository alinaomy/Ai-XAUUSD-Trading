#!/usr/bin/env python3
"""
Ensemble Trading System
Combines multiple RL models for better trading decisions
"""

import pandas as pd
import numpy as np
from stable_baselines3 import PPO, TD3, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from trading_env import TradingEnv
import os
import json

class EnsembleTrader:
    """
    Ensemble trading system that combines predictions from multiple RL models
    """

    def __init__(self, models_config=None):
        if models_config is None:
            self.models_config = {
                'ppo': {'algorithm': 'PPO', 'policy': 'MlpPolicy', 'timesteps': 10000},
                'td3': {'algorithm': 'TD3', 'policy': 'MlpPolicy', 'timesteps': 10000},
                'sac': {'algorithm': 'SAC', 'policy': 'MlpPolicy', 'timesteps': 10000}
            }
        else:
            self.models_config = models_config

        self.models = {}
        self.model_weights = {}

    def create_ensemble_env(self, df):
        """Create environment for ensemble training"""
        def make_env():
            return TradingEnv(df)
        return DummyVecEnv([make_env])

    def train_ensemble(self, train_df, save_path='./ensemble_models/'):
        """Train all models in the ensemble"""
        os.makedirs(save_path, exist_ok=True)

        for model_name, config in self.models_config.items():
            print(f"Training {model_name.upper()} model...")

            # Get algorithm class
            if config['algorithm'] == 'PPO':
                model_class = PPO
            elif config['algorithm'] == 'TD3':
                model_class = TD3
            elif config['algorithm'] == 'SAC':
                model_class = SAC
            else:
                raise ValueError(f"Unknown algorithm: {config['algorithm']}")

            # Create environment
            env = self.create_ensemble_env(train_df)

            # Initialize model
            model = model_class(config['policy'], env, verbose=0)

            # Train model
            model.learn(total_timesteps=config['timesteps'])

            # Save model
            model_path = os.path.join(save_path, f'{model_name}_model.zip')
            model.save(model_path)

            # Store model
            self.models[model_name] = model
            self.model_weights[model_name] = 1.0  # Equal weights initially

            print(f"✅ {model_name.upper()} model trained and saved")

        # Save ensemble configuration
        config_path = os.path.join(save_path, 'ensemble_config.json')
        with open(config_path, 'w') as f:
            json.dump({
                'models': list(self.models.keys()),
                'weights': self.model_weights,
                'config': self.models_config
            }, f, indent=2)

        print("🎯 Ensemble training completed!")

    def load_ensemble(self, load_path='./ensemble_models/'):
        """Load trained ensemble models"""
        config_path = os.path.join(load_path, 'ensemble_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)

        self.model_weights = config['weights']

        for model_name in config['models']:
            model_path = os.path.join(load_path, f'{model_name}_model.zip')

            # Get algorithm class from config
            model_config = config['config'][model_name]
            algorithm_name = model_config['algorithm']

            if algorithm_name == 'PPO':
                model_class = PPO
            elif algorithm_name == 'TD3':
                model_class = TD3
            elif algorithm_name == 'SAC':
                model_class = SAC
            else:
                raise ValueError(f"Unknown algorithm: {algorithm_name}")

            # Load model
            self.models[model_name] = model_class.load(model_path)

        print(f"✅ Loaded ensemble with {len(self.models)} models")

    def predict_ensemble(self, observation, method='weighted_vote'):
        """Get ensemble prediction with confidence score"""
        predictions = {}
        confidences = {}

        for model_name, model in self.models.items():
            action, _states = model.predict(observation, deterministic=True)
            predictions[model_name] = action[0]

            # Calculate confidence based on action magnitude and model agreement
            confidences[model_name] = abs(action[0])

        if method == 'weighted_vote':
            action, confidence = self._weighted_vote_with_confidence(predictions, confidences)
        elif method == 'average':
            action, confidence = self._average_prediction_with_confidence(predictions)
        else:
            action, confidence = self._majority_vote_with_confidence(predictions)

        return action, confidence

    def _weighted_vote_with_confidence(self, predictions, confidences):
        """Weighted voting based on model confidence with confidence score"""
        buy_votes = 0
        sell_votes = 0
        hold_votes = 0
        total_weight = 0

        for model_name, pred in predictions.items():
            weight = confidences[model_name]
            total_weight += weight

            if pred > 0.1:
                buy_votes += weight
            elif pred < -0.1:
                sell_votes += weight
            else:
                hold_votes += weight

        # Calculate overall confidence as the strength of the winning vote
        max_votes = max(buy_votes, sell_votes, hold_votes)
        confidence = max_votes / total_weight if total_weight > 0 else 0

        # Return the action with highest weighted votes
        if buy_votes > sell_votes and buy_votes > hold_votes:
            return 0.5 * confidence, confidence  # Scale action by confidence
        elif sell_votes > buy_votes and sell_votes > hold_votes:
            return -0.5 * confidence, confidence  # Scale action by confidence
        else:
            return 0.0, confidence

    def _average_prediction_with_confidence(self, predictions):
        """Simple average of all predictions with confidence"""
        avg_prediction = np.mean(list(predictions.values()))

        # Confidence based on agreement (lower variance = higher confidence)
        variance = np.var(list(predictions.values()))
        confidence = max(0, 1 - variance)  # Scale variance to confidence

        return avg_prediction, confidence

    def _majority_vote_with_confidence(self, predictions):
        """Majority voting with confidence"""
        buy_count = sum(1 for p in predictions.values() if p > 0.1)
        sell_count = sum(1 for p in predictions.values() if p < -0.1)
        hold_count = len(predictions) - buy_count - sell_count

        total_votes = len(predictions)
        max_count = max(buy_count, sell_count, hold_count)
        confidence = max_count / total_votes

        if buy_count > sell_count and buy_count > hold_count:
            return 0.5, confidence
        elif sell_count > buy_count and sell_count > hold_count:
            return -0.5, confidence
        else:
            return 0.0, confidence

def main():
    """Main ensemble training function"""
    # Load data
    df = pd.read_csv('xauusd_data.csv', parse_dates=['date'], index_col='date')

    # Split data
    train_end = int(len(df) * 0.8)
    train_df = df.iloc[:train_end]

    # Create and train ensemble
    ensemble = EnsembleTrader()
    ensemble.train_ensemble(train_df)

    print("\n🚀 Ensemble Trading System Ready!")
    print("Models trained: PPO, TD3, SAC")
    print("Prediction method: Weighted voting")
    print("Ready for live trading!")

if __name__ == "__main__":
    main()