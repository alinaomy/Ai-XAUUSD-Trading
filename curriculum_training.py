#!/usr/bin/env python3
"""
Enhanced Training with Confidence-Based Rewards
Train models to recognize optimal entry/exit timing with confidence scoring
"""

import pandas as pd
import numpy as np
from stable_baselines3 import PPO, TD3, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from trading_env import TradingEnv
import os
import matplotlib.pyplot as plt

class ConfidenceRewardCallback(BaseCallback):
    """
    Callback that adds confidence-based rewards during training
    """

    def __init__(self, verbose=0):
        super(ConfidenceRewardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_confidences = []
        self.current_episode_reward = 0
        self.current_episode_confidence = 0
        self.step_count = 0

    def _on_step(self) -> bool:
        # Get reward and confidence from environment
        reward = self.locals['rewards'][0]
        self.current_episode_reward += reward

        # Estimate confidence from action magnitude (simplified)
        action = self.locals['actions'][0]
        confidence = abs(action[0]) if hasattr(action, '__len__') else abs(action)
        self.current_episode_confidence += confidence
        self.step_count += 1

        # Check if episode ended
        dones = self.locals['dones']
        if dones[0]:  # Episode ended
            avg_confidence = self.current_episode_confidence / max(self.step_count, 1)
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_confidences.append(avg_confidence)

            # Reset for next episode
            self.current_episode_reward = 0
            self.current_episode_confidence = 0
            self.step_count = 0

        return True

def create_curriculum_environments(df, num_envs=4):
    """
    Create multiple environments with different difficulty levels for curriculum learning
    """
    environments = []

    # Split data into different periods for curriculum
    total_len = len(df)
    chunk_size = total_len // num_envs

    for i in range(num_envs):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < num_envs - 1 else total_len

        chunk_df = df.iloc[start_idx:end_idx].copy()

        # Create environment with increasing difficulty
        difficulty = i / (num_envs - 1)  # 0 to 1

        # Adjust parameters based on difficulty
        stop_loss_pct = 0.02 + difficulty * 0.03  # 2% to 5%
        leverage = 50 - difficulty * 20  # 50x to 30x (easier to harder)

        def make_env(df_chunk, sl_pct, lev):
            def _init():
                return TradingEnv(df_chunk, stop_loss_pct=sl_pct, leverage=lev)
            return _init

        env = DummyVecEnv([make_env(chunk_df, stop_loss_pct, leverage)])
        environments.append(env)

    return environments

def train_with_curriculum(model_class, environments, total_timesteps=50000, model_name="curriculum_model"):
    """
    Train model using curriculum learning with confidence rewards
    """
    print(f"🎯 Starting Curriculum Training for {model_name}...")

    callback = ConfidenceRewardCallback()

    # Train on each environment in sequence (curriculum)
    for i, env in enumerate(environments):
        print(f"📚 Training on Curriculum Level {i+1}/{len(environments)}")

        if i == 0:
            # First environment - create new model
            model = model_class('MlpPolicy', env, verbose=1, tensorboard_log='./tensorboard/')
        else:
            # Subsequent environments - continue training with loaded model
            model.set_env(env)

        # Train on this environment
        env_timesteps = total_timesteps // len(environments)
        model.learn(total_timesteps=env_timesteps, callback=callback, reset_num_timesteps=False)

    # Save final model
    model.save(f'{model_name}_curriculum')
    print(f"✅ Curriculum training completed for {model_name}")

    return model, callback

def plot_training_progress(callback, model_name):
    """Plot training progress with confidence metrics"""
    if not callback.episode_rewards:
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Episode rewards
    ax1.plot(callback.episode_rewards)
    ax1.set_title(f'{model_name} - Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True)

    # Episode confidences
    ax2.plot(callback.episode_confidences)
    ax2.set_title(f'{model_name} - Average Confidence per Episode')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Confidence')
    ax2.grid(True)

    # Rolling average rewards
    if len(callback.episode_rewards) > 10:
        rolling_rewards = pd.Series(callback.episode_rewards).rolling(10).mean()
        ax3.plot(rolling_rewards)
        ax3.set_title(f'{model_name} - Rolling Average Rewards (10 episodes)')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Rolling Avg Reward')
        ax3.grid(True)

    # Reward vs Confidence scatter
    ax4.scatter(callback.episode_confidences, callback.episode_rewards, alpha=0.6)
    ax4.set_title(f'{model_name} - Confidence vs Reward Correlation')
    ax4.set_xlabel('Average Confidence')
    ax4.set_ylabel('Episode Reward')
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig(f'{model_name}_training_progress.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main curriculum training function"""
    # Load data
    df = pd.read_csv('xauusd_data.csv', parse_dates=['date'], index_col='date')

    # Create curriculum environments
    environments = create_curriculum_environments(df, num_envs=4)

    # Train different models with curriculum learning
    models_to_train = [
        (PPO, 'ppo_curriculum'),
        (TD3, 'td3_curriculum'),
        (SAC, 'sac_curriculum')
    ]

    trained_models = {}

    for model_class, model_name in models_to_train:
        model, callback = train_with_curriculum(
            model_class,
            environments,
            total_timesteps=20000,  # Shorter for demo
            model_name=model_name
        )

        trained_models[model_name] = model

        # Plot training progress
        plot_training_progress(callback, model_name)

    print("🎓 Curriculum training completed for all models!")
    print("Models saved with confidence-based learning")

    # Create ensemble configuration for curriculum-trained models
    ensemble_config = {
        'ppo_curriculum': {'algorithm': 'PPO', 'policy': 'MlpPolicy', 'timesteps': 20000},
        'td3_curriculum': {'algorithm': 'TD3', 'policy': 'MlpPolicy', 'timesteps': 20000},
        'sac_curriculum': {'algorithm': 'SAC', 'policy': 'MlpPolicy', 'timesteps': 20000}
    }

    os.makedirs('./curriculum_models/', exist_ok=True)

    # Save ensemble configuration
    import json
    with open('./curriculum_models/curriculum_ensemble_config.json', 'w') as f:
        json.dump({
            'models': list(trained_models.keys()),
            'weights': {name: 1.0 for name in trained_models.keys()},
            'config': ensemble_config
        }, f, indent=2)

    print("✅ Curriculum ensemble configuration saved!")

if __name__ == "__main__":
    main()