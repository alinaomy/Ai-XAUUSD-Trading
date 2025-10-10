import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym import spaces
import torch.nn.functional as F

class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    """
    Transformer-based feature extractor for trading data.
    Uses multi-head attention to capture temporal dependencies.
    """
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super(TransformerFeaturesExtractor, self).__init__(observation_space, features_dim)

        # Observation space contains: [prices(10), indicators(15), position, balance] = 27 features
        self.input_dim = observation_space.shape[0]  # 27
        self.sequence_length = 10  # Price history length        # Embedding layers
        self.price_embedding = nn.Linear(1, 64)
        self.indicator_embedding = nn.Linear(15, 64)  # 15 technical indicators
        self.position_balance_embedding = nn.Linear(2, 64)  # position and balance  # RSI, MACD, MACD_signal, position, balance
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(self.sequence_length, 64))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=8,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(192, features_dim),  # 64*3 = 192 input features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(features_dim, features_dim)
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]

        # Split observations: first 10 are prices, next 15 are indicators, last 2 are position and balance
        prices = observations[:, :10].unsqueeze(-1)  # [batch, 10, 1]
        indicators = observations[:, 10:-2]  # [batch, 15] - all indicators
        position_balance = observations[:, -2:]  # [batch, 2] - position and balance

        # Embed prices with positional encoding
        price_embeddings = self.price_embedding(prices)  # [batch, 10, 64]
        price_embeddings = price_embeddings + self.positional_encoding.unsqueeze(0).expand(batch_size, -1, -1)

        # Apply transformer to price sequence
        transformer_output = self.transformer(price_embeddings)  # [batch, 10, 64]
        pooled_prices = transformer_output.mean(dim=1)  # [batch, 64]

        # Embed indicators
        indicator_embeddings = self.indicator_embedding(indicators)  # [batch, 64]

        # Embed position and balance
        position_balance_embeddings = self.position_balance_embedding(position_balance)  # [batch, 64]

        # Combine all features
        combined_features = torch.cat([pooled_prices, indicator_embeddings, position_balance_embeddings], dim=-1)  # [batch, 192]

        # Final projection to features_dim (128)
        output = self.output_projection(combined_features)  # [batch, features_dim]

        return output


class TransformerTradingPolicy(ActorCriticPolicy):
    """
    Custom Transformer-based policy for trading.
    """

    def __init__(self, *args, **kwargs):
        # Custom feature extractor
        kwargs['features_extractor_class'] = TransformerFeaturesExtractor
        kwargs['features_extractor_kwargs'] = {'features_dim': 128}

        super(TransformerTradingPolicy, self).__init__(*args, **kwargs)

# No need for policy registration - we'll use the class directly