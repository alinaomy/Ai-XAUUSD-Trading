#!/usr/bin/env python3
"""
Market Regime Detection System
Identifies trending vs ranging markets and provides regime-adaptive trading parameters
"""

import pandas as pd
import numpy as np
from enum import Enum
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class MarketRegime(Enum):
    """Market regime classifications"""
    STRONG_BULL = "strong_bull"
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    STRONG_BEAR = "strong_bear"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"

class RegimeParameters:
    """Trading parameters optimized for each market regime"""

    def __init__(self):
        # Base parameters for each regime
        self.regime_params = {
            MarketRegime.STRONG_BULL: {
                'profit_targets': [0.015, 0.03, 0.06, 0.12],  # Higher targets in strong trends
                'trailing_stop_pct': 0.03,  # Tighter stops to capture momentum
                'position_size_multiplier': 1.5,  # Larger positions in strong trends
                'max_holding_time': 48,  # Longer holding in trends
                'breakeven_trigger': 0.02,  # Later breakeven
                'min_confidence_threshold': 0.25,  # Lower threshold for strong trends
            },
            MarketRegime.BULL_TREND: {
                'profit_targets': [0.012, 0.025, 0.05, 0.10],
                'trailing_stop_pct': 0.025,
                'position_size_multiplier': 1.2,
                'max_holding_time': 36,
                'breakeven_trigger': 0.018,
                'min_confidence_threshold': 0.3,
            },
            MarketRegime.BEAR_TREND: {
                'profit_targets': [0.012, 0.025, 0.05, 0.10],
                'trailing_stop_pct': 0.025,
                'position_size_multiplier': 1.2,
                'max_holding_time': 36,
                'breakeven_trigger': 0.018,
                'min_confidence_threshold': 0.3,
            },
            MarketRegime.STRONG_BEAR: {
                'profit_targets': [0.015, 0.03, 0.06, 0.12],
                'trailing_stop_pct': 0.03,
                'position_size_multiplier': 1.5,
                'max_holding_time': 48,
                'breakeven_trigger': 0.02,
                'min_confidence_threshold': 0.25,
            },
            MarketRegime.RANGING: {
                'profit_targets': [0.008, 0.015, 0.03, 0.06],  # Lower targets in ranging markets
                'trailing_stop_pct': 0.02,  # Tighter stops to protect against whipsaws
                'position_size_multiplier': 0.7,  # Smaller positions in ranging markets
                'max_holding_time': 12,  # Shorter holding periods
                'breakeven_trigger': 0.012,  # Earlier breakeven
                'min_confidence_threshold': 0.4,  # Higher threshold for ranging markets
            },
            MarketRegime.HIGH_VOLATILITY: {
                'profit_targets': [0.02, 0.04, 0.08, 0.15],  # Higher targets for volatility
                'trailing_stop_pct': 0.04,  # Wider stops for volatility
                'position_size_multiplier': 0.6,  # Smaller positions in high volatility
                'max_holding_time': 8,  # Very short holding periods
                'breakeven_trigger': 0.025,  # Later breakeven for volatility
                'min_confidence_threshold': 0.5,  # Much higher threshold
            },
            MarketRegime.LOW_VOLATILITY: {
                'profit_targets': [0.005, 0.01, 0.02, 0.04],  # Lower targets in low volatility
                'trailing_stop_pct': 0.015,  # Tighter stops
                'position_size_multiplier': 1.0,  # Normal positions
                'max_holding_time': 24,  # Normal holding periods
                'breakeven_trigger': 0.008,  # Earlier breakeven
                'min_confidence_threshold': 0.35,  # Moderate threshold
            }
        }

    def get_parameters(self, regime: MarketRegime) -> Dict:
        """Get trading parameters for a specific regime"""
        return self.regime_params.get(regime, self.regime_params[MarketRegime.RANGING])

class MarketRegimeDetector:
    """
    Advanced market regime detection using multiple indicators
    """

    def __init__(self, lookback_periods: List[int] = [20, 50, 100]):
        self.lookback_periods = lookback_periods
        self.regime_history = []
        self.parameters = RegimeParameters()

    def detect_regime(self, price_data: pd.DataFrame, current_idx: int = -1) -> Tuple[MarketRegime, Dict]:
        """
        Detect current market regime using comprehensive analysis

        Args:
            price_data: DataFrame with OHLC and technical indicators
            current_idx: Current index in the data (default: latest)

        Returns:
            Tuple of (detected_regime, regime_parameters)
        """
        if len(price_data) < max(self.lookback_periods):
            return MarketRegime.RANGING, self.parameters.get_parameters(MarketRegime.RANGING)

        # Extract current window
        if current_idx == -1:
            current_idx = len(price_data) - 1

        # Calculate regime indicators
        trend_strength = self._calculate_trend_strength(price_data, current_idx)
        volatility = self._calculate_volatility(price_data, current_idx)
        momentum = self._calculate_momentum(price_data, current_idx)
        volume_trend = self._calculate_volume_trend(price_data, current_idx)

        # Determine primary regime
        regime = self._classify_regime(trend_strength, volatility, momentum, volume_trend)

        # Get regime-specific parameters
        regime_params = self.parameters.get_parameters(regime)

        # Store regime history
        self.regime_history.append({
            'timestamp': price_data.index[current_idx] if hasattr(price_data.index, '__getitem__') else current_idx,
            'regime': regime,
            'trend_strength': trend_strength,
            'volatility': volatility,
            'momentum': momentum
        })

        return regime, regime_params

    def _calculate_trend_strength(self, data: pd.DataFrame, idx: int) -> float:
        """Calculate trend strength using ADX-like indicator"""
        if 'High' not in data.columns or 'Low' not in data.columns or 'Close' not in data.columns:
            # Fallback for data without OHLC
            if 'Close' in data.columns:
                prices = data['Close'].iloc[max(0, idx-50):idx+1]
                if len(prices) > 10:
                    # Simple trend strength based on slope
                    x = np.arange(len(prices))
                    slope = np.polyfit(x, prices.values, 1)[0]
                    return abs(slope) / prices.std()  # Normalized slope
            return 0.0

        # Calculate True Range
        high = data['High'].iloc[max(0, idx-50):idx+1]
        low = data['Low'].iloc[max(0, idx-50):idx+1]
        close = data['Close'].iloc[max(0, idx-50):idx+1]

        tr = np.maximum(high - low,
                       np.maximum(abs(high - close.shift(1)),
                                abs(low - close.shift(1))))

        # Calculate Directional Movement
        dm_plus = np.where((high - high.shift(1)) > (low.shift(1) - low),
                          np.maximum(high - high.shift(1), 0), 0)
        dm_minus = np.where((low.shift(1) - low) > (high - high.shift(1)),
                           np.maximum(low.shift(1) - low, 0), 0)

        # Convert to pandas Series for rolling operations
        dm_plus_series = pd.Series(dm_plus, index=high.index)
        dm_minus_series = pd.Series(dm_minus, index=high.index)
        tr_series = pd.Series(tr, index=high.index)

        # Smooth with Wilder's smoothing
        atr = tr_series.rolling(14).mean()
        di_plus = 100 * (dm_plus_series.rolling(14).mean() / atr)
        di_minus = 100 * (dm_minus_series.rolling(14).mean() / atr)

        # ADX (trend strength)
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(14).mean()

        return adx.iloc[-1] if not adx.empty else 0.0

    def _calculate_volatility(self, data: pd.DataFrame, idx: int) -> float:
        """Calculate normalized volatility"""
        if 'Close' not in data.columns:
            return 0.0

        prices = data['Close'].iloc[max(0, idx-50):idx+1]
        if len(prices) < 10:
            return 0.0

        # Calculate multiple volatility measures
        returns = prices.pct_change().dropna()

        # Bollinger Band width (normalized)
        sma = prices.rolling(20).mean()
        std = prices.rolling(20).std()
        bb_width = 2 * std / sma
        bb_volatility = bb_width.iloc[-1] if not bb_width.empty else 0

        # ATR if available
        if 'High' in data.columns and 'Low' in data.columns:
            high = data['High'].iloc[max(0, idx-20):idx+1]
            low = data['Low'].iloc[max(0, idx-20):idx+1]
            close = data['Close'].iloc[max(0, idx-20):idx+1]

            tr = np.maximum(high - low,
                           np.maximum(abs(high - close.shift(1)),
                                    abs(low - close.shift(1))))
            atr = tr.rolling(14).mean()
            atr_volatility = atr.iloc[-1] / prices.iloc[-1] if not atr.empty else 0
        else:
            atr_volatility = 0

        # Return volatility
        ret_volatility = returns.std() * np.sqrt(252)  # Annualized

        # Combine measures
        combined_volatility = (bb_volatility * 0.4 + atr_volatility * 0.4 + ret_volatility * 0.2)

        return combined_volatility

    def _calculate_momentum(self, data: pd.DataFrame, idx: int) -> float:
        """Calculate momentum indicators"""
        if 'Close' not in data.columns:
            return 0.0

        prices = data['Close'].iloc[max(0, idx-50):idx+1]
        if len(prices) < 20:
            return 0.0

        # RSI
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_value = rsi.iloc[-1] if not rsi.empty else 50

        # MACD momentum
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9).mean()
        macd_momentum = (macd - macd_signal).iloc[-1] if not macd.empty else 0

        # Normalize RSI to momentum score (-1 to 1)
        rsi_momentum = (rsi_value - 50) / 50

        # Combine momentum indicators
        combined_momentum = (rsi_momentum * 0.6 + np.sign(macd_momentum) * 0.4)

        return combined_momentum

    def _calculate_volume_trend(self, data: pd.DataFrame, idx: int) -> float:
        """Calculate volume trend if volume data is available"""
        if 'Volume' not in data.columns:
            return 0.0

        volume = data['Volume'].iloc[max(0, idx-50):idx+1]
        if len(volume) < 20:
            return 0.0

        # Volume trend
        volume_sma = volume.rolling(20).mean()
        volume_trend = (volume.iloc[-1] - volume_sma.iloc[-1]) / volume_sma.iloc[-1]

        return volume_trend

    def _classify_regime(self, trend_strength: float, volatility: float,
                        momentum: float, volume_trend: float) -> MarketRegime:
        """
        Classify market regime based on calculated indicators
        """

        # Volatility thresholds
        HIGH_VOL_THRESHOLD = 0.05
        LOW_VOL_THRESHOLD = 0.015

        # Trend strength thresholds
        STRONG_TREND_THRESHOLD = 25
        MODERATE_TREND_THRESHOLD = 20

        # Momentum thresholds
        BULL_MOMENTUM_THRESHOLD = 0.3
        BEAR_MOMENTUM_THRESHOLD = -0.3

        # High volatility regime
        if volatility > HIGH_VOL_THRESHOLD:
            return MarketRegime.HIGH_VOLATILITY

        # Low volatility regime
        if volatility < LOW_VOL_THRESHOLD:
            return MarketRegime.LOW_VOLATILITY

        # Strong trending regimes
        if trend_strength > STRONG_TREND_THRESHOLD:
            if momentum > BULL_MOMENTUM_THRESHOLD:
                return MarketRegime.STRONG_BULL
            elif momentum < BEAR_MOMENTUM_THRESHOLD:
                return MarketRegime.STRONG_BEAR

        # Moderate trending regimes
        if trend_strength > MODERATE_TREND_THRESHOLD:
            if momentum > 0.1:
                return MarketRegime.BULL_TREND
            elif momentum < -0.1:
                return MarketRegime.BEAR_TREND

        # Default to ranging market
        return MarketRegime.RANGING

    def get_regime_statistics(self) -> Dict:
        """Get statistics about regime detection performance"""
        if not self.regime_history:
            return {}

        df = pd.DataFrame(self.regime_history)

        stats = {
            'total_observations': len(df),
            'regime_distribution': df['regime'].value_counts().to_dict(),
            'avg_trend_strength': df['trend_strength'].mean(),
            'avg_volatility': df['volatility'].mean(),
            'avg_momentum': df['momentum'].mean(),
            'regime_transitions': self._calculate_regime_transitions(df)
        }

        return stats

    def _calculate_regime_transitions(self, df: pd.DataFrame) -> Dict:
        """Calculate regime transition frequencies"""
        transitions = {}
        regimes = df['regime'].values

        for i in range(1, len(regimes)):
            from_regime = regimes[i-1]
            to_regime = regimes[i]
            key = f"{from_regime.value}_to_{to_regime.value}"
            transitions[key] = transitions.get(key, 0) + 1

        return transitions

    def get_optimal_parameters_for_regime(self, regime: MarketRegime) -> Dict:
        """Get optimal trading parameters for a specific regime"""
        return self.parameters.get_parameters(regime)

def create_sample_regime_analysis():
    """Create a sample analysis showing regime detection in action"""
    print("🧠 MARKET REGIME DETECTION SYSTEM")
    print("=" * 50)

    # Create sample price data
    np.random.seed(42)
    n_points = 500

    # Generate realistic price data with different regimes
    base_price = 2000
    prices = [base_price]

    for i in range(1, n_points):
        # Different volatility regimes
        if i < 100:  # Low volatility ranging
            volatility = 0.005
            trend = 0.0001
        elif i < 200:  # High volatility
            volatility = 0.02
            trend = 0.0005
        elif i < 350:  # Strong bull trend
            volatility = 0.015
            trend = 0.002
        else:  # Bear trend
            volatility = 0.012
            trend = -0.0015

        # Add momentum component
        momentum = 0.1 * (prices[-1] - prices[-2]) / prices[-2] if len(prices) > 1 else 0

        # Generate price movement
        price_change = trend + np.random.normal(0, volatility) + momentum
        new_price = prices[-1] * (1 + price_change)
        prices.append(max(1800, min(2200, new_price)))

    # Create DataFrame
    data = pd.DataFrame({
        'Close': prices,
        'High': [p * (1 + np.random.uniform(0, 0.005)) for p in prices],
        'Low': [p * (1 - np.random.uniform(0, 0.005)) for p in prices],
        'Volume': np.random.randint(1000, 10000, n_points)
    })

    # Initialize regime detector
    detector = MarketRegimeDetector()

    # Analyze regimes
    regimes = []
    regime_params = []

    print("\n🔍 Analyzing market regimes...")
    for i in range(50, len(data)):  # Start after minimum lookback
        regime, params = detector.detect_regime(data, i)
        regimes.append(regime)
        regime_params.append(params)

        if i % 100 == 0:
            print(f"Processed {i}/{len(data)} data points...")

    # Show regime distribution
    regime_counts = pd.Series([r.value for r in regimes]).value_counts()
    print("\n📊 Regime Distribution:")
    for regime, count in regime_counts.items():
        pct = count / len(regimes) * 100
        print(f"  {regime}: {count} ({pct:.1f}%)")

    # Show example parameters for each regime
    print("\n🎯 Regime-Specific Parameters:")
    unique_regimes = set(regimes)
    for regime in unique_regimes:
        params = detector.get_optimal_parameters_for_regime(regime)
        print(f"\n{regime.value.upper()}:")
        print(f"  Profit Targets: {[f'{t*100:.1f}%' for t in params['profit_targets']]}")
        print(f"  Trailing Stop: {params['trailing_stop_pct']*100:.1f}%")
        print(f"  Position Size Multiplier: {params['position_size_multiplier']:.1f}x")
        print(f"  Max Holding Time: {params['max_holding_time']} hours")
        print(f"  Breakeven Trigger: {params['breakeven_trigger']*100:.1f}%")

    print("\n✅ Market regime detection system ready!")
    print("This will significantly improve trading performance by adapting to market conditions.")

if __name__ == "__main__":
    create_sample_regime_analysis()