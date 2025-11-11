"""
Layer 2: Dynamic State Estimation - Switching Kalman Filter
===========================================================

Provides smoothed trend/mean state estimates with regime-dependent dynamics.
Filters out noise and provides clean state estimates.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from models.switching_kalman_filter_fixed import SwitchingKalmanFilterFixed

from config import KALMAN_CONFIG


class Layer2KalmanFilter:
    """
    Layer 2: Dynamic State Estimation using Switching Kalman Filter
    
    Provides smoothed state estimates and filtered trends,
    handling regime-dependent dynamics.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Layer 2 Switching Kalman Filter
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or KALMAN_CONFIG
        self.model = None
        self.filtered_state_ = None
        self.smoothed_state_ = None
        self.regime_probs_ = None
        
    def estimate_state(self, 
                      data: pd.Series,
                      feature_name: str = 'returns') -> Dict:
        """
        Estimate dynamic state using Switching Kalman Filter
        
        Args:
            data: Time series data
            feature_name: Name of feature being analyzed
            
        Returns:
            Dictionary with state estimation results
        """
        print(f"\n[Layer 2: Switching Kalman Filter]")
        print(f"Estimating dynamic state for {len(data)} samples...")
        
        try:
            # Initialize model
            self.model = SwitchingKalmanFilterFixed(
                n_regimes=self.config.get('n_regimes', 3)
            )
            
            # Fit model
            self.model.fit(data)
            
            # Get filtered and smoothed states
            state_estimates = self.model.get_state_estimates(use_smoothed=False, regime_weighted=True)
            smoothed_estimates = self.model.get_state_estimates(use_smoothed=True, regime_weighted=True)
            self.regime_probs_ = self.model.get_regime_probabilities(use_smoothed=True)
            
            # Extract position (first column)
            self.filtered_state_ = state_estimates.iloc[:, 0] if isinstance(state_estimates, pd.DataFrame) else pd.Series(state_estimates[:, 0], index=data.index)
            self.smoothed_state_ = smoothed_estimates.iloc[:, 0] if isinstance(smoothed_estimates, pd.DataFrame) else pd.Series(smoothed_estimates[:, 0], index=data.index)
            
            # Get current regime
            regimes = self.model.predict_regime(use_smoothed=True)
            current_regime = int(regimes[-1])
            
            # Calculate state statistics
            state_volatility = self._calculate_state_volatility()
            trend_strength = self._calculate_trend_strength()
            
            print(f"✓ State estimation complete")
            print(f"  Current regime: {current_regime}")
            print(f"  State volatility: {state_volatility:.4f}")
            print(f"  Trend strength: {trend_strength:.4f}")
            
            return {
                'filtered_state': self.filtered_state_,
                'smoothed_state': self.smoothed_state_,
                'regime_probabilities': self.regime_probs_,
                'current_regime': current_regime,
                'state_volatility': state_volatility,
                'trend_strength': trend_strength,
                'model': self.model,
                'feature_analyzed': feature_name
            }
            
        except Exception as e:
            print(f"✗ Kalman filter failed: {e}")
            # Return default values
            return {
                'filtered_state': pd.Series(data.values, index=data.index),
                'smoothed_state': pd.Series(data.values, index=data.index),
                'regime_probabilities': pd.DataFrame(
                    np.ones((len(data), self.config.get('n_regimes', 3))) / self.config.get('n_regimes', 3),
                    index=data.index
                ),
                'current_regime': 0,
                'state_volatility': data.std(),
                'trend_strength': 0.0,
                'error': str(e)
            }
    
    def _calculate_state_volatility(self, window: int = 20) -> float:
        """
        Calculate volatility of the smoothed state
        
        Args:
            window: Lookback window
            
        Returns:
            State volatility
        """
        if self.smoothed_state_ is None:
            return 0.0
        
        recent_state = self.smoothed_state_.iloc[-window:]
        volatility = recent_state.std()
        
        return float(volatility)
    
    def _calculate_trend_strength(self, window: int = 50) -> float:
        """
        Calculate strength of current trend
        
        Args:
            window: Lookback window
            
        Returns:
            Trend strength between -1 (strong down) and 1 (strong up)
        """
        if self.smoothed_state_ is None:
            return 0.0
        
        recent_state = self.smoothed_state_.iloc[-window:]
        
        # Simple linear regression slope
        x = np.arange(len(recent_state))
        y = recent_state.values
        
        # Normalize by standard deviation
        slope = np.polyfit(x, y, 1)[0]
        normalized_slope = slope / (recent_state.std() + 1e-10)
        
        # Clip to [-1, 1]
        trend_strength = np.clip(normalized_slope, -1, 1)
        
        return float(trend_strength)
    
    def get_regime_persistence(self, window: int = 20) -> float:
        """
        Calculate regime persistence (stability)
        
        Args:
            window: Lookback window
            
        Returns:
            Persistence score [0-1], higher = more stable
        """
        if self.regime_probs_ is None:
            return 0.5
        
        recent_probs = self.regime_probs_.iloc[-window:]
        
        # Calculate average maximum probability (confidence)
        avg_max_prob = recent_probs.max(axis=1).mean()
        
        return float(avg_max_prob)
    
    def is_state_stable(self, threshold: float = 0.7) -> bool:
        """
        Check if current state is stable
        
        Args:
            threshold: Stability threshold
            
        Returns:
            True if stable, False otherwise
        """
        persistence = self.get_regime_persistence(window=10)
        return persistence > threshold
    
    def get_state_summary(self) -> Dict:
        """
        Get summary of current state
        
        Returns:
            Dictionary with state summary
        """
        if self.smoothed_state_ is None:
            return {}
        
        return {
            'current_state_value': float(self.smoothed_state_.iloc[-1]),
            'state_volatility': self._calculate_state_volatility(),
            'trend_strength': self._calculate_trend_strength(),
            'regime_persistence': self.get_regime_persistence(),
            'is_stable': self.is_state_stable(),
            'n_regimes': self.config.get('n_regimes', 3)
        }


if __name__ == "__main__":
    # Example usage
    print("Layer 2: Switching Kalman Filter Example")
    print("="*60)
    
    # Create sample data with regime switches
    np.random.seed(42)
    
    # Regime 1: Low mean, low variance
    regime1 = np.random.normal(0.0, 0.5, 100)
    
    # Regime 2: High mean, moderate variance
    regime2 = np.random.normal(2.0, 1.0, 100)
    
    # Regime 3: Low mean, high variance
    regime3 = np.random.normal(0.5, 2.0, 100)
    
    data = pd.Series(
        np.concatenate([regime1, regime2, regime3]),
        index=pd.date_range('2023-01-01', periods=300, freq='D')
    )
    
    # Estimate state
    layer2 = Layer2KalmanFilter()
    results = layer2.estimate_state(data, 'synthetic_data')
    
    print(f"\nResults:")
    print(f"  Current regime: {results['current_regime']}")
    print(f"  State volatility: {results['state_volatility']:.4f}")
    print(f"  Trend strength: {results['trend_strength']:.4f}")
    
    print(f"\nState Summary:")
    for key, value in layer2.get_state_summary().items():
        print(f"  {key}: {value}")
    
    print(f"\n✓ Layer 2 complete!")
