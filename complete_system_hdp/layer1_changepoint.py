"""
Layer 1: Fast Break Detection - Bayesian Changepoint Detection
==============================================================

Detects structural breaks in time series using Bayesian methods.
Provides early warning signals of regime transitions.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict
import sys
from pathlib import Path

# Add parent directory to path to import models
sys.path.append(str(Path(__file__).parent.parent))
from models.bayesian_changepoint import BayesianChangepoint

from config import CHANGEPOINT_CONFIG


class Layer1Changepoint:
    """
    Layer 1: Fast Break Detection using Bayesian Changepoint
    
    Detects structural breaks in returns/volatility to provide
    early warning signals for regime changes.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Layer 1 Changepoint Detector
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or CHANGEPOINT_CONFIG
        self.model = None
        self.changepoints_ = None
        self.probabilities_ = None
        
    def detect_breaks(self, 
                     data: pd.Series,
                     feature_name: str = 'returns') -> Dict:
        """
        Detect structural breaks in time series
        
        Args:
            data: Time series data (typically returns)
            feature_name: Name of the feature being analyzed
            
        Returns:
            Dictionary with changepoint detection results
        """
        print(f"\n[Layer 1: Changepoint Detection]")
        print(f"Analyzing {len(data)} samples for structural breaks...")
        
        # Initialize model
        self.model = BayesianChangepoint()
        
        # Detect changepoints
        try:
            self.model.fit(data)
            changepoints = self.model.get_changepoint_locations(
                threshold=self.config.get('threshold', 0.5)
            )
            probabilities = self.model.get_changepoint_probabilities()
            
            self.changepoints_ = changepoints
            self.probabilities_ = probabilities
            
            # Generate changepoint signal
            signal = self._generate_changepoint_signal(data.index, changepoints)
            
            n_breaks = len(changepoints)
            print(f"✓ Detected {n_breaks} structural breaks")
            
            if n_breaks > 0:
                print(f"  Changepoint locations: {changepoints[:5]}..." if n_breaks > 5 else f"  Changepoint locations: {changepoints}")
            
            return {
                'changepoints': changepoints,
                'probabilities': probabilities,
                'signal': signal,
                'n_breaks': n_breaks,
                'model': self.model,
                'feature_analyzed': feature_name
            }
            
        except Exception as e:
            print(f"✗ Changepoint detection failed: {e}")
            return {
                'changepoints': [],
                'probabilities': pd.Series(0, index=data.index),
                'signal': pd.Series(0, index=data.index),
                'n_breaks': 0,
                'error': str(e)
            }
    
    def _generate_changepoint_signal(self, 
                                    index: pd.Index,
                                    changepoints: List[int]) -> pd.Series:
        """
        Generate binary signal indicating changepoint proximity
        
        Args:
            index: Time series index
            changepoints: List of changepoint indices
            
        Returns:
            Series with changepoint signal (1 near breaks, 0 otherwise)
        """
        signal = pd.Series(0, index=index)
        
        # Set signal to 1 within min_distance of changepoints
        min_distance = self.config.get('min_distance', 10)
        
        for cp in changepoints:
            if cp < len(signal):
                # Mark region around changepoint
                start = max(0, cp - min_distance // 2)
                end = min(len(signal), cp + min_distance // 2)
                signal.iloc[start:end] = 1
        
        return signal
    
    def get_recent_break_probability(self, window: int = 20) -> float:
        """
        Get average changepoint probability in recent window
        
        Args:
            window: Lookback window
            
        Returns:
            Average changepoint probability
        """
        if self.probabilities_ is None:
            return 0.0
        
        return float(self.probabilities_.iloc[-window:].mean())
    
    def is_in_transition(self, lookback: int = 10) -> bool:
        """
        Check if currently in transition period (near recent changepoint)
        
        Args:
            lookback: Number of periods to look back
            
        Returns:
            True if in transition, False otherwise
        """
        if self.changepoints_ is None or len(self.changepoints_) == 0:
            return False
        
        # Check if any changepoint in recent lookback
        recent_cp = [cp for cp in self.changepoints_ if cp >= len(self.probabilities_) - lookback]
        
        return len(recent_cp) > 0
    
    def get_time_since_last_break(self) -> int:
        """
        Get number of periods since last detected changepoint
        
        Returns:
            Periods since last break, or -1 if no breaks detected
        """
        if self.changepoints_ is None or len(self.changepoints_) == 0:
            return -1
        
        last_cp = max(self.changepoints_)
        time_since = len(self.probabilities_) - last_cp
        
        return time_since
    
    def get_statistics(self) -> Dict:
        """
        Get changepoint detection statistics
        
        Returns:
            Dictionary with statistics
        """
        if self.changepoints_ is None:
            return {}
        
        stats = {
            'n_changepoints': len(self.changepoints_),
            'avg_segment_length': self._calculate_avg_segment_length(),
            'time_since_last_break': self.get_time_since_last_break(),
            'in_transition': self.is_in_transition(),
            'recent_break_probability': self.get_recent_break_probability()
        }
        
        return stats
    
    def _calculate_avg_segment_length(self) -> float:
        """Calculate average length of segments between changepoints"""
        if len(self.changepoints_) < 2:
            return float(len(self.probabilities_))
        
        # Add start and end points
        points = [0] + list(self.changepoints_) + [len(self.probabilities_)]
        
        # Calculate segment lengths
        lengths = [points[i+1] - points[i] for i in range(len(points)-1)]
        
        return float(np.mean(lengths))


if __name__ == "__main__":
    # Example usage
    print("Layer 1: Changepoint Detection Example")
    print("="*60)
    
    # Create sample data with changepoints
    np.random.seed(42)
    
    # Segment 1: Low volatility
    seg1 = np.random.normal(0.001, 0.01, 100)
    
    # Segment 2: High volatility (regime change)
    seg2 = np.random.normal(0.002, 0.03, 100)
    
    # Segment 3: Negative mean (regime change)
    seg3 = np.random.normal(-0.002, 0.015, 100)
    
    data = pd.Series(
        np.concatenate([seg1, seg2, seg3]),
        index=pd.date_range('2023-01-01', periods=300, freq='D')
    )
    
    # Detect changepoints
    layer1 = Layer1Changepoint()
    results = layer1.detect_breaks(data, 'synthetic_returns')
    
    print(f"\nResults:")
    print(f"  Changepoints detected: {results['n_breaks']}")
    print(f"  Locations: {results['changepoints']}")
    print(f"\nStatistics:")
    for key, value in layer1.get_statistics().items():
        print(f"  {key}: {value}")
    
    print(f"\n✓ Layer 1 complete!")
