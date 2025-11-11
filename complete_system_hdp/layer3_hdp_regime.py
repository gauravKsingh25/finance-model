"""
Layer 3: Regime Classification - Sticky HDP-HMM (CORE LAYER ★)
==============================================================

The main regime classification layer using Sticky HDP-HMM.
Automatically discovers macro regimes with persistent transitions.
Integrates entropy/chaos metrics for additional regime characterization.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from models.hdp_hmm import HDPHMM
from models.chaos_metrics import HurstExponent, EntropyMetrics

from config import HDP_CONFIG, CHAOS_CONFIG


class Layer3HDPRegime:
    """
    Layer 3: Regime Classification using Sticky HDP-HMM ★
    
    This is the core regime detection layer that:
    1. Automatically discovers the number of macro regimes
    2. Uses sticky transitions for regime persistence
    3. Integrates chaos/entropy metrics for characterization
    4. Produces macro regime IDs (e.g., State [0, 1, 2, 3])
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Layer 3 Sticky HDP-HMM
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or HDP_CONFIG
        self.hdp_model = None
        self.hurst_calculator = None
        self.entropy_calculator = None
        
        # Results
        self.regime_sequence_ = None
        self.regime_probs_ = None
        self.n_regimes_ = None
        self.hurst_values_ = None
        self.entropy_values_ = None
        
    def detect_regimes(self, 
                      data: pd.Series,
                      feature_name: str = 'aggregated_signal') -> Dict:
        """
        Detect macro regimes using Sticky HDP-HMM
        
        Args:
            data: Time series data (can be returns, features, or aggregated signal)
            feature_name: Name of input feature
            
        Returns:
            Dictionary with regime detection results
        """
        print(f"\n[Layer 3: Sticky HDP-HMM Regime Classification] ★")
        print(f"Detecting macro regimes from {len(data)} samples...")
        print(f"Configuration: truncation={self.config['truncation']}, kappa={self.config['kappa']}")
        
        try:
            # Initialize Sticky HDP-HMM
            self.hdp_model = HDPHMM(
                truncation=self.config['truncation'],
                alpha=self.config['alpha'],
                gamma=self.config['gamma'],
                kappa=self.config['kappa'],  # STICKY parameter
                max_iter=self.config['max_iter'],
                random_state=self.config.get('random_state', 42)
            )
            
            # Fit model
            self.hdp_model.fit(data)
            
            # Get regime sequence and convert to Series with correct index
            regime_array = self.hdp_model.predict_regime()
            # Use the data's index directly (it should match the regime array length)
            self.regime_sequence_ = pd.Series(regime_array, index=data.index if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series) else None)
            self.regime_probs_ = self.hdp_model.get_regime_probabilities()
            self.n_regimes_ = self.hdp_model.n_active_regimes_
            
            # Get regime statistics
            regime_stats = self.hdp_model.get_regime_statistics()
            model_info = self.hdp_model.get_model_info()
            
            # Calculate chaos/entropy metrics
            chaos_metrics = self._calculate_chaos_metrics(data)
            
            # Calculate regime characteristics
            regime_characteristics = self._characterize_regimes(regime_stats, chaos_metrics)
            
            # Get transition matrix
            transition_matrix = self.hdp_model.get_transition_matrix(active_only=True)
            
            print(f"✓ Regime detection complete")
            print(f"  Discovered regimes: {self.n_regimes_}")
            print(f"  Active regime indices: {model_info['active_regime_indices']}")
            print(f"  Current regime: {self.regime_sequence_.iloc[-1]}")
            
            # Print regime statistics
            print(f"\n  Regime Statistics:")
            for regime_name, stats in regime_stats.items():
                print(f"    {regime_name}: μ={stats['mean']:.4f}, σ={stats['std_dev']:.4f}, {stats['percentage']:.1f}%")
            
            return {
                'regime_sequence': self.regime_sequence_,
                'regime_probabilities': self.regime_probs_,
                'n_regimes': self.n_regimes_,
                'current_regime': int(self.regime_sequence_.iloc[-1]),
                'regime_stats': regime_stats,
                'regime_characteristics': regime_characteristics,
                'transition_matrix': transition_matrix,
                'transition_probs': transition_matrix,  # Alias for aggregator
                'model_info': model_info,
                'chaos_metrics': chaos_metrics,
                'model': self.hdp_model,
                'feature_analyzed': feature_name
            }
            
        except Exception as e:
            print(f"✗ HDP-HMM regime detection failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Return fallback single regime
            return {
                'regime_sequence': pd.Series(0, index=data.index),
                'regime_probabilities': pd.DataFrame(1.0, index=data.index, columns=['Regime_0']),
                'n_regimes': 1,
                'current_regime': 0,
                'error': str(e)
            }
    
    def _calculate_chaos_metrics(self, data: pd.Series) -> Dict:
        """
        Calculate chaos and entropy metrics
        
        Args:
            data: Time series data
            
        Returns:
            Dictionary with chaos metrics
        """
        print(f"  Calculating chaos/entropy metrics...")
        
        metrics = {}
        
        try:
            # Hurst Exponent
            min_samples = min(50, len(data) // 2)
            self.hurst_calculator = HurstExponent(min_samples=min_samples)
            hurst_value = self.hurst_calculator.calculate(data)
            
            metrics['hurst_exponent'] = hurst_value
            metrics['current_hurst'] = float(hurst_value)
            metrics['avg_hurst'] = float(hurst_value)
            
            # Interpret Hurst
            current_hurst = metrics['current_hurst']
            if current_hurst < 0.4:
                metrics['hurst_interpretation'] = 'Mean-reverting'
            elif current_hurst > 0.6:
                metrics['hurst_interpretation'] = 'Trending'
            else:
                metrics['hurst_interpretation'] = 'Random walk'
            
            print(f"    Hurst exponent: {current_hurst:.3f} ({metrics['hurst_interpretation']})")
            
        except Exception as e:
            print(f"    Warning: Hurst calculation failed: {e}")
            metrics['hurst_exponent'] = 0.5
            metrics['current_hurst'] = 0.5
            metrics['avg_hurst'] = 0.5
            metrics['hurst_interpretation'] = 'Unknown'
        
        try:
            # Shannon Entropy
            min_samples = min(30, len(data) // 2)
            self.entropy_calculator = EntropyMetrics(min_samples=min_samples)
            entropy_value = self.entropy_calculator.calculate_shannon_entropy(data)
            
            metrics['shannon_entropy'] = entropy_value
            metrics['current_entropy'] = float(entropy_value)
            metrics['avg_entropy'] = float(entropy_value)
            
            print(f"    Shannon entropy: {metrics['current_entropy']:.3f}")
            
        except Exception as e:
            print(f"    Warning: Entropy calculation failed: {e}")
            metrics['shannon_entropy'] = pd.Series(0, index=data.index)
            metrics['current_entropy'] = 0.0
            metrics['avg_entropy'] = 0.0
        
        return metrics
    
    def _characterize_regimes(self, 
                             regime_stats: Dict,
                             chaos_metrics: Dict) -> Dict:
        """
        Characterize each regime with descriptive labels
        
        Args:
            regime_stats: Regime statistics from HDP-HMM
            chaos_metrics: Chaos/entropy metrics
            
        Returns:
            Dictionary with regime characteristics
        """
        characteristics = {}
        
        for regime_name, stats in regime_stats.items():
            mean = stats['mean']
            volatility = stats['std_dev']
            
            # Determine trend direction
            if mean > 0.001:
                trend = 'Bullish'
            elif mean < -0.001:
                trend = 'Bearish'
            else:
                trend = 'Neutral'
            
            # Determine volatility level (relative to average)
            avg_vol = np.mean([s['std_dev'] for s in regime_stats.values()])
            if volatility > avg_vol * 1.2:
                vol_level = 'High-Vol'
            elif volatility < avg_vol * 0.8:
                vol_level = 'Low-Vol'
            else:
                vol_level = 'Medium-Vol'
            
            # Combine for regime label
            characteristics[regime_name] = {
                'trend': trend,
                'volatility_level': vol_level,
                'label': f"{trend} {vol_level}",
                'mean': mean,
                'std_dev': volatility,
                'percentage': stats['percentage']
            }
        
        return characteristics
    
    def get_current_regime_info(self) -> Dict:
        """
        Get detailed information about current regime
        
        Returns:
            Dictionary with current regime info
        """
        if self.regime_sequence_ is None:
            return {}
        
        current_regime_id = int(self.regime_sequence_.iloc[-1])
        regime_name = f"Regime_{current_regime_id}"
        
        # Get probability
        current_prob = float(self.regime_probs_.iloc[-1, current_regime_id])
        
        # Get regime stats
        regime_stats = self.hdp_model.get_regime_statistics()
        stats = regime_stats.get(regime_name, {})
        
        return {
            'regime_id': current_regime_id,
            'regime_name': regime_name,
            'probability': current_prob,
            'mean': stats.get('mean', 0.0),
            'std_dev': stats.get('std_dev', 0.0),
            'percentage': stats.get('percentage', 0.0)
        }
    
    def get_regime_stability(self, window: int = 20) -> float:
        """
        Calculate regime stability (sticky transitions working)
        
        Args:
            window: Lookback window
            
        Returns:
            Stability score [0-1]
        """
        if self.regime_sequence_ is None:
            return 0.0
        
        recent_regimes = self.regime_sequence_.iloc[-window:]
        n_transitions = np.sum(np.diff(recent_regimes) != 0)
        
        # Stability = 1 - (transitions / max_possible_transitions)
        stability = 1.0 - (n_transitions / (len(recent_regimes) - 1))
        
        return float(stability)
    
    def predict_next_regime(self) -> Tuple[int, float]:
        """
        Predict most likely next regime
        
        Returns:
            Tuple of (regime_id, probability)
        """
        if self.hdp_model is None:
            return 0, 0.0
        
        current_regime = int(self.regime_sequence_.iloc[-1])
        transition_matrix = self.hdp_model.get_transition_matrix(active_only=True)
        
        # Get transition probabilities from current regime
        next_probs = transition_matrix[current_regime, :]
        most_likely = int(np.argmax(next_probs))
        probability = float(next_probs[most_likely])
        
        return most_likely, probability
    
    def get_regime_summary(self) -> Dict:
        """
        Get comprehensive regime summary
        
        Returns:
            Dictionary with regime summary
        """
        if self.regime_sequence_ is None:
            return {}
        
        current_info = self.get_current_regime_info()
        next_regime, next_prob = self.predict_next_regime()
        
        return {
            'current_regime_id': current_info['regime_id'],
            'current_regime_name': current_info['regime_name'],
            'current_probability': current_info['probability'],
            'n_discovered_regimes': self.n_regimes_,
            'regime_stability': self.get_regime_stability(),
            'predicted_next_regime': next_regime,
            'next_regime_probability': next_prob,
            'sticky_parameter_kappa': self.config['kappa']
        }


if __name__ == "__main__":
    # Example usage
    print("Layer 3: Sticky HDP-HMM Regime Classification Example")
    print("="*60)
    
    # Create sample data with multiple regimes
    np.random.seed(42)
    
    # Regime 1: Low return, low vol
    regime1 = np.random.normal(0.0005, 0.01, 150)
    
    # Regime 2: High return, moderate vol (Bull)
    regime2 = np.random.normal(0.0015, 0.015, 150)
    
    # Regime 3: Negative return, high vol (Bear)
    regime3 = np.random.normal(-0.001, 0.025, 150)
    
    data = pd.Series(
        np.concatenate([regime1, regime2, regime3, regime1]),
        index=pd.date_range('2023-01-01', periods=600, freq='D')
    )
    
    # Detect regimes
    layer3 = Layer3HDPRegime()
    results = layer3.detect_regimes(data, 'synthetic_returns')
    
    print(f"\nResults:")
    print(f"  Discovered regimes: {results['n_regimes']}")
    print(f"  Current regime: {results['current_regime']}")
    
    print(f"\nRegime Characteristics:")
    for regime, char in results['regime_characteristics'].items():
        print(f"  {regime}: {char['label']} (μ={char['mean']:.4f}, σ={char['std_dev']:.4f})")
    
    print(f"\nChaos Metrics:")
    print(f"  Current Hurst: {results['chaos_metrics']['current_hurst']:.3f} ({results['chaos_metrics']['hurst_interpretation']})")
    print(f"  Current Entropy: {results['chaos_metrics']['current_entropy']:.3f}")
    
    print(f"\nRegime Summary:")
    for key, value in layer3.get_regime_summary().items():
        print(f"  {key}: {value}")
    
    print(f"\n✓ Layer 3 complete! (Core HDP-HMM layer)")
