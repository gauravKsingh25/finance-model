"""
Layer C: State Aggregation
===========================

Combines outputs from all parallel sensors into unified state representation.
Produces final regime classification and confidence metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import AGGREGATION_CONFIG


class StateAggregator:
    """
    State Aggregation Engine
    
    Combines signals from all layers:
    - Layer 1: Changepoint signals
    - Layer 2: Kalman state estimates  
    - Layer 3: HDP-HMM regimes (CORE)
    - Layer 4: Structural awareness (TICC, Hawkes, GARCH)
    
    Produces:
    - Unified regime classification
    - Confidence scores
    - Consensus metrics
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize State Aggregator
        
        Args:
            config: Aggregation configuration
        """
        self.config = config or AGGREGATION_CONFIG
        
        # Weights for each layer
        self.weights = self.config.get('layer_weights', {
            'layer1_changepoint': 0.15,
            'layer2_kalman': 0.20,
            'layer3_hdp': 0.40,  # CORE layer gets highest weight
            'layer4_structural': 0.25
        })
        
        # Results
        self.aggregated_state_ = None
        self.regime_classification_ = None
        self.confidence_scores_ = None
        
    def aggregate_states(self,
                        layer1_results: Dict,
                        layer2_results: Dict,
                        layer3_results: Dict,
                        layer4_results: Dict,
                        index: pd.Index) -> Dict:
        """
        Aggregate all layer outputs into unified state
        
        Args:
            layer1_results: Changepoint detection results
            layer2_results: Kalman filter results
            layer3_results: HDP-HMM results (CORE)
            layer4_results: Structural awareness results
            index: Time index
            
        Returns:
            Dictionary with aggregated state and regime classification
        """
        print(f"\n[Layer C: State Aggregation]")
        print(f"Aggregating signals from all layers...")
        
        # Extract signals from each layer
        signals = self._extract_signals(
            layer1_results, layer2_results, layer3_results, layer4_results, index
        )
        
        # Compute weighted aggregation
        aggregated_signal = self._compute_weighted_aggregation(signals)
        
        # Determine final regime classification
        regime_classification = self._classify_regimes(
            layer3_results, aggregated_signal, index
        )
        
        # Calculate confidence scores
        confidence = self._calculate_confidence(signals, regime_classification)
        
        # Consensus analysis
        consensus = self._analyze_consensus(signals)
        
        # Store results
        self.aggregated_state_ = aggregated_signal
        self.regime_classification_ = regime_classification
        self.confidence_scores_ = confidence
        
        print(f"✓ State aggregation complete")
        print(f"  Current regime: {regime_classification['current_regime']}")
        print(f"  Confidence: {confidence['overall_confidence']:.2%}")
        print(f"  Consensus: {consensus['consensus_level']}")
        
        return {
            'aggregated_signal': aggregated_signal,
            'regime_classification': regime_classification,
            'confidence': confidence,
            'consensus': consensus,
            'raw_signals': signals
        }
    
    def _extract_signals(self,
                        layer1_results: Dict,
                        layer2_results: Dict,
                        layer3_results: Dict,
                        layer4_results: Dict,
                        index: pd.Index) -> Dict:
        """
        Extract signals from all layers
        
        Returns:
            Dictionary with normalized signals from each layer
        """
        signals = {}
        
        # Layer 1: Changepoint signals
        if 'changepoints' in layer1_results:
            cp_signal = pd.Series(0.0, index=index)
            changepoints = layer1_results['changepoints']
            if len(changepoints) > 0:
                cp_signal.iloc[changepoints] = 1.0
            signals['layer1_changepoint'] = cp_signal
        else:
            signals['layer1_changepoint'] = pd.Series(0.0, index=index)
        
        # Layer 2: Kalman regime probabilities
        if 'regime_probabilities' in layer2_results:
            # Use highest regime probability as signal strength
            regime_probs = layer2_results['regime_probabilities']
            if isinstance(regime_probs, np.ndarray):
                kalman_signal = pd.Series(
                    regime_probs.max(axis=1) if regime_probs.ndim > 1 else regime_probs,
                    index=index
                )
            else:
                kalman_signal = regime_probs
            signals['layer2_kalman'] = kalman_signal
        else:
            signals['layer2_kalman'] = pd.Series(0.5, index=index)
        
        # Layer 3: HDP-HMM regimes (CORE)
        if 'regime_sequence' in layer3_results:
            regime_seq = layer3_results['regime_sequence']
            # Normalize regime IDs to [0, 1] range
            if len(regime_seq) > 0:
                unique_regimes = np.unique(regime_seq)
                regime_norm = (regime_seq - regime_seq.min()) / (regime_seq.max() - regime_seq.min() + 1e-10)
                hdp_signal = pd.Series(regime_norm, index=index)
            else:
                hdp_signal = pd.Series(0.5, index=index)
            signals['layer3_hdp'] = hdp_signal
        else:
            signals['layer3_hdp'] = pd.Series(0.5, index=index)
        
        # Layer 4: Structural awareness aggregated signal
        if 'aggregated_signal' in layer4_results:
            signals['layer4_structural'] = layer4_results['aggregated_signal']
        else:
            signals['layer4_structural'] = pd.Series(0.5, index=index)
        
        return signals
    
    def _compute_weighted_aggregation(self, signals: Dict) -> pd.Series:
        """
        Compute weighted aggregation of all signals
        
        Args:
            signals: Dictionary of signals from each layer
            
        Returns:
            Aggregated signal
        """
        # Initialize aggregated signal
        aggregated = pd.Series(0.0, index=signals['layer3_hdp'].index)
        
        # Weighted sum
        for layer_name, signal in signals.items():
            weight = self.weights.get(layer_name, 0.0)
            # Use assignment instead of += to avoid pandas inplace issue
            aggregated = aggregated + (weight * signal)
        
        # Normalize to [0, 1]
        max_val = aggregated.max()
        if isinstance(max_val, pd.Series):
            max_val = max_val.iloc[0]
        max_val = float(max_val)
        
        if max_val > 0:
            min_val = aggregated.min()
            if isinstance(min_val, pd.Series):
                min_val = min_val.iloc[0]
            min_val = float(min_val)
            aggregated = (aggregated - min_val) / (max_val - min_val)
        
        return aggregated
    
    def _classify_regimes(self,
                         layer3_results: Dict,
                         aggregated_signal: pd.Series,
                         index: pd.Index) -> Dict:
        """
        Determine final regime classification
        
        Uses Layer 3 (HDP-HMM) as primary classifier, enhanced by aggregated signal
        
        Args:
            layer3_results: HDP-HMM results
            aggregated_signal: Aggregated signal from all layers
            index: Time index
            
        Returns:
            Dictionary with regime classification
        """
        # Get HDP-HMM regime sequence (primary classifier)
        regime_sequence = layer3_results.get('regime_sequence', pd.Series(0, index=index))
        
        # Get regime characteristics from HDP
        regime_chars = layer3_results.get('regime_characteristics', {})
        
        # Current regime
        current_regime_id = int(regime_sequence.iloc[-1])
        current_regime_char = regime_chars.get(current_regime_id, {})
        
        # Map to semantic labels - ensure aggregated value is float
        last_agg = aggregated_signal.iloc[-1]
        if isinstance(last_agg, pd.Series):
            last_agg = float(last_agg.iloc[0])
        else:
            last_agg = float(last_agg)
        regime_labels = self._map_regime_labels(current_regime_char, last_agg)
        
        # Transition probability
        if 'transition_probs' in layer3_results:
            trans_matrix = layer3_results['transition_probs']
            # Handle case where regime_id may be larger than matrix size
            if current_regime_id < trans_matrix.shape[0] and current_regime_id < trans_matrix.shape[1]:
                transition_prob = trans_matrix[current_regime_id, current_regime_id]
            else:
                # If regime_id out of bounds, use default high persistence
                transition_prob = 0.9
        else:
            transition_prob = 0.7  # Default
        
        return {
            'current_regime': regime_labels['label'],
            'regime_id': current_regime_id,
            'regime_description': regime_labels['description'],
            'regime_sequence': regime_sequence,
            'transition_probability': transition_prob,
            'characteristics': current_regime_char
        }
    
    def _map_regime_labels(self, 
                          regime_char: Dict,
                          aggregated_value: float) -> Dict:
        """
        Map regime characteristics to semantic labels
        
        Args:
            regime_char: Regime characteristics from HDP-HMM
            aggregated_value: Current aggregated signal value
            
        Returns:
            Dictionary with regime label and description
        """
        # Get characteristics
        hurst = regime_char.get('hurst_exponent', 0.5)
        entropy = regime_char.get('entropy', 1.0)
        
        # Classification logic
        if hurst < 0.4 and entropy > 1.5:
            label = 'Mean-Reverting'
            description = 'Anti-persistent, mean-reverting behavior with high uncertainty'
        elif hurst > 0.6 and entropy < 1.0:
            label = 'Trending'
            description = 'Persistent trending behavior with low uncertainty'
        elif entropy > 1.8:
            label = 'Chaotic'
            description = 'High entropy, unpredictable regime'
        elif aggregated_value > 0.7:
            label = 'High-Stress'
            description = 'Elevated structural signals across multiple sensors'
        elif aggregated_value < 0.3:
            label = 'Low-Stress'
            description = 'Calm market conditions with low structural signals'
        else:
            label = 'Neutral'
            description = 'Balanced market regime'
        
        return {
            'label': label,
            'description': description
        }
    
    def _calculate_confidence(self,
                            signals: Dict,
                            regime_classification: Dict) -> Dict:
        """
        Calculate confidence scores for regime classification
        
        Args:
            signals: Raw signals from all layers
            regime_classification: Regime classification results
            
        Returns:
            Dictionary with confidence metrics
        """
        # 1. Signal consistency (how aligned are the signals?)
        signal_values = []
        for s in signals.values():
            val = s.iloc[-1]
            # Extract scalar value if it's a Series
            if isinstance(val, pd.Series):
                val = float(val.iloc[0])
            else:
                val = float(val)
            signal_values.append(val)
        
        signal_std = np.std(signal_values)
        consistency_score = 1.0 - min(signal_std, 1.0)  # Lower std = higher consistency
        
        # 2. Transition stability (how stable is current regime?)
        transition_prob = regime_classification.get('transition_probability', 0.5)
        
        # 3. Layer 3 (HDP) confidence
        regime_char = regime_classification.get('characteristics', {})
        hdp_confidence = 1.0 - abs(regime_char.get('hurst_exponent', 0.5) - 0.5) * 2  # Confidence higher when hurst is extreme
        
        # Overall confidence (weighted average)
        overall_confidence = (
            0.4 * consistency_score +
            0.4 * transition_prob +
            0.2 * hdp_confidence
        )
        
        return {
            'overall_confidence': overall_confidence,
            'consistency_score': consistency_score,
            'transition_stability': transition_prob,
            'hdp_confidence': hdp_confidence
        }
    
    def _analyze_consensus(self, signals: Dict) -> Dict:
        """
        Analyze consensus across all layers
        
        Args:
            signals: Raw signals from all layers
            
        Returns:
            Dictionary with consensus analysis
        """
        # Current signal values
        current_values = []
        for s in signals.values():
            val = s.iloc[-1]
            # Extract scalar value if it's a Series
            if isinstance(val, pd.Series):
                val = float(val.iloc[0])
            else:
                val = float(val)
            current_values.append(val)
        
        # Calculate statistics
        mean_signal = np.mean(current_values)
        std_signal = np.std(current_values)
        
        # Consensus level
        if std_signal < 0.2:
            consensus_level = 'High'
        elif std_signal < 0.4:
            consensus_level = 'Moderate'
        else:
            consensus_level = 'Low'
        
        # Disagreeing layers (outliers)
        disagreeing = []
        for layer_name, signal in signals.items():
            val = signal.iloc[-1]
            if isinstance(val, pd.Series):
                val = float(val.iloc[0])
            else:
                val = float(val)
            if abs(val - mean_signal) > 2 * std_signal:
                disagreeing.append(layer_name)
        
        return {
            'consensus_level': consensus_level,
            'mean_signal': mean_signal,
            'signal_std': std_signal,
            'disagreeing_layers': disagreeing
        }
    
    def get_aggregation_summary(self) -> Dict:
        """
        Get summary of state aggregation
        
        Returns:
            Dictionary with aggregation summary
        """
        if self.regime_classification_ is None:
            return {'status': 'not_fitted'}
        
        return {
            'current_regime': self.regime_classification_['current_regime'],
            'regime_id': self.regime_classification_['regime_id'],
            'confidence': self.confidence_scores_['overall_confidence'],
            'transition_probability': self.regime_classification_['transition_probability'],
            'regime_description': self.regime_classification_['regime_description']
        }


if __name__ == "__main__":
    # Example usage
    print("Layer C: State Aggregation Example")
    print("="*60)
    
    # Create sample results from each layer
    n_samples = 100
    index = pd.date_range('2023-01-01', periods=n_samples, freq='D')
    
    # Layer 1: Changepoints
    layer1_results = {
        'changepoints': np.array([20, 50, 80]),
        'probabilities': np.array([0.9, 0.85, 0.88])
    }
    
    # Layer 2: Kalman
    layer2_results = {
        'regime_probabilities': np.random.rand(n_samples, 2),
        'filtered_states': np.random.randn(n_samples, 2)
    }
    
    # Layer 3: HDP-HMM (CORE)
    regime_seq = np.concatenate([
        np.zeros(30), np.ones(40), np.zeros(30)
    ])
    layer3_results = {
        'regime_sequence': pd.Series(regime_seq, index=index),
        'regime_characteristics': {
            0: {'hurst_exponent': 0.35, 'entropy': 1.6},
            1: {'hurst_exponent': 0.65, 'entropy': 0.8}
        },
        'transition_probs': np.array([[0.95, 0.05], [0.1, 0.9]])
    }
    
    # Layer 4: Structural
    layer4_results = {
        'aggregated_signal': pd.Series(np.random.rand(n_samples) * 0.5, index=index)
    }
    
    # Aggregate states
    aggregator = StateAggregator()
    results = aggregator.aggregate_states(
        layer1_results, layer2_results, layer3_results, layer4_results, index
    )
    
    print(f"\nAggregation Results:")
    print(f"  Regime: {results['regime_classification']['current_regime']}")
    print(f"  Description: {results['regime_classification']['regime_description']}")
    print(f"  Confidence: {results['confidence']['overall_confidence']:.2%}")
    print(f"  Consensus: {results['consensus']['consensus_level']}")
    
    print(f"\nSummary:")
    for key, value in aggregator.get_aggregation_summary().items():
        print(f"  {key}: {value}")
    
    print(f"\n✓ State aggregation complete!")
