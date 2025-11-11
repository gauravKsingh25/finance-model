"""
Regime Detection Engine - Complete Pipeline Orchestrator
=========================================================

Complete end-to-end regime detection system integrating all layers:
- Layer A: Feature Engineering
- Layer 1: Changepoint Detection  
- Layer 2: Switching Kalman Filter
- Layer 3: Sticky HDP-HMM (CORE)
- Layer 4: Structural Awareness (TICC, Hawkes, GARCH)
- Layer C: State Aggregation

As shown in the architecture diagram.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List
import sys
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent.parent))

# Import all layers
from feature_engineering import FeatureEngineer
from layer1_changepoint import Layer1Changepoint
from layer2_kalman import Layer2KalmanFilter
from layer3_hdp_regime import Layer3HDPRegime
from layer4_structural import Layer4Structural
from state_aggregator import StateAggregator

from config import (
    FEATURE_CONFIG, CHANGEPOINT_CONFIG, KALMAN_CONFIG,
    HDP_CONFIG, TICC_CONFIG, HAWKES_CONFIG, GAS_CONFIG,
    AGGREGATION_CONFIG
)


class RegimeDetectionEngine:
    """
    Complete Regime Detection Pipeline
    
    Orchestrates all layers of the architecture to produce
    final regime classification with high confidence.
    """
    
    def __init__(self,
                 feature_config: Optional[Dict] = None,
                 changepoint_config: Optional[Dict] = None,
                 kalman_config: Optional[Dict] = None,
                 hdp_config: Optional[Dict] = None,
                 ticc_config: Optional[Dict] = None,
                 hawkes_config: Optional[Dict] = None,
                 gas_config: Optional[Dict] = None,
                 aggregation_config: Optional[Dict] = None):
        """
        Initialize Complete Regime Detection Engine
        
        Args:
            feature_config: Feature engineering configuration
            changepoint_config: Changepoint detection configuration
            kalman_config: Kalman filter configuration
            hdp_config: HDP-HMM configuration
            ticc_config: TICC configuration
            hawkes_config: Hawkes configuration
            gas_config: GAS/GARCH configuration
            aggregation_config: Aggregation configuration
        """
        print("="*80)
        print("INITIALIZING REGIME DETECTION ENGINE")
        print("="*80)
        
        # Store configurations
        self.feature_config = feature_config or FEATURE_CONFIG
        self.changepoint_config = changepoint_config or CHANGEPOINT_CONFIG
        self.kalman_config = kalman_config or KALMAN_CONFIG
        self.hdp_config = hdp_config or HDP_CONFIG
        self.ticc_config = ticc_config or TICC_CONFIG
        self.hawkes_config = hawkes_config or HAWKES_CONFIG
        self.gas_config = gas_config or GAS_CONFIG
        self.aggregation_config = aggregation_config or AGGREGATION_CONFIG
        
        # Initialize all layers
        print("\nInitializing layers...")
        self.feature_engineer = FeatureEngineer(self.feature_config)
        self.layer1_changepoint = Layer1Changepoint(self.changepoint_config)
        self.layer2_kalman = Layer2KalmanFilter(self.kalman_config)
        self.layer3_hdp = Layer3HDPRegime(self.hdp_config)
        self.layer4_structural = Layer4Structural(
            self.ticc_config, self.hawkes_config, self.gas_config
        )
        self.state_aggregator = StateAggregator(self.aggregation_config)
        
        print("✓ All layers initialized")
        
        # Results storage
        self.results_ = None
        
    def detect_regimes(self,
                      data: pd.DataFrame,
                      price_col: str = 'close',
                      multi_asset_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Run complete regime detection pipeline
        
        Args:
            data: Main DataFrame with OHLCV data
            price_col: Name of price column
            multi_asset_data: Optional DataFrame with multiple asset returns
            
        Returns:
            Dictionary with complete pipeline results
        """
        print("\n" + "="*80)
        print("RUNNING COMPLETE REGIME DETECTION PIPELINE")
        print("="*80)
        print(f"Data shape: {data.shape}")
        print(f"Date range: {data.index[0]} to {data.index[-1]}")
        
        # Layer A: Feature Engineering
        print("\n" + "-"*80)
        print("LAYER A: FEATURE ENGINEERING")
        print("-"*80)
        features = self.feature_engineer.extract_all_features(
            data, price_col, multi_asset_data
        )
        
        # Layer 1: Changepoint Detection
        print("\n" + "-"*80)
        print("LAYER 1: CHANGEPOINT DETECTION")
        print("-"*80)
        layer1_results = self.layer1_changepoint.detect_breaks(
            features['returns_series']
        )
        
        # Layer 2: Switching Kalman Filter
        print("\n" + "-"*80)
        print("LAYER 2: SWITCHING KALMAN FILTER")
        print("-"*80)
        layer2_results = self.layer2_kalman.estimate_state(
            features['returns_series']
        )
        
        # Layer 3: Sticky HDP-HMM (CORE)
        print("\n" + "-"*80)
        print("LAYER 3: STICKY HDP-HMM (★ CORE LAYER)")
        print("-"*80)
        # Use first feature column or create composite signal
        if isinstance(features['feature_matrix'], pd.DataFrame) and len(features['feature_matrix'].columns) > 0:
            hdp_input = features['feature_matrix'].iloc[:, 0]  # Use first feature
        else:
            hdp_input = features['returns_series']  # Fallback to returns
        
        layer3_results = self.layer3_hdp.detect_regimes(hdp_input)
        
        # Layer 4: Structural Awareness
        print("\n" + "-"*80)
        print("LAYER 4: STRUCTURAL AWARENESS")
        print("-"*80)
        # Prepare data for Layer 4
        if multi_asset_data is not None:
            layer4_data = multi_asset_data.copy()
            layer4_data['returns'] = features['returns_series']
            multi_asset = True
        else:
            layer4_data = data.copy()
            layer4_data['returns'] = features['returns_series']
            multi_asset = False
        
        layer4_results = self.layer4_structural.detect_structural_changes(
            layer4_data, returns_col='returns', multi_asset=multi_asset
        )
        
        # Layer C: State Aggregation
        print("\n" + "-"*80)
        print("LAYER C: STATE AGGREGATION")
        print("-"*80)
        aggregation_results = self.state_aggregator.aggregate_states(
            layer1_results, layer2_results, layer3_results,
            layer4_results, data.index
        )
        
        # Compile complete results
        self.results_ = {
            'timestamp': datetime.now(),
            'data_info': {
                'n_samples': len(data),
                'start_date': data.index[0],
                'end_date': data.index[-1],
                'multi_asset': multi_asset
            },
            'features': features,
            'layer1_changepoint': layer1_results,
            'layer2_kalman': layer2_results,
            'layer3_hdp': layer3_results,
            'layer4_structural': layer4_results,
            'aggregation': aggregation_results,
            'final_regime': aggregation_results['regime_classification'],
            'confidence': aggregation_results['confidence'],
            'consensus': aggregation_results['consensus']
        }
        
        # Print final summary
        self._print_final_summary()
        
        return self.results_
    
    def _print_final_summary(self):
        """Print final summary of regime detection"""
        print("\n" + "="*80)
        print("FINAL REGIME DETECTION SUMMARY")
        print("="*80)
        
        final_regime = self.results_['final_regime']
        confidence = self.results_['confidence']
        consensus = self.results_['consensus']
        
        print(f"\n★ CURRENT REGIME: {final_regime['current_regime']}")
        print(f"  Regime ID: {final_regime['regime_id']}")
        print(f"  Description: {final_regime['regime_description']}")
        print(f"\n★ CONFIDENCE: {confidence['overall_confidence']:.1%}")
        print(f"  Signal Consistency: {confidence['consistency_score']:.1%}")
        print(f"  Transition Stability: {confidence['transition_stability']:.1%}")
        print(f"  HDP Confidence: {confidence['hdp_confidence']:.1%}")
        print(f"\n★ CONSENSUS: {consensus['consensus_level']}")
        print(f"  Signal Std: {consensus['signal_std']:.3f}")
        
        if consensus['disagreeing_layers']:
            print(f"  ⚠ Disagreeing layers: {', '.join(consensus['disagreeing_layers'])}")
        
        # Layer-specific summaries
        print(f"\nLayer Summaries:")
        print(f"  Layer 1: {len(self.results_['layer1_changepoint']['changepoints'])} changepoints detected")
        print(f"  Layer 2: Current regime {self.results_['layer2_kalman']['current_regime']}")
        print(f"  Layer 3: Regime {final_regime['regime_id']} (kappa={self.hdp_config['kappa']})")
        
        layer4 = self.results_['layer4_structural']
        if layer4['ticc'].get('available'):
            print(f"  Layer 4 TICC: {layer4['ticc']['n_regimes']} correlation regimes")
        if layer4['hawkes'].get('available'):
            print(f"  Layer 4 Hawkes: {layer4['hawkes']['fragility_level']} fragility")
        if layer4['garch'].get('available'):
            print(f"  Layer 4 GARCH: {layer4['garch']['current_regime']} volatility")
        
        print("\n" + "="*80)
        print("✓ REGIME DETECTION COMPLETE")
        print("="*80)
    
    def get_regime_sequence(self) -> pd.Series:
        """
        Get complete regime sequence over time
        
        Returns:
            Series with regime classification at each timestamp
        """
        if self.results_ is None:
            raise ValueError("Must run detect_regimes() first")
        
        return self.results_['final_regime']['regime_sequence']
    
    def get_confidence_over_time(self) -> pd.DataFrame:
        """
        Get confidence scores over time
        
        Returns:
            DataFrame with confidence metrics over time
        """
        if self.results_ is None:
            raise ValueError("Must run detect_regimes() first")
        
        # For now, return current confidence
        # Could be extended to track confidence at each timestamp
        conf = self.results_['confidence']
        
        return pd.DataFrame({
            'overall_confidence': [conf['overall_confidence']],
            'consistency_score': [conf['consistency_score']],
            'transition_stability': [conf['transition_stability']],
            'hdp_confidence': [conf['hdp_confidence']]
        })
    
    def export_results(self, output_path: str):
        """
        Export complete results to CSV files
        
        Args:
            output_path: Directory path for output files
        """
        if self.results_ is None:
            raise ValueError("Must run detect_regimes() first")
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nExporting results to {output_path}/...")
        
        # 1. Regime sequence
        regime_seq = self.get_regime_sequence()
        regime_seq.to_csv(output_path / 'regime_sequence.csv', header=['regime'])
        
        # 2. Final regime classification
        final_regime = self.results_['final_regime']
        pd.DataFrame([{
            'timestamp': self.results_['timestamp'],
            'current_regime': final_regime['current_regime'],
            'regime_id': final_regime['regime_id'],
            'description': final_regime['regime_description'],
            'transition_probability': final_regime['transition_probability'],
            'confidence': self.results_['confidence']['overall_confidence']
        }]).to_csv(output_path / 'current_regime.csv', index=False)
        
        # 3. Layer outputs
        layers_summary = pd.DataFrame([{
            'layer1_changepoints': len(self.results_['layer1_changepoint']['changepoints']),
            'layer2_regime': self.results_['layer2_kalman']['current_regime'],
            'layer3_regime_id': final_regime['regime_id'],
            'layer4_ticc_available': self.results_['layer4_structural']['ticc'].get('available', False),
            'layer4_hawkes_available': self.results_['layer4_structural']['hawkes'].get('available', False),
            'layer4_garch_available': self.results_['layer4_structural']['garch'].get('available', False),
            'consensus_level': self.results_['consensus']['consensus_level']
        }])
        layers_summary.to_csv(output_path / 'layers_summary.csv', index=False)
        
        print(f"✓ Results exported to {output_path}/")
    
    def get_summary(self) -> Dict:
        """
        Get concise summary of current regime state
        
        Returns:
            Dictionary with key regime information
        """
        if self.results_ is None:
            return {'status': 'not_fitted'}
        
        final_regime = self.results_['final_regime']
        
        return {
            'current_regime': final_regime['current_regime'],
            'regime_id': final_regime['regime_id'],
            'confidence': self.results_['confidence']['overall_confidence'],
            'consensus': self.results_['consensus']['consensus_level'],
            'transition_probability': final_regime['transition_probability'],
            'description': final_regime['regime_description']
        }


if __name__ == "__main__":
    # Example usage
    print("Regime Detection Engine - Complete Pipeline Test")
    print("="*80)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 300
    
    # Generate regime-switching price series
    regime1 = np.random.randn(100).cumsum() + 100  # Trending
    regime2 = np.random.randn(100) * 0.5 + regime1[-1]  # Mean-reverting
    regime3 = np.random.randn(100).cumsum() + regime2[-1]  # Trending again
    
    prices = np.concatenate([regime1, regime2, regime3])
    
    data = pd.DataFrame({
        'close': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'volume': np.random.randint(1000, 10000, n_samples)
    }, index=pd.date_range('2023-01-01', periods=n_samples, freq='D'))
    
    # Initialize engine
    engine = RegimeDetectionEngine()
    
    # Run complete pipeline
    results = engine.detect_regimes(data, price_col='close')
    
    # Get summary
    print(f"\nEngine Summary:")
    for key, value in engine.get_summary().items():
        print(f"  {key}: {value}")
    
    # Export results
    engine.export_results('output_results')
    
    print(f"\n✓ Complete pipeline test finished!")
