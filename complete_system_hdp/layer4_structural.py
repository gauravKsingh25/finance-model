"""
Layer 4: Structural Awareness - TICC, Hawkes, GAS Models
========================================================

Detects structural changes in correlation, market fragility, and volatility.
Provides multiple perspectives on market regime structure.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from models.ticc_clustering import TICCClustering
from models.hawkes_process_fixed import HawkesProcess
from models.garch_volatility import GARCHVolatilityRegime

from config import TICC_CONFIG, HAWKES_CONFIG, GAS_CONFIG


class Layer4Structural:
    """
    Layer 4: Structural Awareness
    
    Combines multiple structural detection methods:
    1. TICC - Correlation structure changes
    2. Hawkes Process - Market fragility/reflexivity
    3. GAS/GARCH Models - Volatility regimes
    """
    
    def __init__(self, 
                 ticc_config: Optional[Dict] = None,
                 hawkes_config: Optional[Dict] = None,
                 gas_config: Optional[Dict] = None):
        """
        Initialize Layer 4 Structural Awareness
        
        Args:
            ticc_config: TICC configuration
            hawkes_config: Hawkes configuration
            gas_config: GAS/GARCH configuration
        """
        self.ticc_config = ticc_config or TICC_CONFIG
        self.hawkes_config = hawkes_config or HAWKES_CONFIG
        self.gas_config = gas_config or GAS_CONFIG
        
        # Models
        self.ticc_model = None
        self.hawkes_model = None
        self.garch_model = None
        
        # Results
        self.correlation_regimes_ = None
        self.hawkes_intensity_ = None
        self.volatility_regimes_ = None
        
    def detect_structural_changes(self,
                                  data: pd.DataFrame,
                                  returns_col: str = 'returns',
                                  multi_asset: bool = False) -> Dict:
        """
        Detect structural changes using all Layer 4 sensors
        
        Args:
            data: DataFrame with market data
            returns_col: Name of returns column
            multi_asset: Whether data contains multiple assets
            
        Returns:
            Dictionary with all structural detection results
        """
        print(f"\n[Layer 4: Structural Awareness]")
        print(f"Detecting structural changes...")
        
        results = {}
        
        # 1. TICC - Correlation Structure (if multi-asset data available)
        if multi_asset and data.shape[1] > 1:
            ticc_results = self._detect_correlation_structure(data)
            results['ticc'] = ticc_results
        else:
            print(f"  ⊘ TICC skipped (single asset data)")
            results['ticc'] = {'available': False}
        
        # 2. Hawkes Process - Market Fragility
        hawkes_results = self._detect_market_fragility(data, returns_col)
        results['hawkes'] = hawkes_results
        
        # 3. GARCH - Volatility Regimes  
        garch_results = self._detect_volatility_regimes(data, returns_col)
        results['garch'] = garch_results
        
        # Aggregate structural signal
        structural_signal = self._aggregate_structural_signals(results, data.index)
        results['aggregated_signal'] = structural_signal
        
        print(f"✓ Structural awareness detection complete")
        
        return results
    
    def _detect_correlation_structure(self, data: pd.DataFrame) -> Dict:
        """
        Detect correlation structure changes using TICC
        
        Args:
            data: Multi-asset returns data
            
        Returns:
            Dictionary with TICC results
        """
        print(f"  [TICC] Analyzing correlation structure...")
        
        try:
            self.ticc_model = TICCClustering(
                n_clusters=self.ticc_config['n_clusters'],
                window_size=self.ticc_config['window_size'],
                lambda_parameter=self.ticc_config['lambda_parameter'],
                beta=self.ticc_config['beta'],
                max_iter=self.ticc_config['max_iter']
            )
            
            # Fit TICC
            self.ticc_model.fit(data)
            
            # Get correlation regimes
            self.correlation_regimes_ = self.ticc_model.predict()
            
            # Get statistics
            ticc_stats = self.ticc_model.get_cluster_statistics()
            model_info = self.ticc_model.get_model_info()
            transitions = self.ticc_model.get_regime_transitions()
            
            print(f"    ✓ Found {model_info['n_clusters_found']} correlation regimes")
            print(f"    Transitions: {model_info['n_transitions']}")
            
            return {
                'available': True,
                'correlation_regimes': pd.Series(self.correlation_regimes_, index=data.index),
                'n_regimes': model_info['n_clusters_found'],
                'statistics': ticc_stats,
                'transitions': transitions,
                'model': self.ticc_model
            }
            
        except Exception as e:
            print(f"    ✗ TICC failed: {e}")
            return {
                'available': False,
                'error': str(e)
            }
    
    def _detect_market_fragility(self, 
                                data: pd.DataFrame,
                                returns_col: str) -> Dict:
        """
        Detect market fragility using Hawkes Process
        
        Args:
            data: DataFrame with returns
            returns_col: Name of returns column
            
        Returns:
            Dictionary with Hawkes results
        """
        print(f"  [Hawkes] Analyzing market fragility/reflexivity...")
        
        try:
            # Get returns
            if returns_col in data.columns:
                returns = data[returns_col]
            else:
                # Calculate returns from close price
                returns = data.iloc[:, 0].pct_change().dropna()
            
            # Convert returns to events (extreme moves)
            threshold = returns.abs().quantile(0.75)  # Top 25% moves
            events = np.where(returns.abs() > threshold)[0]
            
            if len(events) < 10:
                print(f"    ⊘ Insufficient events for Hawkes analysis")
                return {'available': False, 'reason': 'insufficient_events'}
            
            # Initialize Hawkes model
            self.hawkes_model = HawkesProcess()
            
            # Fit model (events are indices, not need max_time)
            self.hawkes_model.fit(events.astype(float))
            
            # Get intensity at each time point
            self.hawkes_intensity_ = np.array([
                self.hawkes_model.get_intensity_at_time(float(t)) 
                for t in range(len(returns))
            ])
            hawkes_stats = self.hawkes_model.get_statistics()
            
            # Market fragility score (higher branching = more fragile)
            fragility_score = hawkes_stats.get('branching_ratio', 0.0)
            
            if fragility_score > 0.8:
                fragility_level = 'High'
            elif fragility_score > 0.5:
                fragility_level = 'Moderate'
            else:
                fragility_level = 'Low'
            
            print(f"    ✓ Market fragility: {fragility_level} (branching={fragility_score:.3f})")
            
            return {
                'available': True,
                'intensity': pd.Series(self.hawkes_intensity_, index=returns.index),
                'fragility_score': fragility_score,
                'fragility_level': fragility_level,
                'statistics': hawkes_stats,
                'n_events': len(events),
                'model': self.hawkes_model
            }
            
        except Exception as e:
            print(f"    ✗ Hawkes failed: {e}")
            return {
                'available': False,
                'error': str(e)
            }
    
    def _detect_volatility_regimes(self,
                                  data: pd.DataFrame,
                                  returns_col: str) -> Dict:
        """
        Detect volatility regimes using GARCH
        
        Args:
            data: DataFrame with returns
            returns_col: Name of returns column
            
        Returns:
            Dictionary with GARCH results
        """
        print(f"  [GARCH] Analyzing volatility regimes...")
        
        try:
            # Get returns
            if returns_col in data.columns:
                returns = data[returns_col]
            else:
                returns = data.iloc[:, 0].pct_change().dropna()
            
            # Initialize GARCH model
            self.garch_model = GARCHVolatilityRegime(
                p=self.gas_config.get('p', 1),
                q=self.gas_config.get('q', 1)
            )
            
            # Fit model
            self.garch_model.fit(returns)
            
            # Get volatility regimes
            self.volatility_regimes_ = self.garch_model.predict_regimes()
            vol_forecast = self.garch_model.forecast_volatility(horizon=1)
            
            # Current regime (convert to string to avoid formatting issues)
            current_regime = str(self.volatility_regimes_.iloc[-1])
            
            print(f"    ✓ Current volatility regime: {current_regime}")
            print(f"    Forecasted volatility: {vol_forecast:.4f}")
            
            return {
                'available': True,
                'volatility_regimes': self.volatility_regimes_,
                'current_regime': current_regime,
                'forecast': vol_forecast,
                'model': self.garch_model
            }
            
        except Exception as e:
            print(f"    ✗ GARCH failed: {e}")
            # Fallback: simple rolling volatility
            returns = data[returns_col] if returns_col in data.columns else data.iloc[:, 0].pct_change()
            rolling_vol = returns.rolling(20).std()
            median_vol = rolling_vol.median()
            
            vol_regime = pd.Series('Low-Vol', index=data.index)
            vol_regime[rolling_vol > median_vol] = 'High-Vol'
            
            return {
                'available': False,
                'error': str(e),
                'fallback_regime': vol_regime,
                'fallback_volatility': rolling_vol
            }
    
    def _aggregate_structural_signals(self,
                                     results: Dict,
                                     index: pd.Index) -> pd.Series:
        """
        Aggregate all structural signals into single metric
        
        Args:
            results: Dictionary with all structural results
            index: Time index
            
        Returns:
            Aggregated structural change signal
        """
        # Initialize aggregated signal
        signal = pd.Series(0.0, index=index)
        
        # Weight for each component
        weights = {
            'ticc': 0.4,
            'hawkes': 0.3,
            'garch': 0.3
        }
        
        # TICC contribution (regime transitions)
        if results['ticc'].get('available', False):
            corr_regimes = results['ticc']['correlation_regimes']
            # Normalize regime changes to [0, 1]
            regime_changes = (corr_regimes.diff() != 0).astype(float)
            signal += weights['ticc'] * regime_changes
        
        # Hawkes contribution (intensity)
        if results['hawkes'].get('available', False):
            intensity = results['hawkes']['intensity']
            # Normalize intensity to [0, 1]
            intensity_norm = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-10)
            signal += weights['hawkes'] * intensity_norm
        
        # GARCH contribution (volatility regime)
        if results['garch'].get('available', False):
            vol_regime = results['garch']['volatility_regimes']
            # Convert to binary: 1 for High-Vol, 0 for Low-Vol
            vol_signal = (vol_regime == 'High-Vol').astype(float)
            signal += weights['garch'] * vol_signal
        
        # Renormalize
        if signal.max() > 0:
            signal = signal / signal.max()
        
        return signal
    
    def get_structural_summary(self) -> Dict:
        """
        Get summary of structural awareness
        
        Returns:
            Dictionary with structural summary
        """
        summary = {
            'ticc_available': self.ticc_model is not None,
            'hawkes_available': self.hawkes_model is not None,
            'garch_available': self.garch_model is not None
        }
        
        if self.ticc_model:
            summary['ticc_n_regimes'] = len(np.unique(self.correlation_regimes_))
            summary['ticc_current_regime'] = int(self.correlation_regimes_[-1])
        
        if self.hawkes_model:
            summary['hawkes_fragility'] = self.hawkes_model.get_statistics().get('branching_ratio', 0.0)
        
        if self.garch_model:
            summary['garch_current_regime'] = self.garch_model.get_current_regime()
        
        return summary


if __name__ == "__main__":
    # Example usage
    print("Layer 4: Structural Awareness Example")
    print("="*60)
    
    # Create sample multi-asset data
    np.random.seed(42)
    n_samples = 300
    n_assets = 5
    
    # Generate correlated returns with regime change
    regime1_returns = np.random.multivariate_normal(
        mean=np.zeros(n_assets),
        cov=np.eye(n_assets) * 0.01 + 0.005,  # Low correlation
        size=150
    )
    
    regime2_returns = np.random.multivariate_normal(
        mean=np.zeros(n_assets),
        cov=np.eye(n_assets) * 0.01 + 0.02,  # High correlation
        size=150
    )
    
    returns = np.vstack([regime1_returns, regime2_returns])
    
    data = pd.DataFrame(
        returns,
        columns=[f'Asset_{i}' for i in range(n_assets)],
        index=pd.date_range('2023-01-01', periods=n_samples, freq='D')
    )
    
    # Add returns column for single-asset analysis
    data['returns'] = data['Asset_0']
    
    # Detect structural changes
    layer4 = Layer4Structural()
    results = layer4.detect_structural_changes(data, returns_col='returns', multi_asset=True)
    
    print(f"\nResults:")
    if results['ticc'].get('available'):
        print(f"  TICC: {results['ticc']['n_regimes']} correlation regimes")
    
    if results['hawkes'].get('available'):
        print(f"  Hawkes: {results['hawkes']['fragility_level']} market fragility")
    
    if results['garch'].get('available'):
        print(f"  GARCH: {results['garch']['current_regime']} volatility regime")
    
    print(f"\nStructural Summary:")
    for key, value in layer4.get_structural_summary().items():
        print(f"  {key}: {value}")
    
    print(f"\n✓ Layer 4 complete!")
