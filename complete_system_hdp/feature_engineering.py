"""
Layer A: Feature Engineering - Input Feature Extraction
=======================================================

Extracts comprehensive features from market data for the regime detection pipeline.
Implements all feature types shown in the architecture diagram:
- Intraday Features
- Multi-Asset Features  
- Daily Features
- Weekly Features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

from config import FEATURE_CONFIG


class FeatureEngineer:
    """
    Comprehensive feature extraction for regime detection
    
    Extracts features across multiple timeframes and asset dimensions
    to feed into the parallel sensor stack.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Feature Engineer
        
        Args:
            config: Feature configuration dict (defaults to FEATURE_CONFIG)
        """
        self.config = config or FEATURE_CONFIG
        self.features_ = None
        
    def extract_all_features(self, 
                            data: pd.DataFrame,
                            price_col: str = 'close',
                            volume_col: str = 'volume',
                            multi_asset_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Extract all features from market data
        
        Args:
            data: DataFrame with OHLCV data
            price_col: Name of price column
            volume_col: Name of volume column
            multi_asset_data: Optional multi-asset DataFrame
            
        Returns:
            Dictionary with 'feature_matrix', 'returns_series', and optionally 'multi_asset_features'
        """
        features = pd.DataFrame(index=data.index)
        
        print("Extracting features...")
        
        # Calculate returns first (needed everywhere)
        returns = data[price_col].pct_change().dropna()
        
        # Daily features (base features)
        daily_feats = self._extract_daily_features(data, price_col, volume_col)
        features = pd.concat([features, daily_feats], axis=1)
        
        # Intraday features (if applicable)
        if 'time' in data.columns or data.index.freq is not None:
            intraday_feats = self._extract_intraday_features(data, price_col)
            features = pd.concat([features, intraday_feats], axis=1)
        
        # Weekly/longer-term features
        weekly_feats = self._extract_weekly_features(data, price_col)
        features = pd.concat([features, weekly_feats], axis=1)
        
        # Technical indicators
        tech_feats = self._extract_technical_features(data, price_col, volume_col)
        features = pd.concat([features, tech_feats], axis=1)
        
        # Statistical features
        stat_feats = self._extract_statistical_features(data, price_col)
        features = pd.concat([features, stat_feats], axis=1)
        
        # Drop NaN rows
        features = features.dropna()
        
        self.features_ = features
        
        print(f"✓ Extracted {features.shape[1]} features from {features.shape[0]} samples")
        
        # Prepare return dictionary
        result = {
            'feature_matrix': features,
            'returns_series': returns.loc[features.index]  # Align with features
        }
        
        # Add multi-asset features if provided
        if multi_asset_data is not None:
            multi_feats = self.extract_multi_asset_features(multi_asset_data, price_col)
            result['multi_asset_features'] = multi_feats
        
        return result
    
    def extract_multi_asset_features(self, 
                                    multi_asset_data,
                                    price_col: str = 'close') -> pd.DataFrame:
        """
        Extract multi-asset correlation and portfolio features
        
        Args:
            multi_asset_data: Dict of {asset_name: DataFrame} OR DataFrame with multiple columns
            price_col: Name of price column (only used if dict input)
            
        Returns:
            DataFrame with multi-asset features
        """
        # Handle both DataFrame and Dict input
        if isinstance(multi_asset_data, pd.DataFrame):
            # Already a DataFrame with multiple assets
            returns_df = multi_asset_data.pct_change() if 'returns' not in multi_asset_data.columns else multi_asset_data
            returns_df = returns_df.dropna()
        else:
            # Dict of DataFrames
            returns_dict = {}
            for asset_name, df in multi_asset_data.items():
                returns = df[price_col].pct_change()
                returns_dict[asset_name] = returns
            returns_df = pd.DataFrame(returns_dict).dropna()
        
        features = pd.DataFrame(index=returns_df.index)
        
        # Rolling correlation matrix features
        window = self.config.get('short_window', 20)
        
        # Average correlation
        corr_values = []
        for i in range(window, len(returns_df)):
            window_data = returns_df.iloc[i-window:i]
            corr_matrix = window_data.corr()
            # Average off-diagonal correlation
            mask = np.ones_like(corr_matrix, dtype=bool)
            np.fill_diagonal(mask, False)
            avg_corr = corr_matrix.values[mask].mean()
            corr_values.append(avg_corr)
        
        features['avg_correlation'] = np.nan
        features.iloc[window:, features.columns.get_loc('avg_correlation')] = corr_values
        
        # Portfolio variance
        port_var = returns_df.rolling(window).var().mean(axis=1)
        features['portfolio_variance'] = port_var
        
        # PCA - first eigenvalue (market factor)
        if returns_df.shape[1] >= 3:
            pca_values = []
            for i in range(window, len(returns_df)):
                window_data = returns_df.iloc[i-window:i].dropna()
                if len(window_data) > 5:
                    pca = PCA(n_components=min(3, window_data.shape[1]))
                    pca.fit(window_data)
                    pca_values.append(pca.explained_variance_ratio_[0])
                else:
                    pca_values.append(np.nan)
            
            features['pca_first_component'] = np.nan
            features.iloc[window:, features.columns.get_loc('pca_first_component')] = pca_values
        
        # Diversification ratio
        features['diversification_ratio'] = 1 / (1 + features['avg_correlation'])
        
        return features.dropna()
    
    def _extract_daily_features(self, 
                               data: pd.DataFrame,
                               price_col: str,
                               volume_col: str) -> pd.DataFrame:
        """Extract daily OHLCV features"""
        features = pd.DataFrame(index=data.index)
        
        # Returns
        features['returns'] = data[price_col].pct_change()
        features['log_returns'] = np.log(data[price_col] / data[price_col].shift(1))
        
        # Volume
        if volume_col in data.columns:
            features['volume'] = data[volume_col]
            features['volume_change'] = data[volume_col].pct_change()
            features['volume_ma_ratio'] = data[volume_col] / data[volume_col].rolling(20).mean()
        
        # Price ranges
        if 'high' in data.columns and 'low' in data.columns:
            features['high_low_range'] = (data['high'] - data['low']) / data[price_col]
            features['true_range'] = self._calculate_true_range(data)
        
        if 'open' in data.columns:
            features['close_open_range'] = (data[price_col] - data['open']) / data['open']
        
        return features
    
    def _extract_intraday_features(self, 
                                  data: pd.DataFrame,
                                  price_col: str) -> pd.DataFrame:
        """Extract intraday features"""
        features = pd.DataFrame(index=data.index)
        
        # Realized volatility (5-min, 15-min, 30-min, 60-min)
        for window in self.config.get('intraday_windows', [5, 15, 30, 60]):
            returns = data[price_col].pct_change()
            rv = returns.rolling(window).std() * np.sqrt(252 * 78)  # Annualized
            features[f'realized_vol_{window}min'] = rv
        
        # Intraday momentum
        features['intraday_momentum'] = data[price_col] / data[price_col].shift(30) - 1
        
        # Price acceleration
        features['price_acceleration'] = data[price_col].diff().diff()
        
        return features
    
    def _extract_weekly_features(self, 
                                data: pd.DataFrame,
                                price_col: str) -> pd.DataFrame:
        """Extract weekly/longer-term features"""
        features = pd.DataFrame(index=data.index)
        
        windows = [
            self.config.get('short_window', 20),
            self.config.get('medium_window', 50),
            self.config.get('long_window', 200)
        ]
        
        for window in windows:
            # Trend strength (distance from moving average)
            ma = data[price_col].rolling(window).mean()
            features[f'trend_strength_{window}d'] = (data[price_col] - ma) / ma
            
            # Volatility regime
            features[f'volatility_{window}d'] = data[price_col].pct_change().rolling(window).std()
            
            # Momentum
            features[f'momentum_{window}d'] = data[price_col] / data[price_col].shift(window) - 1
        
        return features
    
    def _extract_technical_features(self, 
                                   data: pd.DataFrame,
                                   price_col: str,
                                   volume_col: str) -> pd.DataFrame:
        """Extract technical indicators"""
        features = pd.DataFrame(index=data.index)
        
        # RSI
        features['rsi_14'] = self._calculate_rsi(data[price_col], 14)
        
        # MACD
        macd, signal = self._calculate_macd(data[price_col])
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_histogram'] = macd - signal
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(data[price_col])
        features['bb_position'] = (data[price_col] - bb_lower) / (bb_upper - bb_lower)
        features['bb_width'] = (bb_upper - bb_lower) / bb_middle
        
        # ATR
        if 'high' in data.columns and 'low' in data.columns:
            features['atr_14'] = self._calculate_atr(data, 14)
        
        return features
    
    def _extract_statistical_features(self, 
                                     data: pd.DataFrame,
                                     price_col: str) -> pd.DataFrame:
        """Extract statistical features"""
        features = pd.DataFrame(index=data.index)
        
        returns = data[price_col].pct_change()
        
        # Rolling statistics
        for window in [20, 50]:
            features[f'skewness_{window}d'] = returns.rolling(window).apply(skew, raw=True)
            features[f'kurtosis_{window}d'] = returns.rolling(window).apply(kurtosis, raw=True)
            features[f'variance_{window}d'] = returns.rolling(window).var()
        
        return features
    
    # Helper functions
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, 
                       prices: pd.Series,
                       fast: int = 12,
                       slow: int = 26,
                       signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    def _calculate_bollinger_bands(self, 
                                  prices: pd.Series,
                                  period: int = 20,
                                  num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        return upper, middle, lower
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        tr = self._calculate_true_range(data)
        atr = tr.rolling(window=period).mean()
        return atr
    
    def _calculate_true_range(self, data: pd.DataFrame) -> pd.Series:
        """Calculate True Range"""
        high = data['high']
        low = data['low']
        close_prev = data['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close_prev)
        tr3 = abs(low - close_prev)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr
    
    def get_feature_names(self) -> List[str]:
        """Get list of extracted feature names"""
        if self.features_ is not None:
            return self.features_.columns.tolist()
        return []
    
    def get_feature_importance_proxy(self) -> pd.Series:
        """
        Calculate feature importance proxy using variance
        Higher variance features may be more informative
        """
        if self.features_ is None:
            raise ValueError("No features extracted yet. Call extract_all_features first.")
        
        # Normalize features
        normalized = (self.features_ - self.features_.mean()) / self.features_.std()
        
        # Use variance as proxy for importance
        importance = normalized.var().sort_values(ascending=False)
        
        return importance


if __name__ == "__main__":
    # Example usage
    print("Feature Engineering Example")
    print("="*60)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=500, freq='D')
    
    sample_data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(500) * 2),
        'open': 100 + np.cumsum(np.random.randn(500) * 2),
        'high': 102 + np.cumsum(np.random.randn(500) * 2),
        'low': 98 + np.cumsum(np.random.randn(500) * 2),
        'volume': np.random.randint(1000000, 5000000, 500)
    }, index=dates)
    
    # Extract features
    engineer = FeatureEngineer()
    features = engineer.extract_all_features(sample_data)
    
    print(f"\nExtracted features shape: {features.shape}")
    print(f"\nFeature names (first 10):")
    for i, name in enumerate(engineer.get_feature_names()[:10]):
        print(f"  {i+1}. {name}")
    
    print(f"\n✓ Feature Engineering complete!")
