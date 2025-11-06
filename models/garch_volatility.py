"""
Stream 2: Volatility Regime Detection using GARCH(1,1) Model
This model detects volatility-based regimes (Low-Vol vs High-Vol states)
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')


class GARCHVolatilityRegime:
    """
    GARCH(1,1) Model for Volatility Regime Detection
    
    Estimates volatility and classifies into regimes:
    - State 0: "Low-Vol" (volatility below threshold)
    - State 1: "High-Vol" (volatility above threshold)
    """
    
    def __init__(self, p: int = 1, q: int = 1, vol_percentile: float = 75):
        """
        Initialize GARCH model
        
        Args:
            p: GARCH lag order
            q: ARCH lag order
            vol_percentile: Percentile for high/low vol threshold (default: 75th)
        """
        self.p = p
        self.q = q
        self.vol_percentile = vol_percentile
        self.model = None
        self.results = None
        self.threshold = None
        self.regime_names = {0: "Low-Vol", 1: "High-Vol"}
        
    def fit(self, returns: pd.Series, scale: float = 100) -> 'GARCHVolatilityRegime':
        """
        Fit the GARCH model
        
        Args:
            returns: Time series of returns
            scale: Scaling factor for numerical stability
            
        Returns:
            self
        """
        # Prepare data - remove NaN values
        returns_clean = returns.dropna()
        
        # Scale returns for numerical stability
        returns_scaled = returns_clean * scale
        
        try:
            # Fit GARCH(p, q) model
            self.model = arch_model(
                returns_scaled,
                vol='Garch',
                p=self.p,
                q=self.q,
                rescale=False
            )
            
            self.results = self.model.fit(disp='off', show_warning=False)
            
            # Store the index for alignment
            self.index = returns_clean.index
            self.scale = scale
            
            # Calculate threshold based on estimated volatility
            estimated_vol = self.results.conditional_volatility / scale
            self.threshold = np.percentile(estimated_vol, self.vol_percentile)
            
            print(f"GARCH({self.p},{self.q}) model fitted successfully")
            print(f"AIC: {self.results.aic:.2f}, BIC: {self.results.bic:.2f}")
            print(f"Volatility threshold (High/Low): {self.threshold:.6f}")
            
        except Exception as e:
            print(f"Error fitting GARCH model: {e}")
            raise
        
        return self
    
    def get_estimated_volatility(self, annualize: bool = True) -> pd.Series:
        """
        Get estimated conditional volatility
        
        Args:
            annualize: Whether to annualize volatility
            
        Returns:
            Series of estimated volatility
        """
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        vol = self.results.conditional_volatility / self.scale
        
        if annualize:
            vol = vol * np.sqrt(252)
        
        vol.index = self.index
        vol.name = 'estimated_volatility'
        
        return vol
    
    def predict_regimes(self, custom_threshold: Optional[float] = None) -> pd.Series:
        """
        Predict volatility regime based on threshold
        
        Args:
            custom_threshold: Custom threshold (if None, uses percentile-based threshold)
            
        Returns:
            Series of regime predictions (0: Low-Vol, 1: High-Vol)
        """
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        vol = self.get_estimated_volatility(annualize=False)
        threshold = custom_threshold if custom_threshold is not None else self.threshold
        
        # 0 = Low-Vol, 1 = High-Vol
        regimes = (vol > threshold).astype(int)
        regimes.name = 'volatility_regime'
        
        return regimes
    
    def predict_regime_id(self) -> pd.Series:
        """
        Predict regime and return as string ID
        
        Returns:
            Series of regime IDs with descriptive names
        """
        regimes = self.predict_regimes()
        regime_ids = regimes.map(self.regime_names)
        
        return regime_ids
    
    def get_regime_statistics(self) -> Dict:
        """
        Get statistics for each volatility regime
        
        Returns:
            Dictionary with regime statistics
        """
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        vol = self.get_estimated_volatility(annualize=False)
        regimes = self.predict_regimes()
        
        stats = {}
        
        for regime_id in [0, 1]:
            regime_vol = vol[regimes == regime_id]
            
            if len(regime_vol) > 0:
                stats[f'Regime_{regime_id}'] = {
                    'name': self.regime_names[regime_id],
                    'mean_volatility': regime_vol.mean(),
                    'std_volatility': regime_vol.std(),
                    'min_volatility': regime_vol.min(),
                    'max_volatility': regime_vol.max(),
                    'annualized_volatility': regime_vol.mean() * np.sqrt(252),
                    'n_observations': len(regime_vol),
                    'percentage': (len(regime_vol) / len(vol)) * 100
                }
            else:
                stats[f'Regime_{regime_id}'] = {
                    'name': self.regime_names[regime_id],
                    'n_observations': 0,
                    'percentage': 0
                }
        
        stats['threshold'] = self.threshold
        stats['threshold_annualized'] = self.threshold * np.sqrt(252)
        
        return stats
    
    def forecast_volatility(self, horizon: int = 1) -> pd.Series:
        """
        Forecast future volatility
        
        Args:
            horizon: Number of periods to forecast
            
        Returns:
            Series of forecasted volatility
        """
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        forecast = self.results.forecast(horizon=horizon)
        forecast_var = forecast.variance.iloc[-1] / (self.scale ** 2)
        forecast_vol = np.sqrt(forecast_var)
        
        return forecast_vol
    
    def get_current_regime(self) -> Tuple[int, str, float]:
        """
        Get the current (most recent) regime
        
        Returns:
            Tuple of (regime_number, regime_name, current_volatility)
        """
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        regimes = self.predict_regimes()
        vol = self.get_estimated_volatility(annualize=False)
        
        current_regime = regimes.iloc[-1]
        current_vol = vol.iloc[-1]
        regime_name = self.regime_names[current_regime]
        
        return current_regime, regime_name, current_vol
    
    def get_model_parameters(self) -> Dict:
        """
        Get GARCH model parameters
        
        Returns:
            Dictionary of model parameters
        """
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        params = {
            'omega': self.results.params['omega'],
            'alpha[1]': self.results.params['alpha[1]'],
            'beta[1]': self.results.params['beta[1]'],
            'persistence': self.results.params['alpha[1]'] + self.results.params['beta[1]'],
            'unconditional_vol': np.sqrt(self.results.params['omega'] / 
                                        (1 - self.results.params['alpha[1]'] - 
                                         self.results.params['beta[1]'])) / self.scale
        }
        
        return params
    
    def summary(self) -> str:
        """
        Get model summary
        
        Returns:
            String representation of model summary
        """
        if self.results is None:
            return "Model not fitted yet."
        
        return str(self.results.summary())
