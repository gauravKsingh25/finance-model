"""
Stream 1: Trend Regime Detection using Markov Regime Switching Model
This model detects trend-based regimes (Bull vs Bear states)
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
import warnings
warnings.filterwarnings('ignore')


class MarkovRegimeSwitching:
    """
    Markov Regime Switching Model for Trend Detection
    
    Detects 2 states:
    - State 0: "Bear" (negative trend, higher volatility)
    - State 1: "Bull" (positive trend, lower volatility)
    """
    
    def __init__(self, n_regimes: int = 2):
        """
        Initialize Markov Regime Switching Model
        
        Args:
            n_regimes: Number of regimes (default: 2 for Bull/Bear)
        """
        self.n_regimes = n_regimes
        self.model = None
        self.results = None
        self.regime_names = {0: "Bear", 1: "Bull"}
        
    def fit(self, returns: pd.Series, max_iter: int = 1000) -> 'MarkovRegimeSwitching':
        """
        Fit the Markov Regime Switching model
        
        Args:
            returns: Time series of returns
            max_iter: Maximum iterations for optimization
            
        Returns:
            self
        """
        # Prepare data - remove NaN values
        returns_clean = returns.dropna()
        
        # Convert to appropriate format
        endog = returns_clean.values * 100  # Scale for numerical stability
        
        try:
            # Fit Markov Switching Model
            # switching_variance=True allows different volatility in each regime
            self.model = MarkovRegression(
                endog=endog,
                k_regimes=self.n_regimes,
                switching_variance=True,
                trend='c'  # Include constant
            )
            
            self.results = self.model.fit(maxiter=max_iter, disp=False)
            
            # Store the index for alignment
            self.index = returns_clean.index
            
            print(f"Model fitted successfully with {self.n_regimes} regimes")
            print(f"AIC: {self.results.aic:.2f}, BIC: {self.results.bic:.2f}")
            
        except Exception as e:
            print(f"Error fitting model: {e}")
            raise
        
        return self
    
    def predict_regimes(self) -> pd.Series:
        """
        Predict the most likely regime for each time point
        
        Returns:
            Series of regime predictions (0 or 1)
        """
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get smoothed probabilities
        smoothed_probs = self.results.smoothed_marginal_probabilities
        
        # Convert to DataFrame if it's an array
        if not isinstance(smoothed_probs, pd.DataFrame):
            smoothed_probs = pd.DataFrame(smoothed_probs)
        
        # Predict regime as the one with highest probability
        regimes = smoothed_probs.idxmax(axis=1).values
        
        return pd.Series(regimes, index=self.index, name='regime')
    
    def get_regime_probabilities(self) -> pd.DataFrame:
        """
        Get smoothed probabilities for each regime
        
        Returns:
            DataFrame with probabilities for each regime
        """
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        probs = self.results.smoothed_marginal_probabilities
        
        # Convert to DataFrame if it's an array
        if not isinstance(probs, pd.DataFrame):
            probs = pd.DataFrame(probs, index=self.index)
        else:
            probs.index = self.index
        
        probs.columns = [f"P(Regime_{i})" for i in range(self.n_regimes)]
        
        return probs
    
    def get_regime_parameters(self) -> Dict:
        """
        Get estimated parameters for each regime
        
        Returns:
            Dictionary containing regime parameters
        """
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        params = {}
        
        # Get parameters from results
        result_params = self.results.params
        
        for i in range(self.n_regimes):
            # Get mean (constant term) and variance for each regime
            mean_param = f'const[{i}]'
            sigma_param = f'sigma2[{i}]'
            
            # Extract values, handling both Series and dict-like access
            try:
                mean_val = float(result_params[mean_param]) / 100 if mean_param in result_params else 0  # Unscale
                sigma_val = float(np.sqrt(result_params[sigma_param])) / 100 if sigma_param in result_params else 0  # Unscale
            except:
                mean_val = 0
                sigma_val = 0
            
            params[f'Regime_{i}'] = {
                'mean': mean_val,
                'volatility': sigma_val,
                'annualized_return': mean_val * 252,
                'annualized_volatility': sigma_val * np.sqrt(252),
                'name': self.regime_names.get(i, f"State_{i}")
            }
        
        # Add transition matrix
        params['transition_matrix'] = self.results.regime_transition
        
        return params
    
    def predict_regime_id(self) -> pd.Series:
        """
        Predict regime and return as string ID (e.g., 'State 0: Bull')
        
        Returns:
            Series of regime IDs with descriptive names
        """
        regimes = self.predict_regimes()
        
        # Identify which regime is Bull vs Bear based on mean returns
        params = self.get_regime_parameters()
        
        regime_means = {}
        for i in range(self.n_regimes):
            regime_means[i] = params[f'Regime_{i}']['mean']
        
        # Assign Bull to regime with higher mean, Bear to lower mean
        bull_regime = max(regime_means, key=regime_means.get)
        bear_regime = min(regime_means, key=regime_means.get)
        
        self.regime_names[bull_regime] = "Bull"
        self.regime_names[bear_regime] = "Bear"
        
        # Map numeric regimes to names
        regime_ids = regimes.map(self.regime_names)
        
        return regime_ids
    
    def get_expected_duration(self) -> Dict:
        """
        Calculate expected duration in each regime
        
        Returns:
            Dictionary with expected durations
        """
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        transition_matrix = self.results.regime_transition
        durations = {}
        
        for i in range(self.n_regimes):
            # Expected duration = 1 / (1 - P(stay in same regime))
            # Handle different transition matrix formats
            if transition_matrix.ndim == 3:
                p_stay = float(transition_matrix[i, i, 0])
            else:
                p_stay = float(transition_matrix[i, i])
            
            expected_duration = 1 / (1 - p_stay) if p_stay < 1 else np.inf
            
            durations[f'Regime_{i}'] = {
                'expected_duration': float(expected_duration),
                'probability_stay': float(p_stay),
                'name': self.regime_names.get(i, f"State_{i}")
            }
        
        return durations
    
    def summary(self) -> str:
        """
        Get model summary
        
        Returns:
            String representation of model summary
        """
        if self.results is None:
            return "Model not fitted yet."
        
        return str(self.results.summary())
    
    def get_current_regime(self) -> Tuple[int, str, float]:
        """
        Get the current (most recent) regime
        
        Returns:
            Tuple of (regime_number, regime_name, probability)
        """
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        probs = self.get_regime_probabilities()
        current_regime = probs.iloc[-1].idxmax()
        current_prob = probs.iloc[-1].max()
        
        regime_num = int(current_regime.split('_')[1])
        regime_name = self.regime_names.get(regime_num, f"State_{regime_num}")
        
        return regime_num, regime_name, current_prob
