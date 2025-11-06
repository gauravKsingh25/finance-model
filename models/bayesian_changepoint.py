"""
Bayesian Changepoint Detection (BCD)
Detects structural breaks in time series using Bayesian methods
Purpose: The Alarm - Signals when market regime is about to change
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class BayesianChangepoint:
    """
    Bayesian Changepoint Detection Model
    
    Detects structural breaks in time series data
    Outputs: Probability of changepoint at each time step
    """
    
    def __init__(self, hazard_rate: float = 0.01):
        """
        Initialize Bayesian Changepoint Detector
        
        Args:
            hazard_rate: Prior probability of changepoint at each step (default: 1%)
        """
        self.hazard_rate = hazard_rate
        self.changepoint_probs = None
        self.run_lengths = None
        self.data = None
        
    def fit(self, data: pd.Series) -> 'BayesianChangepoint':
        """
        Fit the changepoint detection model
        
        Args:
            data: Time series data
            
        Returns:
            self
        """
        self.data = data.dropna().values
        n = len(self.data)
        
        # Initialize
        self.run_lengths = np.zeros((n + 1, n + 1))
        self.run_lengths[0, 0] = 1
        
        # Changepoint probabilities
        self.changepoint_probs = np.zeros(n)
        
        # Online Bayesian changepoint detection
        for t in range(n):
            # Predictive probabilities
            predprobs = self._pred_prob(self.data[:t+1], self.run_lengths[:t+1, t])
            
            # Growth probabilities (no changepoint)
            growth_probs = self.run_lengths[:t+1, t] * predprobs * (1 - self.hazard_rate)
            
            # Changepoint probability (new run starts)
            cp_prob = np.sum(self.run_lengths[:t+1, t] * predprobs * self.hazard_rate)
            
            # Update run length distribution
            self.run_lengths[1:t+2, t+1] = growth_probs
            self.run_lengths[0, t+1] = cp_prob
            
            # Normalize
            self.run_lengths[:t+2, t+1] /= np.sum(self.run_lengths[:t+2, t+1])
            
            # Store changepoint probability
            self.changepoint_probs[t] = cp_prob / (cp_prob + np.sum(growth_probs))
        
        print(f"Bayesian Changepoint Detection fitted on {n} observations")
        print(f"Detected {np.sum(self.changepoint_probs > 0.5)} significant changepoints (prob > 50%)")
        
        return self
    
    def _pred_prob(self, data: np.ndarray, run_lengths: np.ndarray) -> np.ndarray:
        """
        Calculate predictive probabilities using Student-t distribution
        
        Args:
            data: Observed data
            run_lengths: Current run length distribution
            
        Returns:
            Predictive probabilities
        """
        if len(data) == 0:
            return np.array([1.0])
        
        n = len(run_lengths)
        probs = np.zeros(n)
        
        for r in range(n):
            if r == 0:
                # No history, use uninformative prior
                probs[r] = 1.0
            else:
                # Use last r observations
                if r > len(data):
                    r = len(data)
                recent_data = data[-r:]
                
                # Student-t parameters
                mean = np.mean(recent_data)
                var = np.var(recent_data) if len(recent_data) > 1 else 1.0
                df = max(len(recent_data) - 1, 1)
                
                # Calculate probability of current observation
                if var > 0:
                    t_stat = (data[-1] - mean) / np.sqrt(var)
                    probs[r] = stats.t.pdf(t_stat, df)
                else:
                    probs[r] = 1.0
        
        # Avoid numerical issues
        probs = np.maximum(probs, 1e-10)
        
        return probs
    
    def get_changepoint_probabilities(self) -> pd.Series:
        """
        Get changepoint probabilities for each time step
        
        Returns:
            Series of changepoint probabilities
        """
        if self.changepoint_probs is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return pd.Series(self.changepoint_probs, name='changepoint_prob')
    
    def detect_changepoints(self, threshold: float = 0.5) -> pd.Series:
        """
        Detect changepoints above threshold
        
        Args:
            threshold: Probability threshold for changepoint detection
            
        Returns:
            Binary series (1 = changepoint, 0 = no changepoint)
        """
        if self.changepoint_probs is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        changepoints = (self.changepoint_probs > threshold).astype(int)
        return pd.Series(changepoints, name='changepoint_detected')
    
    def get_changepoint_locations(self, threshold: float = 0.5) -> List[int]:
        """
        Get indices of detected changepoints
        
        Args:
            threshold: Probability threshold
            
        Returns:
            List of changepoint indices
        """
        changepoints = self.detect_changepoints(threshold)
        return list(np.where(changepoints == 1)[0])
    
    def get_statistics(self) -> Dict:
        """
        Get changepoint detection statistics
        
        Returns:
            Dictionary with statistics
        """
        if self.changepoint_probs is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        stats_dict = {
            'n_observations': len(self.changepoint_probs),
            'mean_cp_prob': float(np.mean(self.changepoint_probs)),
            'max_cp_prob': float(np.max(self.changepoint_probs)),
            'n_significant_cp_50': int(np.sum(self.changepoint_probs > 0.5)),
            'n_significant_cp_75': int(np.sum(self.changepoint_probs > 0.75)),
            'n_significant_cp_90': int(np.sum(self.changepoint_probs > 0.90)),
            'hazard_rate': self.hazard_rate
        }
        
        return stats_dict
    
    def get_current_changepoint_prob(self) -> float:
        """
        Get current (most recent) changepoint probability
        
        Returns:
            Current changepoint probability
        """
        if self.changepoint_probs is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return float(self.changepoint_probs[-1])
    
    def plot_changepoints(self) -> None:
        """Plot data with changepoint probabilities"""
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            
            # Plot data
            ax1.plot(self.data, 'b-', linewidth=1)
            ax1.set_ylabel('Value')
            ax1.set_title('Time Series Data')
            ax1.grid(True, alpha=0.3)
            
            # Plot changepoint probabilities
            ax2.plot(self.changepoint_probs, 'r-', linewidth=1)
            ax2.axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='50% threshold')
            ax2.fill_between(range(len(self.changepoint_probs)), 
                            0, self.changepoint_probs, 
                            where=(self.changepoint_probs > 0.5),
                            color='red', alpha=0.3, label='Changepoint detected')
            ax2.set_ylabel('Changepoint Probability')
            ax2.set_xlabel('Time')
            ax2.set_title('Changepoint Detection Probabilities')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim([0, 1])
            
            plt.tight_layout()
            plt.savefig('reports/bayesian_changepoint_detection.png', dpi=150, bbox_inches='tight')
            print("Plot saved to reports/bayesian_changepoint_detection.png")
            
        except ImportError:
            print("Matplotlib not available for plotting")
