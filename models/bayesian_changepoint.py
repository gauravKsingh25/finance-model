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
        Fit the changepoint detection model using online Bayesian algorithm
        
        Args:
            data: Time series data
            
        Returns:
            self
        """
        self.data = data.dropna().values
        n = len(self.data)
        
        # Initialize run length distribution
        # R[t, r] = P(run length at time t is r)
        max_rl = n + 1
        R = np.zeros((max_rl, n + 1))
        R[0, 0] = 1.0  # At t=0, run length is 0 with probability 1
        
        # Store changepoint probabilities
        self.changepoint_probs = np.zeros(n)
        
        # Message passing - Online Bayesian changepoint detection
        for t in range(n):
            # Evaluate predictive probability for each possible run length
            predprobs = self._pred_prob(self.data[:t+1], R[:t+1, t])
            
            # Calculate growth probabilities (no changepoint)
            # P(r_t+1 = r+1 | X_1:t) ∝ P(x_t | r_t, X_1:t-1) * P(r_t | X_1:t-1) * (1-H)
            growth_probs = R[:t+1, t] * predprobs * (1 - self.hazard_rate)
            
            # Calculate changepoint probability (run length returns to 0)
            # P(r_t+1 = 0 | X_1:t) = sum over r of P(x_t | r_t, X_1:t-1) * P(r_t | X_1:t-1) * H
            cp_prob = np.sum(R[:t+1, t] * predprobs * self.hazard_rate)
            
            # Update run length distribution for time t+1
            R[1:t+2, t+1] = growth_probs  # Shift up (run lengths grow by 1)
            R[0, t+1] = cp_prob  # New run starts
            
            # Normalize to ensure probabilities sum to 1
            normalizer = np.sum(R[:t+2, t+1])
            if normalizer > 0:
                R[:t+2, t+1] /= normalizer
            else:
                R[0, t+1] = 1.0  # Fallback
            
            # Store changepoint probability
            # This is the probability that a changepoint occurred at time t
            total_mass = cp_prob + np.sum(growth_probs)
            if total_mass > 0:
                self.changepoint_probs[t] = cp_prob / total_mass
            else:
                self.changepoint_probs[t] = self.hazard_rate
        
        # Store final run length distribution
        self.run_lengths = R
        
        n_significant = np.sum(self.changepoint_probs > 0.5)
        print(f"Bayesian Changepoint Detection fitted on {n} observations")
        print(f"Detected {n_significant} significant changepoints (prob > 50%)")
        print(f"Mean changepoint probability: {np.mean(self.changepoint_probs):.4f}")
        print(f"Max changepoint probability: {np.max(self.changepoint_probs):.4f}")
        
        return self
    
    def _pred_prob(self, data: np.ndarray, run_lengths: np.ndarray) -> np.ndarray:
        """
        Calculate predictive probabilities using Student-t distribution
        
        Args:
            data: Observed data up to current time
            run_lengths: Current run length distribution
            
        Returns:
            Predictive probabilities for each possible run length
        """
        if len(data) == 0:
            return np.array([1.0])
        
        n = len(run_lengths)
        probs = np.zeros(n)
        current_obs = data[-1]
        
        for r in range(n):
            if r == 0:
                # No history - use uninformative prior
                # Use global statistics
                if len(data) > 1:
                    global_mean = np.mean(data[:-1])
                    global_std = np.std(data[:-1])
                    if global_std > 0:
                        probs[r] = stats.norm.pdf(current_obs, global_mean, global_std)
                    else:
                        probs[r] = 1.0
                else:
                    probs[r] = 1.0
            else:
                # Use last r observations (not including current)
                start_idx = max(0, len(data) - r - 1)
                end_idx = len(data) - 1
                
                if end_idx > start_idx:
                    recent_data = data[start_idx:end_idx]
                    
                    # Bayesian predictive distribution (Student-t)
                    n_obs = len(recent_data)
                    
                    if n_obs >= 1:
                        # Compute sufficient statistics
                        mean_est = np.mean(recent_data)
                        
                        if n_obs > 1:
                            var_est = np.var(recent_data, ddof=1)
                            
                            if var_est > 0:
                                # Student-t predictive distribution
                                # Scale accounts for estimation uncertainty
                                scale = np.sqrt(var_est * (1 + 1/n_obs))
                                df = n_obs - 1
                                
                                # Calculate standardized value
                                t_val = (current_obs - mean_est) / scale
                                probs[r] = stats.t.pdf(t_val, df) / scale
                            else:
                                # Zero variance - use narrow normal
                                probs[r] = stats.norm.pdf(current_obs, mean_est, 1e-6)
                        else:
                            # Single observation - use broad prior
                            probs[r] = stats.norm.pdf(current_obs, mean_est, 1.0)
                    else:
                        probs[r] = 1.0
                else:
                    probs[r] = 1.0
        
        # Avoid numerical issues
        probs = np.maximum(probs, 1e-100)
        probs = np.minimum(probs, 1e100)
        
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
    
    def plot_changepoints(self, title: str = "Bayesian Changepoint Detection", 
                         save_path: str = 'reports/bayesian_changepoint_detection.png',
                         threshold: float = 0.5) -> None:
        """
        Plot data with changepoint probabilities
        
        Args:
            title: Title for the plot
            save_path: Path to save the figure
            threshold: Threshold for changepoint detection visualization
        """
        try:
            import matplotlib.pyplot as plt
            
            # Create figure with better layout
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
            
            # Get changepoint locations
            changepoint_locs = self.get_changepoint_locations(threshold=threshold)
            
            # Plot 1: Time Series Data with changepoints marked
            ax1.plot(self.data, 'b-', linewidth=1.5, label='Data', alpha=0.7)
            
            # Mark changepoints on the data
            if len(changepoint_locs) > 0:
                ax1.scatter(changepoint_locs, self.data[changepoint_locs], 
                           color='red', s=100, marker='v', 
                           label=f'Changepoints (n={len(changepoint_locs)})',
                           zorder=5, edgecolors='darkred', linewidths=2)
                
                # Add vertical lines at changepoints
                for cp in changepoint_locs:
                    ax1.axvline(x=cp, color='red', linestyle='--', alpha=0.3, linewidth=1.5)
            
            ax1.set_ylabel('Value', fontsize=12, fontweight='bold')
            ax1.set_title(f'{title}\nTime Series Data', fontsize=14, fontweight='bold')
            ax1.legend(loc='best', fontsize=10)
            ax1.grid(True, alpha=0.3, linestyle='--')
            
            # Plot 2: Changepoint Detection Probabilities
            time_indices = np.arange(len(self.changepoint_probs))
            
            # Plot probabilities
            ax2.plot(time_indices, self.changepoint_probs, 'b-', 
                    linewidth=1.5, label='Changepoint Probability', alpha=0.7)
            
            # Add threshold line
            ax2.axhline(y=threshold, color='darkgreen', linestyle='--', 
                       linewidth=2, alpha=0.7, label=f'{threshold*100:.0f}% threshold')
            
            # Fill area above threshold
            ax2.fill_between(time_indices, 
                            threshold, self.changepoint_probs, 
                            where=(self.changepoint_probs > threshold),
                            color='red', alpha=0.3, label='Changepoint detected')
            
            # Mark detected changepoints
            if len(changepoint_locs) > 0:
                ax2.scatter(changepoint_locs, self.changepoint_probs[changepoint_locs],
                           color='red', s=100, marker='o', zorder=5,
                           edgecolors='darkred', linewidths=2)
            
            ax2.set_ylabel('Changepoint Probability', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Time', fontsize=12, fontweight='bold')
            ax2.set_title('Changepoint Detection Probabilities', fontsize=14, fontweight='bold')
            ax2.legend(loc='best', fontsize=10)
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.set_ylim([0, 1.0])
            ax2.set_xlim([0, len(self.changepoint_probs)])
            
            # Add statistics box
            stats_text = f"Changepoints detected: {len(changepoint_locs)}\n"
            stats_text += f"Mean probability: {np.mean(self.changepoint_probs):.4f}\n"
            stats_text += f"Max probability: {np.max(self.changepoint_probs):.4f}"
            
            ax2.text(0.02, 0.98, stats_text, 
                    transform=ax2.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=9, family='monospace')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ High-quality plot saved to {save_path}")
            plt.close()
            
        except ImportError:
            print("Matplotlib not available for plotting")
        except Exception as e:
            print(f"Error creating plot: {e}")
