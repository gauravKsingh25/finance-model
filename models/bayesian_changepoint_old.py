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
    
    def __init__(self, hazard_rate: float = 0.01, sensitivity: float = 1.0):
        """
        Initialize Bayesian Changepoint Detector
        
        Args:
            hazard_rate: Prior probability of changepoint at each step (default: 1%)
            sensitivity: Sensitivity parameter for detection (higher = more sensitive)
        """
        self.hazard_rate = hazard_rate
        self.sensitivity = sensitivity
        self.changepoint_probs = None
        self.run_lengths = None
        self.data = None
        
    def fit(self, data: pd.Series) -> 'BayesianChangepoint':
        """
        Fit the changepoint detection model using online Bayesian algorithm
        Based on Adams & MacKay (2007) Bayesian Online Changepoint Detection
        
        Args:
            data: Time series data
            
        Returns:
            self
        """
        self.data = data.dropna().values
        n = len(self.data)
        
        # Initialize
        # R[r, t] = P(run length = r at time t)
        R = np.zeros((n + 1, n + 1))
        R[0, 0] = 1.0
        
        # Store changepoint probabilities
        self.changepoint_probs = np.zeros(n)
        
        # Sufficient statistics for Gaussian model
        # Using conjugate Normal-Gamma prior with data-driven initialization
        if n > 10:
            mu0 = np.mean(self.data[:10])
            sigma0_sq = np.var(self.data[:10]) * self.sensitivity
        else:
            mu0 = 0
            sigma0_sq = np.var(self.data) if n > 1 else 1.0
            
        kappa0 = 0.1  # Low confidence in prior mean
        alpha0 = 2.0  # Shape parameter
        beta0 = sigma0_sq * (alpha0 - 1)  # Scale parameter
        
        # Online Bayesian changepoint detection
        for t in range(n):
            # Allocate space for new probabilities
            R_new = np.zeros(t + 2)
            
            # Evaluate predictive probability for each run length hypothesis
            for r in range(t + 1):
                if R[r, t] == 0:
                    continue
                    
                # Calculate posterior parameters for this run length
                if r == 0:
                    # No observations in this run yet
                    mu_n = mu0
                    kappa_n = kappa0
                    alpha_n = alpha0
                    beta_n = beta0
                else:
                    # Get data for this run
                    run_data = self.data[max(0, t-r):t]
                    
                    if len(run_data) > 0:
                        # Update parameters
                        n_obs = len(run_data)
                        data_mean = np.mean(run_data)
                        
                        kappa_n = kappa0 + n_obs
                        mu_n = (kappa0 * mu0 + n_obs * data_mean) / kappa_n
                        alpha_n = alpha0 + n_obs / 2
                        
                        if n_obs > 1:
                            data_var = np.var(run_data, ddof=0)
                            beta_n = beta0 + 0.5 * n_obs * data_var + \
                                    0.5 * kappa0 * n_obs * (data_mean - mu0)**2 / kappa_n
                        else:
                            beta_n = beta0
                    else:
                        mu_n = mu0
                        kappa_n = kappa0
                        alpha_n = alpha0
                        beta_n = beta0
                
                # Predictive probability (Student-t)
                x = self.data[t]
                df = 2 * alpha_n
                loc = mu_n
                scale = np.sqrt(beta_n * (kappa_n + 1) / (alpha_n * kappa_n))
                
                # Avoid numerical issues
                scale = max(scale, 1e-10)
                
                # Calculate log probability to avoid underflow
                z = (x - loc) / scale
                try:
                    log_pred_prob = stats.t.logpdf(z, df) - np.log(scale)
                    pred_prob = np.exp(log_pred_prob)
                except:
                    pred_prob = 1e-10
                
                pred_prob = np.clip(pred_prob, 1e-100, 1e100)
                
                # Update run length distribution
                # Growth: run length increases by 1 (no changepoint)
                R_new[r + 1] += R[r, t] * pred_prob * (1 - self.hazard_rate)
                
                # Changepoint: run length resets to 0
                R_new[0] += R[r, t] * pred_prob * self.hazard_rate
            
            # Normalize
            sum_R = np.sum(R_new)
            if sum_R > 0:
                R_new /= sum_R
            else:
                R_new[0] = 1.0
            
            # Store new run length distribution
            R[:t+2, t+1] = R_new
            
            # Changepoint probability at time t+1 is proportional to mass at r=0
            # But we want the probability that a changepoint JUST occurred
            # This is the ratio of changepoint mass to total mass before normalization
            cp_mass = R_new[0] * sum_R if sum_R > 0 else 0
            growth_mass = np.sum(R_new[1:]) * sum_R if sum_R > 0 else 0
            total_mass = cp_mass + growth_mass
            
            if total_mass > 0:
                self.changepoint_probs[t] = cp_mass / total_mass
            else:
                self.changepoint_probs[t] = 0.0
        
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
        Calculate predictive probabilities using Gaussian approximation
        
        Args:
            data: Observed data up to current time
            run_lengths: Current run length distribution
            
        Returns:
            Predictive probabilities for each possible run length
        """
        if len(data) <= 1:
            return np.ones(len(run_lengths))
        
        n = len(run_lengths)
        probs = np.ones(n)
        current_obs = data[-1]
        
        # Use Gaussian approximation for speed and stability
        for r in range(n):
            if r == 0:
                # No history - use global statistics
                if len(data) > 2:
                    global_mean = np.mean(data[:-1])
                    global_std = np.std(data[:-1])
                    if global_std > 1e-10:
                        probs[r] = np.exp(-0.5 * ((current_obs - global_mean) / global_std) ** 2)
                    else:
                        probs[r] = 1.0
                else:
                    probs[r] = 1.0
            else:
                # Use last min(r, len(data)-1) observations
                lookback = min(r, len(data) - 1)
                if lookback >= 1:
                    start_idx = len(data) - 1 - lookback
                    recent_data = data[start_idx:len(data)-1]
                    
                    if len(recent_data) > 0:
                        mean_est = np.mean(recent_data)
                        
                        if len(recent_data) > 1:
                            std_est = np.std(recent_data, ddof=1)
                            # Add small regularization
                            std_est = max(std_est, 1e-10)
                        else:
                            # Single observation - use broader uncertainty
                            std_est = np.std(data) if len(data) > 1 else 1.0
                        
                        # Gaussian likelihood
                        z_score = (current_obs - mean_est) / std_est
                        probs[r] = np.exp(-0.5 * z_score ** 2) / (std_est * np.sqrt(2 * np.pi))
                    else:
                        probs[r] = 1.0
                else:
                    probs[r] = 1.0
        
        # Avoid numerical issues
        probs = np.clip(probs, 1e-100, 1e100)
        
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
