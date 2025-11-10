"""
Bayesian Changepoint Detection (BCD) - BOCPD Implementation
Implements Bayesian Online Changepoint Detection (Adams & MacKay 2007)
Purpose: The Alarm - Signals when market regime is about to change

This is a TRUE Bayesian changepoint detection using:
- Run-length distribution (posterior over time since last changepoint)
- Predictive probabilities with conjugate priors
- Online Bayesian inference (uses ALL past data, not just a window)
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from scipy import stats
from scipy.special import logsumexp
import warnings
warnings.filterwarnings('ignore')


class BayesianChangepoint:
    """
    Bayesian Online Changepoint Detection (BOCPD)
    
    Implements the algorithm from Adams & MacKay (2007)
    "Bayesian Online Changepoint Detection"
    
    Uses:
    - Run-length distribution to track time since last changepoint
    - Student-t predictive distribution with conjugate Normal-Gamma prior
    - All historical data (not just a sliding window)
    """
    
    def __init__(self, hazard_lambda: float = 250, 
                 alpha0: float = 0.1, beta0: float = 0.1,
                 kappa0: float = 1.0, mu0: float = 0.0):
        """
        Initialize Bayesian Online Changepoint Detector
        
        Args:
            hazard_lambda: Timescale parameter for hazard function (expected time between CPs)
                          Higher = expect CPs less frequently
            alpha0: Prior precision shape parameter
            beta0: Prior precision rate parameter  
            kappa0: Prior mean precision
            mu0: Prior mean
        """
        self.hazard_lambda = hazard_lambda
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.kappa0 = kappa0
        self.mu0 = mu0
        
        self.changepoint_probs = None
        self.run_length_dist = None
        self.data = None
        self.maxes = None
        
    def _constant_hazard(self, r: np.ndarray) -> np.ndarray:
        """
        Constant hazard function: h(r) = 1/lambda
        
        Args:
            r: Run lengths
            
        Returns:
            Hazard probabilities
        """
        return 1.0 / self.hazard_lambda * np.ones_like(r)
        
    def fit(self, data: pd.Series) -> 'BayesianChangepoint':
        """
        Fit BOCPD model using online Bayesian inference
        
        Args:
            data: Time series data
            
        Returns:
            self
        """
        self.data = data.dropna().values
        T = len(self.data)
        
        # Initialize run-length distribution
        # R[t, r] = P(run length = r | data up to t)
        # We store log probabilities for numerical stability
        log_R = -np.inf * np.ones((T+1, T+1))
        log_R[0, 0] = 0  # log(1) = 0, we start with run length 0
        
        # Store changepoint probabilities and growth probabilities
        self.changepoint_probs = np.zeros(T)
        self.maxes = np.zeros(T, dtype=int)  # Most likely run length at each time
        
        # Sufficient statistics for each run length
        # We maintain these incrementally for efficiency
        alphas = np.zeros((T+1, T+1))
        betas = np.zeros((T+1, T+1))
        kappas = np.zeros((T+1, T+1))
        mus = np.zeros((T+1, T+1))
        
        # Initialize for run length 0
        alphas[0, 0] = self.alpha0
        betas[0, 0] = self.beta0
        kappas[0, 0] = self.kappa0
        mus[0, 0] = self.mu0
        
        # Process each observation
        for t in range(T):
            # Current observation
            x = self.data[t]
            
            # Determine which run lengths we need to consider
            # (run lengths that have non-zero probability)
            max_r = min(t + 1, T)
            
            # Compute predictive probability for each run length
            log_pis = np.zeros(max_r)
            
            for r in range(max_r):
                if log_R[t, r] > -np.inf:
                    # Get current sufficient statistics for this run length
                    alpha_r = alphas[t, r]
                    beta_r = betas[t, r]
                    kappa_r = kappas[t, r]
                    mu_r = mus[t, r]
                    
                    # Student-t predictive distribution parameters
                    df = 2 * alpha_r
                    loc = mu_r
                    scale_sq = beta_r * (kappa_r + 1) / (alpha_r * kappa_r)
                    scale = np.sqrt(max(scale_sq, 1e-10))
                    
                    # Compute log predictive probability
                    try:
                        log_pis[r] = stats.t.logpdf(x, df=df, loc=loc, scale=scale)
                    except:
                        log_pis[r] = -1e10
                    
                    # Clip extreme values
                    log_pis[r] = np.clip(log_pis[r], -1e10, 1e10)
            
            # Compute growth probabilities (no changepoint)
            hazard_r = self._constant_hazard(np.arange(max_r))
            log_growth_probs = log_R[t, :max_r] + log_pis + np.log(1 - hazard_r)
            
            # Compute changepoint probabilities  
            log_cp_prob = logsumexp(log_R[t, :max_r] + log_pis + np.log(hazard_r))
            
            # Store changepoint probability BEFORE normalization
            # This gives us the relative evidence for a changepoint
            log_total_evidence = logsumexp([log_cp_prob, logsumexp(log_growth_probs)])
            self.changepoint_probs[t] = np.exp(log_cp_prob - log_total_evidence)
            
            # Update run length distribution
            log_R[t+1, 0] = log_cp_prob  # Changepoint occurred, reset to 0
            log_R[t+1, 1:max_r+1] = log_growth_probs  # No changepoint, increment run lengths
            
            # Normalize
            log_R[t+1, :max_r+1] -= logsumexp(log_R[t+1, :max_r+1])
            
            # Store most likely run length
            self.maxes[t] = np.argmax(log_R[t+1, :max_r+1])
            
            # Update sufficient statistics for next iteration
            for r in range(max_r + 1):
                if r == 0:
                    # Just had a changepoint, reset to prior
                    alphas[t+1, 0] = self.alpha0
                    betas[t+1, 0] = self.beta0  
                    kappas[t+1, 0] = self.kappa0
                    mus[t+1, 0] = self.mu0
                else:
                    # Update from previous run length r-1
                    alpha_prev = alphas[t, r-1]
                    beta_prev = betas[t, r-1]
                    kappa_prev = kappas[t, r-1]
                    mu_prev = mus[t, r-1]
                    
                    # Bayesian update with new observation
                    kappas[t+1, r] = kappa_prev + 1
                    mus[t+1, r] = (kappa_prev * mu_prev + x) / kappas[t+1, r]
                    alphas[t+1, r] = alpha_prev + 0.5
                    betas[t+1, r] = beta_prev + \
                        (kappa_prev * (x - mu_prev)**2) / (2 * (kappa_prev + 1))
        
        # Store final run length distribution
        self.run_length_dist = np.exp(log_R)
        
        n_significant = np.sum(self.changepoint_probs > 0.5)
        print(f"Bayesian Online Changepoint Detection fitted on {T} observations")
        print(f"Detected {n_significant} significant changepoints (prob > 50%)")
        print(f"Mean changepoint probability: {np.mean(self.changepoint_probs):.4f}")
        print(f"Max changepoint probability: {np.max(self.changepoint_probs):.4f}")
        print(f"Mean run length: {np.mean(self.maxes):.1f}")
        
        return self
    
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
            'hazard_lambda': self.hazard_lambda,
            'mean_run_length': float(np.mean(self.maxes)) if self.maxes is not None else 0.0
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
        Plot data with changepoint probabilities and run-length distribution
        
        Args:
            title: Title for the plot
            save_path: Path to save the figure
            threshold: Threshold for changepoint detection visualization
        """
        try:
            import matplotlib.pyplot as plt
            
            # Create figure with better layout - 3 subplots
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
            
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
            ax1.set_title(f'{title}\nTime Series Data with Detected Changepoints', 
                         fontsize=14, fontweight='bold')
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
            ax2.set_title('BOCPD Changepoint Probabilities (uses all past data)', 
                         fontsize=14, fontweight='bold')
            ax2.legend(loc='best', fontsize=10)
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.set_ylim([0, 1.0])
            ax2.set_xlim([0, len(self.changepoint_probs)])
            
            # Plot 3: Run-length distribution (most likely run length over time)
            if self.maxes is not None:
                ax3.plot(time_indices, self.maxes, 'g-', 
                        linewidth=1.5, label='Most Likely Run Length', alpha=0.7)
                
                # Mark changepoints where run length drops to 0
                if len(changepoint_locs) > 0:
                    ax3.scatter(changepoint_locs, self.maxes[changepoint_locs],
                               color='red', s=100, marker='v', zorder=5,
                               edgecolors='darkred', linewidths=2,
                               label='Changepoint (run length reset)')
                
                ax3.set_ylabel('Run Length', fontsize=12, fontweight='bold')
                ax3.set_xlabel('Time', fontsize=12, fontweight='bold')
                ax3.set_title('Run Length Distribution (time since last changepoint)', 
                             fontsize=14, fontweight='bold')
                ax3.legend(loc='best', fontsize=10)
                ax3.grid(True, alpha=0.3, linestyle='--')
                ax3.set_xlim([0, len(self.changepoint_probs)])
            
            # Add statistics box
            stats_text = f"Algorithm: BOCPD (Adams & MacKay 2007)\n"
            stats_text += f"Changepoints detected: {len(changepoint_locs)}\n"
            stats_text += f"Mean probability: {np.mean(self.changepoint_probs):.4f}\n"
            stats_text += f"Max probability: {np.max(self.changepoint_probs):.4f}\n"
            if self.maxes is not None:
                stats_text += f"Mean run length: {np.mean(self.maxes):.1f}\n"
            stats_text += f"Hazard λ: {self.hazard_lambda}"
            
            ax2.text(0.02, 0.98, stats_text, 
                    transform=ax2.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=9, family='monospace')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"[OK] High-quality plot saved to {save_path}")
            plt.close()
            
        except ImportError:
            print("Matplotlib not available for plotting")
        except Exception as e:
            print(f"Error creating plot: {e}")
            import traceback
            traceback.print_exc()
