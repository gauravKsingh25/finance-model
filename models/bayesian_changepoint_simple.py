"""
Bayesian Changepoint Detection - Simplified Working Implementation
Uses cumulative predictive probability to detect regime changes
This is a TRUE Bayesian implementation that uses all past data
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class BayesianChangepoint:
    """
    Bayesian Changepoint Detection using Student-t predictive distribution
    
    At each timestep, we compute:
    1. The predictive probability under the current regime (all past data)
    2. The predictive probability under a NEW regime (just recent data)
    3. Changepoint probability is high when new regime fits better
    """
    
    def __init__(self, min_segment_length: int = 30, sensitivity: float = 2.0):
        """
        Args:
            min_segment_length: Minimum observations to establish a regime
            sensitivity: Sensitivity to changes (higher = more sensitive)
        """
        self.min_segment_length = min_segment_length
        self.sensitivity = sensitivity
        self.changepoint_probs = None
        self.data = None
        self.maxes = None
        
    def fit(self, data: pd.Series) -> 'BayesianChangepoint':
        """Fit the model"""
        self.data = data.dropna().values
        T = len(self.data)
        
        self.changepoint_probs = np.zeros(T)
        self.maxes = np.zeros(T, dtype=int)
        
        # Use first min_segment_length as initialization
        for t in range(self.min_segment_length, T):
            # Compute changepoint probability
            cp_prob = self._compute_changepoint_prob(t)
            self.changepoint_probs[t] = cp_prob
            
            # Estimate current run length (simple heuristic)
            self.maxes[t] = self._estimate_run_length(t)
        
        n_significant = np.sum(self.changepoint_probs > 0.5)
        print(f"Bayesian Changepoint Detection fitted on {T} observations")
        print(f"Detected {n_significant} significant changepoints (prob > 50%)")
        print(f"Mean changepoint probability: {np.mean(self.changepoint_probs):.4f}")
        print(f"Max changepoint probability: {np.max(self.changepoint_probs):.4f}")
        print(f"Mean run length: {np.mean(self.maxes):.1f}")
        
        return self
    
    def _compute_changepoint_prob(self, t: int) -> float:
        """
        Compute changepoint probability at time t
        
        Compares:
        - Long-term model: fit to data[0:t]  
        - Short-term model: fit to data[t-min_segment_length:t]
        
        If short-term fits current point much better, likely a changepoint
        """
        x = self.data[t]
        
        # Long-term statistics (all data up to t)
        long_data = self.data[:t]
        mu_long = np.mean(long_data)
        sigma_long = np.std(long_data, ddof=1)
        if sigma_long < 1e-10:
            sigma_long = 1e-10
        n_long = len(long_data)
        
        # Short-term statistics (recent data)
        start_idx = max(0, t - self.min_segment_length)
        short_data = self.data[start_idx:t]
        mu_short = np.mean(short_data)
        sigma_short = np.std(short_data, ddof=1)
        if sigma_short < 1e-10:
            sigma_short = 1e-10
        n_short = len(short_data)
        
        # Student-t predictive for long-term model
        df_long = max(n_long - 1, 1)
        scale_long = sigma_long * np.sqrt(1 + 1/n_long)
        log_p_long = stats.t.logpdf(x, df=df_long, loc=mu_long, scale=scale_long)
        
        # Student-t predictive for short-term model  
        df_short = max(n_short - 1, 1)
        scale_short = sigma_short * np.sqrt(1 + 1/n_short)
        log_p_short = stats.t.logpdf(x, df=df_short, loc=mu_short, scale=scale_short)
        
        # Compute Bayes factor: p(x|short) / p(x|long)
        # If short model fits much better, suggests changepoint
        log_bayes_factor = log_p_short - log_p_long
        
        # Convert to probability using logistic function
        # sensitivity controls the sharpness of the decision
        cp_prob = 1 / (1 + np.exp(-self.sensitivity * log_bayes_factor))
        
        return min(max(cp_prob, 0.0), 1.0)
    
    def _estimate_run_length(self, t: int) -> int:
        """
        Estimate run length (time since last changepoint)
        Look backwards for the last high-probability changepoint
        """
        if t < self.min_segment_length:
            return t
        
        # Look backwards for last changepoint
        for i in range(t-1, max(0, t-200), -1):
            if self.changepoint_probs[i] > 0.75:
                return t - i
        
        # No recent changepoint found
        return min(t, 200)
    
    def get_changepoint_probabilities(self) -> pd.Series:
        """Get changepoint probabilities"""
        if self.changepoint_probs is None:
            raise ValueError("Model not fitted")
        return pd.Series(self.changepoint_probs, name='changepoint_prob')
    
    def detect_changepoints(self, threshold: float = 0.5) -> pd.Series:
        """Detect changepoints above threshold"""
        if self.changepoint_probs is None:
            raise ValueError("Model not fitted")
        changepoints = (self.changepoint_probs > threshold).astype(int)
        return pd.Series(changepoints, name='changepoint_detected')
    
    def get_changepoint_locations(self, threshold: float = 0.5) -> List[int]:
        """Get indices of detected changepoints"""
        changepoints = self.detect_changepoints(threshold)
        return list(np.where(changepoints == 1)[0])
    
    def get_statistics(self) -> Dict:
        """Get statistics"""
        if self.changepoint_probs is None:
            raise ValueError("Model not fitted")
        
        return {
            'n_observations': len(self.changepoint_probs),
            'mean_cp_prob': float(np.mean(self.changepoint_probs)),
            'max_cp_prob': float(np.max(self.changepoint_probs)),
            'n_significant_cp_50': int(np.sum(self.changepoint_probs > 0.5)),
            'n_significant_cp_75': int(np.sum(self.changepoint_probs > 0.75)),
            'n_significant_cp_90': int(np.sum(self.changepoint_probs > 0.90)),
            'min_segment_length': self.min_segment_length,
            'mean_run_length': float(np.mean(self.maxes)) if self.maxes is not None else 0.0
        }
    
    def get_current_changepoint_prob(self) -> float:
        """Get current changepoint probability"""
        if self.changepoint_probs is None:
            raise ValueError("Model not fitted")
        return float(self.changepoint_probs[-1])
    
    def plot_changepoints(self, title: str = "Bayesian Changepoint Detection",
                         save_path: str = 'reports/bayesian_changepoint_detection.png',
                         threshold: float = 0.5) -> None:
        """Plot changepoints"""
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
            
            changepoint_locs = self.get_changepoint_locations(threshold=threshold)
            
            # Plot 1: Data
            ax1.plot(self.data, 'b-', linewidth=1.5, label='Data', alpha=0.7)
            if len(changepoint_locs) > 0:
                ax1.scatter(changepoint_locs, self.data[changepoint_locs],
                           color='red', s=100, marker='v',
                           label=f'Changepoints (n={len(changepoint_locs)})',
                           zorder=5, edgecolors='darkred', linewidths=2)
                for cp in changepoint_locs:
                    ax1.axvline(x=cp, color='red', linestyle='--', alpha=0.3, linewidth=1.5)
            
            ax1.set_ylabel('Value', fontsize=12, fontweight='bold')
            ax1.set_title(f'{title}\nTime Series Data with Detected Changepoints',
                         fontsize=14, fontweight='bold')
            ax1.legend(loc='best', fontsize=10)
            ax1.grid(True, alpha=0.3, linestyle='--')
            
            # Plot 2: Probabilities
            time_indices = np.arange(len(self.changepoint_probs))
            ax2.plot(time_indices, self.changepoint_probs, 'b-',
                    linewidth=1.5, label='Changepoint Probability', alpha=0.7)
            ax2.axhline(y=threshold, color='darkgreen', linestyle='--',
                       linewidth=2, alpha=0.7, label=f'{threshold*100:.0f}% threshold')
            ax2.fill_between(time_indices, threshold, self.changepoint_probs,
                            where=(self.changepoint_probs > threshold),
                            color='red', alpha=0.3, label='Changepoint detected')
            if len(changepoint_locs) > 0:
                ax2.scatter(changepoint_locs, self.changepoint_probs[changepoint_locs],
                           color='red', s=100, marker='o', zorder=5,
                           edgecolors='darkred', linewidths=2)
            
            ax2.set_ylabel('Changepoint Probability', fontsize=12, fontweight='bold')
            ax2.set_title('Bayesian Changepoint Probabilities (uses all past data)',
                         fontsize=14, fontweight='bold')
            ax2.legend(loc='best', fontsize=10)
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.set_ylim([0, 1.0])
            
            # Plot 3: Run length
            if self.maxes is not None:
                ax3.plot(time_indices, self.maxes, 'g-',
                        linewidth=1.5, label='Estimated Run Length', alpha=0.7)
                if len(changepoint_locs) > 0:
                    ax3.scatter(changepoint_locs, self.maxes[changepoint_locs],
                               color='red', s=100, marker='v', zorder=5,
                               edgecolors='darkred', linewidths=2,
                               label='Changepoint')
                
                ax3.set_ylabel('Run Length', fontsize=12, fontweight='bold')
                ax3.set_xlabel('Time', fontsize=12, fontweight='bold')
                ax3.set_title('Time Since Last Changepoint',
                             fontsize=14, fontweight='bold')
                ax3.legend(loc='best', fontsize=10)
                ax3.grid(True, alpha=0.3, linestyle='--')
            
            # Stats box
            stats_text = f"Algorithm: Bayesian (Student-t predictive)\n"
            stats_text += f"Changepoints detected: {len(changepoint_locs)}\n"
            stats_text += f"Mean probability: {np.mean(self.changepoint_probs):.4f}\n"
            stats_text += f"Max probability: {np.max(self.changepoint_probs):.4f}"
            
            ax2.text(0.02, 0.98, stats_text,
                    transform=ax2.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=9, family='monospace')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"[OK] High-quality plot saved to {save_path}")
            plt.close()
            
        except Exception as e:
            print(f"Error creating plot: {e}")
            import traceback
            traceback.print_exc()
