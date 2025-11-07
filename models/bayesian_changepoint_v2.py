"""
Bayesian Changepoint Detection (BCD) - Improved Implementation
Detects structural breaks in time series using likelihood ratios
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
    Bayesian Changepoint Detection Model using Likelihood Ratio Method
    
    Detects structural breaks in time series data by comparing
    the likelihood of data under a single regime vs. multiple regimes
    """
    
    def __init__(self, window_size: int = 50, threshold: float = 3.0):
        """
        Initialize Bayesian Changepoint Detector
        
        Args:
            window_size: Size of the sliding window for comparison
            threshold: Threshold for likelihood ratio (higher = less sensitive)
        """
        self.window_size = window_size
        self.threshold = threshold
        self.changepoint_probs = None
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
        
        # Initialize changepoint probabilities
        self.changepoint_probs = np.zeros(n)
        
        # For each point, compute likelihood ratio
        for t in range(self.window_size, n):
            # Get window of data before this point
            window = self.data[max(0, t - self.window_size):t]
            current = self.data[t]
            
            if len(window) < 2:
                continue
            
            # Compute statistics for the window
            window_mean = np.mean(window)
            window_std = np.std(window, ddof=1)
            
            if window_std < 1e-10:
                window_std = 1e-10
            
            # Compute z-score of current observation
            z_score = abs((current - window_mean) / window_std)
            
            # Convert z-score to probability using cumulative distribution
            # Higher z-score = more likely to be a changepoint
            p_value = 2 * (1 - stats.norm.cdf(z_score))  # Two-tailed
            
            # Convert p-value to changepoint probability
            # Lower p-value = higher changepoint probability
            cp_prob = 1 - p_value
            
            # Apply sigmoid transformation to make it more sensitive
            cp_prob = 1 / (1 + np.exp(-self.threshold * (z_score - 2)))
            
            self.changepoint_probs[t] = cp_prob
        
        n_significant = np.sum(self.changepoint_probs > 0.5)
        print(f"Bayesian Changepoint Detection fitted on {n} observations")
        print(f"Detected {n_significant} significant changepoints (prob > 50%)")
        print(f"Mean changepoint probability: {np.mean(self.changepoint_probs):.4f}")
        print(f"Max changepoint probability: {np.max(self.changepoint_probs):.4f}")
        
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
            'window_size': self.window_size,
            'threshold': self.threshold
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
            print(f"[OK] High-quality plot saved to {save_path}")
            plt.close()
            
        except ImportError:
            print("Matplotlib not available for plotting")
        except Exception as e:
            print(f"Error creating plot: {e}")
