"""
Entropy and Hurst Exponent
Model-free metrics for detecting chaos, trending, and mean-reversion
Purpose: The Chaos Sensor - Distinguishes trending vs mean-reverting behavior
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class HurstExponent:
    """
    Hurst Exponent Calculator
    
    Measures long-term memory and trend persistence
    - H < 0.5: Mean-reverting (anti-persistent)
    - H = 0.5: Random walk (Brownian motion)
    - H > 0.5: Trending (persistent)
    """
    
    def __init__(self):
        self.hurst = None
        self.regime = None
        self.data = None
        
    def calculate(self, data: pd.Series, method: str = 'rs') -> float:
        """
        Calculate Hurst exponent
        
        Args:
            data: Time series data
            method: 'rs' (R/S analysis) or 'dfa' (Detrended Fluctuation Analysis)
            
        Returns:
            Hurst exponent value
        """
        self.data = data.dropna().values
        
        if method == 'rs':
            self.hurst = self._calculate_rs()
        elif method == 'dfa':
            self.hurst = self._calculate_dfa()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Determine regime
        if self.hurst < 0.4:
            self.regime = "Strong Mean-Reversion"
        elif self.hurst < 0.5:
            self.regime = "Mean-Reversion"
        elif self.hurst < 0.6:
            self.regime = "Random Walk"
        elif self.hurst < 0.7:
            self.regime = "Trending"
        else:
            self.regime = "Strong Trending"
        
        print(f"Hurst Exponent: {self.hurst:.4f}")
        print(f"Regime: {self.regime}")
        
        return self.hurst
    
    def _calculate_rs(self) -> float:
        """
        Calculate Hurst exponent using Rescaled Range (R/S) analysis
        
        Returns:
            Hurst exponent
        """
        n = len(self.data)
        
        # Use multiple window sizes
        min_window = 10
        max_window = n // 2
        window_sizes = np.unique(np.logspace(
            np.log10(min_window), 
            np.log10(max_window), 
            num=20
        ).astype(int))
        
        rs_values = []
        
        for window in window_sizes:
            if window < min_window or window > n - 1:
                continue
            
            # Split data into chunks
            n_chunks = n // window
            rs_chunk = []
            
            for i in range(n_chunks):
                chunk = self.data[i*window:(i+1)*window]
                
                # Calculate mean
                mean = np.mean(chunk)
                
                # Calculate cumulative deviations
                deviations = chunk - mean
                cumsum_dev = np.cumsum(deviations)
                
                # Calculate range
                R = np.max(cumsum_dev) - np.min(cumsum_dev)
                
                # Calculate standard deviation
                S = np.std(chunk)
                
                # R/S ratio
                if S > 0:
                    rs_chunk.append(R / S)
            
            if rs_chunk:
                rs_values.append(np.mean(rs_chunk))
        
        # Linear regression on log-log plot
        if len(rs_values) > 1:
            log_sizes = np.log(window_sizes[:len(rs_values)])
            log_rs = np.log(rs_values)
            
            # Remove any NaN or inf values
            mask = np.isfinite(log_sizes) & np.isfinite(log_rs)
            if np.sum(mask) > 1:
                slope, _ = np.polyfit(log_sizes[mask], log_rs[mask], 1)
                return float(slope)
        
        return 0.5  # Default to random walk if calculation fails
    
    def _calculate_dfa(self) -> float:
        """
        Calculate Hurst exponent using Detrended Fluctuation Analysis
        
        Returns:
            Hurst exponent
        """
        n = len(self.data)
        
        # Integrate the series (cumulative sum of deviations from mean)
        mean = np.mean(self.data)
        y = np.cumsum(self.data - mean)
        
        # Use multiple window sizes
        min_window = 4
        max_window = n // 4
        window_sizes = np.unique(np.logspace(
            np.log10(min_window),
            np.log10(max_window),
            num=15
        ).astype(int))
        
        fluctuations = []
        
        for window in window_sizes:
            if window < min_window or window >= n:
                continue
            
            # Split into segments
            n_segments = n // window
            f_values = []
            
            for i in range(n_segments):
                segment = y[i*window:(i+1)*window]
                
                # Fit linear trend
                x = np.arange(len(segment))
                coeffs = np.polyfit(x, segment, 1)
                trend = np.polyval(coeffs, x)
                
                # Calculate fluctuation
                fluctuation = np.sqrt(np.mean((segment - trend)**2))
                f_values.append(fluctuation)
            
            if f_values:
                fluctuations.append(np.mean(f_values))
        
        # Linear regression on log-log plot
        if len(fluctuations) > 1:
            log_sizes = np.log(window_sizes[:len(fluctuations)])
            log_fluct = np.log(fluctuations)
            
            # Remove any NaN or inf values
            mask = np.isfinite(log_sizes) & np.isfinite(log_fluct)
            if np.sum(mask) > 1:
                slope, _ = np.polyfit(log_sizes[mask], log_fluct[mask], 1)
                return float(slope)
        
        return 0.5  # Default to random walk
    
    def get_regime(self) -> str:
        """Get qualitative regime description"""
        if self.regime is None:
            raise ValueError("Hurst exponent not calculated. Call calculate() first.")
        return self.regime
    
    def get_statistics(self) -> Dict:
        """
        Get Hurst exponent statistics
        
        Returns:
            Dictionary with statistics
        """
        if self.hurst is None:
            raise ValueError("Hurst exponent not calculated. Call calculate() first.")
        
        stats_dict = {
            'hurst_exponent': float(self.hurst),
            'regime': self.regime,
            'is_mean_reverting': self.hurst < 0.5,
            'is_trending': self.hurst > 0.5,
            'is_random_walk': 0.45 < self.hurst < 0.55,
            'persistence_strength': abs(self.hurst - 0.5),
            'interpretation': self._get_interpretation()
        }
        
        return stats_dict
    
    def _get_interpretation(self) -> str:
        """Get detailed interpretation"""
        if self.hurst < 0.4:
            return "Strong mean-reversion: Price tends to reverse direction frequently"
        elif self.hurst < 0.5:
            return "Mean-reversion: Price shows tendency to return to average"
        elif self.hurst < 0.6:
            return "Random walk: Price movements are largely unpredictable"
        elif self.hurst < 0.7:
            return "Trending: Price shows persistence in direction"
        else:
            return "Strong trending: Price has strong directional momentum"


class EntropyMetrics:
    """
    Entropy-based metrics for chaos detection
    """
    
    def __init__(self):
        self.entropy = None
        self.data = None
        
    def calculate_shannon_entropy(self, data: pd.Series, n_bins: int = 50) -> float:
        """
        Calculate Shannon entropy of return distribution
        
        Args:
            data: Time series data
            n_bins: Number of bins for histogram
            
        Returns:
            Shannon entropy value
        """
        self.data = data.dropna().values
        
        # Create histogram
        hist, _ = np.histogram(self.data, bins=n_bins, density=True)
        
        # Normalize to get probabilities
        hist = hist / np.sum(hist)
        
        # Remove zero probabilities
        hist = hist[hist > 0]
        
        # Calculate Shannon entropy
        self.entropy = -np.sum(hist * np.log2(hist))
        
        return float(self.entropy)
    
    def calculate_approximate_entropy(self, data: pd.Series, m: int = 2, r: float = None) -> float:
        """
        Calculate Approximate Entropy (ApEn)
        Measures regularity and unpredictability
        
        Args:
            data: Time series data
            m: Pattern length
            r: Tolerance (default: 0.2 * std)
            
        Returns:
            Approximate entropy value
        """
        self.data = data.dropna().values
        n = len(self.data)
        
        if r is None:
            r = 0.2 * np.std(self.data)
        
        def _max_dist(xi, xj):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _phi(m):
            patterns = np.array([[self.data[j] for j in range(i, i + m)] 
                                for i in range(n - m + 1)])
            C = []
            for i in range(len(patterns)):
                count = sum([1 for j in range(len(patterns)) 
                           if _max_dist(patterns[i], patterns[j]) <= r])
                C.append(count / (n - m + 1.0))
            return np.sum(np.log(C)) / (n - m + 1.0)
        
        self.entropy = abs(_phi(m) - _phi(m + 1))
        
        return float(self.entropy)
    
    def get_statistics(self) -> Dict:
        """Get entropy statistics"""
        if self.entropy is None:
            raise ValueError("Entropy not calculated. Call calculate method first.")
        
        return {
            'entropy': float(self.entropy),
            'interpretation': self._get_interpretation()
        }
    
    def _get_interpretation(self) -> str:
        """Get interpretation of entropy value"""
        if self.entropy < 2.0:
            return "Low entropy: Highly predictable, low chaos"
        elif self.entropy < 4.0:
            return "Moderate entropy: Some predictability"
        else:
            return "High entropy: Highly chaotic, unpredictable"


class ChaosMetrics:
    """
    Combined chaos and trendiness metrics
    """
    
    def __init__(self):
        self.hurst_calculator = HurstExponent()
        self.entropy_calculator = EntropyMetrics()
        
    def analyze(self, data: pd.Series) -> Dict:
        """
        Perform complete chaos analysis
        
        Args:
            data: Time series data
            
        Returns:
            Dictionary with all metrics
        """
        # Calculate Hurst exponent
        hurst = self.hurst_calculator.calculate(data)
        
        # Calculate entropy
        shannon_entropy = self.entropy_calculator.calculate_shannon_entropy(data)
        
        # Combined metrics
        results = {
            'hurst_exponent': hurst,
            'regime': self.hurst_calculator.get_regime(),
            'shannon_entropy': shannon_entropy,
            'chaos_metric': shannon_entropy / 10.0,  # Normalize
            'is_chaotic': shannon_entropy > 4.0,
            'is_mean_reverting': hurst < 0.5,
            'is_trending': hurst > 0.5,
            'combined_score': self._calculate_combined_score(hurst, shannon_entropy)
        }
        
        return results
    
    def _calculate_combined_score(self, hurst: float, entropy: float) -> float:
        """
        Calculate combined chaos/trend score
        
        Returns:
            Score from 0-1
        """
        # Normalize entropy to 0-1 scale
        norm_entropy = min(entropy / 10.0, 1.0)
        
        # Combine with Hurst (higher = more trending/chaotic)
        score = (abs(hurst - 0.5) * 2 + norm_entropy) / 2
        
        return float(score)
