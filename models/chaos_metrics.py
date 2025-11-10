"""
Entropy and Hurst Exponent - FIXED VERSION
Model-free metrics for detecting chaos, trending, and mean-reversion
Purpose: The Chaos Sensor - Distinguishes trending vs mean-reverting behavior

FIXES:
1. Added comprehensive edge case handling
2. Improved robustness to outliers and missing data
3. Better numerical stability
4. Input validation
5. Fallback mechanisms
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class HurstExponent:
    """
    Hurst Exponent Calculator - Fixed Implementation
    
    Measures long-term memory and trend persistence
    - H < 0.5: Mean-reverting (anti-persistent)
    - H = 0.5: Random walk (Brownian motion)
    - H > 0.5: Trending (persistent)
    
    Improvements:
    - Handles missing data
    - Robust to outliers
    - Better numerical stability
    - Minimum data requirements enforced
    """
    
    def __init__(self, min_samples: int = 50):
        """
        Initialize Hurst calculator
        
        Parameters
        ----------
        min_samples : int, default=50
            Minimum number of samples required
        """
        self.min_samples = min_samples
        self.hurst = None
        self.regime = None
        self.data = None
        self.method_used = None
        
    def calculate(self, data: pd.Series, method: str = 'rs', 
                 handle_outliers: bool = True) -> float:
        """
        Calculate Hurst exponent
        
        Parameters
        ----------
        data : pd.Series
            Time series data
        method : str, default='rs'
            'rs' (R/S analysis) or 'dfa' (Detrended Fluctuation Analysis)
        handle_outliers : bool, default=True
            Whether to winsorize outliers
            
        Returns
        -------
        hurst : float
            Hurst exponent value
        """
        # Edge case: Empty or all-NaN data
        if data is None or len(data) == 0:
            raise ValueError("Data cannot be empty")
        
        # Handle missing data
        clean_data = data.dropna()
        
        if len(clean_data) == 0:
            raise ValueError("All data is NaN")
        
        # Edge case: Too few samples
        if len(clean_data) < self.min_samples:
            warnings.warn(
                f"Only {len(clean_data)} samples (minimum {self.min_samples} recommended). "
                f"Results may be unreliable."
            )
            if len(clean_data) < 10:
                raise ValueError(f"Need at least 10 samples, got {len(clean_data)}")
        
        # Handle outliers
        if handle_outliers:
            clean_data = self._winsorize(clean_data)
        
        # Edge case: Zero variance
        if np.var(clean_data) == 0:
            warnings.warn("Data has zero variance. Returning H=0.5 (random walk)")
            self.hurst = 0.5
            self.regime = "Random Walk"
            self.data = clean_data.values
            self.method_used = method
            return 0.5
        
        self.data = clean_data.values
        self.method_used = method
        
        try:
            if method == 'rs':
                self.hurst = self._calculate_rs()
            elif method == 'dfa':
                self.hurst = self._calculate_dfa()
            else:
                raise ValueError(f"Unknown method: {method}. Use 'rs' or 'dfa'")
        except Exception as e:
            warnings.warn(f"Calculation failed: {e}. Defaulting to H=0.5")
            self.hurst = 0.5
        
        # Ensure Hurst is in valid range
        self.hurst = np.clip(self.hurst, 0.0, 1.0)
        
        # Determine regime
        self._determine_regime()
        
        print(f"Hurst Exponent ({method}): {self.hurst:.4f}")
        print(f"Regime: {self.regime}")
        
        return self.hurst
    
    def _winsorize(self, data: pd.Series, limits: Tuple[float, float] = (0.05, 0.05)) -> pd.Series:
        """
        Winsorize data to handle outliers
        
        Parameters
        ----------
        data : pd.Series
            Input data
        limits : tuple
            Lower and upper percentile limits
        
        Returns
        -------
        winsorized_data : pd.Series
            Data with outliers winsorized
        """
        lower_pct, upper_pct = limits
        lower_val = data.quantile(lower_pct)
        upper_val = data.quantile(1 - upper_pct)
        
        return data.clip(lower=lower_val, upper=upper_val)
    
    def _determine_regime(self):
        """Determine regime based on Hurst exponent"""
        if self.hurst < 0.35:
            self.regime = "Strong Mean-Reversion"
        elif self.hurst < 0.45:
            self.regime = "Mean-Reversion"
        elif self.hurst < 0.55:
            self.regime = "Random Walk"
        elif self.hurst < 0.65:
            self.regime = "Weak Trending"
        elif self.hurst < 0.75:
            self.regime = "Trending"
        else:
            self.regime = "Strong Trending"
    
    def _calculate_rs(self) -> float:
        """
        Calculate Hurst exponent using Rescaled Range (R/S) analysis
        Improved with better numerical stability
        
        Returns
        -------
        hurst : float
            Hurst exponent
        """
        n = len(self.data)
        
        # Use multiple window sizes
        min_window = max(10, int(np.sqrt(n)))
        max_window = n // 2
        
        if max_window < min_window:
            # Too few data points
            warnings.warn("Data too short for reliable R/S analysis")
            return 0.5
        
        window_sizes = np.unique(np.logspace(
            np.log10(min_window), 
            np.log10(max_window), 
            num=min(20, max_window - min_window + 1)
        ).astype(int))
        
        rs_values = []
        valid_sizes = []
        
        for window in window_sizes:
            if window < min_window or window > n - 1:
                continue
            
            # Split data into chunks
            n_chunks = n // window
            rs_chunk = []
            
            for i in range(n_chunks):
                chunk = self.data[i*window:(i+1)*window]
                
                # Edge case: Check chunk validity
                if len(chunk) < 2:
                    continue
                
                # Calculate mean
                mean = np.mean(chunk)
                
                # Calculate cumulative deviations
                deviations = chunk - mean
                cumsum_dev = np.cumsum(deviations)
                
                # Calculate range
                R = np.max(cumsum_dev) - np.min(cumsum_dev)
                
                # Calculate standard deviation
                S = np.std(chunk, ddof=1) if len(chunk) > 1 else np.std(chunk)
                
                # Edge case: Zero standard deviation
                if S == 0 or not np.isfinite(S):
                    continue
                
                # R/S ratio
                rs_ratio = R / S
                
                if np.isfinite(rs_ratio) and rs_ratio > 0:
                    rs_chunk.append(rs_ratio)
            
            if rs_chunk:
                rs_values.append(np.mean(rs_chunk))
                valid_sizes.append(window)
        
        # Edge case: Not enough valid points
        if len(rs_values) < 2:
            warnings.warn("Insufficient valid R/S points. Defaulting to H=0.5")
            return 0.5
        
        # Linear regression on log-log plot
        try:
            log_sizes = np.log(valid_sizes)
            log_rs = np.log(rs_values)
            
            # Remove any NaN or inf values
            mask = np.isfinite(log_sizes) & np.isfinite(log_rs)
            
            if np.sum(mask) < 2:
                warnings.warn("Insufficient valid points for regression")
                return 0.5
            
            slope, intercept = np.polyfit(log_sizes[mask], log_rs[mask], 1)
            
            # Ensure reasonable value
            if not np.isfinite(slope) or slope < 0 or slope > 1:
                warnings.warn(f"Unrealistic Hurst value {slope:.2f}, capping to valid range")
                slope = np.clip(slope, 0.0, 1.0)
            
            return float(slope)
            
        except Exception as e:
            warnings.warn(f"Regression failed: {e}")
            return 0.5
    
    def _calculate_dfa(self) -> float:
        """
        Calculate Hurst exponent using Detrended Fluctuation Analysis
        Improved with better edge case handling
        
        Returns
        -------
        hurst : float
            Hurst exponent
        """
        n = len(self.data)
        
        # Integrate the series (cumulative sum of deviations from mean)
        mean = np.mean(self.data)
        y = np.cumsum(self.data - mean)
        
        # Use multiple window sizes
        min_window = max(4, int(np.sqrt(n) / 2))
        max_window = n // 4
        
        if max_window < min_window:
            warnings.warn("Data too short for DFA")
            return 0.5
        
        window_sizes = np.unique(np.logspace(
            np.log10(min_window),
            np.log10(max_window),
            num=min(15, max_window - min_window + 1)
        ).astype(int))
        
        fluctuations = []
        valid_sizes = []
        
        for window in window_sizes:
            if window < min_window or window >= n:
                continue
            
            # Split into segments
            n_segments = n // window
            
            if n_segments < 1:
                continue
            
            f_values = []
            
            for i in range(n_segments):
                segment = y[i*window:(i+1)*window]
                
                # Edge case: Check segment validity
                if len(segment) < 2:
                    continue
                
                # Fit linear trend
                x = np.arange(len(segment))
                
                try:
                    coeffs = np.polyfit(x, segment, 1)
                    trend = np.polyval(coeffs, x)
                    
                    # Calculate fluctuation
                    fluctuation = np.sqrt(np.mean((segment - trend)**2))
                    
                    if np.isfinite(fluctuation) and fluctuation > 0:
                        f_values.append(fluctuation)
                except:
                    continue
            
            if f_values:
                fluctuations.append(np.mean(f_values))
                valid_sizes.append(window)
        
        # Edge case: Not enough valid points
        if len(fluctuations) < 2:
            warnings.warn("Insufficient valid DFA points")
            return 0.5
        
        # Linear regression on log-log plot
        try:
            log_sizes = np.log(valid_sizes)
            log_fluct = np.log(fluctuations)
            
            # Remove any NaN or inf values
            mask = np.isfinite(log_sizes) & np.isfinite(log_fluct)
            
            if np.sum(mask) < 2:
                warnings.warn("Insufficient valid points for DFA regression")
                return 0.5
            
            slope, _ = np.polyfit(log_sizes[mask], log_fluct[mask], 1)
            
            # Ensure reasonable value
            if not np.isfinite(slope) or slope < 0 or slope > 1:
                warnings.warn(f"Unrealistic DFA Hurst value {slope:.2f}")
                slope = np.clip(slope, 0.0, 1.0)
            
            return float(slope)
            
        except Exception as e:
            warnings.warn(f"DFA regression failed: {e}")
            return 0.5
    
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
    Entropy-based metrics for chaos detection - Fixed Implementation
    
    Improvements:
    - Handles missing data
    - Robust to outliers
    - Better bin selection
    - Input validation
    """
    
    def __init__(self, min_samples: int = 30):
        """
        Initialize entropy calculator
        
        Parameters
        ----------
        min_samples : int, default=30
            Minimum number of samples required
        """
        self.min_samples = min_samples
        self.entropy = None
        self.data = None
        
    def calculate_shannon_entropy(self, data: pd.Series, n_bins: int = None,
                                 handle_outliers: bool = True) -> float:
        """
        Calculate Shannon entropy of return distribution
        
        Parameters
        ----------
        data : pd.Series
            Time series data
        n_bins : int, optional
            Number of bins for histogram (auto-selected if None)
        handle_outliers : bool, default=True
            Whether to winsorize outliers
            
        Returns
        -------
        entropy : float
            Shannon entropy value
        """
        # Edge case: Empty or all-NaN data
        if data is None or len(data) == 0:
            raise ValueError("Data cannot be empty")
        
        # Handle missing data
        clean_data = data.dropna()
        
        if len(clean_data) == 0:
            raise ValueError("All data is NaN")
        
        # Edge case: Too few samples
        if len(clean_data) < self.min_samples:
            warnings.warn(
                f"Only {len(clean_data)} samples (minimum {self.min_samples} recommended)"
            )
            if len(clean_data) < 5:
                raise ValueError(f"Need at least 5 samples, got {len(clean_data)}")
        
        # Handle outliers
        if handle_outliers:
            lower_val = clean_data.quantile(0.05)
            upper_val = clean_data.quantile(0.95)
            clean_data = clean_data.clip(lower=lower_val, upper=upper_val)
        
        # Edge case: Zero variance
        if np.var(clean_data) == 0:
            warnings.warn("Data has zero variance. Entropy = 0")
            self.entropy = 0.0
            self.data = clean_data.values
            return 0.0
        
        self.data = clean_data.values
        
        # Auto-select number of bins if not specified
        if n_bins is None:
            # Use Sturges' rule with modifications
            n_bins = max(10, min(100, int(np.log2(len(self.data)) + 1) * 3))
        
        try:
            # Create histogram
            hist, _ = np.histogram(self.data, bins=n_bins, density=True)
            
            # Normalize to get probabilities
            hist = hist + 1e-10  # Add small constant to avoid log(0)
            hist = hist / np.sum(hist)
            
            # Remove near-zero probabilities
            hist = hist[hist > 1e-10]
            
            # Calculate Shannon entropy
            self.entropy = -np.sum(hist * np.log2(hist))
            
            # Ensure finite
            if not np.isfinite(self.entropy):
                warnings.warn("Entropy calculation resulted in non-finite value")
                self.entropy = 0.0
            
            return float(self.entropy)
            
        except Exception as e:
            warnings.warn(f"Shannon entropy calculation failed: {e}")
            self.entropy = 0.0
            return 0.0
    
    def calculate_approximate_entropy(self, data: pd.Series, m: int = 2, 
                                     r: float = None) -> float:
        """
        Calculate Approximate Entropy (ApEn)
        Measures regularity and unpredictability
        
        Parameters
        ----------
        data : pd.Series
            Time series data
        m : int, default=2
            Pattern length
        r : float, optional
            Tolerance (default: 0.2 * std)
            
        Returns
        -------
        entropy : float
            Approximate entropy value
        """
        # Handle missing data
        clean_data = data.dropna()
        
        if len(clean_data) == 0:
            raise ValueError("All data is NaN")
        
        # Edge case: Too few samples for ApEn
        min_required = m + 2
        if len(clean_data) < min_required:
            raise ValueError(f"Need at least {min_required} samples for m={m}")
        
        self.data = clean_data.values
        n = len(self.data)
        
        # Set tolerance
        if r is None:
            std = np.std(self.data)
            if std == 0:
                warnings.warn("Zero standard deviation. Using r=0.1")
                r = 0.1
            else:
                r = 0.2 * std
        
        def _max_dist(xi, xj):
            """Maximum distance between patterns"""
            try:
                return max([abs(float(ua) - float(va)) for ua, va in zip(xi, xj)])
            except:
                return np.inf
        
        def _phi(m_val):
            """Calculate phi(m)"""
            try:
                # Create patterns
                patterns = []
                for i in range(n - m_val + 1):
                    pattern = [self.data[j] for j in range(i, i + m_val)]
                    patterns.append(pattern)
                
                if not patterns:
                    return 0.0
                
                # Calculate correlations
                C = []
                for i, pattern_i in enumerate(patterns):
                    count = 0
                    for pattern_j in patterns:
                        if _max_dist(pattern_i, pattern_j) <= r:
                            count += 1
                    
                    if count > 0:
                        C.append(count / len(patterns))
                
                if not C or len(C) == 0:
                    return 0.0
                
                # Filter out zeros and calculate log sum
                C = [c for c in C if c > 0]
                if not C:
                    return 0.0
                
                return np.sum(np.log(C)) / len(C)
                
            except Exception as e:
                warnings.warn(f"Phi calculation failed: {e}")
                return 0.0
        
        try:
            phi_m = _phi(m)
            phi_m_plus_1 = _phi(m + 1)
            
            self.entropy = abs(phi_m - phi_m_plus_1)
            
            if not np.isfinite(self.entropy):
                warnings.warn("ApEn calculation resulted in non-finite value")
                self.entropy = 0.0
            
            return float(self.entropy)
            
        except Exception as e:
            warnings.warn(f"Approximate entropy calculation failed: {e}")
            self.entropy = 0.0
            return 0.0
    
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
