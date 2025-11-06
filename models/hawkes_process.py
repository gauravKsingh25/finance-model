"""
Hawkes Process Model
Self-exciting point process for detecting market fragility and cascading events
Purpose: The Fragility Sensor - Detects clustering of events and market stress
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class HawkesProcess:
    """
    Hawkes Self-Exciting Process
    
    Models clustering and self-excitation in event data
    Outputs: Fragility score based on self-excitation intensity
    """
    
    def __init__(self, alpha: float = 0.5, beta: float = 1.0, mu: float = 0.1):
        """
        Initialize Hawkes Process
        
        Args:
            alpha: Excitation parameter (strength of self-excitation)
            beta: Decay parameter (how fast excitement decays)
            mu: Baseline intensity
        """
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.event_times = None
        self.intensities = None
        self.fitted = False
        
    def fit(self, event_times: np.ndarray, optimize: bool = True) -> 'HawkesProcess':
        """
        Fit Hawkes process to event times
        
        Args:
            event_times: Array of event occurrence times
            optimize: Whether to optimize parameters via MLE
            
        Returns:
            self
        """
        self.event_times = np.sort(event_times)
        
        if optimize and len(event_times) > 10:
            # Maximum likelihood estimation
            self._fit_mle()
        
        # Calculate intensities
        self._calculate_intensities()
        
        self.fitted = True
        
        print(f"Hawkes Process fitted on {len(self.event_times)} events")
        print(f"Parameters: μ={self.mu:.4f}, α={self.alpha:.4f}, β={self.beta:.4f}")
        print(f"Branching ratio (stability): {self.alpha/self.beta:.4f}")
        
        return self
    
    def fit_from_returns(self, returns: pd.Series, threshold: float = 2.0) -> 'HawkesProcess':
        """
        Fit Hawkes process from returns by detecting extreme events
        
        Args:
            returns: Time series of returns
            threshold: Number of standard deviations to define extreme event
            
        Returns:
            self
        """
        # Detect extreme events (large absolute returns)
        std = returns.std()
        extreme_events = np.abs(returns) > (threshold * std)
        
        # Get event times (indices where extreme events occur)
        event_times = np.where(extreme_events)[0]
        
        print(f"Detected {len(event_times)} extreme events (>{threshold}σ)")
        
        if len(event_times) < 5:
            print("Warning: Too few events detected. Using lower threshold.")
            event_times = np.where(np.abs(returns) > (std))[0]
            print(f"Detected {len(event_times)} events (>1σ)")
        
        return self.fit(event_times, optimize=True)
    
    def _fit_mle(self):
        """Fit parameters using Maximum Likelihood Estimation"""
        
        def neg_log_likelihood(params):
            """Negative log-likelihood for optimization"""
            mu, alpha, beta = params
            
            # Ensure parameters are valid
            if mu <= 0 or alpha <= 0 or beta <= 0 or alpha >= beta:
                return 1e10
            
            T = self.event_times[-1]
            n = len(self.event_times)
            
            # Compensator term
            compensator = mu * T
            for i, ti in enumerate(self.event_times):
                compensator += (alpha / beta) * (1 - np.exp(-beta * (T - ti)))
            
            # Log-intensity sum
            log_sum = 0
            for i, ti in enumerate(self.event_times):
                intensity = mu
                for j in range(i):
                    intensity += alpha * np.exp(-beta * (ti - self.event_times[j]))
                log_sum += np.log(intensity) if intensity > 0 else -1e10
            
            # Negative log-likelihood
            return compensator - log_sum
        
        # Initial guess
        x0 = [self.mu, self.alpha, self.beta]
        
        # Bounds
        bounds = [(1e-6, 10), (1e-6, 5), (1e-6, 10)]
        
        # Optimize
        try:
            result = minimize(neg_log_likelihood, x0, bounds=bounds, method='L-BFGS-B')
            if result.success:
                self.mu, self.alpha, self.beta = result.x
        except:
            print("MLE optimization failed, using default parameters")
    
    def _calculate_intensities(self):
        """Calculate intensity at each event time"""
        n = len(self.event_times)
        self.intensities = np.zeros(n)
        
        for i in range(n):
            ti = self.event_times[i]
            intensity = self.mu
            
            # Add excitation from previous events
            for j in range(i):
                intensity += self.alpha * np.exp(-self.beta * (ti - self.event_times[j]))
            
            self.intensities[i] = intensity
    
    def get_intensity_at_time(self, t: float) -> float:
        """
        Calculate intensity at specific time
        
        Args:
            t: Time point
            
        Returns:
            Intensity at time t
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        intensity = self.mu
        
        # Add contributions from all past events
        past_events = self.event_times[self.event_times < t]
        for ti in past_events:
            intensity += self.alpha * np.exp(-self.beta * (t - ti))
        
        return float(intensity)
    
    def get_fragility_score(self) -> float:
        """
        Calculate market fragility score
        
        Returns:
            Fragility score (0-1, higher = more fragile)
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Branching ratio: ratio of self-excitation to decay
        branching_ratio = self.alpha / self.beta
        
        # Normalize to 0-1 scale (ratio > 1 means explosive)
        fragility = min(branching_ratio, 1.0)
        
        return float(fragility)
    
    def get_excitation_level(self) -> str:
        """
        Get qualitative excitation level
        
        Returns:
            String describing excitation level
        """
        fragility = self.get_fragility_score()
        
        if fragility < 0.3:
            return "Low"
        elif fragility < 0.6:
            return "Moderate"
        elif fragility < 0.9:
            return "High"
        else:
            return "Critical"
    
    def get_statistics(self) -> Dict:
        """
        Get Hawkes process statistics
        
        Returns:
            Dictionary with statistics
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        branching_ratio = self.alpha / self.beta
        
        stats_dict = {
            'n_events': len(self.event_times),
            'mu_baseline': float(self.mu),
            'alpha_excitation': float(self.alpha),
            'beta_decay': float(self.beta),
            'branching_ratio': float(branching_ratio),
            'is_stable': branching_ratio < 1.0,
            'fragility_score': self.get_fragility_score(),
            'excitation_level': self.get_excitation_level(),
            'mean_intensity': float(np.mean(self.intensities)),
            'max_intensity': float(np.max(self.intensities)),
            'avg_inter_event_time': float(np.mean(np.diff(self.event_times))) if len(self.event_times) > 1 else 0
        }
        
        return stats_dict
    
    def predict_next_event_probability(self, time_horizon: float = 1.0) -> float:
        """
        Predict probability of event in next time period
        
        Args:
            time_horizon: Time period to look ahead
            
        Returns:
            Probability of at least one event
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        current_time = self.event_times[-1]
        current_intensity = self.get_intensity_at_time(current_time)
        
        # Approximate probability using intensity
        # P(N(t+h) - N(t) >= 1) ≈ 1 - exp(-λ(t) * h)
        prob = 1 - np.exp(-current_intensity * time_horizon)
        
        return float(prob)
