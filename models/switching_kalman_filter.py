"""
Switching Kalman Filter (SKF)
Estimates hidden states with regime-dependent dynamics
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


class SwitchingKalmanFilter:
    """
    Switching Kalman Filter for regime-dependent state estimation
    
    Models asset returns as having different trend/mean-reversion dynamics
    depending on the current regime.
    
    Parameters
    ----------
    n_regimes : int, default=2
        Number of regimes (e.g., trending vs mean-reverting)
    state_dim : int, default=2
        Dimension of state vector (position, velocity, etc.)
    max_iter : int, default=100
        Maximum EM iterations
    """
    
    def __init__(self, n_regimes=2, state_dim=2, max_iter=100):
        self.n_regimes = n_regimes
        self.state_dim = state_dim
        self.max_iter = max_iter
        
        # Model parameters (regime-dependent)
        self.A = {}  # State transition matrices
        self.C = {}  # Observation matrices
        self.Q = {}  # Process noise covariances
        self.R = {}  # Observation noise covariances
        
        # Regime parameters
        self.transition_matrix = None  # Regime transition probabilities
        self.initial_probs = None
        
        # Results
        self.regime_probs_ = None
        self.states_ = None
        self.fitted_ = False
        
    def _initialize_parameters(self, data):
        """Initialize model parameters"""
        n_obs = len(data)
        
        # Initialize regime-dependent parameters
        for regime in range(self.n_regimes):
            if regime == 0:
                # Trending regime: momentum
                self.A[regime] = np.array([[1.0, 1.0], [0.0, 0.95]])
            else:
                # Mean-reverting regime
                self.A[regime] = np.array([[1.0, 0.0], [0.0, -0.5]])
            
            self.C[regime] = np.array([[1.0, 0.0]])  # Observe only position
            self.Q[regime] = np.eye(self.state_dim) * 0.01  # Process noise
            self.R[regime] = np.array([[0.1]])  # Observation noise
        
        # Initialize regime transition matrix (symmetric)
        prob = 1.0 / self.n_regimes
        self.transition_matrix = np.full((self.n_regimes, self.n_regimes), 0.05)
        np.fill_diagonal(self.transition_matrix, 0.95)
        
        # Initial regime probabilities
        self.initial_probs = np.ones(self.n_regimes) / self.n_regimes
    
    def _kalman_filter_step(self, x_pred, P_pred, y, regime):
        """Single Kalman filter update step"""
        C = self.C[regime]
        R = self.R[regime]
        
        # Innovation
        y_pred = C @ x_pred
        innovation = y - y_pred
        
        # Innovation covariance
        S = C @ P_pred @ C.T + R
        
        # Kalman gain
        K = P_pred @ C.T @ np.linalg.inv(S)
        
        # State update
        x_upd = x_pred + K @ innovation
        P_upd = P_pred - K @ S @ K.T
        
        # Log-likelihood
        log_lik = norm.logpdf(innovation[0, 0], 0, np.sqrt(S[0, 0]))
        
        return x_upd, P_upd, log_lik
    
    def _kalman_predict_step(self, x, P, regime):
        """Kalman filter prediction step"""
        A = self.A[regime]
        Q = self.Q[regime]
        
        x_pred = A @ x
        P_pred = A @ P @ A.T + Q
        
        return x_pred, P_pred
    
    def _forward_pass(self, data):
        """Forward pass: filtering"""
        n_obs = len(data)
        
        # Storage for filtered estimates
        regime_probs = np.zeros((n_obs, self.n_regimes))
        filtered_states = {}
        filtered_covs = {}
        
        for regime in range(self.n_regimes):
            filtered_states[regime] = np.zeros((n_obs, self.state_dim, 1))
            filtered_covs[regime] = np.zeros((n_obs, self.state_dim, self.state_dim))
        
        # Initial state
        x_init = np.array([[data[0]], [0.0]])
        P_init = np.eye(self.state_dim) * 1.0
        
        # Initialize for each regime
        for regime in range(self.n_regimes):
            filtered_states[regime][0] = x_init
            filtered_covs[regime][0] = P_init
        
        regime_probs[0] = self.initial_probs
        
        # Forward filtering
        for t in range(1, n_obs):
            y = np.array([[data[t]]])
            
            # Prediction for each regime
            pred_probs = regime_probs[t-1] @ self.transition_matrix
            
            likelihoods = np.zeros(self.n_regimes)
            
            for regime in range(self.n_regimes):
                # Predict
                x_pred, P_pred = self._kalman_predict_step(
                    filtered_states[regime][t-1],
                    filtered_covs[regime][t-1],
                    regime
                )
                
                # Update
                x_upd, P_upd, log_lik = self._kalman_filter_step(
                    x_pred, P_pred, y, regime
                )
                
                filtered_states[regime][t] = x_upd
                filtered_covs[regime][t] = P_upd
                likelihoods[regime] = np.exp(log_lik)
            
            # Update regime probabilities
            regime_probs[t] = pred_probs * likelihoods
            regime_probs[t] /= np.sum(regime_probs[t])
        
        return regime_probs, filtered_states, filtered_covs
    
    def fit(self, data):
        """
        Fit Switching Kalman Filter
        
        Parameters
        ----------
        data : pd.Series or np.ndarray
            Univariate time series
        
        Returns
        -------
        self
        """
        if isinstance(data, pd.Series):
            data = data.values
        
        # Initialize parameters
        self._initialize_parameters(data)
        
        # Run forward pass (simplified - full EM would iterate)
        regime_probs, filtered_states, filtered_covs = self._forward_pass(data)
        
        self.regime_probs_ = regime_probs
        self.states_ = filtered_states
        self.fitted_ = True
        
        print(f"Switching Kalman Filter fitted on {len(data)} observations")
        print(f"Regimes: {self.n_regimes}, State dimension: {self.state_dim}")
        
        return self
    
    def predict_regime(self):
        """Get most likely regime at each time point"""
        if not self.fitted_:
            raise ValueError("Model not fitted yet.")
        
        return np.argmax(self.regime_probs_, axis=1)
    
    def get_regime_probabilities(self):
        """Get regime probabilities over time"""
        if not self.fitted_:
            raise ValueError("Model not fitted yet.")
        
        return pd.DataFrame(
            self.regime_probs_,
            columns=[f'Regime_{i}' for i in range(self.n_regimes)]
        )
    
    def get_filtered_state(self, regime=None):
        """
        Get filtered state estimates
        
        Parameters
        ----------
        regime : int, optional
            If specified, return state for this regime.
            If None, return weighted average across regimes.
        
        Returns
        -------
        state : np.ndarray
            State estimates (position, velocity, etc.)
        """
        if not self.fitted_:
            raise ValueError("Model not fitted yet.")
        
        n_obs = len(self.regime_probs_)
        
        if regime is not None:
            # Return state for specific regime
            states = self.states_[regime]
            return states.reshape(n_obs, self.state_dim)
        else:
            # Weighted average across regimes
            weighted_state = np.zeros((n_obs, self.state_dim))
            for r in range(self.n_regimes):
                states = self.states_[r].reshape(n_obs, self.state_dim)
                weights = self.regime_probs_[:, r:r+1]
                weighted_state += states * weights
            
            return weighted_state
    
    def get_trend_estimate(self):
        """Get estimated trend (velocity component)"""
        state = self.get_filtered_state()
        return state[:, 1]  # Velocity component
    
    def get_level_estimate(self):
        """Get estimated level (position component)"""
        state = self.get_filtered_state()
        return state[:, 0]  # Position component
    
    def get_regime_statistics(self):
        """Get statistics for each regime"""
        if not self.fitted_:
            raise ValueError("Model not fitted yet.")
        
        regimes = self.predict_regime()
        stats = {}
        
        for r in range(self.n_regimes):
            mask = regimes == r
            avg_prob = np.mean(self.regime_probs_[mask, r]) if np.any(mask) else 0
            
            stats[f"Regime_{r}"] = {
                'count': np.sum(mask),
                'percentage': np.sum(mask) / len(regimes) * 100,
                'avg_probability': avg_prob,
                'regime_type': 'Trending' if r == 0 else 'Mean-Reverting'
            }
        
        return stats
    
    def get_model_parameters(self):
        """Get model parameters"""
        if not self.fitted_:
            raise ValueError("Model not fitted yet.")
        
        params = {
            'n_regimes': self.n_regimes,
            'state_dim': self.state_dim,
            'transition_matrix': self.transition_matrix
        }
        
        for r in range(self.n_regimes):
            params[f'regime_{r}_dynamics'] = {
                'A': self.A[r].tolist(),
                'momentum_persistence': self.A[r][1, 1]
            }
        
        return params
