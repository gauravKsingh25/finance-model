"""
Hierarchical Dirichlet Process Hidden Markov Model (HDP-HMM)
Automatically discovers number of regimes from data
"""
import numpy as np
import pandas as pd
from scipy.stats import norm, invgamma
from scipy.special import logsumexp
import warnings
warnings.filterwarnings('ignore')


class HDPHMM:
    """
    Hierarchical Dirichlet Process Hidden Markov Model
    
    Automatically discovers the number of market regimes from data
    using a non-parametric Bayesian approach.
    
    Parameters
    ----------
    truncation : int, default=10
        Maximum number of regimes to consider (truncation level)
    alpha : float, default=1.0
        Concentration parameter for regime transitions
    gamma : float, default=1.0
        Concentration parameter for global distribution
    max_iter : int, default=100
        Maximum Gibbs sampling iterations
    """
    
    def __init__(self, truncation=10, alpha=1.0, gamma=1.0, max_iter=100):
        self.truncation = truncation  # Max regimes
        self.alpha = alpha
        self.gamma = gamma
        self.max_iter = max_iter
        
        # Model parameters
        self.means_ = None
        self.variances_ = None
        self.transition_probs_ = None
        self.regime_probs_ = None
        self.n_active_regimes_ = None
        
        self.fitted_ = False
    
    def _initialize_parameters(self, data):
        """Initialize parameters using k-means-like approach"""
        # Initialize with evenly spaced means
        data_min, data_max = np.min(data), np.max(data)
        self.means_ = np.linspace(data_min, data_max, self.truncation)
        
        # Initialize variances
        data_var = np.var(data)
        self.variances_ = np.full(self.truncation, data_var)
        
        # Initialize transition probabilities (sticky)
        self.transition_probs_ = np.zeros((self.truncation, self.truncation))
        for i in range(self.truncation):
            # Stick-breaking construction (approximate)
            probs = np.random.dirichlet(np.ones(self.truncation) * self.alpha / self.truncation)
            # Make diagonal dominant (sticky)
            probs[i] += 0.5
            probs /= np.sum(probs)
            self.transition_probs_[i] = probs
    
    def _emission_probability(self, obs, regime):
        """Compute emission probability for observation given regime"""
        mean = self.means_[regime]
        var = self.variances_[regime]
        return norm.pdf(obs, mean, np.sqrt(var))
    
    def _forward_algorithm(self, data):
        """Forward algorithm for HMM"""
        n_obs = len(data)
        n_states = self.truncation
        
        # Forward probabilities
        alpha = np.zeros((n_obs, n_states))
        
        # Initial probabilities (uniform)
        initial_probs = np.ones(n_states) / n_states
        
        # First observation
        for s in range(n_states):
            alpha[0, s] = initial_probs[s] * self._emission_probability(data[0], s)
        
        alpha[0] /= np.sum(alpha[0])
        
        # Forward recursion
        for t in range(1, n_obs):
            for s in range(n_states):
                # Sum over previous states
                alpha[t, s] = np.sum(alpha[t-1] * self.transition_probs_[:, s])
                alpha[t, s] *= self._emission_probability(data[t], s)
            
            # Normalize to prevent underflow
            alpha[t] /= np.sum(alpha[t])
        
        return alpha
    
    def _backward_algorithm(self, data):
        """Backward algorithm for HMM"""
        n_obs = len(data)
        n_states = self.truncation
        
        # Backward probabilities
        beta = np.zeros((n_obs, n_states))
        
        # Initialize (last time point)
        beta[-1] = 1.0
        
        # Backward recursion
        for t in range(n_obs - 2, -1, -1):
            for s in range(n_states):
                for s_next in range(n_states):
                    beta[t, s] += (self.transition_probs_[s, s_next] * 
                                   self._emission_probability(data[t+1], s_next) * 
                                   beta[t+1, s_next])
            
            # Normalize
            if np.sum(beta[t]) > 0:
                beta[t] /= np.sum(beta[t])
        
        return beta
    
    def _baum_welch_iteration(self, data):
        """Single Baum-Welch (EM) iteration"""
        n_obs = len(data)
        n_states = self.truncation
        
        # E-step: Forward-backward
        alpha = self._forward_algorithm(data)
        beta = self._backward_algorithm(data)
        
        # Compute gamma (state probabilities)
        gamma = alpha * beta
        gamma /= np.sum(gamma, axis=1, keepdims=True)
        
        # Compute xi (transition probabilities)
        xi = np.zeros((n_obs - 1, n_states, n_states))
        for t in range(n_obs - 1):
            for s in range(n_states):
                for s_next in range(n_states):
                    xi[t, s, s_next] = (alpha[t, s] * 
                                        self.transition_probs_[s, s_next] *
                                        self._emission_probability(data[t+1], s_next) *
                                        beta[t+1, s_next])
            
            # Normalize
            xi_sum = np.sum(xi[t])
            if xi_sum > 0:
                xi[t] /= xi_sum
        
        # M-step: Update parameters
        
        # Update means and variances
        for s in range(n_states):
            gamma_sum = np.sum(gamma[:, s])
            
            if gamma_sum > 1e-10:
                # Update mean
                self.means_[s] = np.sum(gamma[:, s] * data) / gamma_sum
                
                # Update variance
                diff = data - self.means_[s]
                self.variances_[s] = np.sum(gamma[:, s] * diff**2) / gamma_sum
                
                # Add small constant for numerical stability
                self.variances_[s] = max(self.variances_[s], 1e-6)
        
        # Update transition probabilities with Dirichlet prior
        for s in range(n_states):
            xi_sum = np.sum(xi[:, s, :], axis=0)
            # Add Dirichlet prior (sticky HDP-HMM)
            prior = np.ones(n_states) * self.alpha / n_states
            prior[s] += self.alpha  # Sticky
            
            self.transition_probs_[s] = (xi_sum + prior) / (np.sum(xi_sum) + np.sum(prior))
        
        return gamma
    
    def _prune_regimes(self, threshold=0.01):
        """Identify and count active regimes"""
        if self.regime_probs_ is None:
            return 0
        
        # Count usage of each regime
        regime_usage = np.sum(self.regime_probs_, axis=0) / len(self.regime_probs_)
        
        # Count regimes used more than threshold
        active_regimes = np.sum(regime_usage > threshold)
        
        return active_regimes
    
    def fit(self, data):
        """
        Fit HDP-HMM model
        
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
        
        # Run Baum-Welch (EM) iterations
        for iteration in range(self.max_iter):
            gamma = self._baum_welch_iteration(data)
            
            # Check for convergence (simplified)
            if iteration > 0 and iteration % 10 == 0:
                active = self._prune_regimes()
                if iteration % 20 == 0:
                    print(f"Iteration {iteration}: {active} active regimes")
        
        self.regime_probs_ = gamma
        self.n_active_regimes_ = self._prune_regimes()
        self.fitted_ = True
        
        print(f"\nHDP-HMM fitted on {len(data)} observations")
        print(f"Discovered {self.n_active_regimes_} active regimes (truncation: {self.truncation})")
        
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
            columns=[f'Regime_{i}' for i in range(self.truncation)]
        )
    
    def get_active_regimes(self, threshold=0.01):
        """
        Get list of active regimes
        
        Parameters
        ----------
        threshold : float
            Minimum usage threshold to consider regime active
        
        Returns
        -------
        active_regimes : list
            List of active regime indices
        """
        if not self.fitted_:
            raise ValueError("Model not fitted yet.")
        
        regime_usage = np.sum(self.regime_probs_, axis=0) / len(self.regime_probs_)
        active = np.where(regime_usage > threshold)[0]
        
        return active.tolist()
    
    def get_regime_statistics(self):
        """Get statistics for each active regime"""
        if not self.fitted_:
            raise ValueError("Model not fitted yet.")
        
        active_regimes = self.get_active_regimes()
        stats = {}
        
        for r in active_regimes:
            mask = self.predict_regime() == r
            
            stats[f"Regime_{r}"] = {
                'mean': self.means_[r],
                'variance': self.variances_[r],
                'count': np.sum(mask),
                'percentage': np.sum(mask) / len(mask) * 100,
                'avg_probability': np.mean(self.regime_probs_[:, r])
            }
        
        return stats
    
    def get_model_info(self):
        """Get model information"""
        if not self.fitted_:
            raise ValueError("Model not fitted yet.")
        
        return {
            'truncation': self.truncation,
            'n_active_regimes': self.n_active_regimes_,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'active_regime_indices': self.get_active_regimes()
        }
    
    def get_transition_matrix(self, active_only=True):
        """
        Get transition probability matrix
        
        Parameters
        ----------
        active_only : bool
            If True, return only transitions between active regimes
        
        Returns
        -------
        transition_matrix : np.ndarray
        """
        if not self.fitted_:
            raise ValueError("Model not fitted yet.")
        
        if active_only:
            active = self.get_active_regimes()
            return self.transition_probs_[np.ix_(active, active)]
        else:
            return self.transition_probs_
