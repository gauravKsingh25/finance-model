"""
Hierarchical Dirichlet Process Hidden Markov Model (HDP-HMM)
Automatically discovers number of regimes from data using Gibbs sampling
and Dirichlet Process priors.

This is a proper implementation using:
- Stick-breaking construction for infinite state space
- Gibbs sampling for posterior inference
- Hierarchical Dirichlet Process priors
"""
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import norm, invgamma, multivariate_normal
from scipy.special import logsumexp, digamma
import warnings
warnings.filterwarnings('ignore')


class HDPHMM:
    """
    Hierarchical Dirichlet Process Hidden Markov Model
    
    Automatically discovers the number of market regimes from data
    using a non-parametric Bayesian approach with Gibbs sampling.
    
    The HDP-HMM uses a hierarchical Dirichlet process prior to allow
    an infinite number of hidden states, automatically inferring the
    appropriate number from data through Gibbs sampling.
    
    Parameters
    ----------
    truncation : int, default=10
        Truncation level for stick-breaking (max number of states to consider)
    alpha : float, default=1.0
        Concentration parameter for state transitions (DP parameter)
    gamma : float, default=1.0
        Concentration parameter for global distribution (top-level DP)
    max_iter : int, default=100
        Number of Gibbs sampling iterations
    kappa : float, default=1.0
        Sticky parameter for self-transitions (kappa-HDP-HMM)
    random_state : int, optional
        Random seed for reproducibility
        
    Attributes
    ----------
    means_ : np.ndarray
        Emission means for each state
    variances_ : np.ndarray
        Emission variances for each state
    beta_ : np.ndarray
        Stick-breaking weights (global distribution)
    pi_ : np.ndarray
        Transition probability matrix
    state_sequence_ : np.ndarray
        Most likely state sequence (Viterbi path)
    n_active_regimes_ : int
        Number of active/used regimes
    fitted_ : bool
        Whether the model has been fitted
    """
    
    def __init__(
        self, 
        truncation: int = 10,
        alpha: float = 1.0,
        gamma: float = 1.0,
        max_iter: int = 100,
        kappa: float = 1.0,
        random_state: Optional[int] = None
    ):
        self.truncation = truncation
        self.alpha = alpha
        self.gamma = gamma
        self.max_iter = max_iter
        self.kappa = kappa
        self.random_state = random_state
        
        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)
        
        # Model parameters (to be learned)
        self.means_: Optional[np.ndarray] = None
        self.variances_: Optional[np.ndarray] = None
        self.beta_: Optional[np.ndarray] = None  # Stick-breaking weights
        self.pi_: Optional[np.ndarray] = None    # Transition matrix
        self.state_sequence_: Optional[np.ndarray] = None
        self.regime_probs_: Optional[np.ndarray] = None
        self.n_active_regimes_: Optional[int] = None
        
        # Hyperparameters for emission distributions (conjugate priors)
        self.mu0: float = 0.0      # Prior mean
        self.sigma0: float = 1.0   # Prior variance on mean
        self.a0: float = 2.0       # Prior shape for variance (inverse gamma)
        self.b0: float = 1.0       # Prior scale for variance (inverse gamma)
        
        self.fitted_ = False
    
    def _stick_breaking(self, v: np.ndarray) -> np.ndarray:
        """
        Convert stick-breaking variables to probability weights.
        
        Parameters
        ----------
        v : np.ndarray
            Stick-breaking proportions (each in [0,1])
            
        Returns
        -------
        beta : np.ndarray
            Probability weights that sum to 1
        """
        beta = np.zeros(len(v))
        remaining = 1.0
        
        for k in range(len(v) - 1):
            beta[k] = v[k] * remaining
            remaining *= (1 - v[k])
        beta[-1] = remaining
        
        return beta
    
    def _sample_stick_breaking_weights(self, counts: np.ndarray) -> np.ndarray:
        """
        Sample stick-breaking weights from posterior.
        
        Parameters
        ----------
        counts : np.ndarray
            Usage counts for each state
            
        Returns
        -------
        beta : np.ndarray
            Sampled probability weights
        """
        v = np.zeros(self.truncation)
        
        for k in range(self.truncation - 1):
            # Count observations in state k and beyond
            nk = counts[k]
            nk_plus = np.sum(counts[k+1:])
            
            # Sample from Beta posterior
            a = 1.0 + nk
            b = self.gamma + nk_plus
            v[k] = np.random.beta(a, b)
        
        v[-1] = 1.0  # Last stick gets all remaining
        
        return self._stick_breaking(v)
    
    def _initialize_parameters(self, data: np.ndarray) -> None:
        """
        Initialize parameters using data statistics.
        
        Parameters
        ----------
        data : np.ndarray
            Univariate time series data
        """
        n_obs = len(data)
        
        # Check for constant or near-constant data
        data_var = np.var(data)
        if data_var < 1e-8:
            data_var = 1e-2  # Set minimum variance for constant data
        
        # Initialize hyperparameters based on data
        self.mu0 = np.mean(data)
        self.sigma0 = max(data_var, 1e-2)
        self.b0 = max(data_var, 1e-2)
        
        # Initialize means from data quantiles
        quantiles = np.linspace(0, 1, self.truncation)
        self.means_ = np.quantile(data, quantiles)
        
        # Initialize variances
        self.variances_ = np.full(self.truncation, max(data_var, 1e-2))
        
        # Initialize stick-breaking weights (uniform-ish)
        v = np.random.beta(1, self.gamma, self.truncation)
        v[-1] = 1.0
        self.beta_ = self._stick_breaking(v)
        
        # Initialize transition matrix using sticky HDP prior
        self.pi_ = np.zeros((self.truncation, self.truncation))
        for i in range(self.truncation):
            # Dirichlet with sticky bias
            concentration = self.alpha * self.beta_ + self.kappa * (np.arange(self.truncation) == i)
            self.pi_[i, :] = np.random.dirichlet(concentration)
        
        # Initialize state sequence randomly
        self.state_sequence_ = np.random.choice(
            self.truncation, 
            size=n_obs, 
            p=self.beta_
        )
    
    def _emission_log_prob(self, obs: float, state: int) -> float:
        """
        Compute log emission probability.
        
        Parameters
        ----------
        obs : float
            Observation value
        state : int
            State index
            
        Returns
        -------
        log_prob : float
            Log probability of observation given state
        """
        mean = self.means_[state]
        var = max(self.variances_[state], 1e-6)  # Numerical stability
        
        # Log of normal pdf
        log_prob = -0.5 * np.log(2 * np.pi * var) - 0.5 * ((obs - mean) ** 2) / var
        
        # Handle numerical issues
        if np.isnan(log_prob) or np.isinf(log_prob):
            log_prob = -1e10  # Very small probability
        
        return log_prob
    
    def _sample_state_sequence(self, data: np.ndarray) -> None:
        """
        Sample state sequence using forward-filtering backward-sampling (OPTIMIZED).
        
        This implements the Gibbs sampling step for the hidden states
        given current parameters.
        
        Parameters
        ----------
        data : np.ndarray
            Observation sequence
        """
        n_obs = len(data)
        n_states = self.truncation
        
        # Precompute all emission log probabilities (vectorized)
        log_emissions = np.zeros((n_obs, n_states))
        for k in range(n_states):
            mean = self.means_[k]
            var = max(self.variances_[k], 1e-6)
            log_emissions[:, k] = -0.5 * np.log(2 * np.pi * var) - 0.5 * ((data - mean) ** 2) / var
        
        # Replace NaN/inf with very small probability
        log_emissions = np.nan_to_num(log_emissions, nan=-1e10, posinf=-1e10, neginf=-1e10)
        
        # Forward pass: compute filtering distributions
        log_forward = np.zeros((n_obs, n_states))
        
        # Initial distribution
        log_forward[0, :] = np.log(self.beta_ + 1e-10) + log_emissions[0, :]
        log_forward[0, :] -= logsumexp(log_forward[0, :])
        
        # Precompute log transition matrix
        log_pi = np.log(self.pi_ + 1e-10)
        
        # Forward recursion (vectorized)
        for t in range(1, n_obs):
            # Vectorized computation of message passing
            log_forward[t, :] = logsumexp(log_forward[t-1, :, np.newaxis] + log_pi, axis=0)
            log_forward[t, :] += log_emissions[t, :]
            log_forward[t, :] -= logsumexp(log_forward[t, :])
        
        # Backward sampling
        self.state_sequence_ = np.zeros(n_obs, dtype=int)
        
        # Sample final state
        final_probs = np.exp(log_forward[-1, :] - logsumexp(log_forward[-1, :]))
        final_probs = np.maximum(final_probs, 0)  # Ensure non-negative
        final_probs /= np.sum(final_probs)
        self.state_sequence_[-1] = np.random.choice(n_states, p=final_probs)
        
        # Backward recursion
        for t in range(n_obs - 2, -1, -1):
            # Conditional probability p(s_t | s_{t+1}, data)
            next_state = self.state_sequence_[t + 1]
            log_probs = log_forward[t, :] + log_pi[:, next_state]
            
            probs = np.exp(log_probs - logsumexp(log_probs))
            probs = np.maximum(probs, 0)  # Ensure non-negative
            probs /= np.sum(probs)
            
            self.state_sequence_[t] = np.random.choice(n_states, p=probs)
    
    def _sample_emission_parameters(self, data: np.ndarray) -> None:
        """
        Sample emission parameters (means and variances) from posterior.
        
        Uses conjugate Normal-Inverse-Gamma priors for Gaussian emissions.
        
        Parameters
        ----------
        data : np.ndarray
            Observation sequence
        """
        for k in range(self.truncation):
            # Get observations assigned to state k
            mask = (self.state_sequence_ == k)
            n_k = np.sum(mask)
            
            if n_k > 0:
                obs_k = data[mask]
                
                # Posterior parameters for Normal-Inverse-Gamma
                # Update for mean (normal posterior)
                posterior_var = 1.0 / (1.0 / self.sigma0 + n_k / self.variances_[k])
                posterior_mean = posterior_var * (
                    self.mu0 / self.sigma0 + 
                    np.sum(obs_k) / self.variances_[k]
                )
                
                # Update for variance (inverse gamma posterior)
                a_post = self.a0 + n_k / 2.0
                squared_diff = np.sum((obs_k - self.means_[k]) ** 2)
                b_post = self.b0 + squared_diff / 2.0
                
                # Sample variance from inverse gamma
                self.variances_[k] = invgamma.rvs(a_post, scale=b_post)
                self.variances_[k] = max(self.variances_[k], 1e-6)  # Numerical stability
                
                # Sample mean from normal
                self.means_[k] = np.random.normal(posterior_mean, np.sqrt(posterior_var))
            else:
                # Sample from prior if no observations
                self.variances_[k] = invgamma.rvs(self.a0, scale=self.b0)
                self.variances_[k] = max(self.variances_[k], 1e-6)
                self.means_[k] = np.random.normal(self.mu0, np.sqrt(self.sigma0))
    
    def _sample_transition_matrix(self) -> None:
        """
        Sample transition matrix from posterior using Dirichlet.
        
        Uses the sticky HDP-HMM formulation with kappa parameter.
        """
        # Count transitions
        transition_counts = np.zeros((self.truncation, self.truncation))
        
        for t in range(len(self.state_sequence_) - 1):
            from_state = self.state_sequence_[t]
            to_state = self.state_sequence_[t + 1]
            transition_counts[from_state, to_state] += 1
        
        # Sample each row of transition matrix
        for i in range(self.truncation):
            # Dirichlet posterior with sticky bias
            concentration = (
                self.alpha * self.beta_ + 
                self.kappa * (np.arange(self.truncation) == i) +
                transition_counts[i, :]
            )
            
            # Ensure positive concentrations
            concentration = np.maximum(concentration, 1e-10)
            
            self.pi_[i, :] = np.random.dirichlet(concentration)
    
    def _sample_global_weights(self) -> None:
        """
        Sample global stick-breaking weights beta from posterior.
        """
        # Count state usage
        state_counts = np.zeros(self.truncation)
        for k in range(self.truncation):
            state_counts[k] = np.sum(self.state_sequence_ == k)
        
        # Sample stick-breaking weights
        self.beta_ = self._sample_stick_breaking_weights(state_counts)
    
    def _gibbs_iteration(self, data: np.ndarray) -> None:
        """
        Single Gibbs sampling iteration.
        
        Sequentially samples:
        1. State sequence (forward-filtering backward-sampling)
        2. Emission parameters (from conjugate posteriors)
        3. Transition matrix (from Dirichlet posteriors)
        4. Global weights (stick-breaking)
        
        Parameters
        ----------
        data : np.ndarray
            Observation sequence
        """
        # Sample state sequence
        self._sample_state_sequence(data)
        
        # Sample emission parameters
        self._sample_emission_parameters(data)
        
        # Sample transition matrix
        self._sample_transition_matrix()
        
        # Sample global weights
        self._sample_global_weights()
    
    def _compute_regime_probabilities(self, data: np.ndarray) -> np.ndarray:
        """
        Compute marginal state probabilities using forward-backward (OPTIMIZED).
        
        Parameters
        ----------
        data : np.ndarray
            Observation sequence
            
        Returns
        -------
        gamma : np.ndarray
            State probabilities (n_obs x n_states)
        """
        n_obs = len(data)
        n_states = self.truncation
        
        # Precompute all emission log probabilities
        log_emissions = np.zeros((n_obs, n_states))
        for k in range(n_states):
            mean = self.means_[k]
            var = max(self.variances_[k], 1e-6)
            log_emissions[:, k] = -0.5 * np.log(2 * np.pi * var) - 0.5 * ((data - mean) ** 2) / var
        
        log_emissions = np.nan_to_num(log_emissions, nan=-1e10, posinf=-1e10, neginf=-1e10)
        
        # Forward pass
        log_forward = np.zeros((n_obs, n_states))
        log_forward[0, :] = np.log(self.beta_ + 1e-10) + log_emissions[0, :]
        log_forward[0, :] -= logsumexp(log_forward[0, :])
        
        log_pi = np.log(self.pi_ + 1e-10)
        
        for t in range(1, n_obs):
            log_forward[t, :] = logsumexp(log_forward[t-1, :, np.newaxis] + log_pi, axis=0)
            log_forward[t, :] += log_emissions[t, :]
            log_forward[t, :] -= logsumexp(log_forward[t, :])
        
        # Backward pass
        log_backward = np.zeros((n_obs, n_states))
        log_backward[-1, :] = 0.0
        
        for t in range(n_obs - 2, -1, -1):
            log_backward[t, :] = logsumexp(
                log_pi + log_emissions[t+1, :] + log_backward[t+1, :], 
                axis=1
            )
        
        # Compute gamma
        log_gamma = log_forward + log_backward
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)
        gamma = np.nan_to_num(gamma, nan=0.0)
        
        # Renormalize each row
        row_sums = gamma.sum(axis=1, keepdims=True)
        gamma = gamma / np.maximum(row_sums, 1e-10)
        
        return gamma
    
    def fit(self, data: pd.Series) -> 'HDPHMM':
        """
        Fit HDP-HMM model using Gibbs sampling.
        
        Parameters
        ----------
        data : pd.Series or np.ndarray
            Univariate time series data
        
        Returns
        -------
        self : HDPHMM
            Fitted model instance
        """
        if isinstance(data, pd.Series):
            data = data.values
        
        data = np.asarray(data).flatten()
        
        if len(data) < 10:
            raise ValueError("Need at least 10 observations to fit HDP-HMM")
        
        print(f"Fitting HDP-HMM with Gibbs sampling on {len(data)} observations...")
        print(f"Parameters: truncation={self.truncation}, alpha={self.alpha}, "
              f"gamma={self.gamma}, kappa={self.kappa}")
        
        # Initialize parameters
        self._initialize_parameters(data)
        
        # Gibbs sampling (with optimized iterations)
        print_interval = max(1, self.max_iter // 5)  # Print 5 times max
        
        for iteration in range(self.max_iter):
            self._gibbs_iteration(data)
            
            # Print progress less frequently
            if iteration == 0 or (iteration + 1) % print_interval == 0:
                n_active = len(np.unique(self.state_sequence_))
                print(f"  Iteration {iteration + 1}/{self.max_iter}: "
                      f"{n_active} active states")
        
        # Compute final regime probabilities
        self.regime_probs_ = self._compute_regime_probabilities(data)
        
        # Count active regimes
        unique_states = np.unique(self.state_sequence_)
        self.n_active_regimes_ = len(unique_states)
        
        self.fitted_ = True
        
        print(f"✓ HDP-HMM fitted: {self.n_active_regimes_} active regimes")
        
        return self
    
    def predict_regime(self) -> np.ndarray:
        """
        Get most likely regime at each time point.
        
        Returns
        -------
        regimes : np.ndarray
            Most likely regime indices
        """
        if not self.fitted_:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        return self.state_sequence_.copy()
    
    def get_regime_probabilities(self) -> pd.DataFrame:
        """
        Get regime probabilities over time.
        
        Returns
        -------
        probs : pd.DataFrame
            DataFrame with regime probabilities for each time step
        """
        if not self.fitted_:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        return pd.DataFrame(
            self.regime_probs_,
            columns=[f'Regime_{i}' for i in range(self.truncation)]
        )
    
    def get_active_regimes(self, threshold: float = 0.01) -> List[int]:
        """
        Get list of active regimes.
        
        Parameters
        ----------
        threshold : float, default=0.01
            Minimum usage threshold to consider regime active
        
        Returns
        -------
        active_regimes : List[int]
            List of active regime indices
        """
        if not self.fitted_:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        regime_usage = np.sum(self.regime_probs_, axis=0) / len(self.regime_probs_)
        active = np.where(regime_usage > threshold)[0]
        
        return active.tolist()
    
    def get_regime_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for each active regime.
        
        Returns
        -------
        stats : Dict[str, Dict[str, float]]
            Dictionary mapping regime names to their statistics
        """
        if not self.fitted_:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        active_regimes = self.get_active_regimes()
        stats = {}
        
        for r in active_regimes:
            mask = self.state_sequence_ == r
            n_obs = np.sum(mask)
            
            stats[f"Regime_{r}"] = {
                'mean': float(self.means_[r]),
                'variance': float(self.variances_[r]),
                'std_dev': float(np.sqrt(self.variances_[r])),
                'count': int(n_obs),
                'percentage': float(n_obs / len(mask) * 100),
                'avg_probability': float(np.mean(self.regime_probs_[:, r])),
                'beta_weight': float(self.beta_[r])
            }
        
        return stats
    
    def get_model_info(self) -> Dict[str, any]:
        """
        Get model information and hyperparameters.
        
        Returns
        -------
        info : Dict[str, any]
            Dictionary with model information
        """
        if not self.fitted_:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        return {
            'truncation': self.truncation,
            'n_active_regimes': self.n_active_regimes_,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'kappa': self.kappa,
            'max_iter': self.max_iter,
            'active_regime_indices': self.get_active_regimes(),
            'random_state': self.random_state
        }
    
    def get_transition_matrix(self, active_only: bool = True) -> np.ndarray:
        """
        Get transition probability matrix.
        
        Parameters
        ----------
        active_only : bool, default=True
            If True, return only transitions between active regimes
        
        Returns
        -------
        transition_matrix : np.ndarray
            Transition probability matrix
        """
        if not self.fitted_:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        if active_only:
            active = self.get_active_regimes()
            return self.pi_[np.ix_(active, active)]
        else:
            return self.pi_.copy()
    
    def get_global_weights(self) -> np.ndarray:
        """
        Get global stick-breaking weights (beta).
        
        Returns
        -------
        beta : np.ndarray
            Global probability weights from stick-breaking
        """
        if not self.fitted_:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        return self.beta_.copy()

