"""
Standard Hidden Markov Model (HMM) Implementation
For comparison with Sticky HDP-HMM in the TICC regime detection pipeline
Uses hmmlearn library for robust implementation
"""
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
from hmmlearn import hmm
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


class StandardHMM:
    """
    Standard Hidden Markov Model with Gaussian emissions
    
    This is a traditional HMM with fixed number of states, used for
    comparison with the Sticky HDP-HMM. Unlike HDP-HMM, this requires
    pre-specification of the number of states.
    
    Parameters
    ----------
    n_components : int, default=3
        Number of hidden states (must be specified in advance)
    covariance_type : str, default='full'
        Type of covariance parameters ('spherical', 'diag', 'full', 'tied')
    n_iter : int, default=100
        Maximum number of iterations for Baum-Welch EM algorithm
    random_state : int, optional
        Random seed for reproducibility
    min_covar : float, default=1e-3
        Minimum covariance for numerical stability
        
    Attributes
    ----------
    means_ : np.ndarray
        Emission means for each state
    covars_ : np.ndarray
        Emission covariances for each state
    transmat_ : np.ndarray
        Transition probability matrix
    startprob_ : np.ndarray
        Initial state distribution
    state_sequence_ : np.ndarray
        Most likely state sequence (Viterbi path)
    fitted_ : bool
        Whether the model has been fitted
    """
    
    def __init__(
        self, 
        n_components: int = 3,
        covariance_type: str = 'full',
        n_iter: int = 100,
        random_state: Optional[int] = None,
        min_covar: float = 1e-3
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self.min_covar = min_covar
        
        # Initialize hmmlearn model
        self.model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
            min_covar=min_covar,
            init_params='stmc',  # Initialize start, transition, means, covariances
            params='stmc'  # Update all parameters
        )
        
        # Model parameters (populated after fitting)
        self.means_: Optional[np.ndarray] = None
        self.covars_: Optional[np.ndarray] = None
        self.transmat_: Optional[np.ndarray] = None
        self.startprob_: Optional[np.ndarray] = None
        self.state_sequence_: Optional[np.ndarray] = None
        self.regime_probs_: Optional[np.ndarray] = None
        
        self.fitted_ = False
    
    def fit(self, data: pd.Series) -> 'StandardHMM':
        """
        Fit HMM model using Baum-Welch EM algorithm.
        
        Parameters
        ----------
        data : pd.Series or np.ndarray
            Univariate time series data
        
        Returns
        -------
        self : StandardHMM
            Fitted model instance
        """
        if isinstance(data, pd.Series):
            data = data.values
        
        data = np.asarray(data).flatten()
        
        if len(data) < 10:
            raise ValueError("Need at least 10 observations to fit HMM")
        
        # Reshape for hmmlearn (expects 2D: n_samples x n_features)
        X = data.reshape(-1, 1)
        
        print(f"Fitting Standard HMM with {self.n_components} states on {len(data)} observations...")
        
        try:
            # Fit the model
            self.model.fit(X)
            
            # Extract learned parameters
            self.means_ = self.model.means_.flatten()
            self.covars_ = self.model.covars_.flatten()
            self.transmat_ = self.model.transmat_
            self.startprob_ = self.model.startprob_
            
            # Decode most likely state sequence (Viterbi)
            self.state_sequence_ = self.model.predict(X)
            
            # Compute posterior probabilities (forward-backward)
            self.regime_probs_ = self.model.predict_proba(X)
            
            self.fitted_ = True
            
            # Calculate convergence metrics
            converged = self.model.monitor_.converged if hasattr(self.model, 'monitor_') else True
            n_iter_used = self.model.n_iter if hasattr(self.model, 'n_iter') else self.n_iter
            
            print(f"✓ Standard HMM fitted: {self.n_components} states")
            print(f"  Converged: {converged}, Iterations: {n_iter_used}")
            print(f"  Log-likelihood: {self.model.score(X):.2f}")
            
        except Exception as e:
            print(f"Error fitting HMM: {e}")
            raise
        
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
            columns=[f'Regime_{i}' for i in range(self.n_components)]
        )
    
    def get_regime_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for each regime.
        
        Returns
        -------
        stats : Dict[str, Dict[str, float]]
            Dictionary mapping regime names to their statistics
        """
        if not self.fitted_:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        stats = {}
        
        for r in range(self.n_components):
            mask = self.state_sequence_ == r
            n_obs = np.sum(mask)
            
            stats[f"Regime_{r}"] = {
                'mean': float(self.means_[r]),
                'variance': float(self.covars_[r]),
                'std_dev': float(np.sqrt(self.covars_[r])),
                'count': int(n_obs),
                'percentage': float(n_obs / len(mask) * 100),
                'avg_probability': float(np.mean(self.regime_probs_[:, r])),
                'start_prob': float(self.startprob_[r])
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
            'n_components': self.n_components,
            'covariance_type': self.covariance_type,
            'n_iter': self.n_iter,
            'random_state': self.random_state,
            'converged': self.model.monitor_.converged if hasattr(self.model, 'monitor_') else True
        }
    
    def get_transition_matrix(self) -> np.ndarray:
        """
        Get transition probability matrix.
        
        Returns
        -------
        transition_matrix : np.ndarray
            Transition probability matrix (n_components x n_components)
        """
        if not self.fitted_:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        return self.transmat_.copy()
    
    def get_stationary_distribution(self) -> np.ndarray:
        """
        Compute stationary distribution of the Markov chain.
        
        Returns
        -------
        stationary : np.ndarray
            Stationary distribution over states
        """
        if not self.fitted_:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        # Find eigenvector corresponding to eigenvalue 1
        eigenvalues, eigenvectors = np.linalg.eig(self.transmat_.T)
        
        # Find index of eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        
        # Get corresponding eigenvector and normalize
        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / np.sum(stationary)
        
        return stationary
    
    def calculate_metrics(self, data: np.ndarray) -> Dict[str, float]:
        """
        Calculate model performance metrics.
        
        Parameters
        ----------
        data : np.ndarray
            Original data used for fitting
            
        Returns
        -------
        metrics : Dict[str, float]
            Dictionary of performance metrics
        """
        if not self.fitted_:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        X = data.reshape(-1, 1)
        
        # Calculate AIC and BIC
        log_likelihood = self.model.score(X)
        n_samples = len(data)
        
        # Number of free parameters
        # Means: n_components, Variances: n_components
        # Transition matrix: n_components * (n_components - 1) 
        # Start probabilities: n_components - 1
        n_params = (
            self.n_components +  # means
            self.n_components +  # variances
            self.n_components * (self.n_components - 1) +  # transition matrix
            (self.n_components - 1)  # start probabilities
        )
        
        aic = 2 * n_params - 2 * log_likelihood
        bic = np.log(n_samples) * n_params - 2 * log_likelihood
        
        # Calculate regime transition metrics
        transitions = np.sum(np.diff(self.state_sequence_) != 0)
        avg_regime_duration = n_samples / (transitions + 1)
        
        return {
            'log_likelihood': log_likelihood,
            'aic': aic,
            'bic': bic,
            'n_parameters': n_params,
            'n_transitions': transitions,
            'avg_regime_duration': avg_regime_duration
        }
