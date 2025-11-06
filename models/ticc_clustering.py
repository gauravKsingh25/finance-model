"""
TICC (Toeplitz Inverse Covariance-based Clustering)
Detects correlation structure changes across multiple time series
"""
import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLassoCV
from sklearn.cluster import KMeans
from scipy.linalg import block_diag
import warnings
warnings.filterwarnings('ignore')


class TICCClustering:
    """
    TICC: Toeplitz Inverse Covariance-based Clustering
    
    Identifies temporal segments where assets have similar correlation structures.
    Useful for detecting regime changes in multi-asset portfolios.
    
    Parameters
    ----------
    n_clusters : int, default=3
        Number of correlation regimes to discover
    window_size : int, default=10
        Size of temporal window for computing correlations
    lambda_parameter : float, default=11e-2
        Sparsity parameter for inverse covariance estimation
    beta : float, default=400
        Temporal smoothness parameter
    max_iter : int, default=100
        Maximum iterations for optimization
    """
    
    def __init__(self, n_clusters=3, window_size=10, lambda_parameter=11e-2, 
                 beta=400, max_iter=100):
        self.n_clusters = n_clusters
        self.window_size = window_size
        self.lambda_parameter = lambda_parameter
        self.beta = beta
        self.max_iter = max_iter
        
        self.cluster_centers_ = None
        self.labels_ = None
        self.fitted_ = False
        self.n_assets_ = None
        self.inv_cov_matrices_ = {}
        
    def _create_block_toeplitz(self, data):
        """Create block-Toeplitz matrix from multivariate time series"""
        n_samples, n_assets = data.shape
        n_windows = n_samples - self.window_size + 1
        
        # Create stacked windows
        stacked_data = np.zeros((n_windows, n_assets * self.window_size))
        
        for i in range(n_windows):
            window = data[i:i + self.window_size, :]
            stacked_data[i, :] = window.T.flatten()
        
        return stacked_data
    
    def _compute_inverse_covariance(self, data_segment):
        """Compute sparse inverse covariance matrix"""
        try:
            # Use GraphicalLasso for sparse inverse covariance
            model = GraphicalLassoCV(alphas=4, max_iter=100)
            model.fit(data_segment)
            return model.precision_
        except:
            # Fallback to simple inverse
            cov = np.cov(data_segment.T)
            # Add small diagonal for numerical stability
            cov += np.eye(cov.shape[0]) * 1e-4
            return np.linalg.inv(cov)
    
    def _initialize_clusters(self, stacked_data):
        """Initialize clusters using K-means"""
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        initial_labels = kmeans.fit_predict(stacked_data)
        return initial_labels
    
    def _compute_cluster_parameters(self, stacked_data, labels):
        """Compute inverse covariance for each cluster"""
        inv_covs = {}
        
        for k in range(self.n_clusters):
            cluster_data = stacked_data[labels == k]
            
            if len(cluster_data) > 0:
                inv_covs[k] = self._compute_inverse_covariance(cluster_data)
            else:
                # Empty cluster - use identity
                dim = stacked_data.shape[1]
                inv_covs[k] = np.eye(dim)
        
        return inv_covs
    
    def _assign_labels(self, stacked_data, inv_covs):
        """Assign labels based on likelihood and temporal smoothness"""
        n_windows = len(stacked_data)
        labels = np.zeros(n_windows, dtype=int)
        
        # Compute log-likelihoods for each cluster
        log_likelihoods = np.zeros((n_windows, self.n_clusters))
        
        for k in range(self.n_clusters):
            for i in range(n_windows):
                x = stacked_data[i, :]
                theta = inv_covs[k]
                
                # Log-likelihood (up to constant)
                log_lik = -0.5 * x.T @ theta @ x
                log_likelihoods[i, k] = log_lik
        
        # Dynamic programming for temporal smoothness
        # Simple greedy approach: maximize likelihood with smoothness penalty
        for i in range(n_windows):
            if i == 0:
                labels[i] = np.argmax(log_likelihoods[i, :])
            else:
                # Penalize switching
                scores = log_likelihoods[i, :].copy()
                for k in range(self.n_clusters):
                    if k != labels[i-1]:
                        scores[k] -= self.beta / n_windows
                labels[i] = np.argmax(scores)
        
        return labels
    
    def fit(self, data):
        """
        Fit TICC model to multivariate time series
        
        Parameters
        ----------
        data : pd.DataFrame or np.ndarray
            Multivariate time series (T x N) where T is time, N is number of assets
        
        Returns
        -------
        self
        """
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        self.n_assets_ = data.shape[1]
        
        # Create block-Toeplitz representation
        stacked_data = self._create_block_toeplitz(data)
        
        # Initialize clusters
        labels = self._initialize_clusters(stacked_data)
        
        # EM-like iterations
        for iteration in range(self.max_iter):
            # M-step: Update cluster parameters
            inv_covs = self._compute_cluster_parameters(stacked_data, labels)
            
            # E-step: Reassign labels
            new_labels = self._assign_labels(stacked_data, inv_covs)
            
            # Check convergence
            if np.array_equal(labels, new_labels):
                print(f"TICC converged after {iteration + 1} iterations")
                break
            
            labels = new_labels
        
        # Extend labels to match original data length
        extended_labels = np.zeros(len(data), dtype=int)
        extended_labels[:self.window_size-1] = labels[0]
        extended_labels[self.window_size-1:] = labels
        
        self.labels_ = extended_labels
        self.inv_cov_matrices_ = inv_covs
        self.fitted_ = True
        
        return self
    
    def predict(self, data=None):
        """
        Predict cluster assignments
        
        Returns
        -------
        labels : np.ndarray
            Cluster assignments for each time point
        """
        if not self.fitted_:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        return self.labels_
    
    def get_cluster_statistics(self):
        """Get statistics for each cluster"""
        if not self.fitted_:
            raise ValueError("Model not fitted yet.")
        
        stats = {}
        unique_labels = np.unique(self.labels_)
        
        for label in unique_labels:
            mask = self.labels_ == label
            stats[f"Cluster_{label}"] = {
                'count': np.sum(mask),
                'percentage': np.sum(mask) / len(self.labels_) * 100,
                'avg_duration': self._compute_avg_duration(mask)
            }
        
        return stats
    
    def _compute_avg_duration(self, mask):
        """Compute average duration of regime"""
        durations = []
        current_duration = 0
        
        for i in range(len(mask)):
            if mask[i]:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                    current_duration = 0
        
        if current_duration > 0:
            durations.append(current_duration)
        
        return np.mean(durations) if durations else 0
    
    def get_correlation_structure(self, cluster_id):
        """
        Get correlation structure for a specific cluster
        
        Parameters
        ----------
        cluster_id : int
            Cluster ID
        
        Returns
        -------
        correlation : np.ndarray
            Correlation matrix (approximation from inverse covariance)
        """
        if cluster_id not in self.inv_cov_matrices_:
            raise ValueError(f"Cluster {cluster_id} not found")
        
        inv_cov = self.inv_cov_matrices_[cluster_id]
        
        # Extract first block (contemporaneous correlations)
        block_size = self.n_assets_
        theta_0 = inv_cov[:block_size, :block_size]
        
        # Convert inverse covariance to correlation (approximation)
        # Correlation ≈ -θ_ij / sqrt(θ_ii * θ_jj) for i≠j
        diag_sqrt = np.sqrt(np.diag(theta_0))
        corr = -theta_0 / np.outer(diag_sqrt, diag_sqrt)
        np.fill_diagonal(corr, 1.0)
        
        return corr
    
    def get_regime_transitions(self):
        """Get transition points between regimes"""
        if not self.fitted_:
            raise ValueError("Model not fitted yet.")
        
        transitions = []
        for i in range(1, len(self.labels_)):
            if self.labels_[i] != self.labels_[i-1]:
                transitions.append({
                    'index': i,
                    'from': self.labels_[i-1],
                    'to': self.labels_[i]
                })
        
        return transitions
    
    def get_model_info(self):
        """Get model information"""
        if not self.fitted_:
            raise ValueError("Model not fitted yet.")
        
        return {
            'n_clusters': self.n_clusters,
            'window_size': self.window_size,
            'n_assets': self.n_assets_,
            'n_transitions': len(self.get_regime_transitions()),
            'fitted': self.fitted_
        }
