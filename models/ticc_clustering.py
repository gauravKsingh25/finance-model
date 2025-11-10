"""
Improved TICC-like Clustering for Regime Detection
Uses temporal correlation structure changes to detect market regimes.
This is a robust implementation focused on accurate regime detection.
"""
import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLassoCV, empirical_covariance
from sklearn.cluster import KMeans
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class TICCClustering:
    """
    Improved TICC-like Clustering for Temporal Regime Detection
    
    This implementation focuses on detecting correlation structure changes
    across multiple time series with reduced false positives.
    
    Key improvements:
    - Better initialization using correlation distance
    - Robust inverse covariance estimation
    - Adaptive temporal smoothing
    - Outlier-resistant clustering
    
    Parameters
    ----------
    n_clusters : int, default=3
        Number of correlation regimes to discover
    window_size : int, default=10
        Size of temporal window for computing correlations
    lambda_parameter : float, default=11e-2
        Sparsity parameter for inverse covariance estimation
    beta : float, default=600
        Temporal smoothness parameter (higher = fewer transitions)
    max_iter : int, default=50
        Maximum iterations for optimization
    convergence_threshold : float, default=0.95
        Convergence threshold for label stability
    min_cluster_size : int, default=5
        Minimum size for a valid cluster
    """
    
    def __init__(self, n_clusters=3, window_size=10, lambda_parameter=11e-2, 
                 beta=1000, max_iter=50, convergence_threshold=0.95, 
                 min_cluster_size=5):
        self.n_clusters = n_clusters
        self.window_size = window_size
        self.lambda_parameter = lambda_parameter
        self.beta = beta
        self.max_iter = max_iter
        self.convergence_threshold = convergence_threshold
        self.min_cluster_size = min_cluster_size
        
        self.cluster_centers_ = None
        self.labels_ = None
        self.fitted_ = False
        self.n_assets_ = None
        self.inv_cov_matrices_ = {}
        self.cov_matrices_ = {}
        self.scaler_ = None
    
    def _create_block_toeplitz(self, data):
        """
        Create block-Toeplitz-like feature matrix from multivariate time series
        Uses rolling windows with standardized data
        """
        n_samples, n_assets = data.shape
        n_windows = n_samples - self.window_size + 1
        
        # Standardize the data first
        if self.scaler_ is None:
            self.scaler_ = StandardScaler()
            data_scaled = self.scaler_.fit_transform(data)
        else:
            data_scaled = self.scaler_.transform(data)
        
        # Create stacked windows
        stacked_data = np.zeros((n_windows, n_assets * self.window_size))
        
        for i in range(n_windows):
            window = data_scaled[i:i + self.window_size, :]
            # Stack as flattened vector
            stacked_data[i, :] = window.T.flatten()
        
        return stacked_data, data_scaled
    
    def _compute_robust_covariance(self, data_segment, use_graphical_lasso=True):
        """
        Compute robust covariance/inverse covariance matrix
        
        Parameters
        ----------
        data_segment : np.ndarray
            Data segment for covariance computation
        use_graphical_lasso : bool
            Whether to use GraphicalLasso for sparse estimation
        
        Returns
        -------
        cov : np.ndarray
            Covariance matrix
        inv_cov : np.ndarray
            Inverse covariance (precision) matrix
        """
        if len(data_segment) < 3:
            # Too few samples - return identity
            dim = data_segment.shape[1]
            return np.eye(dim), np.eye(dim)
        
        try:
            if use_graphical_lasso and len(data_segment) > 10:
                # Use GraphicalLasso for sparse inverse covariance
                model = GraphicalLassoCV(
                    alphas=[self.lambda_parameter * i for i in [0.5, 1.0, 2.0]], 
                    max_iter=100,
                    cv=min(3, len(data_segment) // 3)
                )
                model.fit(data_segment)
                return model.covariance_, model.precision_
            else:
                # Use empirical covariance with regularization
                cov = empirical_covariance(data_segment, assume_centered=False)
                # Add regularization for numerical stability
                reg_cov = cov + np.eye(cov.shape[0]) * 1e-4
                inv_cov = np.linalg.inv(reg_cov)
                return cov, inv_cov
        except:
            # Fallback to regularized empirical covariance
            cov = np.cov(data_segment.T)
            reg_cov = cov + np.eye(cov.shape[0]) * 1e-3
            inv_cov = np.linalg.inv(reg_cov)
            return reg_cov, inv_cov
    
    def _initialize_clusters(self, stacked_data):
        """
        Initialize clusters using correlation-based K-means
        Uses robust initialization to avoid poor local minima
        """
        # Try multiple initializations and pick best one
        best_labels = None
        best_inertia = np.inf
        
        for _ in range(5):  # Try 5 random initializations
            kmeans = KMeans(
                n_clusters=self.n_clusters, 
                n_init=10,
                max_iter=300,
                random_state=None
            )
            labels = kmeans.fit_predict(stacked_data)
            
            if kmeans.inertia_ < best_inertia:
                best_inertia = kmeans.inertia_
                best_labels = labels
        
        # Ensure all clusters have minimum size
        best_labels = self._enforce_min_cluster_size(best_labels, stacked_data)
        
        return best_labels
    
    def _enforce_min_cluster_size(self, labels, data):
        """
        Ensure all clusters meet minimum size requirement
        Merge small clusters into nearest larger cluster
        """
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        while np.any(counts < self.min_cluster_size):
            # Find smallest cluster
            small_idx = np.argmin(counts)
            small_label = unique_labels[small_idx]
            
            if len(unique_labels) == 1:
                # Can't merge if only one cluster left
                break
            
            # Find nearest cluster based on centroid distance
            small_cluster_data = data[labels == small_label]
            small_centroid = np.mean(small_cluster_data, axis=0)
            
            min_dist = np.inf
            merge_target = None
            
            for other_label in unique_labels:
                if other_label != small_label:
                    other_data = data[labels == other_label]
                    other_centroid = np.mean(other_data, axis=0)
                    dist = np.linalg.norm(small_centroid - other_centroid)
                    
                    if dist < min_dist:
                        min_dist = dist
                        merge_target = other_label
            
            # Merge small cluster into target
            labels[labels == small_label] = merge_target
            
            # Relabel to be consecutive
            unique_labels_new = np.unique(labels)
            label_map = {old: new for new, old in enumerate(unique_labels_new)}
            labels = np.array([label_map[l] for l in labels])
            
            # Update unique labels and counts
            unique_labels, counts = np.unique(labels, return_counts=True)
        
        return labels
    
    def _compute_cluster_parameters(self, stacked_data, labels):
        """
        Compute covariance and inverse covariance for each cluster
        
        Returns
        -------
        cov_matrices : dict
            Covariance matrices for each cluster
        inv_cov_matrices : dict
            Inverse covariance matrices for each cluster
        """
        cov_matrices = {}
        inv_cov_matrices = {}
        
        for k in range(self.n_clusters):
            cluster_data = stacked_data[labels == k]
            
            if len(cluster_data) >= self.min_cluster_size:
                cov, inv_cov = self._compute_robust_covariance(cluster_data)
                cov_matrices[k] = cov
                inv_cov_matrices[k] = inv_cov
            else:
                # Small or empty cluster - use identity
                dim = stacked_data.shape[1]
                cov_matrices[k] = np.eye(dim)
                inv_cov_matrices[k] = np.eye(dim)
        
        return cov_matrices, inv_cov_matrices
    
    def _compute_log_likelihood(self, x, inv_cov, cov):
        """
        Compute log-likelihood of observation under Gaussian with given precision
        
        Uses stable computation to avoid numerical issues
        """
        try:
            # Compute log determinant safely
            sign, log_det = np.linalg.slogdet(inv_cov)
            if sign <= 0:
                log_det = 0
            
            # Mahalanobis distance
            mahal_dist = x.T @ inv_cov @ x
            
            # Log-likelihood
            d = len(x)
            log_lik = 0.5 * log_det - 0.5 * mahal_dist - 0.5 * d * np.log(2 * np.pi)
            
            return log_lik
        except:
            # Fallback to simple distance
            return -0.5 * np.linalg.norm(x) ** 2
    
    def _assign_labels(self, stacked_data, inv_covs, covs):
        """
        Assign labels using log-likelihood with temporal smoothness
        
        Uses forward-backward dynamic programming for global optimization
        """
        n_windows = len(stacked_data)
        
        # Compute log-likelihoods for each cluster
        log_likelihoods = np.zeros((n_windows, self.n_clusters))
        
        for k in range(self.n_clusters):
            for i in range(n_windows):
                x = stacked_data[i, :]
                log_lik = self._compute_log_likelihood(x, inv_covs[k], covs[k])
                log_likelihoods[i, k] = log_lik
        
        # Normalize log-likelihoods to avoid numerical issues
        log_likelihoods = log_likelihoods - np.max(log_likelihoods, axis=1, keepdims=True)
        
        # Dynamic programming with temporal smoothness
        # Forward pass
        forward_scores = np.zeros((n_windows, self.n_clusters))
        forward_path = np.zeros((n_windows, self.n_clusters), dtype=int)
        
        # Initialize
        forward_scores[0, :] = log_likelihoods[0, :]
        
        # Adaptive beta based on data characteristics
        adaptive_beta = self.beta / n_windows
        
        for t in range(1, n_windows):
            for k in range(self.n_clusters):
                # Score for staying in same regime
                stay_score = forward_scores[t-1, k] + log_likelihoods[t, k]
                
                # Score for switching regimes (penalized)
                switch_scores = forward_scores[t-1, :] + log_likelihoods[t, k] - adaptive_beta
                switch_scores[k] += adaptive_beta  # No penalty for staying
                
                # Take best previous state
                best_prev = np.argmax(switch_scores)
                forward_scores[t, k] = switch_scores[best_prev]
                forward_path[t, k] = best_prev
        
        # Backward pass to get optimal path
        labels = np.zeros(n_windows, dtype=int)
        labels[-1] = np.argmax(forward_scores[-1, :])
        
        for t in range(n_windows - 2, -1, -1):
            labels[t] = forward_path[t + 1, labels[t + 1]]
        
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
        
        # Handle edge cases
        if len(data) < self.window_size + self.min_cluster_size:
            raise ValueError(
                f"Data length ({len(data)}) must be at least "
                f"window_size + min_cluster_size ({self.window_size + self.min_cluster_size})"
            )
        
        self.n_assets_ = data.shape[1]
        
        # Create block-Toeplitz representation
        stacked_data, data_scaled = self._create_block_toeplitz(data)
        
        # Initialize clusters
        labels = self._initialize_clusters(stacked_data)
        
        # EM-like iterations
        prev_labels = None
        for iteration in range(self.max_iter):
            # M-step: Update cluster parameters
            cov_matrices, inv_cov_matrices = self._compute_cluster_parameters(
                stacked_data, labels
            )
            
            # E-step: Reassign labels
            new_labels = self._assign_labels(stacked_data, inv_cov_matrices, cov_matrices)
            
            # Check convergence using label stability
            if prev_labels is not None:
                stability = np.mean(labels == prev_labels)
                if stability >= self.convergence_threshold:
                    print(f"TICC converged after {iteration + 1} iterations (stability: {stability:.3f})")
                    break
            
            prev_labels = labels.copy()
            labels = new_labels
        
        # Store final parameters
        self.cov_matrices_, self.inv_cov_matrices_ = self._compute_cluster_parameters(
            stacked_data, labels
        )
        
        # Extend labels to match original data length
        # Use backward fill for initial window
        extended_labels = np.zeros(len(data), dtype=int)
        extended_labels[:self.window_size-1] = labels[0]
        extended_labels[self.window_size-1:] = labels
        
        # Post-process: Remove spurious transitions (noise reduction)
        extended_labels = self._smooth_labels(extended_labels)
        
        self.labels_ = extended_labels
        self.fitted_ = True
        
        # Update effective number of clusters
        self.n_clusters = len(np.unique(self.labels_))
        
        return self
    
    def _smooth_labels(self, labels, min_regime_length=5):
        """
        Smooth labels to remove spurious short regimes
        
        Parameters
        ----------
        labels : np.ndarray
            Raw labels
        min_regime_length : int
            Minimum length for a regime to be valid
        
        Returns
        -------
        smoothed_labels : np.ndarray
            Smoothed labels with spurious transitions removed
        """
        smoothed = labels.copy()
        n = len(labels)
        
        i = 0
        while i < n:
            current_label = smoothed[i]
            
            # Find end of current regime
            j = i
            while j < n and smoothed[j] == current_label:
                j += 1
            
            regime_length = j - i
            
            # If regime is too short, merge with neighbors
            if regime_length < min_regime_length and i > 0 and j < n:
                # Determine which neighbor to merge with
                prev_label = smoothed[i-1] if i > 0 else current_label
                next_label = smoothed[j] if j < n else current_label
                
                # Merge with the label that appears more frequently around this regime
                if prev_label == next_label:
                    merge_label = prev_label
                else:
                    # Count occurrences in a local window
                    window_start = max(0, i - 10)
                    window_end = min(n, j + 10)
                    window = smoothed[window_start:window_end]
                    
                    prev_count = np.sum(window == prev_label)
                    next_count = np.sum(window == next_label)
                    
                    merge_label = prev_label if prev_count >= next_count else next_label
                
                smoothed[i:j] = merge_label
            
            i = j
        
        return smoothed
    
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
            Correlation matrix for the contemporaneous period
        """
        if cluster_id not in self.cov_matrices_:
            raise ValueError(f"Cluster {cluster_id} not found")
        
        cov = self.cov_matrices_[cluster_id]
        
        # Extract first block (contemporaneous correlations)
        block_size = self.n_assets_
        cov_0 = cov[:block_size, :block_size]
        
        # Convert covariance to correlation
        diag_sqrt = np.sqrt(np.diag(cov_0))
        # Avoid division by zero
        diag_sqrt = np.where(diag_sqrt > 1e-10, diag_sqrt, 1.0)
        corr = cov_0 / np.outer(diag_sqrt, diag_sqrt)
        np.fill_diagonal(corr, 1.0)
        
        # Clip to valid correlation range
        corr = np.clip(corr, -1.0, 1.0)
        
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
        
        transitions = self.get_regime_transitions()
        
        return {
            'n_clusters_requested': self.n_clusters,
            'n_clusters_found': len(np.unique(self.labels_)),
            'window_size': self.window_size,
            'n_assets': self.n_assets_,
            'n_transitions': len(transitions),
            'beta': self.beta,
            'convergence_threshold': self.convergence_threshold,
            'fitted': self.fitted_,
            'avg_transition_spacing': len(self.labels_) / max(1, len(transitions))
        }

