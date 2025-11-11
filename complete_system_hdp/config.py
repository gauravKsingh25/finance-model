"""
System Configuration for Complete Regime Detection Pipeline
===========================================================

Centralized configuration for all components of the multi-layered
regime detection system using Sticky HDP-HMM.
"""

# ============================================================================
# LAYER 1: FAST BREAK DETECTION (Bayesian Changepoint)
# ============================================================================

CHANGEPOINT_CONFIG = {
    'prior_scale': 0.1,           # Prior scale for changepoint detection
    'threshold': 0.5,              # Probability threshold for changepoint
    'min_distance': 10,            # Minimum distance between changepoints
    'online_mode': False           # Use online or batch detection
}


# ============================================================================
# LAYER 2: DYNAMIC STATE ESTIMATION (Switching Kalman Filter)
# ============================================================================

KALMAN_CONFIG = {
    'n_regimes': 3,                # Number of regimes for switching
    'transition_covariance': 0.1,  # Process noise
    'observation_covariance': 1.0, # Measurement noise
    'em_iterations': 50,           # EM algorithm iterations
    'smooth': True                 # Apply smoothing
}


# ============================================================================
# LAYER 3: REGIME CLASSIFICATION (STICKY HDP-HMM) ★
# ============================================================================

HDP_CONFIG = {
    'truncation': 8,               # Maximum number of regimes to consider
    'alpha': 1.0,                  # DP concentration (transition)
    'gamma': 1.0,                  # DP concentration (global)
    'kappa': 20.0,                 # STICKY parameter (regime persistence)
    'max_iter': 50,                # Gibbs sampling iterations
    'random_state': 42,            # Reproducibility
    'burn_in': 10                  # Burn-in iterations before sampling
}

# Chaos/Entropy Metrics Configuration (Layer 3 sensors)
CHAOS_CONFIG = {
    'hurst_window': 100,           # Window for Hurst exponent calculation
    'entropy_bins': 10,            # Bins for entropy calculation
    'lyapunov_delay': 1,           # Time delay for Lyapunov estimation
    'lyapunov_dimension': 3        # Embedding dimension
}


# ============================================================================
# LAYER 4: STRUCTURAL AWARENESS
# ============================================================================

# TICC (Toeplitz Inverse Covariance Clustering)
TICC_CONFIG = {
    'n_clusters': 3,               # Initial number of correlation regimes
    'window_size': 10,             # Temporal window for correlation
    'lambda_parameter': 11e-2,     # Sparsity parameter
    'beta': 1000,                  # Temporal smoothness parameter
    'max_iter': 50,                # Maximum iterations
    'convergence_threshold': 0.95, # Label stability threshold
    'min_cluster_size': 5          # Minimum cluster size
}

# Hawkes Process
HAWKES_CONFIG = {
    'kernel': 'exponential',       # Kernel type
    'max_iter': 100,               # Maximum iterations
    'tol': 1e-5,                   # Convergence tolerance
    'baseline_window': 50          # Window for baseline intensity
}

# GAS Models (Generalized Autoregressive Score)
GAS_CONFIG = {
    'volatility_model': 'GARCH',   # Underlying volatility model
    'p': 1,                        # GARCH order p
    'q': 1,                        # GARCH order q
    'distribution': 't',           # Error distribution
    'n_regimes': 2                 # Low-vol / High-vol
}


# ============================================================================
# LAYER C: STATE AGGREGATION
# ============================================================================

AGGREGATION_CONFIG = {
    'regime_definitions': {
        # Map (trend, volatility) -> Final Regime
        ('Bull', 'Low-Vol'): 'Quiet Bull',
        ('Bull', 'High-Vol'): 'Volatile Bull',
        ('Bear', 'Low-Vol'): 'Quiet Bear',
        ('Bear', 'High-Vol'): 'Panic Selloff',
        ('Neutral', 'Low-Vol'): 'Range-bound Calm',
        ('Neutral', 'High-Vol'): 'Range-bound Choppy'
    },
    'confidence_weights': {
        'hdp_regime': 0.35,         # Weight for HDP-HMM regime
        'ticc_correlation': 0.25,   # Weight for TICC correlation
        'kalman_state': 0.20,       # Weight for Kalman state
        'changepoint': 0.10,        # Weight for changepoint signal
        'chaos_metrics': 0.10       # Weight for chaos/entropy
    },
    'transition_penalty': 0.05      # Penalty for regime switching
}


# ============================================================================
# INPUT FEATURES (Layer A)
# ============================================================================

FEATURE_CONFIG = {
    # Intraday Features
    'intraday_windows': [5, 15, 30, 60],  # Minutes
    'intraday_features': [
        'realized_volatility',
        'price_range',
        'volume_surge',
        'momentum'
    ],
    
    # Multi-Asset Features
    'multi_asset_features': [
        'correlation_matrix',
        'pca_eigenvalues',
        'portfolio_variance',
        'diversification_ratio'
    ],
    
    # Daily Features
    'daily_features': [
        'returns',
        'log_returns',
        'volume',
        'high_low_range',
        'close_open_range'
    ],
    
    # Weekly Features
    'weekly_features': [
        'trend_strength',
        'volatility_regime',
        'momentum_indicator',
        'mean_reversion'
    ],
    
    # Lookback periods
    'short_window': 20,
    'medium_window': 50,
    'long_window': 200
}


# ============================================================================
# SYSTEM PARAMETERS
# ============================================================================

SYSTEM_CONFIG = {
    'min_data_points': 100,         # Minimum data for reliable detection
    'max_data_points': 10000,       # Maximum data to process at once
    'parallel_execution': True,     # Run sensors in parallel
    'n_jobs': -1,                   # CPU cores (-1 = all)
    'logging_level': 'INFO',        # DEBUG, INFO, WARNING, ERROR
    'save_intermediate': True,      # Save intermediate results
    'output_format': 'parquet',     # CSV or Parquet
    'cache_enabled': True,          # Cache computed features
    'real_time_mode': False         # Enable real-time processing
}


# ============================================================================
# PERFORMANCE TUNING
# ============================================================================

PERFORMANCE_CONFIG = {
    'use_gpu': False,               # GPU acceleration (if available)
    'batch_size': 1000,             # Batch size for processing
    'memory_limit_mb': 4096,        # Memory limit (MB)
    'checkpoint_interval': 1000,    # Save checkpoints every N samples
    'lazy_loading': True            # Load data on-demand
}


# ============================================================================
# VALIDATION & MONITORING
# ============================================================================

VALIDATION_CONFIG = {
    'enable_validation': True,      # Validate outputs
    'regime_stability_threshold': 0.8,  # Minimum stability
    'max_transitions_per_period': 15,   # Max regime changes
    'confidence_threshold': 0.5,    # Minimum confidence for regime
    'alert_on_transition': True,    # Alert on regime change
    'monitor_metrics': [
        'regime_duration',
        'transition_frequency',
        'confidence_level',
        'model_convergence'
    ]
}


# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

OUTPUT_CONFIG = {
    'save_regime_sequence': True,
    'save_probabilities': True,
    'save_state_vector': True,
    'save_intermediate_layers': True,
    'generate_plots': True,
    'plot_types': [
        'regime_timeline',
        'probability_heatmap',
        'transition_matrix',
        'feature_importance'
    ],
    'output_directory': 'results/',
    'timestamp_format': '%Y%m%d_%H%M%S'
}


# ============================================================================
# REGIME DEFINITIONS (Final Output)
# ============================================================================

REGIME_METADATA = {
    'Quiet Bull': {
        'description': 'Upward trend with low volatility',
        'strategy_suggestion': 'Trend following, long positions',
        'risk_level': 'Low',
        'typical_duration_days': 60
    },
    'Volatile Bull': {
        'description': 'Upward trend with high volatility',
        'strategy_suggestion': 'Careful long positions, tighter stops',
        'risk_level': 'Medium-High',
        'typical_duration_days': 30
    },
    'Quiet Bear': {
        'description': 'Downward trend with low volatility',
        'strategy_suggestion': 'Short positions, defensive',
        'risk_level': 'Medium',
        'typical_duration_days': 40
    },
    'Panic Selloff': {
        'description': 'Downward trend with high volatility',
        'strategy_suggestion': 'Risk-off, hedging, contrarian opportunities',
        'risk_level': 'Very High',
        'typical_duration_days': 15
    },
    'Range-bound Calm': {
        'description': 'Sideways market with low volatility',
        'strategy_suggestion': 'Mean reversion, range trading',
        'risk_level': 'Low',
        'typical_duration_days': 50
    },
    'Range-bound Choppy': {
        'description': 'Sideways market with high volatility',
        'strategy_suggestion': 'Reduce positions, wait for clarity',
        'risk_level': 'Medium',
        'typical_duration_days': 20
    }
}


# ============================================================================
# ENVIRONMENT-SPECIFIC OVERRIDES
# ============================================================================

import os

ENVIRONMENT = os.getenv('REGIME_DETECTION_ENV', 'development')

if ENVIRONMENT == 'production':
    # Production overrides
    HDP_CONFIG['max_iter'] = 100  # More iterations for accuracy
    SYSTEM_CONFIG['logging_level'] = 'WARNING'
    PERFORMANCE_CONFIG['checkpoint_interval'] = 500
    
elif ENVIRONMENT == 'testing':
    # Testing overrides
    HDP_CONFIG['max_iter'] = 20  # Faster for tests
    SYSTEM_CONFIG['logging_level'] = 'DEBUG'
    SYSTEM_CONFIG['save_intermediate'] = False


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_config(component: str) -> dict:
    """
    Get configuration for a specific component
    
    Args:
        component: One of 'hdp', 'ticc', 'kalman', 'changepoint', 
                   'hawkes', 'gas', 'system', 'output'
    
    Returns:
        Dictionary with component configuration
    """
    config_map = {
        'hdp': HDP_CONFIG,
        'ticc': TICC_CONFIG,
        'kalman': KALMAN_CONFIG,
        'changepoint': CHANGEPOINT_CONFIG,
        'hawkes': HAWKES_CONFIG,
        'gas': GAS_CONFIG,
        'chaos': CHAOS_CONFIG,
        'aggregation': AGGREGATION_CONFIG,
        'features': FEATURE_CONFIG,
        'system': SYSTEM_CONFIG,
        'performance': PERFORMANCE_CONFIG,
        'validation': VALIDATION_CONFIG,
        'output': OUTPUT_CONFIG
    }
    
    return config_map.get(component.lower(), {})


def validate_config():
    """Validate all configuration parameters"""
    errors = []
    
    # Validate HDP config
    if HDP_CONFIG['kappa'] < 0:
        errors.append("HDP_CONFIG: kappa must be >= 0")
    
    if HDP_CONFIG['truncation'] < 2:
        errors.append("HDP_CONFIG: truncation must be >= 2")
    
    # Validate TICC config
    if TICC_CONFIG['window_size'] < 2:
        errors.append("TICC_CONFIG: window_size must be >= 2")
    
    # Validate aggregation weights
    weight_sum = sum(AGGREGATION_CONFIG['confidence_weights'].values())
    if abs(weight_sum - 1.0) > 0.01:
        errors.append(f"AGGREGATION_CONFIG: confidence_weights sum to {weight_sum}, should be 1.0")
    
    if errors:
        raise ValueError(f"Configuration validation failed:\n" + "\n".join(errors))
    
    return True


# Validate on import
try:
    validate_config()
except ValueError as e:
    print(f"Warning: {e}")
