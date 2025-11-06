"""
Evaluation metrics for regime detection models
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, silhouette_score


class ModelEvaluator:
    """Evaluate performance of regime detection models"""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_classification(self, 
                               y_true: np.ndarray, 
                               y_pred: np.ndarray,
                               model_name: str = "Model") -> Dict:
        """
        Evaluate classification performance
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'model': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        self.results[model_name] = metrics
        return metrics
    
    def evaluate_regime_stability(self, 
                                 regimes: np.ndarray,
                                 model_name: str = "Model") -> Dict:
        """
        Evaluate stability of regime predictions
        
        Args:
            regimes: Array of regime predictions
            model_name: Name of the model
            
        Returns:
            Dictionary of stability metrics
        """
        # Count regime transitions
        transitions = np.sum(np.diff(regimes) != 0)
        transition_rate = transitions / len(regimes)
        
        # Average regime duration
        regime_lengths = []
        current_regime = regimes[0]
        current_length = 1
        
        for regime in regimes[1:]:
            if regime == current_regime:
                current_length += 1
            else:
                regime_lengths.append(current_length)
                current_regime = regime
                current_length = 1
        regime_lengths.append(current_length)
        
        stability_metrics = {
            'model': model_name,
            'total_transitions': transitions,
            'transition_rate': transition_rate,
            'avg_regime_duration': np.mean(regime_lengths),
            'std_regime_duration': np.std(regime_lengths),
            'min_regime_duration': np.min(regime_lengths),
            'max_regime_duration': np.max(regime_lengths)
        }
        
        return stability_metrics
    
    def evaluate_volatility_prediction(self, 
                                      realized_vol: np.ndarray,
                                      predicted_vol: np.ndarray,
                                      model_name: str = "Model") -> Dict:
        """
        Evaluate volatility prediction accuracy
        
        Args:
            realized_vol: Actual realized volatility
            predicted_vol: Predicted volatility
            model_name: Name of the model
            
        Returns:
            Dictionary of prediction metrics
        """
        # Remove NaN values
        mask = ~(np.isnan(realized_vol) | np.isnan(predicted_vol))
        realized_vol = realized_vol[mask]
        predicted_vol = predicted_vol[mask]
        
        errors = predicted_vol - realized_vol
        pct_errors = (errors / realized_vol) * 100
        
        metrics = {
            'model': model_name,
            'mae': np.mean(np.abs(errors)),
            'mse': np.mean(errors**2),
            'rmse': np.sqrt(np.mean(errors**2)),
            'mape': np.mean(np.abs(pct_errors)),
            'correlation': np.corrcoef(realized_vol, predicted_vol)[0, 1],
            'bias': np.mean(errors)
        }
        
        return metrics
    
    def evaluate_regime_separation(self, 
                                   features: np.ndarray,
                                   regimes: np.ndarray,
                                   model_name: str = "Model") -> Dict:
        """
        Evaluate how well regimes are separated in feature space
        
        Args:
            features: Feature array (n_samples, n_features)
            regimes: Regime labels
            model_name: Name of the model
            
        Returns:
            Dictionary of separation metrics
        """
        if len(np.unique(regimes)) < 2:
            return {'error': 'Need at least 2 regimes for separation metrics'}
        
        # Silhouette score
        try:
            silhouette = silhouette_score(features, regimes)
        except:
            silhouette = None
        
        # Within-regime variance vs between-regime variance
        unique_regimes = np.unique(regimes)
        within_variance = []
        regime_means = []
        
        for regime in unique_regimes:
            regime_data = features[regimes == regime]
            if len(regime_data) > 0:
                within_variance.append(np.var(regime_data))
                regime_means.append(np.mean(regime_data, axis=0))
        
        within_var = np.mean(within_variance) if within_variance else None
        between_var = np.var(regime_means) if len(regime_means) > 1 else None
        
        metrics = {
            'model': model_name,
            'silhouette_score': silhouette,
            'within_regime_variance': within_var,
            'between_regime_variance': between_var,
            'separation_ratio': between_var / within_var if (within_var and between_var and within_var > 0) else None
        }
        
        return metrics
    
    def calculate_sharpe_by_regime(self, 
                                   returns: pd.Series,
                                   regimes: np.ndarray,
                                   risk_free_rate: float = 0.0) -> Dict:
        """
        Calculate Sharpe ratio for each regime
        
        Args:
            returns: Return series
            regimes: Regime labels
            risk_free_rate: Risk-free rate (annualized)
            
        Returns:
            Dictionary with Sharpe ratios per regime
        """
        sharpe_ratios = {}
        
        for regime in np.unique(regimes):
            regime_returns = returns[regimes == regime]
            if len(regime_returns) > 0:
                mean_return = regime_returns.mean() * 252  # Annualized
                std_return = regime_returns.std() * np.sqrt(252)  # Annualized
                
                if std_return > 0:
                    sharpe = (mean_return - risk_free_rate) / std_return
                else:
                    sharpe = 0
                
                sharpe_ratios[f'regime_{regime}'] = {
                    'sharpe_ratio': sharpe,
                    'mean_return': mean_return,
                    'volatility': std_return,
                    'n_observations': len(regime_returns)
                }
        
        return sharpe_ratios
    
    def generate_summary_report(self) -> pd.DataFrame:
        """
        Generate summary report of all evaluated models
        
        Returns:
            DataFrame with summary metrics
        """
        if not self.results:
            return pd.DataFrame()
        
        summary = []
        for model_name, metrics in self.results.items():
            row = {'model': model_name}
            for key, value in metrics.items():
                if key != 'confusion_matrix' and not isinstance(value, (list, np.ndarray)):
                    row[key] = value
            summary.append(row)
        
        return pd.DataFrame(summary)


def calculate_realized_volatility(returns: pd.Series, 
                                  window: int = 20,
                                  annualize: bool = True) -> pd.Series:
    """
    Calculate realized volatility using rolling window
    
    Args:
        returns: Return series
        window: Rolling window size
        annualize: Whether to annualize volatility
        
    Returns:
        Series of realized volatility
    """
    vol = returns.rolling(window=window).std()
    
    if annualize:
        vol = vol * np.sqrt(252)
    
    return vol
