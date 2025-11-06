"""
State Aggregation & Output Engine
Combines outputs from Stream 1 (Trend Regime) and Stream 2 (Volatility Regime)
to generate final regime classification
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class StateVector:
    """Combined state vector from all sensors"""
    trend_regime: str  # "Bull" or "Bear"
    volatility_regime: str  # "Low-Vol" or "High-Vol"
    trend_probability: float
    volatility_value: float
    timestamp: pd.Timestamp


class RegimeDefinitionEngine:
    """
    Defines final market regimes based on combined state vector
    
    Final Regimes:
    1. "Quiet Bull" - Bull trend + Low volatility
    2. "Volatile Bull" - Bull trend + High volatility  
    3. "Quiet Bear" - Bear trend + Low volatility
    4. "Panic Selloff" - Bear trend + High volatility
    """
    
    def __init__(self):
        self.regime_definitions = {
            ('Bull', 'Low-Vol'): 'Quiet Bull',
            ('Bull', 'High-Vol'): 'Volatile Bull',
            ('Bear', 'Low-Vol'): 'Quiet Bear',
            ('Bear', 'High-Vol'): 'Panic Selloff'
        }
        
        self.regime_descriptions = {
            'Quiet Bull': 'Upward trend with low volatility - Ideal for trend following',
            'Volatile Bull': 'Upward trend with high volatility - Choppy upward movement',
            'Quiet Bear': 'Downward trend with low volatility - Steady decline',
            'Panic Selloff': 'Downward trend with high volatility - Market stress/crisis'
        }
        
    def define_regime(self, 
                     trend_regime: str, 
                     volatility_regime: str) -> str:
        """
        Define final regime based on component regimes
        
        Args:
            trend_regime: Output from Markov Switching model
            volatility_regime: Output from GARCH model
            
        Returns:
            Final regime classification
        """
        regime_key = (trend_regime, volatility_regime)
        return self.regime_definitions.get(regime_key, 'Unknown')
    
    def get_regime_description(self, regime: str) -> str:
        """Get description of a regime"""
        return self.regime_descriptions.get(regime, 'No description available')


class StateAggregator:
    """
    Aggregates outputs from multiple models into combined state vector
    """
    
    def __init__(self):
        self.regime_engine = RegimeDefinitionEngine()
        self.state_history = []
        
    def create_state_vector(self,
                           trend_regime: str,
                           volatility_regime: str,
                           trend_probability: float,
                           volatility_value: float,
                           timestamp: pd.Timestamp) -> StateVector:
        """
        Create a state vector from model outputs
        
        Args:
            trend_regime: Trend regime from Markov model
            volatility_regime: Volatility regime from GARCH model
            trend_probability: Probability of current trend regime
            volatility_value: Current volatility value
            timestamp: Timestamp of observation
            
        Returns:
            StateVector object
        """
        return StateVector(
            trend_regime=trend_regime,
            volatility_regime=volatility_regime,
            trend_probability=trend_probability,
            volatility_value=volatility_value,
            timestamp=timestamp
        )
    
    def aggregate_states(self,
                        trend_regimes: pd.Series,
                        volatility_regimes: pd.Series,
                        trend_probabilities: Optional[pd.DataFrame] = None,
                        volatility_values: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Aggregate model outputs into combined state vectors
        
        Args:
            trend_regimes: Series of trend regime predictions
            volatility_regimes: Series of volatility regime predictions
            trend_probabilities: DataFrame of trend regime probabilities
            volatility_values: Series of volatility values
            
        Returns:
            DataFrame with combined state vectors and final regimes
        """
        # Align all series
        common_index = trend_regimes.index.intersection(volatility_regimes.index)
        
        trend_regimes = trend_regimes.loc[common_index]
        volatility_regimes = volatility_regimes.loc[common_index]
        
        if trend_probabilities is not None:
            trend_probabilities = trend_probabilities.loc[common_index]
            # Get max probability for current regime
            trend_probs = trend_probabilities.max(axis=1)
        else:
            trend_probs = pd.Series(1.0, index=common_index)
        
        if volatility_values is not None:
            volatility_values = volatility_values.loc[common_index]
        else:
            volatility_values = pd.Series(0.0, index=common_index)
        
        # Create combined DataFrame
        combined = pd.DataFrame({
            'trend_regime': trend_regimes,
            'volatility_regime': volatility_regimes,
            'trend_probability': trend_probs,
            'volatility_value': volatility_values
        }, index=common_index)
        
        # Define final regimes
        combined['final_regime'] = combined.apply(
            lambda row: self.regime_engine.define_regime(
                row['trend_regime'], 
                row['volatility_regime']
            ),
            axis=1
        )
        
        # Add regime descriptions
        combined['regime_description'] = combined['final_regime'].apply(
            self.regime_engine.get_regime_description
        )
        
        # Store state history
        for idx, row in combined.iterrows():
            state = self.create_state_vector(
                trend_regime=row['trend_regime'],
                volatility_regime=row['volatility_regime'],
                trend_probability=row['trend_probability'],
                volatility_value=row['volatility_value'],
                timestamp=idx
            )
            self.state_history.append(state)
        
        return combined
    
    def get_regime_statistics(self, combined_states: pd.DataFrame) -> Dict:
        """
        Calculate statistics for final regimes
        
        Args:
            combined_states: DataFrame from aggregate_states
            
        Returns:
            Dictionary with regime statistics
        """
        stats = {}
        
        for regime in combined_states['final_regime'].unique():
            regime_data = combined_states[combined_states['final_regime'] == regime]
            
            stats[regime] = {
                'count': len(regime_data),
                'percentage': (len(regime_data) / len(combined_states)) * 100,
                'avg_trend_probability': regime_data['trend_probability'].mean(),
                'avg_volatility': regime_data['volatility_value'].mean(),
                'description': self.regime_engine.get_regime_description(regime)
            }
        
        return stats
    
    def get_regime_transitions(self, combined_states: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate regime transition matrix
        
        Args:
            combined_states: DataFrame from aggregate_states
            
        Returns:
            DataFrame with transition probabilities
        """
        regimes = combined_states['final_regime']
        unique_regimes = regimes.unique()
        
        # Create transition matrix
        transition_matrix = pd.DataFrame(
            0.0, 
            index=unique_regimes, 
            columns=unique_regimes
        )
        
        for i in range(len(regimes) - 1):
            from_regime = regimes.iloc[i]
            to_regime = regimes.iloc[i + 1]
            transition_matrix.loc[from_regime, to_regime] += 1
        
        # Normalize to get probabilities
        transition_matrix = transition_matrix.div(transition_matrix.sum(axis=1), axis=0)
        
        return transition_matrix
    
    def get_current_state(self, combined_states: pd.DataFrame) -> Dict:
        """
        Get current (most recent) market state
        
        Args:
            combined_states: DataFrame from aggregate_states
            
        Returns:
            Dictionary with current state information
        """
        if len(combined_states) == 0:
            return {}
        
        current = combined_states.iloc[-1]
        
        return {
            'timestamp': combined_states.index[-1],
            'trend_regime': current['trend_regime'],
            'volatility_regime': current['volatility_regime'],
            'final_regime': current['final_regime'],
            'description': current['regime_description'],
            'trend_probability': current['trend_probability'],
            'volatility_value': current['volatility_value']
        }
    
    def export_regime_output(self, 
                            combined_states: pd.DataFrame, 
                            output_file: str) -> None:
        """
        Export final regime output to CSV
        
        Args:
            combined_states: DataFrame from aggregate_states
            output_file: Path to output file
        """
        combined_states.to_csv(output_file)
        print(f"Regime output exported to {output_file}")
