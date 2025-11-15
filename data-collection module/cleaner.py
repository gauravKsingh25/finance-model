# cleaner.py
"""
Data cleaning and validation utilities
"""
import pandas as pd
import numpy as np
from typing import Optional

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean financial data by removing invalid values and duplicates
    
    Args:
        df: Raw DataFrame with OHLCV data
        
    Returns:
        Cleaned DataFrame
    """
    # Flatten multi-level columns if present (from yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        # Get the first level of column names (the actual OHLCV names)
        df.columns = df.columns.get_level_values(0)
    
    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Drop rows with NaN values
    df = df.dropna()
    
    # Remove duplicate timestamps, keep the last occurrence
    df = df[~df.index.duplicated(keep='last')]
    
    # Ensure data is sorted by index (timestamp)
    df = df.sort_index()
    
    # Validate OHLC logic (High >= Low, etc.)
    df = validate_ohlc(df)
    
    return df

def validate_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate OHLC data integrity
    
    Args:
        df: DataFrame with OHLC columns
        
    Returns:
        Validated DataFrame
    """
    if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
        # Remove rows where High < Low (invalid)
        invalid_mask = df['High'] < df['Low']
        if invalid_mask.any():
            print(f"⚠️  Removed {invalid_mask.sum()} rows with invalid OHLC data")
            df = df[~invalid_mask]
        
        # Remove rows where Close/Open outside High/Low range
        invalid_mask = (
            (df['Close'] > df['High']) | 
            (df['Close'] < df['Low']) |
            (df['Open'] > df['High']) | 
            (df['Open'] < df['Low'])
        )
        if invalid_mask.any():
            print(f"⚠️  Removed {invalid_mask.sum()} rows with Close/Open outside High/Low range")
            df = df[~invalid_mask]
    
    return df

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic derived features
    
    Args:
        df: DataFrame with OHLC data
        
    Returns:
        DataFrame with additional features
    """
    if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
        df['Returns'] = df['Close'].pct_change()
        df['Range'] = df['High'] - df['Low']
        df['TypicalPrice'] = (df['High'] + df['Low'] + df['Close']) / 3
    
    return df
