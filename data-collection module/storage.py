# storage.py
"""
Storage utilities for loading and saving data in Parquet format
"""
import pandas as pd
import os
from pathlib import Path

def save_parquet(df: pd.DataFrame, path: str):
    """
    Save DataFrame to Parquet format with automatic directory creation
    
    Args:
        df: DataFrame to save
        path: File path for saving
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path)
    print(f"💾 Saved data to {path}")

def load_parquet(path: str) -> pd.DataFrame:
    """
    Load DataFrame from Parquet file
    
    Args:
        path: File path to load from
        
    Returns:
        DataFrame with loaded data
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_parquet(path)

def ensure_directories(base_dir: str):
    """
    Create necessary directory structure
    
    Args:
        base_dir: Base directory for data storage
    """
    dirs = [
        f"{base_dir}/raw",
        f"{base_dir}/hourly",
        f"{base_dir}/4hour",
        f"{base_dir}/daily",
        f"{base_dir}/weekly",
        "./logs"
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
