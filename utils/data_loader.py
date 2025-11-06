"""
Data loading and preprocessing utilities for finance models
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import os


class DataLoader:
    """Load and preprocess stock and index data"""
    
    def __init__(self, data_dir: str = None):
        """
        Initialize DataLoader
        
        Args:
            data_dir: Base directory containing data folders
        """
        if data_dir is None:
            self.base_dir = Path(__file__).parent.parent
        else:
            self.base_dir = Path(data_dir)
            
        self.stocks_dir = self.base_dir / "stocks data"
        self.indexes_dir = self.base_dir / "indexes data"
        
    def load_stock(self, symbol: str) -> pd.DataFrame:
        """
        Load stock data from CSV
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            
        Returns:
            DataFrame with stock data
        """
        file_path = self.stocks_dir / f"{symbol}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Stock data not found: {file_path}")
            
        df = pd.read_csv(file_path)
        return self._preprocess_data(df)
    
    def load_index(self, index_name: str) -> pd.DataFrame:
        """
        Load index data from CSV
        
        Args:
            index_name: Index name (e.g., 'NIFTY 50')
            
        Returns:
            DataFrame with index data
        """
        file_path = self.indexes_dir / f"{index_name}_minute.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Index data not found: {file_path}")
            
        df = pd.read_csv(file_path)
        return self._preprocess_data(df)
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess loaded data
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        # Standardize column names (handle different formats)
        df.columns = df.columns.str.strip().str.lower()
        
        # Try to identify datetime column
        datetime_cols = ['date', 'datetime', 'timestamp', 'time']
        dt_col = None
        for col in datetime_cols:
            if col in df.columns:
                dt_col = col
                break
        
        if dt_col:
            df[dt_col] = pd.to_datetime(df[dt_col])
            df.set_index(dt_col, inplace=True)
            df.sort_index(inplace=True)
        
        # Remove any duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        return df
    
    def calculate_returns(self, df: pd.DataFrame, 
                         price_col: str = 'close',
                         log_returns: bool = True) -> pd.Series:
        """
        Calculate returns from price data
        
        Args:
            df: DataFrame with price data
            price_col: Name of price column
            log_returns: If True, calculate log returns; else simple returns
            
        Returns:
            Series of returns
        """
        if price_col not in df.columns:
            raise ValueError(f"Column '{price_col}' not found in DataFrame")
        
        prices = df[price_col]
        
        if log_returns:
            returns = np.log(prices / prices.shift(1))
        else:
            returns = prices.pct_change()
        
        return returns.dropna()
    
    def resample_to_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample intraday data to daily OHLC
        
        Args:
            df: DataFrame with intraday data
            
        Returns:
            Daily OHLC DataFrame
        """
        # Create OHLC resampling dictionary
        ohlc_dict = {}
        
        if 'open' in df.columns:
            ohlc_dict['open'] = 'first'
        if 'high' in df.columns:
            ohlc_dict['high'] = 'max'
        if 'low' in df.columns:
            ohlc_dict['low'] = 'min'
        if 'close' in df.columns:
            ohlc_dict['close'] = 'last'
        if 'volume' in df.columns:
            ohlc_dict['volume'] = 'sum'
        
        daily = df.resample('D').agg(ohlc_dict)
        return daily.dropna()
    
    def get_available_stocks(self) -> List[str]:
        """Get list of available stock symbols"""
        if not self.stocks_dir.exists():
            return []
        
        stocks = []
        for file in self.stocks_dir.glob("*.csv"):
            stocks.append(file.stem)
        return sorted(stocks)
    
    def get_available_indexes(self) -> List[str]:
        """Get list of available index names"""
        if not self.indexes_dir.exists():
            return []
        
        indexes = []
        for file in self.indexes_dir.glob("*_minute.csv"):
            index_name = file.stem.replace('_minute', '')
            indexes.append(index_name)
        return sorted(indexes)
    
    def create_cleaned_dataset(self, 
                              symbols: List[str],
                              output_dir: str,
                              data_type: str = 'stock',
                              resample_freq: str = 'D') -> None:
        """
        Create cleaned dataset for testing
        
        Args:
            symbols: List of symbols to process
            output_dir: Directory to save cleaned data
            data_type: 'stock' or 'index'
            resample_freq: Resampling frequency ('D' for daily, 'H' for hourly)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for symbol in symbols:
            try:
                print(f"Processing {symbol}...")
                
                # Load data
                if data_type == 'stock':
                    df = self.load_stock(symbol)
                else:
                    df = self.load_index(symbol)
                
                # Resample if needed
                if resample_freq == 'D' and len(df) > 1000:
                    df = self.resample_to_daily(df)
                
                # Calculate returns
                if 'close' in df.columns:
                    df['returns'] = self.calculate_returns(df, 'close', log_returns=True)
                    df['simple_returns'] = self.calculate_returns(df, 'close', log_returns=False)
                
                # Save cleaned data
                output_file = output_path / f"{symbol}_cleaned.csv"
                df.to_csv(output_file)
                print(f"  Saved to {output_file}")
                
            except Exception as e:
                print(f"  Error processing {symbol}: {e}")


def load_sample_data(n_samples: int = 1000) -> Tuple[pd.Series, pd.Series]:
    """
    Generate synthetic sample data for testing
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        Tuple of (prices, returns)
    """
    np.random.seed(42)
    
    # Generate random walk prices with regime changes
    returns = []
    current_regime = 0
    
    for i in range(n_samples):
        # Change regime every ~200 samples
        if i % 200 == 0:
            current_regime = 1 - current_regime
        
        # Different return distributions for different regimes
        if current_regime == 0:
            # Bull regime: positive drift, low volatility
            ret = np.random.normal(0.0005, 0.01)
        else:
            # Bear regime: negative drift, high volatility
            ret = np.random.normal(-0.0003, 0.02)
        
        returns.append(ret)
    
    returns = pd.Series(returns)
    prices = pd.Series(100 * np.exp(returns.cumsum()))
    
    return prices, returns
