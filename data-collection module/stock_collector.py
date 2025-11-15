"""
Stock Data Collector
Multi-source data collection for Indian stocks with intelligent fallback
Priority: Yahoo Finance -> Twelve Data -> NSEpy -> FMP (deprecated)
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional
import os
from dotenv import load_dotenv
import time

from indian_stocks import INDIAN_STOCKS
from config import CONFIG
from storage import save_parquet, load_parquet

load_dotenv()

class StockCollector:
    """
    Stock data collector with multi-source fallback
    Handles individual stock data collection similar to index collector
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or CONFIG
        self.config['twelvedata_api_key'] = os.getenv('TWELVEDATA_API_KEY')
        self.twelve_data_client = None
        self.stocks = INDIAN_STOCKS
        
    def fetch_yfinance(self, symbol: str, start: str, end: str, interval: str = '1m') -> Optional[pd.DataFrame]:
        """
        Fetch stock data from Yahoo Finance
        
        Args:
            symbol: Yahoo Finance symbol (e.g., 'RELIANCE.NS')
            start: Start date
            end: End date  
            interval: Data interval (1m, 5m, 15m, 30m, 1h, 1d)
            
        Returns:
            DataFrame with OHLCV data or None
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Yahoo Finance requires datetime objects for intraday
            if interval in ['1m', '5m', '15m', '30m', '1h']:
                # For intraday, use period instead of dates
                df = ticker.history(period='1d', interval=interval)
            else:
                df = ticker.history(start=start, end=end, interval=interval)
            
            if df is not None and not df.empty:
                # Remove timezone info for consistency
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                return df
            
            return None
            
        except Exception as e:
            if 'No data found' not in str(e):
                print(f"⚠️ Yahoo Finance error for {symbol}: {str(e)}")
            return None
    
    def fetch_twelvedata(self, symbol: str, start: str, end: str, interval: str = '1min') -> Optional[pd.DataFrame]:
        """
        Fetch data from Twelve Data API
        Note: Free tier limited to 8 calls/minute
        """
        if not self.config.get('twelvedata_api_key'):
            return None
            
        try:
            from twelvedata import TDClient
            
            if self.twelve_data_client is None:
                self.twelve_data_client = TDClient(apikey=self.config['twelvedata_api_key'])
            
            # Remove .NS/.BO suffix for TwelveData
            td_symbol = symbol.replace('.NS', '').replace('.BO', '')
            
            ts = self.twelve_data_client.time_series(
                symbol=td_symbol,
                interval=interval,
                start_date=start,
                end_date=end,
                timezone='Asia/Kolkata'
            )
            
            df = ts.as_pandas()
            
            if df is not None and not df.empty:
                # Rename columns to match standard format
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                return df
            
            return None
            
        except ImportError:
            return None
        except Exception as e:
            if 'limit' in str(e).lower() or 'rate' in str(e).lower():
                print(f"⚠️ Twelve Data rate limit reached")
            return None
    
    def fetch_nselib(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        """
        Fetch data from NSEpy (daily data only)
        """
        try:
            from nsepy import get_history
            from datetime import datetime
            
            start_date = datetime.strptime(start, '%Y-%m-%d')
            end_date = datetime.strptime(end, '%Y-%m-%d')
            
            df = get_history(
                symbol=symbol,
                start=start_date,
                end=end_date
            )
            
            if df is not None and not df.empty:
                # Rename columns to match standard format
                column_map = {
                    'Open': 'Open',
                    'High': 'High',
                    'Low': 'Low',
                    'Close': 'Close',
                    'Volume': 'Volume'
                }
                df = df.rename(columns=column_map)
                return df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            return None
            
        except ImportError:
            return None
        except Exception as e:
            return None
    
    def fetch_with_fallback(self, stock_config: Dict, start: str, end: str, interval: str = '1m') -> Optional[pd.DataFrame]:
        """
        Fetch stock data with intelligent fallback
        Priority: Yahoo Finance -> Twelve Data (rate limited)
        
        Args:
            stock_config: Stock configuration from INDIAN_STOCKS
            start: Start date
            end: End date
            interval: Data interval
            
        Returns:
            DataFrame with OHLCV data or None
        """
        # Try Yahoo Finance first (best for Indian stocks)
        if stock_config.get('yahoo'):
            df = self.fetch_yfinance(stock_config['yahoo'], start, end, interval)
            if df is not None and not df.empty:
                return df
        
        # Fallback to Twelve Data (8 calls/min limit)
        if interval in ['1m', '1min'] and self.config.get('twelvedata_api_key'):
            symbol = stock_config.get('nse') or stock_config.get('symbol')
            if symbol:
                df = self.fetch_twelvedata(symbol, start, end, '1min')
                if df is not None and not df.empty:
                    return df
        
        return None
    
    def fetch_stock_minute_data(self, symbol: str, save_to_file: bool = True) -> Optional[pd.DataFrame]:
        """
        Fetch 1-minute data for a single stock
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE')
            save_to_file: Whether to save to parquet file
            
        Returns:
            DataFrame with minute data or None
        """
        if symbol not in self.stocks:
            print(f"❌ Unknown stock: {symbol}")
            return None
        
        stock_config = self.stocks[symbol]
        
        # Get current time in IST
        now = datetime.now()
        start = now.strftime('%Y-%m-%d')
        end = (now + timedelta(days=1)).strftime('%Y-%m-%d')
        
        print(f"Fetching {symbol} ({stock_config['yahoo']})...")
        
        # Fetch data with fallback
        df = self.fetch_with_fallback(stock_config, start, end, interval='1m')
        
        if df is None or df.empty:
            print(f"No data for {symbol}")
            return None
        
        print(f"Got {len(df)} minute bars for {symbol}")
        
        # Save to parquet if requested
        if save_to_file:
            filename = f"{stock_config['yahoo'].replace('.NS', '').replace('.BO', '')}_1m_streaming.parquet"
            # Use absolute path
            base_dir = os.path.dirname(os.path.abspath(__file__))
            filepath = os.path.join(base_dir, 'stocks', 'streaming', filename)
            
            # Load existing data and append if file exists
            try:
                if os.path.exists(filepath):
                    existing_df = load_parquet(filepath)
                    if existing_df is not None:
                        df = pd.concat([existing_df, df])
                        df = df[~df.index.duplicated(keep='last')]
                        df = df.sort_index()
                
                save_parquet(df, filepath)
                print(f"Saved to {filepath}")
            except Exception as e:
                print(f"Save error: {e}")
        
        return df
    
    def fetch_multiple_stocks(self, symbols: list, delay: float = 0.5) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks with delay to avoid rate limits
        
        Args:
            symbols: List of stock symbols
            delay: Delay between requests in seconds
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}
        
        for i, symbol in enumerate(symbols, 1):
            print(f"\n[{i}/{len(symbols)}] Processing {symbol}...")
            
            try:
                df = self.fetch_stock_minute_data(symbol, save_to_file=True)
                if df is not None:
                    results[symbol] = df
                
                # Delay to avoid rate limits
                if i < len(symbols):
                    time.sleep(delay)
                    
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
                continue
        
        print(f"\n{'='*60}")
        print(f"Successfully fetched {len(results)}/{len(symbols)} stocks")
        print(f"{'='*60}")
        
        return results

if __name__ == '__main__':
    # Test with a few stocks
    collector = StockCollector()
    
    test_stocks = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK']
    print(f"Testing with {len(test_stocks)} stocks...\n")
    
    results = collector.fetch_multiple_stocks(test_stocks, delay=1.0)
    
    print(f"\nSummary:")
    for symbol, df in results.items():
        print(f"   {symbol:15} : {len(df)} bars")
