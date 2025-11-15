# collector.py
"""
Multi-source data collection with intelligent fallback
Supports: Yahoo Finance, Twelve Data, NSEpy
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import time
import os
from dotenv import load_dotenv

from config import CONFIG
from storage import save_parquet, load_parquet

# Load environment variables
load_dotenv()

class DataCollector:
    """
    Intelligent data collector with multi-source support
    Priority: Yahoo Finance -> Twelve Data (FMP deprecated Aug 2025)
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or CONFIG
        self.config['twelvedata_api_key'] = os.getenv('TWELVEDATA_API_KEY')
        self.twelve_data_client = None
        
    def fetch_yfinance(self, symbol: str, start: str, end: str, interval: str = '1m') -> Optional[pd.DataFrame]:
        """
        Fetch data from Yahoo Finance
        
        Args:
            symbol: Yahoo Finance symbol (e.g., '^NSEI')
            start: Start date
            end: End date
            interval: Data interval
            
        Returns:
            DataFrame with OHLCV data or None
        """
        try:
            print(f"📊 Fetching from Yahoo Finance: {symbol}")
            df = yf.download(symbol, interval=interval, start=start, end=end, progress=False)
            
            if df.empty:
                print(f"⚠️  No data returned from Yahoo Finance")
                return None
                
            df.dropna(inplace=True)
            return df
        except Exception as e:
            print(f"❌ Yahoo Finance error: {str(e)}")
            return None
    
    def fetch_fmp(self, symbol: str, start: str, end: str, interval: str = '1min') -> Optional[pd.DataFrame]:
        """
        Fetch data from Financial Modeling Prep (FMP)
        
        Args:
            symbol: FMP symbol (e.g., '^NSEI')
            start: Start date  
            end: End date
            interval: Data interval (1min, 5min, 15min, 30min, 1hour, 4hour)
            
        Returns:
            DataFrame with OHLCV data or None
        """
        if not self.config.get('fmp_api_key'):
            print(f"⚠️  FMP API key not configured")
            return None
            
        try:
            import requests
            
            print(f"📊 Fetching from FMP: {symbol}")
            
            # FMP uses different interval format
            interval_map = {
                '1m': '1min',
                '5m': '5min', 
                '15m': '15min',
                '30m': '30min',
                '1h': '1hour',
                '4h': '4hour',
                '1d': '1day'
            }
            
            fmp_interval = interval_map.get(interval, interval)
            
            # FMP historical intraday endpoint
            url = f"https://financialmodelingprep.com/api/v3/historical-chart/{fmp_interval}/{symbol}"
            params = {
                'apikey': self.config['fmp_api_key'],
                'from': start,
                'to': end
            }
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code != 200:
                print(f"❌ FMP API error: Status {response.status_code}")
                return None
            
            data = response.json()
            
            if not data or len(data) == 0:
                print(f"⚠️  No data returned from FMP")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Rename columns to match standard format
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            
            # Standardize column names
            column_map = {
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }
            df.rename(columns=column_map, inplace=True)
            
            # Ensure we have required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if all(col in df.columns for col in required_cols):
                return df[required_cols].sort_index()
            else:
                print(f"⚠️  FMP data missing required columns")
                return None
                
        except ImportError:
            print(f"⚠️  requests package not installed")
            return None
        except Exception as e:
            print(f"❌ FMP error: {str(e)}")
            return None
    
    def fetch_twelvedata(self, symbol: str, start: str, end: str, interval: str = '1min') -> Optional[pd.DataFrame]:
        """
        Fetch data from Twelve Data (backup source)
        
        Args:
            symbol: Twelve Data symbol
            start: Start date
            end: End date
            interval: Data interval
            
        Returns:
            DataFrame with OHLCV data or None
        """
        if not self.config['twelvedata_api_key']:
            print("⚠️  Twelve Data API key not configured")
            return None
            
        try:
            # Lazy import
            from twelvedata import TDClient
            
            if not self.twelve_data_client:
                self.twelve_data_client = TDClient(apikey=self.config['twelvedata_api_key'])
            
            print(f"📊 Fetching from Twelve Data: {symbol}")
            
            ts = self.twelve_data_client.time_series(
                symbol=symbol,
                interval=interval,
                start_date=start,
                end_date=end,
                outputsize=5000
            )
            
            df = ts.as_pandas()
            
            if df.empty:
                print(f"⚠️  No data returned from Twelve Data")
                return None
            
            # Rename columns to match standard format
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            return df
        except ImportError:
            print("⚠️  twelvedata package not installed. Install with: pip install twelvedata")
            return None
        except Exception as e:
            print(f"❌ Twelve Data error: {str(e)}")
            return None
    
    def fetch_nselib(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        """
        Fetch daily data from NSEpy (fallback for daily data only)
        
        Args:
            symbol: NSE symbol
            start: Start date
            end: End date
            
        Returns:
            DataFrame with OHLCV data or None
        """
        try:
            # Lazy import
            from nsepy import get_history
            
            print(f"📊 Fetching from NSEpy: {symbol}")
            
            start_dt = datetime.strptime(start, "%Y-%m-%d")
            end_dt = datetime.strptime(end, "%Y-%m-%d")
            
            # NSEpy is for daily data only
            df = get_history(
                symbol=symbol,
                start=start_dt,
                end=end_dt
            )
            
            if df.empty:
                print(f"⚠️  No data returned from NSEpy")
                return None
            
            # Standardize column names
            df = df.rename(columns={
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            })
            
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        except ImportError:
            print("⚠️  nsepy package not installed. Install with: pip install nsepy")
            return None
        except Exception as e:
            print(f"❌ NSEpy error: {str(e)}")
            return None
    
    def fetch_with_fallback(self, index_config: Dict, start: str, end: str, interval: str = '1m') -> Optional[pd.DataFrame]:
        """
        Fetch data with intelligent fallback across sources
        Priority: Yahoo Finance -> Twelve Data (rate limited)
        
        Args:
            index_config: Index configuration from INDIAN_INDICES
            start: Start date
            end: End date
            interval: Data interval
            
        Returns:
            DataFrame with OHLCV data or None
        """
        # Try Yahoo Finance first (free, unlimited, best for Indian indices)
        if index_config.get('yahoo'):
            print(f"📊 Trying Yahoo Finance: {index_config['yahoo']}")
            df = self.fetch_yfinance(index_config['yahoo'], start, end, interval)
            if df is not None and not df.empty:
                print(f"✅ Yahoo Finance successful: {len(df)} rows")
                return df
            print(f"⚠️  Yahoo Finance failed, trying fallback...")
        
        # Fallback to Twelve Data (8 calls/min limit on free tier)
        if interval in ['1m', '1min'] and self.config.get('twelvedata_api_key'):
            if index_config.get('twelvedata'):
                print(f"📊 Trying Twelve Data: {index_config['twelvedata']}")
                td_interval = '1min' if interval == '1m' else interval
                df = self.fetch_twelvedata(index_config['twelvedata'], start, end, td_interval)
                if df is not None and not df.empty:
                    print(f"✅ Twelve Data successful: {len(df)} rows")
                    return df
                print(f"⚠️  Twelve Data failed (check rate limit)")
        
        print(f"❌ All data sources failed for {index_config.get('nse', 'Unknown')}")
        return None
    
    def update_data(self, symbol: str = None, nse_symbol: str = None):
        """
        Update data for a given symbol with incremental fetching
        
        Args:
            symbol: Yahoo Finance symbol (default from config)
            nse_symbol: NSE symbol for fallback (default from config)
        """
        symbol = symbol or self.config["symbol"]
        nse_symbol = nse_symbol or self.config.get("nse_symbol", "NIFTY")
        base_dir = self.config["base_dir"]
        data_path = f"{base_dir}/raw/{symbol.replace('^', '')}_1m.parquet"
        
        try:
            existing = load_parquet(data_path)
            last_timestamp = existing.index[-1]
            start = (last_timestamp + timedelta(minutes=1)).strftime('%Y-%m-%d')
            print(f"📂 Existing data found. Last timestamp: {last_timestamp}")
        except FileNotFoundError:
            existing = pd.DataFrame()
            start = self.config["start_date"]
            print(f"🆕 No existing data. Starting from: {start}")
        
        end = datetime.today().strftime('%Y-%m-%d')
        print(f"🔄 Fetching data for {symbol} from {start} to {end}")
        
        new_data = pd.DataFrame()
        delta = timedelta(days=self.config["fetch_chunk_days"])
        current = datetime.strptime(start, "%Y-%m-%d")
        
        # Fetch in chunks (Yahoo Finance 1-min has 7-day limit)
        retry_count = 0
        while current < datetime.today():
            chunk_end = min(current + delta, datetime.today())
            
            df = self.fetch_1min_data(
                symbol, 
                start=current.strftime('%Y-%m-%d'), 
                end=chunk_end.strftime('%Y-%m-%d')
            )
            
            if df is not None and not df.empty:
                new_data = pd.concat([new_data, df])
                retry_count = 0
                print(f"✅ Fetched {len(df)} rows for {current.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}")
            else:
                retry_count += 1
                if retry_count >= self.config["max_retries"]:
                    print(f"⚠️  Max retries reached. Skipping chunk.")
                    retry_count = 0
                else:
                    print(f"⏳ Retrying in {self.config['retry_delay_seconds']} seconds...")
                    time.sleep(self.config['retry_delay_seconds'])
                    continue
            
            current += delta
            time.sleep(1)  # Rate limiting
        
        if not new_data.empty:
            combined = pd.concat([existing, new_data]).sort_index().drop_duplicates()
            save_parquet(combined, data_path)
            print(f"✅ Data updated and saved to {data_path}")
            return combined
        else:
            print("ℹ️  No new data fetched.")
            return existing if not existing.empty else None
    
    def update_multiple_symbols(self, symbols: Dict[str, Dict[str, str]] = None):
        """
        Update data for multiple symbols
        
        Args:
            symbols: Dictionary of symbols with yahoo and nse keys
        """
        symbols = symbols or self.config["symbols"]
        
        for name, sym_config in symbols.items():
            print(f"\n{'='*60}")
            print(f"Processing {name}")
            print(f"{'='*60}")
            
            try:
                self.update_data(
                    symbol=sym_config["yahoo"],
                    nse_symbol=sym_config.get("nse", "")
                )
            except Exception as e:
                print(f"❌ Error processing {name}: {str(e)}")
                continue
