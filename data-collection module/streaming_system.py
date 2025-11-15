# streaming_system.py
"""
Dual-File Streaming System for Indian Market Indices

System Architecture:
--------------------
For each index (e.g., Nifty 50):

File 1: NSEI_1m_streaming.parquet
  - Stores ONLY current week's 1-min data
  - Gets CLEARED every Sunday midnight
  - Acts as temporary streaming buffer
  - Max size: ~7 days × 375 rows/day = ~2,625 rows

File 2: NSEI_daily.parquet  
  - Stores historical daily bars
  - Appends 7 new rows every Sunday midnight
  - NEVER gets cleared
  - Grows forever: Week 1: 7 rows, Week 2: 14 rows, Week 52: 364 rows

Workflow:
---------
Monday-Sunday:
  - Fetch 1-min data every minute
  - Append to NSEI_1m_streaming.parquet
  - File grows: 0 → 375 → 750 → ... → 2,625 rows

Sunday 00:00 (Midnight):
  - Read NSEI_1m_streaming.parquet (full week data)
  - Aggregate to 7 daily bars
  - Append to NSEI_daily.parquet
  - CLEAR NSEI_1m_streaming.parquet (reset to empty)
  - Start fresh for next week
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import os
from typing import Dict, List, Optional
import logging

from storage import load_parquet, save_parquet, ensure_directories
from config import CONFIG
from indian_market_indices import INDIAN_INDICES

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StreamingDataCollector:
    """
    Manages dual-file streaming system for each index
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or CONFIG
        self.base_dir = self.config['base_dir']
        
    def get_streaming_file_path(self, symbol: str) -> str:
        """Get path for streaming 1-min file (cleared weekly)"""
        symbol_clean = symbol.replace('^', '')
        return f"{self.base_dir}/streaming/{symbol_clean}_1m_streaming.parquet"
    
    def get_daily_file_path(self, symbol: str) -> str:
        """Get path for permanent daily file (grows forever)"""
        symbol_clean = symbol.replace('^', '')
        return f"{self.base_dir}/daily/{symbol_clean}_daily.parquet"
    
    def append_1min_data(self, symbol: str, new_data: pd.DataFrame):
        """
        Append new 1-min data to streaming file
        
        Args:
            symbol: Yahoo Finance symbol (e.g., '^NSEI')
            new_data: DataFrame with new 1-min bars
        """
        if new_data is None or new_data.empty:
            logger.warning(f"{symbol}: No data to append")
            return
        
        streaming_file = self.get_streaming_file_path(symbol)
        
        # Flatten multi-level columns if needed
        if isinstance(new_data.columns, pd.MultiIndex):
            new_data.columns = new_data.columns.get_level_values(0)
        
        # Load existing streaming data if exists
        try:
            existing = load_parquet(streaming_file)
            logger.info(f"{symbol}: Loaded existing streaming file ({len(existing):,} rows)")
            
            # Append new data
            combined = pd.concat([existing, new_data])
            combined = combined.sort_index()
            combined = combined[~combined.index.duplicated(keep='last')]
            
            new_rows = len(combined) - len(existing)
            logger.info(f"{symbol}: Adding {new_rows} new rows to streaming file")
            
        except FileNotFoundError:
            logger.info(f"{symbol}: Creating new streaming file")
            combined = new_data
        
        # Save streaming file
        save_parquet(combined, streaming_file)
        
        file_size = os.path.getsize(streaming_file) / 1024  # KB
        logger.info(f"{symbol}: Streaming file now has {len(combined):,} rows ({file_size:.1f} KB)")
        
        return len(combined)
    
    def consolidate_week_and_clear(self, symbol: str) -> int:
        """
        Sunday Midnight Job:
        1. Read full week of 1-min data from streaming file
        2. Aggregate to 7 daily bars
        3. Append to permanent daily file
        4. CLEAR streaming file (reset for next week)
        
        Args:
            symbol: Yahoo Finance symbol
            
        Returns:
            Number of daily bars added
        """
        streaming_file = self.get_streaming_file_path(symbol)
        daily_file = self.get_daily_file_path(symbol)
        
        symbol_clean = symbol.replace('^', '')
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🗓️  WEEKLY CONSOLIDATION: {symbol}")
        logger.info(f"{'='*70}")
        
        # Check if streaming file exists
        if not os.path.exists(streaming_file):
            logger.warning(f"❌ No streaming file found: {streaming_file}")
            return 0
        
        # Load streaming data (full week)
        try:
            df_streaming = load_parquet(streaming_file)
            logger.info(f"📂 Loaded {len(df_streaming):,} rows from streaming file")
            
            # Flatten columns if needed
            if isinstance(df_streaming.columns, pd.MultiIndex):
                df_streaming.columns = df_streaming.columns.get_level_values(0)
            
        except Exception as e:
            logger.error(f"❌ Error loading streaming file: {e}")
            return 0
        
        if df_streaming.empty:
            logger.warning(f"⚠️  Streaming file is empty")
            return 0
        
        # Show date range
        logger.info(f"📅 Date range: {df_streaming.index[0]} to {df_streaming.index[-1]}")
        
        # Aggregate to daily bars
        try:
            daily_bars = df_streaming.resample('1D').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            
            logger.info(f"✅ Created {len(daily_bars)} daily bars:")
            for idx in daily_bars.index:
                logger.info(f"   • {idx.date()}: "
                          f"O={daily_bars.loc[idx, 'Open']:.2f}, "
                          f"H={daily_bars.loc[idx, 'High']:.2f}, "
                          f"L={daily_bars.loc[idx, 'Low']:.2f}, "
                          f"C={daily_bars.loc[idx, 'Close']:.2f}")
                
        except Exception as e:
            logger.error(f"❌ Error aggregating to daily: {e}")
            return 0
        
        # Load existing daily file and append
        try:
            if os.path.exists(daily_file):
                existing_daily = load_parquet(daily_file)
                logger.info(f"📂 Existing daily file has {len(existing_daily):,} rows")
                
                # Flatten columns if needed
                if isinstance(existing_daily.columns, pd.MultiIndex):
                    existing_daily.columns = existing_daily.columns.get_level_values(0)
                
                # Append new daily bars
                combined_daily = pd.concat([existing_daily, daily_bars])
                combined_daily = combined_daily[~combined_daily.index.duplicated(keep='last')]
                combined_daily = combined_daily.sort_index()
                
                rows_added = len(combined_daily) - len(existing_daily)
                logger.info(f"➕ Adding {rows_added} new daily bars")
                
            else:
                logger.info(f"🆕 Creating new daily file")
                combined_daily = daily_bars
                rows_added = len(daily_bars)
            
            # Save consolidated daily file
            save_parquet(combined_daily, daily_file)
            logger.info(f"💾 Saved to {daily_file} (total: {len(combined_daily):,} rows)")
            
        except Exception as e:
            logger.error(f"❌ Error saving daily file: {e}")
            return 0
        
        # CLEAR streaming file (reset for next week)
        try:
            # Create empty DataFrame with same structure
            empty_df = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
            empty_df.index.name = 'Datetime'
            
            save_parquet(empty_df, streaming_file)
            logger.info(f"🧹 CLEARED streaming file - ready for next week!")
            logger.info(f"   File reset: 0 rows")
            
        except Exception as e:
            logger.error(f"❌ Error clearing streaming file: {e}")
        
        logger.info(f"{'='*70}\n")
        
        return rows_added
    
    def get_current_week_summary(self, symbol: str) -> Dict:
        """
        Get summary of current week's streaming data
        
        Args:
            symbol: Yahoo Finance symbol
            
        Returns:
            Dictionary with streaming file stats
        """
        streaming_file = self.get_streaming_file_path(symbol)
        daily_file = self.get_daily_file_path(symbol)
        
        result = {
            'symbol': symbol,
            'streaming_file': streaming_file,
            'daily_file': daily_file,
            'streaming_exists': os.path.exists(streaming_file),
            'daily_exists': os.path.exists(daily_file),
            'streaming_rows': 0,
            'streaming_size_kb': 0,
            'daily_rows': 0,
            'daily_size_kb': 0
        }
        
        # Streaming file stats
        if result['streaming_exists']:
            try:
                df_stream = load_parquet(streaming_file)
                result['streaming_rows'] = len(df_stream)
                result['streaming_size_kb'] = os.path.getsize(streaming_file) / 1024
                
                if not df_stream.empty:
                    result['streaming_start'] = df_stream.index[0]
                    result['streaming_end'] = df_stream.index[-1]
            except:
                pass
        
        # Daily file stats
        if result['daily_exists']:
            try:
                df_daily = load_parquet(daily_file)
                result['daily_rows'] = len(df_daily)
                result['daily_size_kb'] = os.path.getsize(daily_file) / 1024
                
                if not df_daily.empty:
                    result['daily_start'] = df_daily.index[0]
                    result['daily_end'] = df_daily.index[-1]
            except:
                pass
        
        return result
    
    def print_system_status(self, symbols: List[str] = None):
        """
        Print status of all streaming files
        
        Args:
            symbols: List of symbols to check (None = all Indian indices)
        """
        if symbols is None:
            symbols = [info['yahoo'] for info in INDIAN_INDICES.values()]
        
        print("\n" + "="*70)
        print("📊 DUAL-FILE STREAMING SYSTEM STATUS")
        print("="*70)
        
        for symbol in symbols[:10]:  # Show first 10 for brevity
            summary = self.get_current_week_summary(symbol)
            
            print(f"\n{summary['symbol']}:")
            
            if summary['streaming_exists']:
                print(f"  📁 Streaming (1-min):")
                print(f"     • Rows: {summary['streaming_rows']:,}")
                print(f"     • Size: {summary['streaming_size_kb']:.1f} KB")
                if summary['streaming_rows'] > 0:
                    print(f"     • Range: {summary.get('streaming_start')} to {summary.get('streaming_end')}")
            else:
                print(f"  📁 Streaming: Not created yet")
            
            if summary['daily_exists']:
                print(f"  📁 Daily (permanent):")
                print(f"     • Rows: {summary['daily_rows']:,} days")
                print(f"     • Size: {summary['daily_size_kb']:.1f} KB")
                if summary['daily_rows'] > 0:
                    print(f"     • Range: {summary.get('daily_start')} to {summary.get('daily_end')}")
            else:
                print(f"  📁 Daily: Not created yet")
        
        print("\n" + "="*70)


def setup_streaming_directories():
    """Create directory structure for streaming system"""
    base_dir = CONFIG['base_dir']
    
    dirs = [
        f"{base_dir}/streaming",  # Temporary 1-min streaming files
        f"{base_dir}/daily",      # Permanent daily files
        "./logs"
    ]
    
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"✅ Directory structure created")


if __name__ == "__main__":
    # Test the streaming system
    print("="*70)
    print("DUAL-FILE STREAMING SYSTEM - TEST")
    print("="*70)
    
    setup_streaming_directories()
    
    collector = StreamingDataCollector()
    
    # Example: Simulate streaming and consolidation for Nifty 50
    test_symbol = "^NSEI"
    
    print(f"\n1. Current status:")
    collector.print_system_status([test_symbol])
    
    print(f"\n2. Simulating weekly consolidation...")
    rows_added = collector.consolidate_week_and_clear(test_symbol)
    
    print(f"\n3. After consolidation:")
    collector.print_system_status([test_symbol])
    
    print("\n" + "="*70)
    print("Test complete!")
    print("="*70)
