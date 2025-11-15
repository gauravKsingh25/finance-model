# weekly_consolidator.py
"""
Weekly Data Consolidation System
Consolidates 1-minute data into daily files at end of each week
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import os
from typing import Dict, List

from storage import load_parquet, save_parquet
from config import CONFIG
from indian_market_indices import INDIAN_INDICES


class WeeklyConsolidator:
    """
    Manages weekly consolidation of 1-minute data to daily files
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or CONFIG
        self.base_dir = self.config['base_dir']
        
    def get_week_boundaries(self, reference_date: datetime = None):
        """
        Get start and end of the current week (Monday to Sunday)
        
        Args:
            reference_date: Date to calculate week from (default: today)
            
        Returns:
            tuple: (week_start, week_end) as datetime objects
        """
        if reference_date is None:
            reference_date = datetime.now()
        
        # Get Monday of current week (weekday 0)
        days_since_monday = reference_date.weekday()
        week_start = reference_date - timedelta(days=days_since_monday)
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Get Sunday of current week (weekday 6)
        week_end = week_start + timedelta(days=6, hours=23, minutes=59, seconds=59)
        
        return week_start, week_end
    
    def consolidate_week_to_daily(self, symbol: str, week_start: datetime, week_end: datetime):
        """
        Consolidate 1-minute data for a week into daily bars and append to daily file
        
        Args:
            symbol: Symbol to consolidate (e.g., '^NSEI')
            week_start: Start of week
            week_end: End of week
            
        Returns:
            Number of daily rows added
        """
        symbol_clean = symbol.replace('^', '')
        raw_file = f"{self.base_dir}/raw/{symbol_clean}_1m.parquet"
        daily_file = f"{self.base_dir}/daily/{symbol_clean}_daily.parquet"
        
        print(f"\n🔄 Consolidating {symbol} for week {week_start.date()} to {week_end.date()}")
        
        # Check if raw file exists
        if not os.path.exists(raw_file):
            print(f"   ⚠️  No raw data file found: {raw_file}")
            return 0
        
        # Load 1-minute data
        try:
            df_1min = load_parquet(raw_file)
            print(f"   📂 Loaded {len(df_1min):,} rows of 1-min data")
            
            # Flatten multi-level columns if present (from yfinance)
            if isinstance(df_1min.columns, pd.MultiIndex):
                df_1min.columns = df_1min.columns.get_level_values(0)
                print(f"   ✓ Flattened multi-level columns")
                
        except Exception as e:
            print(f"   ❌ Error loading raw data: {e}")
            return 0
        
        # Filter data for this week
        # Convert week boundaries to match DataFrame timezone if needed
        if df_1min.index.tz is not None:
            week_start_tz = week_start.replace(tzinfo=df_1min.index.tz)
            week_end_tz = week_end.replace(tzinfo=df_1min.index.tz)
        else:
            week_start_tz = week_start
            week_end_tz = week_end
        
        week_data = df_1min[(df_1min.index >= week_start_tz) & (df_1min.index <= week_end_tz)]
        
        if week_data.empty:
            print(f"   ℹ️  No data found for this week")
            return 0
        
        print(f"   📊 Found {len(week_data):,} rows for this week")
        
        # Resample to daily (1D)
        daily_bars = week_data.resample('1D').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        if daily_bars.empty:
            print(f"   ⚠️  No daily bars created (possible data issue)")
            return 0
        
        print(f"   ✅ Created {len(daily_bars)} daily bars:")
        for idx in daily_bars.index:
            print(f"      • {idx.date()}: O={daily_bars.loc[idx, 'Open']:.2f}, "
                  f"H={daily_bars.loc[idx, 'High']:.2f}, "
                  f"L={daily_bars.loc[idx, 'Low']:.2f}, "
                  f"C={daily_bars.loc[idx, 'Close']:.2f}")
        
        # Load existing daily file if it exists
        if os.path.exists(daily_file):
            try:
                existing_daily = load_parquet(daily_file)
                print(f"   📂 Existing daily file has {len(existing_daily):,} rows")
                
                # Merge with existing data
                combined = pd.concat([existing_daily, daily_bars])
                combined = combined[~combined.index.duplicated(keep='last')]  # Remove duplicates
                combined = combined.sort_index()
                
                rows_added = len(combined) - len(existing_daily)
                print(f"   ➕ Adding {rows_added} new daily bars")
                
            except Exception as e:
                print(f"   ⚠️  Error loading existing daily file: {e}")
                combined = daily_bars
                rows_added = len(daily_bars)
        else:
            print(f"   🆕 Creating new daily file")
            combined = daily_bars
            rows_added = len(daily_bars)
        
        # Save consolidated daily data
        save_parquet(combined, daily_file)
        print(f"   💾 Saved to {daily_file} (total: {len(combined):,} rows)")
        
        return rows_added
    
    def cleanup_old_1min_data(self, symbol: str, keep_days: int = 30):
        """
        Remove 1-minute data older than keep_days to save space
        (Keep only recent 1-min data, older data is in daily files)
        
        Args:
            symbol: Symbol to clean
            keep_days: Number of days to keep in 1-min file
        """
        symbol_clean = symbol.replace('^', '')
        raw_file = f"{self.base_dir}/raw/{symbol_clean}_1m.parquet"
        
        if not os.path.exists(raw_file):
            return
        
        try:
            df = load_parquet(raw_file)
            original_rows = len(df)
            original_size = os.path.getsize(raw_file) / 1024 / 1024  # MB
            
            # Calculate cutoff date
            cutoff_date = datetime.now() - timedelta(days=keep_days)
            if df.index.tz is not None:
                cutoff_date = cutoff_date.replace(tzinfo=df.index.tz)
            
            # Keep only recent data
            df_recent = df[df.index >= cutoff_date]
            rows_removed = original_rows - len(df_recent)
            
            if rows_removed > 0:
                save_parquet(df_recent, raw_file)
                new_size = os.path.getsize(raw_file) / 1024 / 1024  # MB
                
                print(f"   🧹 Cleaned {symbol}:")
                print(f"      • Removed {rows_removed:,} rows older than {keep_days} days")
                print(f"      • File size: {original_size:.2f} MB → {new_size:.2f} MB "
                      f"(saved {original_size - new_size:.2f} MB)")
            else:
                print(f"   ✅ {symbol}: No old data to remove")
                
        except Exception as e:
            print(f"   ❌ Error cleaning {symbol}: {e}")
    
    def run_weekly_consolidation(self, symbols: List[str] = None, cleanup: bool = True):
        """
        Run weekly consolidation for all or specified symbols
        
        Args:
            symbols: List of symbols to consolidate (None = all)
            cleanup: Whether to remove old 1-min data after consolidation
            
        Returns:
            Dict with consolidation results
        """
        if symbols is None:
            # Get all symbols from indian_market_indices
            symbols = [info["yahoo"] for info in INDIAN_INDICES.values()]
        
        # Get last complete week (previous Monday to Sunday)
        today = datetime.now()
        days_since_monday = today.weekday()
        last_monday = today - timedelta(days=days_since_monday + 7)
        week_start, week_end = self.get_week_boundaries(last_monday)
        
        print("="*70)
        print(f"🗓️  WEEKLY CONSOLIDATION")
        print("="*70)
        print(f"Week: {week_start.date()} to {week_end.date()}")
        print(f"Symbols to process: {len(symbols)}")
        print(f"Cleanup old data: {cleanup}")
        print("="*70)
        
        results = {
            "week_start": week_start,
            "week_end": week_end,
            "symbols_processed": 0,
            "total_daily_rows_added": 0,
            "errors": []
        }
        
        for symbol in symbols:
            try:
                rows_added = self.consolidate_week_to_daily(symbol, week_start, week_end)
                results["symbols_processed"] += 1
                results["total_daily_rows_added"] += rows_added
                
                # Cleanup old 1-min data if requested
                if cleanup:
                    keep_days = self.config['weekly_consolidation']['keep_1min_days']
                    self.cleanup_old_1min_data(symbol, keep_days)
                    
            except Exception as e:
                error_msg = f"{symbol}: {str(e)}"
                results["errors"].append(error_msg)
                print(f"   ❌ Error processing {symbol}: {e}")
        
        # Summary
        print("\n" + "="*70)
        print("📊 CONSOLIDATION SUMMARY")
        print("="*70)
        print(f"✅ Symbols processed: {results['symbols_processed']}/{len(symbols)}")
        print(f"➕ Total daily rows added: {results['total_daily_rows_added']}")
        print(f"❌ Errors: {len(results['errors'])}")
        
        if results['errors']:
            print("\nErrors encountered:")
            for error in results['errors'][:10]:  # Show first 10 errors
                print(f"  • {error}")
        
        print("="*70)
        
        return results
    
    def run_consolidation_for_current_week(self, symbols: List[str] = None):
        """
        Consolidate data for current week (useful for testing)
        
        Args:
            symbols: List of symbols to consolidate
        """
        if symbols is None:
            symbols = [info["yahoo"] for info in INDIAN_INDICES.values()]
        
        week_start, week_end = self.get_week_boundaries()
        
        print("="*70)
        print(f"🗓️  CURRENT WEEK CONSOLIDATION (TEST MODE)")
        print("="*70)
        print(f"Week: {week_start.date()} to {week_end.date()}")
        print(f"Symbols: {len(symbols)}")
        print("="*70)
        
        results = {"symbols_processed": 0, "total_rows": 0}
        
        for symbol in symbols:
            try:
                rows = self.consolidate_week_to_daily(symbol, week_start, week_end)
                results["symbols_processed"] += 1
                results["total_rows"] += rows
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        print(f"\n✅ Processed {results['symbols_processed']} symbols, "
              f"added {results['total_rows']} daily rows")
        
        return results


if __name__ == "__main__":
    # Test the consolidator
    consolidator = WeeklyConsolidator()
    
    # Test with a few symbols
    test_symbols = ["^NSEI", "^NSEBANK", "^CNXIT"]
    
    print("Testing weekly consolidation...")
    consolidator.run_consolidation_for_current_week(test_symbols)
