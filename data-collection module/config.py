# config.py
"""
Configuration for Data Collection Module
Supports multiple symbols and data sources
"""

CONFIG = {
    # Primary symbol configuration
    "symbol": "^NSEI",                      # Nifty 50 (Yahoo Finance format)
    "nse_symbol": "NIFTY",                  # NSE format for NSEpy
    
    # Data collection parameters
    "interval": "1m",                       # 1-min data
    "base_dir": "./data",
    "update_freq_hours": 1,
    "fetch_chunk_days": 7,                  # yfinance 1-min supports 7 days per call
    "start_date": "2025-11-05",            # Recent date (Yahoo 1-min only keeps ~7 days)
    
    # Multiple symbols support - load from indian_market_indices.py
    "symbols": None,  # Will be loaded dynamically from INDIAN_INDICES
    "use_all_indian_indices": True,  # Set to True to collect all NSE/BSE indices
    
    # Aggregation frequencies
    "aggregations": {
        "1H": "hourly",
        "4H": "4hour",
        "1D": "daily",
        "1W": "weekly"
    },
    
    # Weekly consolidation settings
    "weekly_consolidation": {
        "enabled": True,
        "run_day": "Sunday",  # Day to run weekly consolidation
        "run_time": "23:00",  # Time to run (11 PM)
        "keep_1min_days": 30,  # Keep 1-min data for last N days, consolidate older
    },
    
    # Data source preferences (fallback order)
    "data_sources": ["yfinance", "twelvedata", "nselib"],
    
    # API keys (loaded from .env)
    "twelvedata_api_key": None,  # Will be loaded from .env
    
    # Retry and error handling
    "max_retries": 3,
    "retry_delay_seconds": 5,
    
    # Logging
    "log_level": "INFO",
    "log_file": "./logs/data_collection.log"
}
