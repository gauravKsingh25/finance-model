# Data Collection Module - Streaming System

A streamlined system for collecting 1-minute streaming financial data with weekly aggregation capabilities.

## 🎯 Features

- **1-Minute Streaming**: Real-time data collection from Yahoo Finance
- **Automated Scheduling**: Market hours monitoring with automatic data collection
- **Weekly Aggregation**: Consolidate 1-minute data into weekly summaries
- **Data Validation**: OHLC validation and data cleaning utilities
- **Parquet Storage**: Efficient storage and retrieval
- **Multiple Symbols**: Support for multiple Indian market indices

## 📁 Directory Structure

```
data-collection module/
├── config.py                    # Configuration settings
├── collector.py                 # Yahoo Finance data fetching
├── cleaner.py                   # Data validation and cleaning
├── storage.py                   # Parquet storage utilities
├── streaming_system.py          # Core 1-minute streaming collector
├── streaming_scheduler.py       # Automated market hours scheduler
├── weekly_consolidator.py       # Weekly data aggregation
├── indian_market_indices.py     # Indian market index definitions
├── requirements.txt             # Dependencies
├── data/                        # Output directory for collected data
└── logs/                        # System logs
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd "data-collection module"
pip install -r requirements.txt
```

### 2. Run Streaming System

**Start automated streaming (runs during market hours):**
```bash
python streaming_scheduler.py
```

**Manual single collection:**
```bash
python streaming_system.py
```

**Weekly aggregation:**
```bash
python weekly_consolidator.py
```

## 📊 Output Structure

```
data/
├── raw/                         # 1-minute streaming data
│   ├── NIFTY_50_1m.parquet
│   ├── NIFTY_BANK_1m.parquet
│   └── ...
└── weekly/                      # Weekly aggregated data
    ├── NIFTY_50_weekly.parquet
    ├── NIFTY_BANK_weekly.parquet
    └── ...
```

## ⚙️ Configuration

Edit `config.py` to customize:

- **Market Hours**: IST market open/close times (9:15 AM - 3:30 PM)
- **Symbols**: Configured in `indian_market_indices.py`
- **Update Frequency**: 1-minute interval for streaming
- **Storage Paths**: Data and log directories

## 🔧 Usage Examples

### Streaming System

```python
from streaming_system import StreamingDataCollector

# Initialize collector
collector = StreamingDataCollector()

# Collect data for all configured indices
collector.collect_all_indices()
```

### Automated Scheduler

```python
from streaming_scheduler import StreamingScheduler

# Start scheduler (runs during market hours only)
scheduler = StreamingScheduler()
scheduler.run()

# Scheduler automatically:
# - Monitors market hours
# - Collects 1-minute data every minute
# - Stops outside market hours
# - Handles weekends and holidays
```

### Weekly Aggregation

```python
from weekly_consolidator import WeeklyConsolidator

# Aggregate all symbols to weekly data
consolidator = WeeklyConsolidator()
consolidator.consolidate_all_symbols()
```

## � Market Hours

- **Market Open**: 9:15 AM IST
- **Market Close**: 3:30 PM IST
- **Trading Days**: Monday - Friday
- **Streaming Interval**: 1 minute

## 📝 Data Source

- **Primary Source**: Yahoo Finance API
- **Interval**: 1-minute OHLCV data
- **Symbols**: Indian market indices (Nifty 50, Bank Nifty, etc.)
- **API Key**: Not required (free access)

## 🔍 System Components

### streaming_system.py
Core streaming data collector that fetches 1-minute data from Yahoo Finance and stores in Parquet format.

### streaming_scheduler.py
Automated scheduler that runs the streaming system during market hours, ensuring continuous data collection.

### weekly_consolidator.py
Aggregates 1-minute data into weekly OHLCV summaries with volume-weighted statistics.

### collector.py
Low-level data fetching utilities for Yahoo Finance API integration.

### storage.py
Parquet file management utilities for efficient data storage and retrieval.

### cleaner.py
Data validation and cleaning functions for OHLC integrity.

## 🐛 Troubleshooting

**Issue**: No data collected
- **Solution**: Verify market hours (9:15 AM - 3:30 PM IST, Mon-Fri)
- **Solution**: Check internet connection and Yahoo Finance availability

**Issue**: Import errors
- **Solution**: Install dependencies: `pip install -r requirements.txt`

**Issue**: Scheduler not running
- **Solution**: Check logs in `./logs/streaming_system.log`

## 🔄 Maintenance

- **Logs**: Monitor `./logs/streaming_system.log` for collection status
- **Storage**: Parquet files auto-compress for efficiency
- **Cleanup**: Old logs rotate automatically

## � Integration

The collected data is ready for:
- Regime detection models
- Changepoint analysis
- Time-series forecasting
- Technical analysis
- Machine learning pipelines

Output Parquet files can be directly loaded with pandas:
```python
import pandas as pd
df = pd.read_parquet('data/raw/NIFTY_50_1m.parquet')
```
