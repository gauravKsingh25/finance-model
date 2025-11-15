# Stock Data Collection System

Automated streaming data collection for all major Indian stocks.

## Overview

This system collects 1-minute OHLCV data for **100 major NSE F&O stocks** during market hours.

### Features
- ✅ **100 liquid NSE stocks** (NIFTY 50 + NIFTY Next 50 + top F&O stocks)
- ✅ **1-minute streaming data** from Yahoo Finance (free, reliable)
- ✅ **Multi-source fallback**: Yahoo Finance → Twelve Data
- ✅ **Batch processing**: 50 stocks per minute (rotates through all stocks)
- ✅ **Market hours only**: Mon-Fri 9:00 AM - 3:30 PM IST
- ✅ **Parquet storage**: Efficient, compressed data format
- ✅ **Automatic deduplication**: No duplicate timestamps

## Stock Universe

### Coverage (100 stocks)
- **NIFTY 50**: All constituents (Reliance, TCS, HDFC Bank, Infosys, etc.)
- **NIFTY Next 50**: Major mid-cap stocks (Adani Green, Zomato, Paytm, etc.)
- **Top F&O Stocks**: High liquidity stocks across sectors

### Sectors Covered
- Finance (Banks, NBFCs, Insurance)
- IT (TCS, Infosys, Wipro, HCL Tech)
- Energy (Reliance, ONGC, BPCL, Adani)
- Auto (Maruti, Tata Motors, Hero, Bajaj)
- Pharma (Sun Pharma, Dr. Reddy's, Cipla)
- FMCG (HUL, ITC, Nestle, Britannia)
- Metals & Mining (Tata Steel, JSW, Vedanta)
- Infrastructure (L&T, Adani Ports)
- Real Estate (DLF, Godrej Properties)
- And more...

## Files

### Configuration
- `indian_stocks.py` - Stock universe configuration (100 stocks)
- `get_indian_stocks.py` - Script to regenerate stock list

### Core Components
- `stock_collector.py` - Multi-source stock data collector
- `stock_streaming_scheduler.py` - Automated streaming scheduler

### Data Storage
```
stocks/
├── streaming/          # 1-minute intraday data
│   ├── RELIANCE_1m_streaming.parquet
│   ├── TCS_1m_streaming.parquet
│   ├── INFY_1m_streaming.parquet
│   └── ...  (100 files total)
└── daily/             # Daily consolidated data (future)
```

## Usage

### Start Stock Streaming

```bash
# Default batch size (50 stocks/minute)
python stock_streaming_scheduler.py
```

This will:
1. Process 50 stocks per minute in rotation
2. All 100 stocks covered every ~2 minutes
3. Run continuously during market hours (9:00 AM - 3:30 PM IST)
4. Save data to `stocks/streaming/` folder

### Collect Specific Stocks

```python
from stock_collector import StockCollector

collector = StockCollector()

# Fetch specific stocks
stocks = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK']
results = collector.fetch_multiple_stocks(stocks, delay=1.0)

# Access data
for symbol, df in results.items():
    print(f"{symbol}: {len(df)} bars")
    print(df.tail())
```

### Add More Stocks

Edit `indian_stocks.py` to add new stocks:

```python
INDIAN_STOCKS = {
    'NEWSTOCK': {
        'name': 'New Company Ltd',
        'exchange': 'NSE',
        'yahoo': 'NEWSTOCK.NS',
        'nse': 'NEWSTOCK',
        'sector': 'Technology'
    },
    # ... existing stocks
}
```

## Data Sources

### Primary: Yahoo Finance
- ✅ Free, unlimited API calls
- ✅ 1-minute intraday data
- ✅ High reliability for Indian stocks
- ✅ Works for NSE (.NS) and BSE (.BO) stocks

### Fallback: Twelve Data
- ⚠️ Free tier: 8 API calls/minute (rate limited)
- ✅ 1-minute data available
- ⚠️ Insufficient for 100 stocks/minute
- 🎯 Use only when Yahoo Finance fails

## Scheduler Configuration

### Batch Processing
The scheduler processes stocks in batches to:
- Avoid overwhelming Yahoo Finance API
- Distribute load evenly across market hours
- Enable coverage of all 100 stocks every 2 minutes

```python
# Customize batch size
scheduler = StockStreamingScheduler(batch_size=50)  # 50 stocks/minute
scheduler.start()
```

### Schedule
- **Frequency**: Every minute at :00 seconds
- **Days**: Monday to Friday only
- **Hours**: 9:00 AM to 3:30 PM IST
- **Batches**: ~2 batches to cover 100 stocks
- **Cycle**: Complete rotation every 2 minutes

## Data Format

### Parquet Files
Each stock has a streaming parquet file:
```
Symbol_1m_streaming.parquet
```

### Data Schema
```python
Index: DatetimeIndex (timestamp)
Columns:
  - Open: float64
  - High: float64
  - Low: float64
  - Close: float64
  - Volume: int64
```

### Example
```python
import pandas as pd

# Read stock data
df = pd.read_parquet('stocks/streaming/RELIANCE_1m_streaming.parquet')

print(df.head())
#                      Open    High     Low   Close   Volume
# 2025-11-14 09:00:00  1234.5  1236.0  1233.0  1235.5  123456
# 2025-11-14 09:01:00  1235.5  1237.0  1235.0  1236.0  234567
# ...
```

## Performance

### Current System
- ✅ **100 stocks** in stock universe
- ✅ **50 stocks/minute** batch processing
- ✅ **2-minute** full coverage cycle
- ✅ **375 market hours/month** (approx)
- ✅ **~112,500 data points/stock/month** (375 min × 300 bars/day)

### Storage
- **Per stock**: ~15-20 KB per day (compressed parquet)
- **Total daily**: ~1.5-2 MB for all 100 stocks
- **Monthly**: ~30-40 MB total storage

## Monitoring

### Check Running Scheduler
```bash
# View logs
tail -f logs/stock_streaming.log
```

### Verify Data Collection
```bash
# Check streaming folder
ls -lh stocks/streaming/

# Count files
ls stocks/streaming/ | wc -l  # Should be 100
```

### Test Data Integrity
```python
import pandas as pd

# Test read
df = pd.read_parquet('stocks/streaming/RELIANCE_1m_streaming.parquet')

# Check for duplicates
assert df.index.duplicated().sum() == 0, "Duplicate timestamps found!"

# Verify columns
assert list(df.columns) == ['Open', 'High', 'Low', 'Close', 'Volume']

print("✅ Data integrity OK")
```

## Troubleshooting

### No Data Collected
1. Check if market is open (Mon-Fri 9:00-15:30 IST)
2. Verify Yahoo Finance connectivity
3. Check logs: `logs/stock_streaming.log`

### Rate Limiting
- Yahoo Finance: No rate limits observed for Indian stocks
- Twelve Data: 8 calls/minute on free tier (handled automatically)

### Missing Stocks
Some stocks may not be available on Yahoo Finance:
- Verify symbol format (should end with .NS for NSE)
- Check if stock is actively trading
- Try BSE symbol (.BO) as alternative

### Storage Issues
- Parquet files are compressed (~15 KB/stock/day)
- Clean old data periodically if needed
- Consider daily consolidation for long-term storage

## Integration with Models

### Use with HDP Regime Detection
```python
import pandas as pd
from stock_collector import StockCollector

# Collect fresh data
collector = StockCollector()
df = collector.fetch_stock_minute_data('RELIANCE', save_to_file=False)

# Use with your regime detection models
from complete_system_hdp.regime_engine import RegimeEngine
engine = RegimeEngine()
regimes = engine.detect_regimes(df)
```

### Batch Processing for All Stocks
```python
from indian_stocks import INDIAN_STOCKS

results = {}
for symbol in INDIAN_STOCKS.keys():
    df = pd.read_parquet(f'stocks/streaming/{symbol}_1m_streaming.parquet')
    # Your analysis here
    results[symbol] = analyze(df)
```

## Future Enhancements

- [ ] Daily data consolidation
- [ ] Historical data backfill
- [ ] Real-time WebSocket streaming
- [ ] Options chain data collection
- [ ] Corporate actions integration
- [ ] Fundamental data enrichment

## Notes

- **Market hours**: NSE operates 9:00 AM - 3:30 PM IST
- **Holidays**: System automatically skips market holidays (weekends)
- **Data retention**: Keep as needed, ~2 MB/day for all stocks
- **Scalability**: Can handle 200+ stocks with current batch approach

---

**Last Updated**: November 2025
**Status**: ✅ Operational
**Coverage**: 100 major NSE stocks
