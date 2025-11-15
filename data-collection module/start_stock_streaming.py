"""
Quick Start - Stock Data Collection
Run this to start collecting stock data immediately
"""

print("=" * 70)
print("Indian Stock Data Collection System")
print("=" * 70)
print()
print("Coverage: 100 major NSE F&O stocks")
print("Schedule: Every minute during market hours (9:00 AM - 3:30 PM IST)")
print("Storage: stocks/streaming/ folder (parquet format)")
print()
print("=" * 70)
print()

# Import and start scheduler
from stock_streaming_scheduler import StockStreamingScheduler

# Create scheduler with default settings
# - Batch size: 50 stocks per minute
# - Covers all 100 stocks every 2 minutes
scheduler = StockStreamingScheduler(batch_size=50)

print("Starting scheduler...")
print()
print("Press Ctrl+C to stop")
print()

# Start streaming
scheduler.start()
