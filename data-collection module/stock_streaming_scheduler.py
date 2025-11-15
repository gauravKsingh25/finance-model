"""
Stock Streaming Scheduler
Automated data collection for all Indian stocks every minute during market hours
"""

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime, timedelta
import pytz
import time
import logging

from stock_collector import StockCollector
from indian_stocks import INDIAN_STOCKS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/stock_streaming.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StockStreamingScheduler:
    """
    Automated scheduler for streaming stock data collection
    Runs every minute during NSE market hours (9:00 AM - 3:30 PM IST, Mon-Fri)
    """
    
    def __init__(self, batch_size: int = 50):
        self.collector = StockCollector()
        self.scheduler = BlockingScheduler(timezone=pytz.timezone('Asia/Kolkata'))
        self.batch_size = batch_size  # Process stocks in batches to manage load
        self.stock_symbols = list(INDIAN_STOCKS.keys())
        self.total_stocks = len(self.stock_symbols)
        
        logger.info(f"Stock Streaming Scheduler initialized")
        logger.info(f"   Total stocks: {self.total_stocks}")
        logger.info(f"   Batch size: {batch_size} stocks per minute")
        logger.info(f"   Estimated batches: {(self.total_stocks + batch_size - 1) // batch_size}")
    
    def is_market_open(self) -> bool:
        """Check if NSE market is currently open"""
        now = datetime.now(pytz.timezone('Asia/Kolkata'))
        
        # Market closed on weekends
        if now.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        
        # Market hours: 9:00 AM to 3:30 PM IST
        market_start = now.replace(hour=9, minute=0, second=0, microsecond=0)
        market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_start <= now <= market_end
    
    def get_current_batch(self) -> list:
        """
        Get batch of stocks to process based on current minute
        Distributes stocks across market hours for even load distribution
        """
        now = datetime.now(pytz.timezone('Asia/Kolkata'))
        
        # Calculate which batch to process (cycles through all stocks hourly)
        minutes_since_market_open = (now.hour - 9) * 60 + now.minute
        batch_index = minutes_since_market_open % ((self.total_stocks + self.batch_size - 1) // self.batch_size)
        
        start_idx = batch_index * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.total_stocks)
        
        return self.stock_symbols[start_idx:end_idx]
    
    def fetch_minute_data(self):
        """
        Fetch 1-minute data for current batch of stocks
        Synchronized to run at exactly :00 seconds of each minute
        """
        # Wait until exactly :00 seconds
        current_second = datetime.now().second
        if current_second > 0:
            sleep_time = 60 - current_second
            logger.debug(f"⏳ Waiting {sleep_time}s to sync to :00 seconds...")
            time.sleep(sleep_time)
        
        # Check if market is open
        if not self.is_market_open():
            logger.info("Market closed - skipping data collection")
            return
        
        timestamp = datetime.now(pytz.timezone('Asia/Kolkata'))
        logger.info(f"\n{'='*70}")
        logger.info(f"Stock Data Collection - {timestamp.strftime('%Y-%m-%d %H:%M:%S IST')}")
        logger.info(f"{'='*70}")
        
        # Get current batch
        batch = self.get_current_batch()
        logger.info(f"Processing batch: {len(batch)} stocks")
        logger.info(f"   Symbols: {', '.join(batch[:5])}{'...' if len(batch) > 5 else ''}")
        
        # Fetch data for batch
        try:
            results = self.collector.fetch_multiple_stocks(batch, delay=0.1)
            
            success_count = len(results)
            logger.info(f"\nBatch complete: {success_count}/{len(batch)} stocks successful")
            
            if success_count < len(batch):
                failed = set(batch) - set(results.keys())
                logger.warning(f"Failed stocks: {', '.join(list(failed)[:5])}")
        
        except Exception as e:
            logger.error(f"Error in batch processing: {e}", exc_info=True)
    
    def status_check(self):
        """Periodic status check (runs every hour)"""
        now = datetime.now(pytz.timezone('Asia/Kolkata'))
        logger.info(f"\n{'='*70}")
        logger.info(f"Stock Scheduler Status - {now.strftime('%Y-%m-%d %H:%M:%S IST')}")
        logger.info(f"   Total stocks: {self.total_stocks}")
        logger.info(f"   Batch size: {self.batch_size}")
        logger.info(f"   Market open: {self.is_market_open()}")
        logger.info(f"{'='*70}\n")
    
    def start(self):
        """Start the streaming scheduler"""
        logger.info(f"\n{'='*70}")
        logger.info("Starting Stock Streaming Scheduler")
        logger.info(f"{'='*70}")
        logger.info(f"Schedule:")
        logger.info(f"   - Data collection: Every minute at :00 seconds")
        logger.info(f"   - Market hours: Mon-Fri 9:00 AM - 3:30 PM IST")
        logger.info(f"   - Status check: Every hour")
        logger.info(f"{'='*70}\n")
        
        # Schedule minute data collection (runs every minute at :00 seconds)
        self.scheduler.add_job(
            self.fetch_minute_data,
            trigger=CronTrigger(
                day_of_week='mon-fri',
                hour='9-15',
                minute='*',
                second='0',
                timezone='Asia/Kolkata'
            ),
            id='stock_minute_data',
            name='Stock Minute Data Collection',
            max_instances=1
        )
        
        # Schedule status check (every hour)
        self.scheduler.add_job(
            self.status_check,
            trigger=CronTrigger(
                minute='0',
                timezone='Asia/Kolkata'
            ),
            id='status_check',
            name='Status Check'
        )
        
        # Run initial status check
        self.status_check()
        
        try:
            # Start scheduler
            logger.info("Scheduler started - Press Ctrl+C to stop\n")
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("\nStopping Stock Streaming Scheduler...")
            self.scheduler.shutdown()
            logger.info("Scheduler stopped")

if __name__ == '__main__':
    # Start with batch size of 50 stocks per minute
    # This means all ~210 stocks will be processed over ~4-5 minutes, cycling throughout the day
    scheduler = StockStreamingScheduler(batch_size=50)
    scheduler.start()
