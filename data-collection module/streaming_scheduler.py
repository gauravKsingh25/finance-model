# streaming_scheduler.py
"""
Automated Streaming Scheduler for Indian Market Indices

Jobs:
-----
1. EVERY MINUTE (during market hours):
   - Fetch 1-min data for all Indian indices
   - Append to streaming files (NSEI_1m_streaming.parquet, etc.)
   
2. EVERY SUNDAY 00:00 (Midnight):
   - Read all streaming files (7 days of 1-min data)
   - Aggregate to 7 daily bars
   - Append to permanent daily files
   - CLEAR streaming files (reset for next week)

File Structure:
---------------
Streaming (cleared weekly):
  - data/streaming/NSEI_1m_streaming.parquet
  - data/streaming/NSEBANK_1m_streaming.parquet
  - ... (30 indices)

Daily (grows forever):
  - data/daily/NSEI_daily.parquet
  - data/daily/NSEBANK_daily.parquet
  - ... (30 indices)
"""

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime, timedelta
import pytz
import logging
import sys
import time

from collector import DataCollector
from streaming_system import StreamingDataCollector, setup_streaming_directories
from indian_market_indices import INDIAN_INDICES
from config import CONFIG

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/streaming_scheduler.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# IST timezone
IST = pytz.timezone('Asia/Kolkata')


class StreamingScheduler:
    """
    Manages automated streaming and consolidation jobs
    """
    
    def __init__(self):
        self.scheduler = BlockingScheduler(timezone=IST)
        self.data_collector = DataCollector()
        self.streaming_collector = StreamingDataCollector()
        
        # Store index configurations for fallback support
        self.index_configs = INDIAN_INDICES
        
        logger.info(f"✅ StreamingScheduler initialized")
        logger.info(f"📊 Tracking {len(self.index_configs)} Indian indices")
    
    def fetch_minute_data(self):
        """
        JOB 1: Fetch 1-min data and append to streaming files
        Runs: Every minute during market hours
        Synchronized to start exactly at 00 seconds of each minute
        """
        try:
            # Wait until exactly 00 seconds of the current minute
            current_time = datetime.now(IST)
            seconds_to_wait = 60 - current_time.second
            
            if seconds_to_wait < 60:  # Only wait if not already at :00
                logger.info(f"⏱️  Syncing to minute start... waiting {seconds_to_wait} seconds")
                time.sleep(seconds_to_wait)
            
            # Now we're at exactly :00 seconds
            fetch_time = datetime.now(IST)
            logger.info("\n" + "="*70)
            logger.info(f"🔄 MINUTE DATA FETCH - {fetch_time.strftime('%Y-%m-%d %H:%M:%S IST')}")
            logger.info(f"⏰ Exact fetch time: {fetch_time.strftime('%H:%M:%S.%f')[:-3]}")
            logger.info("="*70)
            
            success_count = 0
            failed_symbols = []
            
            for index_name, index_config in self.index_configs.items():
                try:
                    # Fetch latest 1-min data (last 1 day to capture recent bars)
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=1)
                    
                    # Use fallback system
                    new_data = self.data_collector.fetch_with_fallback(
                        index_config,
                        start=start_date.strftime('%Y-%m-%d'),
                        end=end_date.strftime('%Y-%m-%d'),
                        interval='1m'
                    )
                    
                    if new_data is not None and not new_data.empty:
                        # Get primary symbol for file naming
                        symbol = index_config.get('yahoo') or f"FALLBACK_{index_name}"
                        
                        # Append to streaming file
                        rows = self.streaming_collector.append_1min_data(symbol, new_data)
                        logger.info(f"✅ {index_name} ({symbol}): Streaming file now has {rows:,} rows")
                        success_count += 1
                    else:
                        logger.warning(f"⚠️  {index_name}: No new data fetched from any source")
                        failed_symbols.append(index_name)
                        
                except Exception as e:
                    logger.error(f"❌ {index_name}: Error - {e}")
                    failed_symbols.append(index_name)
            
            logger.info(f"\n✅ Minute fetch complete: {success_count}/{len(self.index_configs)} successful")
            if failed_symbols:
                logger.warning(f"⚠️  Failed indices: {', '.join(failed_symbols[:5])}...")
            logger.info("="*70 + "\n")
            
        except Exception as e:
            logger.error(f"❌ Critical error in minute fetch: {e}")
    
    def weekly_consolidation(self):
        """
        JOB 2: Weekly consolidation and clearing
        Runs: Every Sunday 00:00 (midnight)
        
        Process:
        1. Read streaming files (7 days of 1-min data)
        2. Aggregate to 7 daily bars
        3. Append to permanent daily files
        4. CLEAR streaming files
        """
        try:
            logger.info("\n" + "="*70)
            logger.info("📅 WEEKLY CONSOLIDATION STARTED")
            logger.info(f"⏰ Time: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}")
            logger.info("="*70 + "\n")
            
            total_bars_added = 0
            success_count = 0
            
            for index_name, index_config in self.index_configs.items():
                try:
                    # Get symbol for file path
                    symbol = index_config.get('yahoo') or f"FALLBACK_{index_name}"
                    
                    bars_added = self.streaming_collector.consolidate_week_and_clear(symbol)
                    
                    if bars_added > 0:
                        total_bars_added += bars_added
                        success_count += 1
                        logger.info(f"✅ {index_name} ({symbol}): Added {bars_added} daily bars")
                    else:
                        logger.warning(f"⚠️  {index_name}: No data to consolidate")
                        
                except Exception as e:
                    logger.error(f"❌ {index_name}: Error during consolidation - {e}")
            
            logger.info("\n" + "="*70)
            logger.info(f"✅ WEEKLY CONSOLIDATION COMPLETE")
            logger.info(f"📊 Processed: {success_count}/{len(self.index_configs)} indices")
            logger.info(f"📈 Total daily bars added: {total_bars_added}")
            logger.info(f"🧹 Streaming files cleared and reset for next week")
            logger.info("="*70 + "\n")
            
        except Exception as e:
            logger.error(f"❌ Critical error in weekly consolidation: {e}")
    
    def print_status(self):
        """Print current status of all files"""
        logger.info("\n📊 Current System Status:")
        # Show first 5 symbols
        symbols_to_show = [cfg.get('yahoo') or f"FALLBACK_{name}" 
                          for name, cfg in list(self.index_configs.items())[:5]]
        self.streaming_collector.print_system_status(symbols_to_show)
    
    def setup_jobs(self):
        """Configure all scheduled jobs"""
        
        # JOB 1: Fetch minute data every minute during market hours
        # NSE trading hours: 9:15 AM - 3:30 PM IST (Monday-Friday)
        # Synced to start at :00 seconds of each minute
        self.scheduler.add_job(
            func=self.fetch_minute_data,
            trigger=CronTrigger(
                day_of_week='mon-fri',
                hour='9-15',
                minute='*',
                second='0',  # Start exactly at :00 seconds
                timezone=IST
            ),
            id='minute_fetch',
            name='Fetch 1-min data (synced to :00 seconds)',
            max_instances=1,
            replace_existing=True
        )
        logger.info("✅ Job 1: Minute data fetch (Mon-Fri, 9:00-15:30 IST, synced to :00 seconds)")
        
        # JOB 2: Weekly consolidation every Sunday midnight
        self.scheduler.add_job(
            func=self.weekly_consolidation,
            trigger=CronTrigger(
                day_of_week='sun',
                hour=0,
                minute=0,
                timezone=IST
            ),
            id='weekly_consolidation',
            name='Weekly consolidation and clearing',
            max_instances=1,
            replace_existing=True
        )
        logger.info("✅ Job 2: Weekly consolidation (Sunday 00:00 IST)")
        
        # Optional: Status check every hour
        self.scheduler.add_job(
            func=self.print_status,
            trigger=IntervalTrigger(hours=1),
            id='status_check',
            name='Status check',
            max_instances=1,
            replace_existing=True
        )
        logger.info("✅ Job 3: Status check (every hour)")
    
    def start(self):
        """Start the scheduler"""
        try:
            logger.info("\n" + "="*70)
            logger.info("🚀 STREAMING SCHEDULER STARTING")
            logger.info("="*70)
            
            # Setup directories
            setup_streaming_directories()
            
            # Setup jobs
            self.setup_jobs()
            
            # Print scheduled jobs
            logger.info("\n📅 Scheduled jobs:")
            for job in self.scheduler.get_jobs():
                logger.info(f"  • {job.name} (ID: {job.id})")
            
            logger.info("\n✅ Scheduler running... Press Ctrl+C to stop")
            logger.info("="*70 + "\n")
            
            # Start scheduler (blocking)
            self.scheduler.start()
            
            # Print next run times after scheduler starts
            logger.info("\n📅 Next run times:")
            for job in self.scheduler.get_jobs():
                logger.info(f"  • {job.name}: {job.next_run_time}")
            
        except KeyboardInterrupt:
            logger.info("\n⏹️  Scheduler stopped by user")
            self.scheduler.shutdown()
        except Exception as e:
            logger.error(f"❌ Scheduler error: {e}")
            self.scheduler.shutdown()


def test_immediate_fetch():
    """Test function: Fetch data immediately (without waiting for schedule)"""
    logger.info("\n🧪 TEST MODE: Immediate fetch")
    
    setup_streaming_directories()
    scheduler = StreamingScheduler()
    
    # Test with first 3 indices
    test_indices = dict(list(INDIAN_INDICES.items())[:3])
    logger.info(f"Testing with: {list(test_indices.keys())}")
    
    scheduler.index_configs = test_indices
    
    # Fetch data
    scheduler.fetch_minute_data()
    
    # Show status
    scheduler.print_status()


def test_immediate_consolidation():
    """Test function: Consolidate immediately"""
    logger.info("\n🧪 TEST MODE: Immediate consolidation")
    
    setup_streaming_directories()
    scheduler = StreamingScheduler()
    
    # Test with first 3 indices
    test_indices = dict(list(INDIAN_INDICES.items())[:3])
    scheduler.index_configs = test_indices
    
    # Consolidate
    scheduler.weekly_consolidation()
    
    # Show status
    scheduler.print_status()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "test-fetch":
            test_immediate_fetch()
        elif mode == "test-consolidate":
            test_immediate_consolidation()
        else:
            print("Unknown mode. Use: test-fetch or test-consolidate")
    else:
        # Production mode: Start scheduler
        scheduler = StreamingScheduler()
        scheduler.start()
