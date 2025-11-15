# test_scheduler.py
"""
Quick test to verify scheduler setup and fetch data immediately
"""

from streaming_scheduler import StreamingScheduler, setup_streaming_directories
from indian_market_indices import INDIAN_INDICES
import logging
import sys

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_scheduler():
    """Test the scheduler with all indices from the PNG"""
    
    print("\n" + "="*70)
    print("TESTING STREAMING SCHEDULER")
    print("="*70)
    
    # Setup directories
    setup_streaming_directories()
    
    # Create scheduler
    scheduler = StreamingScheduler()
    
    # Show all symbols that will be fetched
    print(f"\n📊 Total indices to fetch: {len(scheduler.index_configs)}")
    print("\nIndices list:")
    for i, (name, info) in enumerate(INDIAN_INDICES.items(), 1):
        yahoo_sym = info.get('yahoo') or 'FALLBACK'
        status = "✅ Yahoo" if info.get('yahoo_works') else "🔄 Fallback"
        print(f"  {i:2d}. {name:25s} - {yahoo_sym:25s} [{status}] ({info['nse']})")
    
    # Test immediate fetch
    print("\n" + "="*70)
    print("🧪 TESTING IMMEDIATE FETCH")
    print("="*70)
    
    scheduler.fetch_minute_data()
    
    # Show status
    print("\n" + "="*70)
    print("📊 SYSTEM STATUS AFTER FETCH")
    print("="*70)
    scheduler.print_status()
    
    print("\n" + "="*70)
    print("✅ TEST COMPLETE")
    print("="*70)

if __name__ == "__main__":
    test_scheduler()
