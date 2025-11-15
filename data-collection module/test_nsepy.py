# test_nsepy.py
"""Test NSEpy capabilities for the failing indices"""
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from nsepy import get_history
from datetime import datetime, timedelta

# Indices that failed on Yahoo Finance
test_indices = {
    "BSE 100": "BSE100",
    "BSE 200": "BSE200", 
    "BSE 500": "BSE500",
    "BSE MidCap": "BSEMID",
    "BSE SmallCap": "BSESML",
    "Nifty 500": "NIFTY 500",
    "Nifty Next 50": "NIFTY NEXT 50",
    "Nifty Midcap 100": "NIFTY MIDCAP 100",
    "Nifty Smallcap 100": "NIFTY SMALLCAP 100",
    "Nifty 50 Value 20": "NIFTY50 VALUE 20",
}

print("="*70)
print("TESTING NSEpy FOR FAILING INDICES")
print("="*70)
print("\nNOTE: NSEpy only provides DAILY data, not 1-minute intraday data")
print("="*70)

end_date = datetime.now()
start_date = end_date - timedelta(days=7)

working_indices = []
failed_indices = []

for name, symbol in test_indices.items():
    try:
        df = get_history(symbol=symbol, start=start_date, end=end_date, index=True)
        
        if df is not None and not df.empty:
            print(f"✅ {name:25s} ({symbol:25s}) - {len(df)} days of data")
            print(f"   Date range: {df.index[0].date()} to {df.index[-1].date()}")
            print(f"   Last Close: {df['Close'].iloc[-1]:.2f}")
            working_indices.append((name, symbol))
        else:
            print(f"❌ {name:25s} ({symbol:25s}) - No data returned")
            failed_indices.append((name, symbol))
            
    except Exception as e:
        print(f"❌ {name:25s} ({symbol:25s}) - Error: {str(e)[:60]}")
        failed_indices.append((name, symbol))
    print()

print("="*70)
print("SUMMARY")
print("="*70)
print(f"✅ Working in NSEpy: {len(working_indices)}/{len(test_indices)}")
print(f"❌ Failed: {len(failed_indices)}/{len(test_indices)}")

if working_indices:
    print("\n📊 Working indices (DAILY data only):")
    for name, symbol in working_indices:
        print(f"  • {name:25s} - {symbol}")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("NSEpy can provide DAILY data for these indices, but:")
print("  ⚠️  NO 1-minute intraday data (only daily OHLC)")
print("  ⚠️  Cannot be used for real-time streaming every minute")
print("  ✅  Can be used for end-of-day consolidation only")
print("="*70)
