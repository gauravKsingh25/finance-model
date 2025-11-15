# verify_fallback_sources.py
"""
Verify which indices are available in NSEpy and Twelve Data
"""
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv('../.env')

# Test indices that failed on Yahoo Finance
failed_indices = {
    "BSE_100": {
        "nselib": "BSE100",
        "twelvedata_symbols": ["BSE:BSE100", "BSE100", "SENSEX100"]
    },
    "BSE_200": {
        "nselib": "BSE200", 
        "twelvedata_symbols": ["BSE:BSE200", "BSE200"]
    },
    "BSE_500": {
        "nselib": "BSE500",
        "twelvedata_symbols": ["BSE:BSE500", "BSE500"]
    },
    "BSE_MIDCAP": {
        "nselib": "BSEMID",
        "twelvedata_symbols": ["BSE:MIDCAP", "BSEMIDCAP"]
    },
    "BSE_SMALLCAP": {
        "nselib": "BSESML",
        "twelvedata_symbols": ["BSE:SMALLCAP", "BSESMALLCAP"]
    },
    "NIFTY_500": {
        "nselib": "NIFTY 500",
        "twelvedata_symbols": ["NIFTY500", "NSE:NIFTY500"]
    },
    "NIFTY_NEXT_50": {
        "nselib": "NIFTY NEXT 50",
        "twelvedata_symbols": ["NIFTYJR", "NSE:NIFTYJR", "NIFTYNEXT50"]
    },
    "NIFTY_MIDCAP_100": {
        "nselib": "NIFTY MIDCAP 100",
        "twelvedata_symbols": ["NIFTYMIDCAP100", "NSE:NIFTYMIDCAP100"]
    },
    "NIFTY_SMALLCAP_100": {
        "nselib": "NIFTY SMALLCAP 100",
        "twelvedata_symbols": ["NIFTYSMLCAP100", "NSE:NIFTYSMLCAP100"]
    },
    "NIFTY_50_VALUE_20": {
        "nselib": "NIFTY50 VALUE 20",
        "twelvedata_symbols": ["NIFTY50VALUE20", "NSE:NIFTY50VALUE20"]
    },
    "NIFTY_50_USD": {
        "nselib": None,
        "twelvedata_symbols": ["NIFTY50USD", "NSE:NIFTY50USD"]
    },
}

print("="*70)
print("VERIFYING FALLBACK DATA SOURCES")
print("="*70)

results = {}

# Test NSEpy (daily data only)
print("\n1. Testing NSEpy (Daily Data)...")
print("-"*70)
try:
    from nsepy import get_history
    from nsepy.history import get_index_list
    
    # Get available indices
    try:
        available_indices = get_index_list()
        print(f"✅ NSEpy available. Found {len(available_indices)} indices")
    except:
        available_indices = []
        print("⚠️  Could not get index list from NSEpy")
    
    for name, config in failed_indices.items():
        if config['nselib']:
            try:
                end = datetime.now()
                start = end - timedelta(days=5)
                df = get_history(symbol=config['nselib'], start=start, end=end, index=True)
                
                if df is not None and not df.empty:
                    print(f"  ✅ {name:25s} - NSEpy: {config['nselib']:20s} ({len(df)} rows)")
                    if name not in results:
                        results[name] = {}
                    results[name]['nselib'] = config['nselib']
                    results[name]['nselib_works'] = True
                else:
                    print(f"  ❌ {name:25s} - NSEpy: {config['nselib']:20s} (no data)")
            except Exception as e:
                print(f"  ❌ {name:25s} - NSEpy: {config['nselib']:20s} (error: {str(e)[:50]})")
        else:
            print(f"  ⊘  {name:25s} - NSEpy: Not available")
            
except ImportError:
    print("❌ NSEpy not installed")

# Test Twelve Data
print("\n2. Testing Twelve Data (1-min data)...")
print("-"*70)

api_key = os.getenv('TWELVEDATA_API_KEY')
if not api_key:
    print("❌ TWELVEDATA_API_KEY not found in .env")
else:
    try:
        from twelvedata import TDClient
        import time
        
        client = TDClient(apikey=api_key)
        print(f"✅ Twelve Data client initialized")
        
        for name, config in failed_indices.items():
            found = False
            for symbol in config['twelvedata_symbols']:
                try:
                    # Test with time series
                    ts = client.time_series(
                        symbol=symbol,
                        interval='1day',
                        outputsize=5
                    )
                    df = ts.as_pandas()
                    
                    if df is not None and not df.empty:
                        print(f"  ✅ {name:25s} - TwelveData: {symbol:20s} ({len(df)} rows)")
                        if name not in results:
                            results[name] = {}
                        results[name]['twelvedata'] = symbol
                        results[name]['twelvedata_works'] = True
                        found = True
                        break
                    
                except Exception as e:
                    error_msg = str(e)[:60]
                    if 'API credits' in error_msg:
                        print(f"  ⚠️  Rate limit hit, waiting 60 seconds...")
                        time.sleep(60)
                    continue
            
            if not found:
                print(f"  ❌ {name:25s} - TwelveData: No working symbol found")
                
    except ImportError:
        print("❌ twelvedata package not installed")
    except Exception as e:
        print(f"❌ Twelve Data error: {e}")

# Print summary
print("\n" + "="*70)
print("SUMMARY - WORKING FALLBACK SOURCES")
print("="*70)

working_count = 0
for name, sources in results.items():
    nse_status = "✅ NSEpy" if sources.get('nselib_works') else ""
    td_status = "✅ TwelveData" if sources.get('twelvedata_works') else ""
    
    if sources.get('nselib_works') or sources.get('twelvedata_works'):
        print(f"{name:25s} - {nse_status:15s} {td_status}")
        working_count += 1

print(f"\n✅ Found {working_count}/{len(failed_indices)} indices in fallback sources")

# List indices with NO fallback
print("\n" + "="*70)
print("INDICES TO REMOVE (No fallback available)")
print("="*70)

to_remove = []
for name in failed_indices.keys():
    if name not in results or (not results[name].get('nselib_works') and not results[name].get('twelvedata_works')):
        to_remove.append(name)
        print(f"  ❌ {name}")

print(f"\nTotal to remove: {len(to_remove)}")
