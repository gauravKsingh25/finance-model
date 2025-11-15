# test_fmp.py
"""Test Financial Modeling Prep API for Indian indices"""
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import requests
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pandas as pd

load_dotenv('../.env')

FMP_API_KEY = os.getenv('FMP_API_KEY')

print("="*70)
print("TESTING FMP (Financial Modeling Prep) API")
print("="*70)
print(f"API Key: {FMP_API_KEY[:10]}..." if FMP_API_KEY else "❌ No API Key")
print("="*70)

# Test symbols for Indian indices
test_symbols = {
    "BSE Sensex": ["^BSESN", "SENSEX", "BSE:SENSEX"],
    "Nifty 50": ["^NSEI", "NIFTY", "NSE:NIFTY"],
    "Nifty 500": ["NIFTY500", "^CNX500", "NSE:NIFTY500"],
    "BSE 100": ["BSE100", "^BSE100"],
    "BSE 500": ["BSE500", "^BSE500"],
}

def test_fmp_intraday(symbol):
    """Test FMP intraday data"""
    try:
        # Try 1-min intraday data
        url = f"https://financialmodelingprep.com/api/v3/historical-chart/1min/{symbol}"
        params = {'apikey': FMP_API_KEY}
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                return True, len(data), data[0] if data else None
        
        return False, 0, None
    except Exception as e:
        return False, 0, str(e)

print("\n📊 Testing 1-minute intraday data...")
print("-"*70)

working = []
for name, symbols in test_symbols.items():
    found = False
    for symbol in symbols:
        success, rows, sample = test_fmp_intraday(symbol)
        if success:
            print(f"✅ {name:25s} - {symbol:20s} ({rows} bars)")
            if sample:
                print(f"   Sample: {sample}")
            working.append((name, symbol))
            found = True
            break
        else:
            print(f"❌ {name:25s} - {symbol:20s} (failed)")
    
    if not found:
        print(f"⚠️  {name:25s} - No working symbol found")
    print()

print("="*70)
print(f"✅ Working: {len(working)}/{len(test_symbols)}")
print("="*70)

# Test what endpoints are available
print("\n🔍 Testing FMP endpoints...")
print("-"*70)

endpoints = {
    "Quote": f"https://financialmodelingprep.com/api/v3/quote/^NSEI",
    "Search": f"https://financialmodelingprep.com/api/v3/search?query=NIFTY",
    "Available Indices": f"https://financialmodelingprep.com/api/v3/symbol/available-indexes",
}

for name, url in endpoints.items():
    try:
        response = requests.get(url, params={'apikey': FMP_API_KEY}, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ {name:25s} - {len(data) if isinstance(data, list) else 'OK'}")
            if name == "Available Indices" and isinstance(data, list):
                indian_indices = [idx for idx in data if 'india' in str(idx).lower() or 'nse' in str(idx).lower() or 'bse' in str(idx).lower()]
                print(f"   Indian indices found: {len(indian_indices)}")
                for idx in indian_indices[:10]:
                    print(f"   • {idx}")
        else:
            print(f"❌ {name:25s} - Status {response.status_code}")
    except Exception as e:
        print(f"❌ {name:25s} - Error: {str(e)[:50]}")

print("\n" + "="*70)
