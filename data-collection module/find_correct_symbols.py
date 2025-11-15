# find_correct_symbols.py
"""
Find correct Yahoo Finance symbols for Indian indices
"""
import yfinance as yf
from datetime import datetime, timedelta
import sys

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Test various symbol formats
test_symbols = {
    "BSE Sensex": ["^BSESN", "SENSEX.BO", "BSE"],
    "BSE 100": ["^BSE100", "BSE-100", "BSE100.BO"],
    "BSE 200": ["^BSE200", "BSE-200", "BSE200.BO"],
    "BSE 500": ["^BSE500", "BSE-500", "BSE500.BO"],
    "BSE MidCap": ["^BSEMID", "BSEMIDCAP.BO", "BSE-MIDCAP"],
    "BSE SmallCap": ["^BSESML", "BSESMALLCAP.BO", "BSE-SMALLCAP"],
    
    "Nifty 50": ["^NSEI", "NIFTY50.NS", "NIFTY"],
    "Nifty 100": ["^CNX100", "NIFTY100.NS"],
    "Nifty 200": ["^CNX200", "NIFTY200.NS"],
    "Nifty 500": ["^CNX500", "NIFTY500.NS"],
    "Nifty Next 50": ["^NIFTYJR", "NIFTYNEXT50.NS", "NIFTYJR.NS"],
    "Nifty Midcap 50": ["^NSEMDCP50", "NIFTYMIDCAP50.NS"],
    "Nifty Midcap 100": ["^CNXMIDCAP", "NIFTYMIDCAP100.NS"],
    "Nifty Midcap 150": ["^NIFTYMIDCAP150", "NIFTYMIDCAP150.NS"],
    "Nifty Smallcap 50": ["^NIFTYSMLCAP50", "NIFTYSMLCAP50.NS"],
    "Nifty Smallcap 100": ["^CNXSMALLCAP", "NIFTYSMLCAP100.NS"],
    "Nifty Smallcap 250": ["^NIFTYSMLCAP250", "NIFTYSMLCAP250.NS"],
    "Nifty 50 Value 20": ["^NIFTY50VALUE20", "NIFTYVALUE20.NS"],
    "Nifty 50 USD": ["^NIFTY50USD", "NIFTY50USD.NS"],
    "India VIX": ["^INDIAVIX", "INDIAVIX.NS"],
}

def test_symbol(symbol):
    """Test if a symbol works with yfinance"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=2)
        
        df = yf.download(symbol, interval='1d', start=start_date, end=end_date, progress=False)
        
        if df is not None and not df.empty:
            return True, len(df)
        return False, 0
    except:
        return False, 0

print("="*70)
print("FINDING CORRECT YAHOO FINANCE SYMBOLS")
print("="*70)

working_symbols = {}

for name, symbols in test_symbols.items():
    print(f"\n{name}:")
    found = False
    for symbol in symbols:
        works, rows = test_symbol(symbol)
        if works:
            print(f"  ✅ {symbol} - WORKS ({rows} rows)")
            working_symbols[name] = symbol
            found = True
            break
        else:
            print(f"  ❌ {symbol} - FAILED")
    
    if not found:
        print(f"  ⚠️  NO WORKING SYMBOL FOUND")

print("\n" + "="*70)
print("SUMMARY - WORKING SYMBOLS")
print("="*70)
for name, symbol in working_symbols.items():
    print(f"{name:25s} -> {symbol}")

print(f"\n✅ Found {len(working_symbols)}/20 working symbols")
