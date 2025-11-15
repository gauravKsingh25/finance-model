# indian_market_indices.py
"""
Indian Market Indices for Data Collection
Complete list from PNG - using Yahoo Finance (reliable & free)
FMP free tier deprecated Aug 31, 2025 - no longer viable

Status:
- ✅ Working: Available on Yahoo Finance with 1-minute data (9 indices)
- ❌ Unavailable: Not available on any free 1-minute data source (11 indices)

Note: Twelve Data has 8 calls/min limit, insufficient for 20 indices.
"""

INDIAN_INDICES = {
    # === WORKING ON YAHOO FINANCE (1-min data) ===
    "BSE_SENSEX": {
        "yahoo": "^BSESN",
        "fmp": "^BSESN",
        "nse": "BSE SENSEX",
        "exchange": "BSE",
        "sector": "Broad Market",
        "status": "✅ Working"
    },
    "NIFTY_50": {
        "yahoo": "^NSEI",
        "fmp": "^NSEI",
        "nse": "NIFTY 50",
        "exchange": "NSE",
        "sector": "Broad Market",
        "status": "✅ Working"
    },
    "NIFTY_100": {
        "yahoo": "^CNX100",
        "fmp": "^CNX100",
        "nse": "NIFTY 100",
        "exchange": "NSE",
        "sector": "Broad Market",
        "status": "✅ Working"
    },
    "NIFTY_200": {
        "yahoo": "^CNX200",
        "fmp": "^CNX200",
        "nse": "NIFTY 200",
        "exchange": "NSE",
        "sector": "Broad Market",
        "status": "✅ Working"
    },
    "NIFTY_MIDCAP_50": {
        "yahoo": "^NSEMDCP50",
        "fmp": "^NSEMDCP50",
        "nse": "NIFTY MIDCAP 50",
        "exchange": "NSE",
        "sector": "Midcap",
        "status": "✅ Working"
    },
    "NIFTY_MIDCAP_100": {
        "yahoo": "^CNXMIDCAP",
        "fmp": "^CNXMIDCAP",
        "nse": "NIFTY MIDCAP 100",
        "exchange": "NSE",
        "sector": "Midcap",
        "status": "🔄 Limited"
    },
    "NIFTY_MIDCAP_150": {
        "yahoo": "NIFTYMIDCAP150.NS",
        "fmp": "NIFTYMIDCAP150.NS",
        "nse": "NIFTY MIDCAP 150",
        "exchange": "NSE",
        "sector": "Midcap",
        "status": "✅ Working"
    },
    "NIFTY_SMALLCAP_50": {
        "yahoo": "NIFTYSMLCAP50.NS",
        "fmp": "NIFTYSMLCAP50.NS",
        "nse": "NIFTY SMALLCAP 50",
        "exchange": "NSE",
        "sector": "Smallcap",
        "status": "✅ Working"
    },
    "NIFTY_SMALLCAP_100": {
        "yahoo": "^CNXSMALLCAP",
        "fmp": "^CNXSMALLCAP",
        "nse": "NIFTY SMALLCAP 100",
        "exchange": "NSE",
        "sector": "Smallcap",
        "status": "🔄 Limited"
    },
    "NIFTY_SMALLCAP_250": {
        "yahoo": "NIFTYSMLCAP250.NS",
        "fmp": "NIFTYSMLCAP250.NS",
        "nse": "NIFTY SMALLCAP 250",
        "exchange": "NSE",
        "sector": "Smallcap",
        "status": "✅ Working"
    },
    "INDIA_VIX": {
        "yahoo": "^INDIAVIX",
        "fmp": "^INDIAVIX",
        "nse": "INDIA VIX",
        "exchange": "NSE",
        "sector": "Volatility",
        "status": "✅ Working"
    },
    "NIFTY_NEXT_50": {
        "yahoo": "^NIFTYJR",
        "fmp": "^NIFTYJR",
        "nse": "NIFTY NEXT 50",
        "exchange": "NSE",
        "sector": "Broad Market",
        "status": "🔄 Limited"
    },
    "NIFTY_500": {
        "yahoo": "^CNX500",
        "fmp": "^CNX500",
        "nse": "NIFTY 500",
        "exchange": "NSE",
        "sector": "Broad Market",
        "status": "🔄 Limited"
    },
    "NIFTY_50_VALUE_20": {
        "yahoo": "^NIFTY50VALUE20",
        "fmp": "^NIFTY50VALUE20",
        "nse": "NIFTY 50 VALUE 20",
        "exchange": "NSE",
        "sector": "Value",
        "status": "🔄 Limited"
    },
    "NIFTY_50_USD": {
        "yahoo": "^NIFTY50USD",
        "fmp": "^NIFTY50USD",
        "nse": "NIFTY 50 USD",
        "exchange": "NSE",
        "sector": "Broad Market",
        "status": "🔄 Limited"
    },
    
    # === BSE INDICES (Not available for 1-min on free sources) ===
    "BSE_100": {
        "yahoo": "^BSE100",
        "fmp": "^BSE100",
        "nse": "BSE-100",
        "exchange": "BSE",
        "sector": "Broad Market",
        "status": "❌ Unavailable"
    },
    "BSE_200": {
        "yahoo": "^BSE200",
        "fmp": "^BSE200",
        "nse": "BSE-200",
        "exchange": "BSE",
        "sector": "Broad Market",
        "status": "❌ Unavailable"
    },
    "BSE_500": {
        "yahoo": "^BSE500",
        "fmp": "^BSE500",
        "nse": "BSE-500",
        "exchange": "BSE",
        "sector": "Broad Market",
        "status": "❌ Unavailable"
    },
    "BSE_MIDCAP": {
        "yahoo": "^BSEMID",
        "fmp": "^BSEMID",
        "nse": "BSE MIDCAP",
        "exchange": "BSE",
        "sector": "Midcap",
        "status": "❌ Unavailable"
    },
    "BSE_SMALLCAP": {
        "yahoo": "^BSESML",
        "fmp": "^BSESML",
        "nse": "BSE SMALLCAP",
        "exchange": "BSE",
        "sector": "Smallcap",
        "status": "❌ Unavailable"
    },
}

# Filter function to get only working indices
def get_working_indices():
    """Get indices that reliably work on Yahoo Finance"""
    return {k: v for k, v in INDIAN_INDICES.items() if v["status"] == "✅ Working"}

def get_all_indices():
    """Get all indices (will attempt with fallbacks)"""
    return INDIAN_INDICES


def get_all_symbols():
    """Get list of all Yahoo Finance symbols"""
    return [info["yahoo"] for info in INDIAN_INDICES.values()]


def get_symbols_by_exchange(exchange):
    """Get symbols filtered by exchange (NSE or BSE)"""
    return {
        name: info 
        for name, info in INDIAN_INDICES.items() 
        if info["exchange"] == exchange
    }


def get_symbols_by_sector(sector):
    """Get symbols filtered by sector"""
    return {
        name: info 
        for name, info in INDIAN_INDICES.items() 
        if info["sector"] == sector
    }


def print_all_indices():
    """Print formatted list of all indices"""
    print("\n" + "="*70)
    print("INDIAN MARKET INDICES - COMPLETE LIST")
    print("="*70)
    
    # Group by exchange
    for exchange in ["NSE", "BSE"]:
        indices = get_symbols_by_exchange(exchange)
        print(f"\n{exchange} INDICES ({len(indices)} total):")
        print("-"*70)
        
        for name, info in indices.items():
            print(f"  • {name:30} {info['yahoo']:15} ({info['sector']})")
    
    print(f"\n{'='*70}")
    print(f"TOTAL INDICES: {len(INDIAN_INDICES)}")
    print("="*70 + "\n")


if __name__ == "__main__":
    print_all_indices()
    
    print("\nExample Usage:")
    print("-"*70)
    print(f"All symbols: {len(get_all_symbols())} indices")
    print(f"NSE only: {len(get_symbols_by_exchange('NSE'))} indices")
    print(f"BSE only: {len(get_symbols_by_exchange('BSE'))} indices")
    print(f"Banking sector: {len(get_symbols_by_sector('Banking'))} indices")
