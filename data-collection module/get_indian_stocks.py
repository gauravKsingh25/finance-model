"""
Fetch complete list of Indian stocks from NSE/BSE
Creates a comprehensive stock list for data collection
"""

import pandas as pd
import requests
from datetime import datetime
import json

def get_nse_stocks():
    """Get all NSE stocks from official NSE website"""
    print("📊 Fetching NSE stocks...")
    
    try:
        # NSE equity list endpoint
        url = "https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O"
        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json'
        }
        
        # Get F&O stocks first (most liquid)
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            stocks = []
            for item in data.get('data', []):
                stocks.append({
                    'symbol': item.get('symbol'),
                    'name': item.get('meta', {}).get('companyName', item.get('symbol')),
                    'exchange': 'NSE',
                    'yahoo': f"{item.get('symbol')}.NS",
                    'nse': item.get('symbol'),
                    'sector': 'FNO'  # F&O stocks
                })
            print(f"✅ Found {len(stocks)} F&O stocks from NSE")
            return stocks
    except Exception as e:
        print(f"⚠️ NSE API error: {e}")
    
    # Fallback: Use a comprehensive list of major NSE stocks
    print("📋 Using comprehensive NSE stock list...")
    return get_major_nse_stocks()

def get_major_nse_stocks():
    """Comprehensive list of major NSE stocks (NIFTY 500 constituents)"""
    # Top 100 most liquid NSE stocks
    major_stocks = [
        # NIFTY 50
        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR', 'ITC', 'SBIN',
        'BHARTIARTL', 'KOTAKBANK', 'LT', 'AXISBANK', 'ASIANPAINT', 'MARUTI', 'TITAN',
        'SUNPHARMA', 'ULTRACEMCO', 'BAJFINANCE', 'NESTLEIND', 'WIPRO', 'HCLTECH',
        'ADANIENT', 'ONGC', 'NTPC', 'POWERGRID', 'JSWSTEEL', 'TATAMOTORS', 'TATASTEEL',
        'COALINDIA', 'M&M', 'INDUSINDBK', 'BAJAJFINSV', 'DIVISLAB', 'TECHM', 'DRREDDY',
        'EICHERMOT', 'APOLLOHOSP', 'BRITANNIA', 'HINDALCO', 'CIPLA', 'GRASIM', 'BPCL',
        'HEROMOTOCO', 'TATACONSUM', 'ADANIPORTS', 'SHRIRAMFIN', 'UPL', 'LTIM', 'SBILIFE',
        'HDFCLIFE',
        
        # NIFTY NEXT 50
        'ADANIGREEN', 'ADANIPOWER', 'AMBUJACEM', 'ATGL', 'BAJAJHLDNG', 'BANDHANBNK',
        'BERGEPAINT', 'BEL', 'BOSCHLTD', 'CHOLAFIN', 'COLPAL', 'DABUR', 'DLF', 'DMART',
        'GODREJCP', 'GAIL', 'HDFCAMC', 'HAVELLS', 'ICICIPRULI', 'INDIGO', 'INDHOTEL',
        'JINDALSTEL', 'LUPIN', 'MARICO', 'MCDOWELL-N', 'MOTHERSON', 'NMDC', 'PAGEIND',
        'PETRONET', 'PIDILITIND', 'PNB', 'RECLTD', 'SBICARD', 'SIEMENS', 'TATAPOWER',
        'TVSMOTOR', 'TORNTPHARM', 'VEDL', 'ZOMATO', 'ZYDUSLIFE', 'NYKAA', 'PAYTM',
        'POLICYBZR', 'IRCTC', 'IRFC', 'PFC', 'IOC', 'CANBK', 'UNIONBANK',
        
        # Additional major stocks
        'BAJAJ-AUTO', 'BIOCON', 'SAIL', 'GMRINFRA', 'LICHSGFIN', 'MPHASIS', 'PERSISTENT',
        'VOLTAS', 'ABB', 'ESCORTS', 'TRENT', 'BANKBARODA', 'IDBI', 'YESBANK', 'IDFCFIRSTB',
        'GODREJPROP', 'PRESTIGE', 'OBEROIRLTY', 'PHOENIXLTD', 'MFSL', 'ABCAPITAL',
        'AUROPHARMA', 'SUNPHARMA', 'ALKEM', 'LAURUSLABS', 'AIAENG', 'ASTRAL', 'CUMMINSIND',
        'HONAUT', 'COFORGE', 'DIXON', 'MINDTREE', 'LTTS', 'INFY', 'OFSS', 'CYIENT',
    ]
    
    stocks = []
    for symbol in sorted(set(major_stocks)):  # Remove duplicates
        stocks.append({
            'symbol': symbol,
            'name': symbol,
            'exchange': 'NSE',
            'yahoo': f"{symbol}.NS",
            'nse': symbol,
            'sector': 'Equity'
        })
    
    print(f"✅ Loaded {len(stocks)} major NSE stocks")
    return stocks

def get_bse_stocks():
    """Get major BSE stocks (SENSEX + BSE 100 constituents)"""
    print("📊 Fetching BSE stocks...")
    
    # Major BSE stocks (SENSEX 30 + top BSE 100)
    bse_stocks = {
        'RELIANCE': '500325',
        'TCS': '532540',
        'HDFCBANK': '500180',
        'INFY': '500209',
        'ICICIBANK': '532174',
        'HINDUNILVR': '500696',
        'ITC': '500875',
        'SBIN': '500112',
        'BHARTIARTL': '532454',
        'KOTAKBANK': '500247',
        'LT': '500510',
        'AXISBANK': '532215',
        'ASIANPAINT': '500820',
        'MARUTI': '532500',
        'TITAN': '500114',
        'SUNPHARMA': '524715',
        'ULTRACEMCO': '532538',
        'BAJFINANCE': '500034',
        'NESTLEIND': '500790',
        'WIPRO': '507685',
        'HCLTECH': '532281',
        'NTPC': '532555',
        'POWERGRID': '532898',
        'JSWSTEEL': '500228',
        'TATAMOTORS': '500570',
        'TATASTEEL': '500470',
        'COALINDIA': '533278',
        'M&M': '500520',
        'INDUSINDBK': '532187',
    }
    
    stocks = []
    for name, bse_code in bse_stocks.items():
        stocks.append({
            'symbol': name,
            'name': name,
            'exchange': 'BSE',
            'yahoo': f"{name}.BO",
            'bse': bse_code,
            'sector': 'Equity'
        })
    
    print(f"✅ Loaded {len(stocks)} BSE stocks")
    return stocks

def create_stock_universe():
    """Create comprehensive stock universe combining NSE and BSE"""
    print("=" * 60)
    print("Creating Indian Stock Universe")
    print("=" * 60)
    
    # Get stocks from both exchanges
    nse_stocks = get_nse_stocks()
    bse_stocks = get_bse_stocks()
    
    # Combine and deduplicate
    all_stocks = {}
    
    # Add NSE stocks (prefer NSE over BSE for liquidity)
    for stock in nse_stocks:
        symbol = stock['symbol']
        all_stocks[symbol] = stock
    
    # Add BSE stocks (only if not in NSE)
    for stock in bse_stocks:
        symbol = stock['symbol']
        if symbol not in all_stocks:
            all_stocks[symbol] = stock
    
    stocks_list = list(all_stocks.values())
    
    # Sort by symbol
    stocks_list = sorted(stocks_list, key=lambda x: x['symbol'])
    
    print(f"\n{'=' * 60}")
    print(f"📊 Total unique stocks: {len(stocks_list)}")
    print(f"   NSE stocks: {len([s for s in stocks_list if s['exchange'] == 'NSE'])}")
    print(f"   BSE stocks: {len([s for s in stocks_list if s['exchange'] == 'BSE'])}")
    print(f"{'=' * 60}\n")
    
    # Save to JSON
    output_file = 'indian_stocks_universe.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stocks_list, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved to {output_file}")
    
    # Also create Python config file
    create_python_config(stocks_list)
    
    # Show sample
    print("\n📋 Sample stocks:")
    for stock in stocks_list[:10]:
        print(f"   {stock['symbol']:15} | {stock['yahoo']:20} | {stock['exchange']}")
    
    return stocks_list

def create_python_config(stocks_list):
    """Create Python configuration file for stocks"""
    config_content = '''# indian_stocks.py
"""
Indian Stock Universe Configuration
Auto-generated list of Indian stocks for data collection

Total Stocks: {total}
NSE Stocks: {nse_count}
BSE Stocks: {bse_count}
Generated: {timestamp}
"""

INDIAN_STOCKS = {{
'''.format(
        total=len(stocks_list),
        nse_count=len([s for s in stocks_list if s['exchange'] == 'NSE']),
        bse_count=len([s for s in stocks_list if s['exchange'] == 'BSE']),
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )
    
    for stock in stocks_list:
        config_content += f'''    '{stock['symbol']}': {{
        'name': '{stock['name']}',
        'exchange': '{stock['exchange']}',
        'yahoo': '{stock['yahoo']}',
        'nse': '{stock.get('nse', '')}',
        'sector': '{stock.get('sector', 'Equity')}'
    }},
'''
    
    config_content += '}\n'
    
    with open('indian_stocks.py', 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"✅ Saved to indian_stocks.py")

if __name__ == '__main__':
    stocks = create_stock_universe()
    print(f"\n✅ Stock universe created successfully!")
    print(f"   Use 'indian_stocks.py' for data collection")
