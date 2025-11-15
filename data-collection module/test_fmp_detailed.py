import requests
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()

api_key = os.getenv('FMP_API_KEY')
print(f"API Key: {api_key[:10]}..." if api_key else "No API key found")
print(f"\n{'='*60}")

# Test 1: Check API key validity with a simple quote
print("\n1. Testing API Key Validity with Quote Endpoint:")
url = f"https://financialmodelingprep.com/api/v3/quote/%5ENSEI?apikey={api_key}"
response = requests.get(url)
print(f"URL: {url[:80]}...")
print(f"Status: {response.status_code}")
print(f"Response: {response.text[:200]}")

# Test 2: Search for available Indian indices
print(f"\n{'='*60}")
print("\n2. Searching for Indian Indices:")
url = f"https://financialmodelingprep.com/api/v3/search?query=nifty&limit=10&apikey={api_key}"
response = requests.get(url)
print(f"Status: {response.status_code}")
print(f"Response: {response.text[:300]}")

# Test 3: Get available markets/exchanges
print(f"\n{'='*60}")
print("\n3. Getting Available Exchanges:")
url = f"https://financialmodelingprep.com/api/v3/symbol/available-indexes?apikey={api_key}"
response = requests.get(url)
print(f"Status: {response.status_code}")
print(f"Response: {response.text[:300]}")

# Test 4: Try historical price (daily) instead of intraday
print(f"\n{'='*60}")
print("\n4. Testing Historical Daily Data (^NSEI):")
url = f"https://financialmodelingprep.com/api/v3/historical-price-full/%5ENSEI?apikey={api_key}"
response = requests.get(url)
print(f"Status: {response.status_code}")
print(f"Response: {response.text[:300]}")

# Test 5: Try different symbol formats
print(f"\n{'='*60}")
print("\n5. Testing Different Symbol Formats:")
symbols_to_test = [
    "NSEI",  # Without ^
    "^NSEI",  # With ^
    "NSE:NIFTY",  # Exchange prefix
    "NIFTY50",  # Alternative name
]

for symbol in symbols_to_test:
    url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}?apikey={api_key}"
    response = requests.get(url)
    print(f"\n{symbol}: Status {response.status_code} | Response: {response.text[:150]}")

# Test 6: Check account info/profile
print(f"\n{'='*60}")
print("\n6. Checking API Account Info:")
url = f"https://financialmodelingprep.com/api/v4/account_info?apikey={api_key}"
response = requests.get(url)
print(f"Status: {response.status_code}")
print(f"Response: {response.text}")

# Test 7: Try intraday with date range
print(f"\n{'='*60}")
print("\n7. Testing Intraday Chart (1min) with Dates:")
# Recent date range
to_date = datetime.now().strftime('%Y-%m-%d')
from_date = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')

url = f"https://financialmodelingprep.com/api/v3/historical-chart/1min/%5ENSEI?from={from_date}&to={to_date}&apikey={api_key}"
response = requests.get(url)
print(f"URL: {url[:100]}...")
print(f"Status: {response.status_code}")
print(f"Response: {response.text[:300]}")

print(f"\n{'='*60}")
print("\nDone!")
