import requests
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('FMP_API_KEY')
print(f"Testing FMP v4 API endpoints...")
print(f"{'='*60}\n")

# Test v4 endpoints (if available for free tier)
endpoints_to_test = [
    # V4 Quote
    ("v4/quote", "https://financialmodelingprep.com/api/v4/quotes/%5ENSEI"),
    
    # V4 Real-time quote
    ("v4/real-time-quote", "https://financialmodelingprep.com/api/v4/stock_price/%5ENSEI"),
    
    # V4 Intraday
    ("v4/intraday", "https://financialmodelingprep.com/api/v4/historical-price-intraday/%5ENSEI?interval=1min"),
    
    # Check what's available on free tier
    ("Available features", "https://financialmodelingprep.com/api/v4/available-traded/list"),
]

for name, url in endpoints_to_test:
    full_url = f"{url}?apikey={api_key}"
    print(f"\n{name}:")
    print(f"URL: {url[:80]}")
    
    try:
        response = requests.get(full_url, timeout=10)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            print(f"✅ SUCCESS!")
            print(f"Response: {response.text[:300]}")
        else:
            error_text = response.text[:200]
            if "upgrade" in error_text.lower() or "subscription" in error_text.lower():
                print(f"💰 REQUIRES PAID PLAN")
            print(f"Response: {error_text}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    print(f"-" * 60)

print("\n\nConclusion:")
print("FMP free tier no longer supports historical/intraday data.")
print("You need a paid subscription or we should use alternative sources.")
