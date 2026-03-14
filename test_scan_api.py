import time
import requests

def test_scan():
    url = "http://localhost:5001/scan"
    payload = {"mode": "INTRADAY", "use_ai": True}
    
    # First request
    print("Sending first request (should take ~40-60 seconds)...")
    start = time.time()
    try:
        response = requests.post(url, json=payload, timeout=120)
        end = time.time()
        print(f"First request took: {end - start:.2f} seconds")
        print(f"Status Code: {response.status_code}")
    except Exception as e:
        print(f"First request failed: {e}")

    # Second request (should be cached)
    print("\nSending second request (should take < 1 second)...")
    start = time.time()
    try:
        response = requests.post(url, json=payload, timeout=10)
        end = time.time()
        print(f"Second request took: {end - start:.2f} seconds")
        print(f"Status Code: {response.status_code}")
    except Exception as e:
        print(f"Second request failed: {e}")

if __name__ == "__main__":
    test_scan()
