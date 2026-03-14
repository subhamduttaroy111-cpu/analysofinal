import time
import yfinance as yf
from backend.config import STOCKS

def test_download(threads):
    start = time.time()
    data = yf.download(STOCKS, period="5d", interval="15m", group_by="ticker", progress=False, threads=threads)
    end = time.time()
    print(f"Threads={threads}: downloaded {len(STOCKS)} stocks in {end - start:.2f} seconds")

if __name__ == "__main__":
    test_download(True)
    # test_download(False) # Skip false since it will be extremely slow
