# ================== CONFIGURATION ==================

# STOCKS list has been significantly reduced for Render deployment
# Render's free tier has a hard 100-second HTTP timeout limit, and scanning
# 494 stocks was taking 3-5+ minutes, causing the frontend to time out.
# These are the Top 50 highly liquid Nifty/BankNifty stocks.

STOCKS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "HINDUNILVR.NS", "LT.NS",
    "BAJFINANCE.NS", "HCLTECH.NS", "MARUTI.NS", "SUNPHARMA.NS", "TATAMOTORS.NS",
    "M&M.NS", "TATASTEEL.NS", "POWERGRID.NS", "KOTAKBANK.NS", "ASIANPAINT.NS",
    "ONGC.NS", "NTPC.NS", "ULTRACEMCO.NS", "NESTLEIND.NS", "TITAN.NS",
    "ADANIENT.NS", "BAJAJFINSV.NS", "ADANIPORTS.NS", "WIPRO.NS", "HAL.NS",
    "GRASIM.NS", "TECHM.NS", "SBILIFE.NS", "BEL.NS", "PFC.NS",
    "HINDALCO.NS", "JSWSTEEL.NS", "ZOMATO.NS", "TRENT.NS", "SIEMENS.NS",
    "DRREDDY.NS", "TATACOMM.NS", "BAJAJ-AUTO.NS", "INDIGO.NS", "TATAPOWER.NS",
    "COALINDIA.NS", "DLF.NS", "CIPLA.NS", "EICHERMOT.NS", "CHOLAFIN.NS"
]

# Mode Configuration
MODE_CONFIG = {
    "INTRADAY": {
        "period": "5d",
        "interval": "15m",
        "min_data_points": 30
    },
    "SWING": {
        "period": "3mo",
        "interval": "1d",
        "min_data_points": 50
    },
    "LONG_TERM": {
        "period": "2y",
        "interval": "1d",
        "min_data_points": 150
    }
}
