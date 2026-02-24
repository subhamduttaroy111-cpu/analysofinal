# STOCKS list has been significantly reduced for Render deployment
# Render's free tier has a hard 100-second HTTP timeout limit.
# Scanning 100 stocks takes about 45-60 seconds, which is safely
# within the timeout limit. 

STOCKS = [
    # Top 50
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "HINDUNILVR.NS", "LT.NS",
    "BAJFINANCE.NS", "HCLTECH.NS", "MARUTI.NS", "SUNPHARMA.NS", "TATAMOTORS.NS",
    "M&M.NS", "TATASTEEL.NS", "POWERGRID.NS", "KOTAKBANK.NS", "ASIANPAINT.NS",
    "ONGC.NS", "NTPC.NS", "ULTRACEMCO.NS", "NESTLEIND.NS", "TITAN.NS",
    "ADANIENT.NS", "BAJAJFINSV.NS", "ADANIPORTS.NS", "WIPRO.NS", "HAL.NS",
    "GRASIM.NS", "TECHM.NS", "SBILIFE.NS", "BEL.NS", "PFC.NS",
    "HINDALCO.NS", "JSWSTEEL.NS", "ZOMATO.NS", "TRENT.NS", "SIEMENS.NS",
    "DRREDDY.NS", "TATACOMM.NS", "BAJAJ-AUTO.NS", "INDIGO.NS", "TATAPOWER.NS",
    "COALINDIA.NS", "DLF.NS", "CIPLA.NS", "EICHERMOT.NS", "CHOLAFIN.NS",
    
    # Next 50 (Nifty 100)
    "BRITANNIA.NS", "GAIL.NS", "APOLLOHOSP.NS", "HEROMOTOCO.NS", "PNB.NS",
    "BANKBARODA.NS", "TVSMOTOR.NS", "AMBUJACEM.NS", "BPCL.NS", "GODREJCP.NS",
    "SHRIRAMFIN.NS", "BOSCHLTD.NS", "MCDOWELL-N.NS", "HDFCLIFE.NS", "TATACONSUM.NS",
    "DABUR.NS", "INDUSINDBK.NS", "VEDL.NS", "HAVELLS.NS", "PIDILITIND.NS",
    "MAXHEALTH.NS", "ICICIPRULI.NS", "CUMMINSIND.NS", "COLPAL.NS", "LUPIN.NS",
    "IOC.NS", "UBL.NS", "CGPOWER.NS", "AUBANK.NS", "MARICO.NS",
    "VOLTAS.NS", "POLYCAB.NS", "MOTHERSON.NS", "SRF.NS", "TORNTPOWER.NS",
    "TORNTPHARM.NS", "JINDALSTEL.NS", "IDFCFIRSTB.NS", "CONCOR.NS", "MUTHOOTFIN.NS",
    "TIINDIA.NS", "PIIND.NS", "SUZLON.NS", "INDHOTEL.NS", "BHEL.NS",
    "TATAELXSI.NS", "PERSISTENT.NS", "DIXON.NS", "ASTRAL.NS", "OBEROIRLTY.NS"
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
