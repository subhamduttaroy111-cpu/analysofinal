import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import time
from flask import jsonify, request
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import STOCKS, MODE_CONFIG
from indicators import add_indicators
from strategies import (
    intraday_logic, swing_logic, longterm_logic,
    intraday_logic_ai, swing_logic_ai, longterm_logic_ai
)
from analyzer import generate_market_analysis
from news_fetcher import get_stock_news

# Load win rates from pre-calculated file
def load_win_rates():
    """Load pre-calculated win rates from JSON file."""
    try:
        win_rates_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'win_rates.json')
        if os.path.exists(win_rates_path):
            with open(win_rates_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading win rates: {e}")
    return None

WIN_RATES_DATA = load_win_rates()

# Request Cache to prevent duplicate downloads
SCAN_CACHE = {}
CACHE_TIMEOUT = 300  # 5 minutes

# ================= HELPER FUNCTIONS =================

def run_multi_timeframe_analysis(symbol):
    """Run AI strategies on all timeframes for a specific stock"""
    analysis = {}
    
    # Define modes: (Name, Period, Interval, Strategy Function)
    modes = [
        ("INTRADAY", "5d", "15m", intraday_logic_ai),
        ("SWING", "3mo", "1d", swing_logic_ai),
        ("LONG_TERM", "2y", "1d", longterm_logic_ai)
    ]
    
    for mode_name, period, interval, strategy_func in modes:
        try:
            # Download specific data for this mode
            df = yf.download(symbol, period=period, interval=interval, progress=False)
            
            # Fix for yfinance returning MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            if df.empty or len(df) < 10:
                analysis[mode_name] = {"available": False, "reason": "Insufficient Data"}
                continue
                
            # Add indicators
            df = add_indicators(df)
            
            # Run Strategy
            score, signal, reasons, sl, tgt, rr = strategy_func(df)
            
            analysis[mode_name] = {
                "available": True,
                "signal": signal,
                "score": score,
                "reasons": reasons
            }
            
        except Exception as e:
            print(f"Error in {mode_name} analysis for {symbol}: {e}")
            analysis[mode_name] = {"available": False, "error": str(e)}
            
    return analysis

# ================= API ROUTES =================

def register_routes(app):
    """Register all API routes"""
    
    @app.route('/scan', methods=['POST'])
    def scan():
        mode = request.json.get('mode')
        use_ai = request.json.get('use_ai', True)

        # Check Cache
        cache_key = f"{mode}_{use_ai}"
        current_time = time.time()
        if cache_key in SCAN_CACHE:
            cached_result, timestamp = SCAN_CACHE[cache_key]
            if current_time - timestamp < CACHE_TIMEOUT:
                print(f"⚡ Returning cached scan result for mode {mode} (Age: {current_time - timestamp:.1f}s)")
                return jsonify(cached_result)

        config = MODE_CONFIG.get(mode, MODE_CONFIG["INTRADAY"])
        period = config["period"]
        interval = config["interval"]
        min_data_points = config["min_data_points"]

        # ⚡ Safe serial download to save memory on 512MB RAM server
        print(f"⚡ Downloading {len(STOCKS)} stocks (threading enabled to speed up)...")
        data = yf.download(
            STOCKS,
            period=period,
            interval=interval,
            group_by='ticker',
            progress=False,
            threads=True # Enabled threading to speed up 494 stocks download
        )

        def process_stock(s):
            try:
                df = data[s].dropna()
                if len(df) < min_data_points:
                    return None

                df = add_indicators(df)

                if use_ai:
                    if mode == "INTRADAY":
                        score, bias, reasons, sl, tgt, rr = intraday_logic_ai(df)
                    elif mode == "SWING":
                        score, bias, reasons, sl, tgt, rr = swing_logic_ai(df)
                    else:
                        score, bias, reasons, sl, tgt, rr = longterm_logic_ai(df)
                else:
                    if mode == "INTRADAY":
                        score, bias, reasons, sl, tgt, rr = intraday_logic(df)
                    elif mode == "SWING":
                        score, bias, reasons, sl, tgt, rr = swing_logic(df)
                    else:
                        score, bias, reasons, sl, tgt, rr = longterm_logic(df)

                current_price = round(float(df['Close'].iloc[-1]), 2)

                if np.isnan(score): score = 0
                if np.isnan(rr): rr = 0
                if np.isnan(sl): sl = current_price * 0.95
                if np.isnan(tgt): tgt = current_price * 1.05

                if score >= 0 and rr >= 0:
                    return {
                        "symbol": s.replace(".NS", ""),
                        "ltp": current_price,
                        "bias": bias,
                        "score": score,
                        "reason": reasons,
                        "execution": {
                            "entry": current_price,
                            "sl": round(float(sl), 2),
                            "target1": round(float(tgt), 2),
                            "rr_ratio": round(float(rr), 2)
                        },
                        "indicators": {
                            "rsi": round(float(df['RSI'].iloc[-1]), 1),
                            "macd": "BUY" if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1] else "SELL",
                            "volume": "HIGH" if df['Volume_Ratio'].iloc[-1] > 1.2 else "NORMAL"
                        },
                        "last_updated": str(df.index[-1])
                    }
            except Exception as e:
                print(f"Error: {s} - {e}")
            return None

        # ⚡ Process 4 stocks simultaneously 
        results = []
        print(f"⚡ Processing {len(STOCKS)} stocks with 4 parallel workers...")
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(process_stock, s): s for s in STOCKS}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)

        final = sorted(results, key=lambda x: x['score'], reverse=True)[:5]
        print(f"✅ Scan done! {len(results)} signals found, top {len(final)} returned")

        mode_key = mode if mode != "LONG_TERM" else "LONG_TERM"
        win_rate_info = None
        if WIN_RATES_DATA and 'modes' in WIN_RATES_DATA:
            win_rate_info = WIN_RATES_DATA['modes'].get(mode_key)

        response_data = {
            "status": "success",
            "data": final,
            "win_rate": win_rate_info
        }

        # Save to Cache
        SCAN_CACHE[cache_key] = (response_data, current_time)

        return jsonify(response_data)

    @app.route('/get_stock_details', methods=['POST'])
    def details():
        symbol = request.json.get('symbol') + ".NS"
        stock = yf.Ticker(symbol)

        info = stock.info
        fundamentals = {
            "sector": info.get('sector', 'N/A'),
            "high52": info.get('fiftyTwoWeekHigh', 'N/A'),
            "low52": info.get('fiftyTwoWeekLow', 'N/A'),
            "marketCap": info.get('marketCap', 'N/A'),
            "pe": info.get('trailingPE', 'N/A')
        }

        return jsonify({
            "status": "success",
            "fundamentals": fundamentals,
            "ai_analysis": run_multi_timeframe_analysis(symbol)
        })
    
    @app.route('/get_news', methods=['POST'])
    def get_news():
        """Fetch recent news for a stock"""
        data = request.json
        symbol = data.get('symbol')
        
        # Fetch news
        news = get_stock_news(symbol + ".NS")
        
        if not news:
            return jsonify({
                "error": True,
                "message": "No news available for this stock"
            })
        
        return jsonify({
            "error": False,
            "news": news
        })

    # ═══════════════════════════════════════════════════════════
    # 3D GLOBE — News Sentiment API (ADDITIVE — new feature)
    # This route is completely independent of all existing logic.
    # ═══════════════════════════════════════════════════════════
    @app.route('/api/sentiment-globe', methods=['GET'])
    def sentiment_globe():
        """Return real-time news sentiment data for the 3D globe."""
        try:
            from sentiment_api import get_globe_data
            return jsonify(get_globe_data())
        except Exception as e:
            print(f"⚠️ Sentiment Globe error: {e}")
            return jsonify({"error": str(e), "hubs": [], "news_points": [], "summary": {}}), 500

    @app.route('/api/market-indices', methods=['GET'])
    def market_indices():
        """Fetch live prices for Nifty, Sensex, and BankNifty"""
        indices = {
            "NIFTY 50": "^NSEI",
            "SENSEX": "^BSESN",
            "BANKNIFTY": "^NSEBANK"
        }
        results = []
        try:
            for name, symbol in indices.items():
                ticker = yf.Ticker(symbol)
                todays_data = ticker.history(period='1d')
                if not todays_data.empty:
                    current_price = todays_data['Close'].iloc[0] # or -1
                    # To be safer with just 1d data:
                    current_price = todays_data['Close'].iloc[-1]
                    prev_close = ticker.info.get('previousClose', current_price)
                    
                    # Sometimes fast history doesn't quickly get previousClose from info
                    # Let's use 5d history to ensure we have the previous close robustly
                    hist = ticker.history(period='5d')
                    if len(hist) > 1:
                        current_price = hist['Close'].iloc[-1]
                        prev_close = hist['Close'].iloc[-2]
                        
                    change = current_price - prev_close
                    change_pct = (change / prev_close) * 100 if prev_close else 0
                    
                    results.append({
                        "name": name,
                        "price": round(current_price, 2),
                        "change": round(change, 2),
                        "change_pct": round(change_pct, 2),
                        "status": "BULLISH" if change >= 0 else "BEARISH"
                    })
            return jsonify({"status": "success", "data": results})
        except Exception as e:
            print(f"Error fetching indices: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500
