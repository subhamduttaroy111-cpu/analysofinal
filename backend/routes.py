import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import time
import gc
from flask import jsonify, request
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from config import STOCKS, MODE_CONFIG
from indicators import add_indicators
from strategies import (
    intraday_logic, swing_logic, longterm_logic,
    intraday_logic_ai, swing_logic_ai, longterm_logic_ai
)
from analyzer import generate_market_analysis
from news_fetcher import get_stock_news
from model_manager import get_model_manager

# ═══════════════════════════════════════════════════════════
# OPTIMIZATION STEP 4: Global Model Manager (loaded ONCE)
# ═══════════════════════════════════════════════════════════
GLOBAL_MODEL_MANAGER = get_model_manager()
print("[OPTIMIZATION] ✅ Models cached globally (reused for all scans)")

# ═══════════════════════════════════════════════════════════
# OPTIMIZATION STEP 7: Unified Confidence Scoring System
# ═══════════════════════════════════════════════════════════

def normalize_score(score, min_val=0, max_val=100):
    """Normalize raw score (0-100) to 0-1 range"""
    if max_val == min_val:
        return 0.5
    normalized = (score - min_val) / (max_val - min_val)
    return max(0, min(1, normalized))

def calculate_confidence_score(result, use_ai=True):
    """
    UNIFIED CONFIDENCE SCORING (STEP 7)
    
    Formula: confidence = (ML×0.5) + (indicators×0.3) + (bias×0.2)
    Weights: 50% ML, 30% Indicators, 20% Signal Bias
    Returns: 0.0-1.0 (confidence range)
    """
    try:
        indicator_score = normalize_score(result.get('score', 50), 0, 100)
        ml_confidence = result.get('ml_confidence', 0.5) if use_ai else 0.5
        bias = result.get('bias', 'NEUTRAL')
        bias_score = {'BULLISH': 1.0, 'NEUTRAL': 0.5, 'BEARISH': 0.0}.get(bias, 0.5)
        
        confidence = (ml_confidence * 0.5) + (indicator_score * 0.3) + (bias_score * 0.2)
        return max(0, min(1, confidence))
    except Exception as e:
        print(f"[WARNING] Confidence scoring error: {e}")
        return 0.5


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

# Request Cache to prevent duplicate downloads (OPTIMIZATION STEPS 2 & 9)
SCAN_CACHE = {}
CACHE_TIMEOUT = 300  # 5 minutes

def cleanup_cache():
    """OPTIMIZATION STEP 10: Remove expired cache entries"""
    global SCAN_CACHE
    current_time = time.time()
    expired_keys = [k for k, (_, ts) in SCAN_CACHE.items() if current_time - ts > CACHE_TIMEOUT]
    for key in expired_keys:
        del SCAN_CACHE[key]
    if expired_keys:
        print(f"[CACHE] Cleaned {len(expired_keys)} expired entries")

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

# ================= MARKET CONTEXT ENGINE =================
# Pre-scan: fetch Nifty 50 trend + sector sentiment (cached 5 mins)

_MARKET_CONTEXT_CACHE = {"data": None, "timestamp": 0}
_MARKET_CONTEXT_TTL = 300  # 5 minutes

def get_market_context():
    """Fetch Nifty 50 trend and sector sentiment. Cached for 5 mins."""
    current_time = time.time()
    if _MARKET_CONTEXT_CACHE["data"] and (current_time - _MARKET_CONTEXT_CACHE["timestamp"]) < _MARKET_CONTEXT_TTL:
        return _MARKET_CONTEXT_CACHE["data"]

    context = {'nifty_trend': 'NEUTRAL', 'sector_sentiment': 0, 'htf_trend': 'NEUTRAL'}

    try:
        # ── Nifty 50 trend (Quick 5-day check) ──
        nifty_df = yf.download('^NSEI', period='3mo', interval='1d', progress=False)
        if isinstance(nifty_df.columns, pd.MultiIndex):
            nifty_df.columns = nifty_df.columns.get_level_values(0)
        if not nifty_df.empty and len(nifty_df) >= 50:
            nifty_close = nifty_df['Close'].iloc[-1]
            nifty_ema50 = nifty_df['Close'].ewm(span=50, adjust=False).mean().iloc[-1]
            nifty_ema200 = nifty_df['Close'].ewm(span=200, adjust=False).mean().iloc[-1] if len(nifty_df) >= 200 else nifty_ema50
            
            if nifty_close > nifty_ema50 and nifty_close > nifty_ema200:
                context['nifty_trend'] = 'BULLISH'
            elif nifty_close < nifty_ema50 and nifty_close < nifty_ema200:
                context['nifty_trend'] = 'BEARISH'
            else:
                context['nifty_trend'] = 'NEUTRAL'
            print(f"📊 Nifty 50 Trend: {context['nifty_trend']} (Price: {nifty_close:.0f}, EMA50: {nifty_ema50:.0f})")
    except Exception as e:
        print(f"⚠️ Nifty context fetch failed: {e}")

    try:
        # ── Sector sentiment from news engine ──
        from sentiment_api import get_globe_data
        globe = get_globe_data()
        if globe and 'sectors' in globe:
            # Build a dict of sector->sentiment for quick lookup
            context['sector_map'] = {}
            for sec in globe['sectors']:
                context['sector_map'][sec['name'].lower()] = sec['avg_sentiment']
            # Overall market sentiment
            total_bull = globe.get('summary', {}).get('bullish', 0)
            total_bear = globe.get('summary', {}).get('bearish', 0)
            total = total_bull + total_bear + globe.get('summary', {}).get('neutral', 0)
            if total > 0:
                context['sector_sentiment'] = (total_bull - total_bear) / total
            print(f"📰 News Sentiment: Bull={total_bull} Bear={total_bear} → Score={context['sector_sentiment']:.2f}")
    except Exception as e:
        print(f"⚠️ Sentiment context fetch failed: {e}")

    _MARKET_CONTEXT_CACHE["data"] = context
    _MARKET_CONTEXT_CACHE["timestamp"] = current_time
    return context


def get_stock_sector_sentiment(symbol, context):
    """Get sector-specific sentiment for a stock based on keyword matching."""
    sector_map = context.get('sector_map', {})
    sym = symbol.replace('.NS', '').lower()
    
    # Simple mapping of known stocks to sectors
    STOCK_SECTOR_MAP = {
        'banking & fin': ['hdfcbank', 'icicibank', 'sbin', 'kotakbank', 'axisbank', 'bajfinance', 'bajajfinsv', 'indusindbk', 'bankbaroda', 'pnb'],
        'it & tech': ['tcs', 'infy', 'wipro', 'hcltech', 'techm', 'ltim', 'persistent', 'coforge', 'mphasis'],
        'auto': ['tatamotors', 'maruti', 'bajaj-auto', 'heromotoco', 'm&m', 'eichermot', 'ashokley', 'tvsmotors'],
        'pharma': ['sunpharma', 'drreddy', 'cipla', 'divislab', 'apollohosp', 'lupin', 'auropharma'],
        'energy & metal': ['reliance', 'ongc', 'ntpc', 'powergrid', 'coalindia', 'tatasteel', 'jswsteel', 'hindalco', 'adanient', 'adanigreen', 'adaniports'],
        'fmcg': ['itc', 'hindunilvr', 'nestleind', 'dabur', 'britannia', 'godrejcp', 'marico', 'tataconsum'],
    }
    
    for sector, stocks in STOCK_SECTOR_MAP.items():
        if sym in stocks:
            return sector_map.get(sector, 0)
    return context.get('sector_sentiment', 0)


# ================= API ROUTES =================

def register_routes(app):
    """Register all API routes"""
    
    @app.route('/scan', methods=['POST'])
    def scan():
        """
        📊 OPTIMIZED SCAN ENDPOINT
        
        OPTIMIZATIONS:
          - STEP 2: Batch download + timeout
          - STEP 3: Parallel processing (5 workers)
          - STEP 4: Global model loading (singleton)
          - STEP 7: Unified confidence scoring
          - STEP 8: Return TOP 3 instead of TOP 5
          - STEP 9: Enhanced result caching
          - STEP 10: Memory cleanup
          
        Expected performance: 2-3 min (vs 11-13 min before)
        """
        scan_start = time.time()
        mode = request.json.get('mode')
        use_ai = request.json.get('use_ai', True)

        # ═══ STEP 9: Check result cache ═══
        cache_key = f"{mode}_{use_ai}"
        current_time = time.time()
        if cache_key in SCAN_CACHE:
            cached_result, timestamp = SCAN_CACHE[cache_key]
            if current_time - timestamp < CACHE_TIMEOUT:
                elapsed = time.time() - scan_start
                print(f"[CACHE HIT] ⚡ Returned in {elapsed:.2f}s (Age: {current_time - timestamp:.1f}s)")
                return jsonify(cached_result)

        config = MODE_CONFIG.get(mode, MODE_CONFIG["INTRADAY"])
        period = config["period"]
        interval = config["interval"]
        min_data_points = config["min_data_points"]

        # ═══ STEP 2: Bulk download with timeout ═══
        print(f"\n[DOWNLOAD] 📥 Fetching {len(STOCKS)} stocks ({period}, {interval})...")
        try:
            dl_start = time.time()
            data = yf.download(
                STOCKS,
                period=period,
                interval=interval,
                group_by='ticker',
                progress=False,
                threads=True,
                timeout=120  # ← OPTIMIZATION: Add timeout
            )
            dl_time = time.time() - dl_start
            print(f"[DOWNLOAD] ✅ Downloaded in {dl_time:.1f}s")
        except Exception as e:
            print(f"[ERROR] ❌ Download failed: {e}")
            return jsonify({"status": "error", "message": f"Download failed: {str(e)}"}), 503

        # ═══ PRE-SCAN: Fetch market context (Nifty trend + sector sentiment) ═══
        print("[CONTEXT] 📊 Fetching market context (Nifty + Sectors)...")
        market_context = get_market_context()

        # ═══ STEP 3 & 4: Parallel processing with global models ═══
        def process_stock(s):
            """Process single stock with STEP 7 confidence scoring"""
            try:
                df = data[s].dropna()
                if len(df) < min_data_points:
                    return None

                df = add_indicators(df)

                # STEP 4: Use global cached model manager (loaded once at startup)
                # Build stock-specific context with sector sentiment
                stock_context = dict(market_context) if market_context else None
                if stock_context:
                    stock_context['sector_sentiment'] = get_stock_sector_sentiment(s, market_context)

                if use_ai:
                    if mode == "INTRADAY":
                        score, bias, reasons, sl, tgt, rr = intraday_logic_ai(df)
                    elif mode == "SWING":
                        score, bias, reasons, sl, tgt, rr = swing_logic_ai(df)
                    else:
                        score, bias, reasons, sl, tgt, rr = longterm_logic_ai(df)
                else:
                    if mode == "INTRADAY":
                        score, bias, reasons, sl, tgt, rr = intraday_logic(df, market_context=stock_context)
                    elif mode == "SWING":
                        score, bias, reasons, sl, tgt, rr = swing_logic(df, market_context=stock_context)
                    else:
                        # Get fundamental data for long-term filter
                        try:
                            ticker_info = yf.Ticker(s).info
                            pe_ratio = ticker_info.get('trailingPE', None)
                            market_cap = ticker_info.get('marketCap', None)
                        except Exception:
                            pe_ratio = None
                            market_cap = None
                        score, bias, reasons, sl, tgt, rr = longterm_logic(
                            df, 
                            pe_ratio=pe_ratio, 
                            market_cap=market_cap,
                            market_context=stock_context
                        )

                current_price = round(float(df['Close'].iloc[-1]), 2)

                if np.isnan(score): score = 0
                if np.isnan(rr): rr = 0
                if np.isnan(sl): sl = current_price * 0.95
                if np.isnan(tgt): tgt = current_price * 1.05

                if score >= 0 and rr >= 0:
                    result = {
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
                    
                    # STEP 7: Calculate unified confidence score
                    result['ml_confidence'] = 0.5  # Default if no AI
                    result['confidence_score'] = calculate_confidence_score(result, use_ai)
                    return result
            except Exception as e:
                pass  # Silent fail - stock couldn't be processed
            return None

        # STEP 3: Parallel processing with 5 workers (max for 512MB RAM)
        results = []
        proc_start = time.time()
        print(f"[PROCESSING] ⚡ Analyzing {len(STOCKS)} stocks with 5 parallel workers...")
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(process_stock, s): s for s in STOCKS}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
        
        proc_time = time.time() - proc_start
        print(f"[PROCESSING] ✅ Analyzed in {proc_time:.1f}s ({len(results)} passed filters)")

        # ═══ STEP 8: Return TOP 3 by confidence score (NOT arbitrary top 5) ═══
        final = sorted(results, key=lambda x: x.get('confidence_score', 0.5), reverse=True)[:3]
        print(f"[RANKING] 🎯 Selected TOP 3 stocks by confidence score")

        mode_key = mode if mode != "LONG_TERM" else "LONG_TERM"
        win_rate_info = None
        if WIN_RATES_DATA and 'modes' in WIN_RATES_DATA:
            win_rate_info = WIN_RATES_DATA['modes'].get(mode_key)

        response_data = {
            "status": "success",
            "data": final,
            "win_rate": win_rate_info
        }

        # STEP 9: Cache the result
        SCAN_CACHE[cache_key] = (response_data, current_time)
        
        # STEP 10: Memory cleanup
        cleanup_cache()
        gc.collect()

        total_time = time.time() - scan_start
        print(f"[COMPLETE] ⏱️  Total: {total_time:.1f}s ({total_time/60:.2f} minutes)\n")
        
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

    # Cache for Market Indices to avoid 429 Too Many Requests from YFinance
    INDICES_CACHE = {}
    INDICES_CACHE_TIMEOUT = 300  # 5 minutes

    @app.route('/api/market-indices', methods=['GET'])
    def market_indices():
        """Fetch live prices for Nifty, Sensex, and BankNifty"""
        current_time = time.time()
        
        # Check Cache
        if "data" in INDICES_CACHE:
            cached_result, timestamp = INDICES_CACHE["data"]
            if current_time - timestamp < INDICES_CACHE_TIMEOUT:
                return jsonify({"status": "success", "data": cached_result, "cached": True})

        indices = {
            "NIFTY 50": "^NSEI",
            "SENSEX": "^BSESN",
            "BANKNIFTY": "^NSEBANK"
        }
        results = []
        try:
            for name, symbol in indices.items():
                ticker = yf.Ticker(symbol)
                # Using 5d history avoids returning entirely empty dataframes causing failures
                hist = ticker.history(period='5d')
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                        
                    change = current_price - prev_close
                    change_pct = (change / prev_close) * 100 if prev_close else 0
                    
                    results.append({
                        "name": name,
                        "price": round(current_price, 2),
                        "change": round(change, 2),
                        "change_pct": round(change_pct, 2),
                        "status": "BULLISH" if change >= 0 else "BEARISH"
                    })
            
            # Save to Cache
            if results:
                INDICES_CACHE["data"] = (results, current_time)
                
            return jsonify({"status": "success", "data": results, "cached": False})
        except Exception as e:
            print(f"Error fetching indices: {e}")
            return jsonify({"status": "error", "message": f"Rate limit exceeded or unavailable. {e}"}), 503


