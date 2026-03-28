# Analyso: Project Technical Report

## 1. PROJECT OVERVIEW
**Analyso** is a free, AI-powered stock market scanner built specifically for Indian markets (NSE and BSE). It aims to provide institutional-grade technical analysis and trading signals for retail traders. 

**Main Purpose and Features:**
- **Mode-based Scanning:** Offers predefined modes for Intraday (15m timeframe), Swing Trading (Daily), and Long-Term Investments (Weekly/Daily).
- **AI-Powered Analysis:** Combines classical technical indicators with Machine Learning (XGBoost) and Deep Learning (LSTM) to generate "Probability Matrix" scores for stocks.
- **Automated Market Analysis:** Integrates with Google's Gemini LLM to write professional, probabilistic market analysis mimicking human experts.
- **Global Sentiment 3D Globe:** Visualizes market sentiment across global financial hubs using real-time RSS feeds parsed through NLP (NLTK VADER).
- **Risk Management:** Calculates dynamic Stop Loss and Target prices using Average True Range (ATR) and factors in Risk-to-Reward ratios.

---

## 2. TECH STACK
- **Frontend Framework:** Vanilla HTML5, CSS3, and JavaScript. No overarching framework like React/Vue, relying on raw DOM manipulation.
- **3D Rendering:** `Globe.gl` (Three.js under the hood) for the interactive 3D sentiment map.
- **Backend Framework:** Python 3 with Flask (`gunicorn` used for production web server).
- **Machine Learning / AI:** 
  - `scikit-learn` & `xgboost` for ML Predictors.
  - `tensorflow` & `keras` for LSTM neural networks.
  - `nltk` (VADER lexicon) for news sentiment analysis.
  - Custom API Integration with Google `generativeai` (Gemini API).
- **Database / Authentication:** Google Firebase Auth (frontend client-side user management). No SQL/NoSQL database in use; relies on transient in-memory Python dictionaries for caching and static JSON files for persisting stats.
- **External Data APIs:** `yfinance` (Yahoo Finance) for historical bars, live quotes, and fundamentals. RSS Parsers (`feedparser`) for financial news. 

---

## 3. FILE STRUCTURE
**Root Directory**
- `requirements.txt` / `package.json`: Project dependencies for Python / Node.js (Firebase).

**Backend (`/backend`):**
- `server.py`: The Flask application entry point and CORS configuration.
- `routes.py`: Defines all API endpoints (`/scan`, `/get_stock_details`, `/get_news`, etc.).
- `analyzer.py`: Integrates with the Google Gemini API to return structured probabilistic text analysis.
- `strategies.py`: Source code for the trading logic. Contains both rule-based and AI-fallback logic functions for different timeframes.
- `indicators.py`: Helper functions to add technical indicators (EMA, MACD, RSI, ATR) to Pandas DataFrames.
- `model_manager.py`, `ml_predictor.py`, `lstm_predictor*.py`: Houses the architecture to load trained ML/LSTM models and evaluate live data.
- `sentiment_api.py`: Connects to multiple global financial RSS feeds and uses VADER for the 3D Globe's data.
- `config.py` & `config_models.py`: Stores lists of analyzed stock tickers and model settings.

**Frontend (`/frontend`):**
- `index.html`: The core scanner dashboard and SEO landing page.
- `globe.html`: Dedicated page rendering the 3D News Sentiment interface.
- `login.html`: Standalone layout for sign-in/up (also overlaid dynamically on index.html).
- `js/script.js`: Main application logic—handles API requests (fetch), rendering the UI grid, and managing modals.
- `js/globe.js`: Renders and updates the interactive `Globe.gl` component.
- `js/firebase-config.js` & `auth.js`: Handles Firebase initialization and user login flow.
- `css/styles.css` & `css/disclaimer.css`: Styles for the UI, maintaining a clean, glass-morphic, modern theme.

---

## 4. BACKEND ANALYSIS (Flask)

### Key Endpoints
1. **`POST /scan`**
   - **Purpose:** Screens a hardcoded list of stocks (`STOCKS` in config) and returns the top 5 setups based on score metrics.
   - **Logic:** Takes `mode` and `use_ai`. Utilizes Python's `ThreadPoolExecutor` to concurrently download data via `yfinance`. Passes data to the specified strategy in `strategies.py`. Results are grouped, sorted descending by score, and cached for 5 minutes in memory (`SCAN_CACHE`).
   
2. **`POST /get_stock_details`**
   - **Purpose:** Fetches in-depth data for a specific stock (e.g., clicked in the UI modal).
   - **Logic:** Calls `yfinance` for fundamental info (Sector, 52wk High/Low, PE, Market Cap) and runs a multi-timeframe analysis across `INTRADAY`, `SWING`, and `LONG_TERM`.

3. **`POST /get_news`**
   - **Purpose:** Retrieves the latest news headlines linked to a user-requested stock symbol via `feedparser`.

4. **`GET /api/sentiment-globe`**
   - **Purpose:** Returns categorized JSON for the frontend 3-D globe.
   - **Logic:** Parses RSS XML from Times, MoneyControl, CNBC, Yahoo, maps keywords to financial Hubs (e.g., Mumbai, Tokyo, New York), applies NLTK VADER sentiment scoring, and aggregates the counts. 

5. **`GET /api/market-indices`**
   - **Purpose:** Gets live price ticks for Nifty 50, Sensex, and BankNifty. Uses an aggressive 5-minute memory cache to prevent rate-limits from Yahoo Finance.

---

## 5. TRADING STRATEGY CODE

- **Indicators Used:** Moving Averages (`EMA 9, 21, 50, 200`), Momentum (`RSI`), Trend validation (`MACD Histogram & Signal lines`), Volatility (`Bollinger Bands`, `ATR`), and Volume (Ratio to moving average).
- **Signal Generation:**
   - **Rule-Based:** Stacking strict conditions. For instance, Swing Trading strictly requires a stacked EMA (`21 > 50 > 200`), positive MACD, and high Volume (`> 1.5x avg`). It penalizes overbought conditions (`RSI > 75`).
   - **AI-Based Fallback:** If `model_manager.is_available()` is true, it routes the DataFrame to an Ensemble ML/LSTM prediction engine which calculates raw probabilities for `BULLISH/BEARISH/NEUTRAL` arrays. High confidence boosts the base score.
- **Risk Management:** Uses Average True Range (ATR) multipliers to programmatically set Stop Losses and Targets (ex. SL = `price - ATR*1.2`, TGT = `price + ATR*2.5`) to gauge the Risk-Reward ratio.
- **Data Source:** Exclusively **yfinance** (Yahoo Finance) for free, delayed/live bars.
- **Accuracy / Backtesting:** There is no live backtesting engine built into the server. However, it pulls pre-calculated win rates from a static JSON (`data/win_rates.json`) to display "Historical Accuracy" in the UI.

---

## 6. FRONTEND ANALYSIS

- **Pages and Components:** 
   - A singular dashboard consisting of a Hero Section, Controls Section (dropdown to pick timeframe), and a highly dynamic Results Grid.
   - Modals are extensively used for deeper dives—featuring TradingView lightweight HTML iframes, dynamic AI-reasoning lists, and an embedded Fundamentals block.
- **Displaying Signals:** 
   - When a scan concludes, it loops through the JSON objects and generates "Stock Cards". If a score is exceptionally high, it marks it as "High Conviction".
   - Colors are rigorously used (Greens/Reds) dependent on the "Bias" (BULLISH/BEARISH).
- **Talking to the Backend:** 
   - Implemented natively using `async/await function` wrappers utilizing the `fetch()` API. It automatically points to the `BACKEND_URL` defined in `js/config.js` (`analysofinal-backend.onrender.com`).

---

## 7. CURRENT PROBLEMS & RISKS

1. **Massive `yfinance` Dependency & Rate Limits:** 
   - Calling `yfinance.download(494 stocks)` simultaneously on every cache miss is extremely dangerous. Yahoo Finance is notorious for silently blacklisting IPs or rate-limiting servers that make bulk concurrent queries.
2. **Performance / Resource Exhaustion:** 
   - A 512MB RAM free-tier Render server might struggle to hold 500 parallel Pandas DataFrames loaded with technical indicator logic in memory, plus bulky LSTM/XGBoost models sequentially running predictions.
   - Initial cold starts have severe latency. The UI warns it can take "8-10 minutes".
3. **Absence of a Real Database:** 
   - Uses completely in-memory variables (`SCAN_CACHE = {}`). If Render spins down the server (which free tiers do after 15 minutes of inactivity), all caches are obliterated. Cold hits from different workers on multi-threaded servers will duplicate API calls. 
4. **Security Vulnerabilities:**
   - The API endpoints (like `/scan`) have absolutely no token verification or JWT checks. A malicious actor could easily script a loop to hit `/scan` indefinitely, blowing out server memory and guaranteeing a Yahoo Finance IP ban.

---

## 8. DEPLOYMENT

- **Backend (Render):**
  - Hosted as a Render Web Service. 
  - Uses `server.py` wrapped inside `waitress` or `gunicorn` (as noted in requirements). 
  - Render handles the Python environment. Concurrency constraints (like `$OMP_NUM_THREADS = 1`) are hard-coded in `server.py` to prevent CPU exhaustion.
- **Frontend (Vercel):**
  - Hosted purely as static architecture (HTML/CSS/JS). Vercel acts as a lightning-fast CDN. 
  - Cross-Origin Resource Sharing (CORS) is enabled on the backend to accept traffic from Vercel.
- **Environment Variables:**
  - `PORT`: Tells Flask which port to bind to.
  - `ALLOWED_ORIGINS`: Used by Flask-CORS to restrict incoming domains.
  - `GEMINI_API_KEY`: Used by `analyzer.py` to prompt the Google generative model.
  - `NLTK_DATA`: Explicit local path overriding to `/tmp` in serverless instances so Vader Lexicon can download properly.
