"""
Configuration file for ML trading pipeline.
Contains settings for all three timeframes: Intraday, Swing, Long-term
"""

import os
import pandas as pd
from typing import List, Dict, Any

# Get current directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# Load stock list from CSV
STOCKS_CSV_PATH = os.path.join(CURRENT_DIR, 'stocks.csv')

def load_stock_list() -> List[str]:
    """
    Load stock list from CSV file.
    
    Returns:
        List of stock symbols with .NS suffix for Yahoo Finance
    """
    try:
        df = pd.read_csv(STOCKS_CSV_PATH)
        return df['Yahoo_Symbol'].tolist()
    except FileNotFoundError:
        print(f"Error: {STOCKS_CSV_PATH} not found")
        return []

# Stock Universe - 200 NSE stocks
STOCK_LIST = load_stock_list()

# ========== INTRADAY CONFIGURATION ==========
INTRADAY_CONFIG: Dict[str, Any] = {
    'timeframe': '5m',
    'period': '60d',              # yfinance limitation
    'interval': '5m',
    'target_pct': 1.5,            # 1.5% target for intraday
    'sl_pct': 0.75,               # 0.75% stop loss
    'holding_period': '1d',        # Close by end of day
    'indicators': {
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'ema_periods': [9, 20, 50],
        'atr_period': 14,
        'volume_ma': 20,
        'bb_period': 20,
        'bb_std': 2
    },
    'features': [
        'rsi', 'macd_signal', 'macd_histogram', 'ema_9', 'ema_20', 'ema_50',
        'volume_ratio', 'atr', 'price_position', 'order_block_distance',
        'liquidity_grab', 'fvg_present', 'time_of_day', 'volatility',
        'trend_strength', 'support_resistance_proximity', 'risk_reward_ratio'
    ],
    'order_block_lookback': 20,    # Look back 20 candles for order blocks
    'liquidity_sensitivity': 0.001  # 0.1% for equal highs/lows detection
}

# ========== SWING CONFIGURATION ==========
SWING_CONFIG: Dict[str, Any] = {
    'timeframe': '1h',             # Primary: hourly candles
    'timeframe_secondary': '1d',   # Secondary: daily for context
    'period': '2y',                # 2 years of data available
    'interval': '1h',
    'target_pct': 5.0,             # 5% target for swing
    'sl_pct': 2.5,                 # 2.5% stop loss
    'holding_period': '15d',       # Hold up to 15 days
    'indicators': {
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'ema_periods': [20, 50, 200],
        'atr_period': 14,
        'volume_ma': 50,
        'bb_period': 20,
        'bb_std': 2
    },
    'features': [
        'rsi', 'macd_signal', 'macd_histogram', 'ema_20', 'ema_50', 'ema_200',
        'volume_ratio', 'atr', 'price_position', 'order_block_distance',
        'liquidity_grab', 'fvg_present', 'day_of_week', 'volatility',
        'trend_strength', 'support_resistance_proximity', 'risk_reward_ratio',
        'weekly_trend', 'sector_strength'
    ],
    'order_block_lookback': 100,   # Look back 100 candles for order blocks
    'liquidity_sensitivity': 0.002  # 0.2% for equal highs/lows detection
}

# ========== LONG-TERM CONFIGURATION ==========
LONGTERM_CONFIG: Dict[str, Any] = {
    'timeframe': '1d',             # Primary: daily candles
    'timeframe_secondary': '1wk',  # Secondary: weekly for context
    'period': '5y',                # 5+ years of data available
    'interval': '1d',
    'target_pct': 15.0,            # 15% target for long-term
    'sl_pct': 7.5,                 # 7.5% stop loss
    'holding_period': '365d',      # Hold up to 12 months
    'indicators': {
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'ema_periods': [50, 100, 200],
        'atr_period': 14,
        'volume_ma': 100,
        'bb_period': 20,
        'bb_std': 2
    },
    'features': [
        'rsi', 'macd_signal', 'macd_histogram', 'ema_50', 'ema_100', 'ema_200',
        'volume_ratio', 'atr', 'price_position', 'order_block_distance',
        'liquidity_grab', 'monthly_trend', 'volatility', 'trend_strength',
        'support_resistance_proximity', 'risk_reward_ratio', 'quarterly_trend',
        'yearly_high_low_position', 'fundamental_score'
    ],
    'order_block_lookback': 200,   # Look back 200 candles for order blocks
    'liquidity_sensitivity': 0.003  # 0.3% for equal highs/lows detection
}

# ========== MODEL TRAINING PARAMETERS ==========
MODEL_CONFIG: Dict[str, Any] = {
    'test_size': 0.15,
    'validation_size': 0.15,
    'random_state': 42,
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 15,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'random_state': 42,
        'n_jobs': -1
    },
    'xgboost': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 8,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    },
    'gradient_boosting': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 8,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'random_state': 42
    }
}

# ========== FILE PATHS ==========
PATHS: Dict[str, str] = {
    'raw_intraday': os.path.join(PROJECT_ROOT, 'data', 'raw', 'intraday'),
    'raw_swing': os.path.join(PROJECT_ROOT, 'data', 'raw', 'swing'),
    'raw_longterm': os.path.join(PROJECT_ROOT, 'data', 'raw', 'longterm'),
    'processed': os.path.join(PROJECT_ROOT, 'data', 'processed'),
    'models': os.path.join(PROJECT_ROOT, 'data', 'models'),
    'logs': os.path.join(CURRENT_DIR, 'logs')
}

# ========== DOWNLOAD SETTINGS ==========
DOWNLOAD_CONFIG: Dict[str, Any] = {
    'retry_attempts': 3,
    'retry_delay': 2,              # seconds
    'request_delay': 0.5,          # 500ms between requests to avoid rate limiting
    'timeout': 30,                 # seconds
    'columns_to_keep': ['Open', 'High', 'Low', 'Close', 'Volume']  # Optimize disk space
}

# ========== LOGGING SETTINGS ==========
LOGGING_CONFIG: Dict[str, Any] = {
    'log_level': 'INFO',
    'log_format': '%(asctime)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S'
}

# Validation
if __name__ == "__main__":
    print(f"Loaded {len(STOCK_LIST)} stocks")
    print(f"First 5 stocks: {STOCK_LIST[:5]}")
    print(f"\nIntraday features: {len(INTRADAY_CONFIG['features'])}")
    print(f"Swing features: {len(SWING_CONFIG['features'])}")
    print(f"Long-term features: {len(LONGTERM_CONFIG['features'])}")
    print(f"\nAll paths configured:")
    for key, path in PATHS.items():
        print(f"  {key}: {path}")
