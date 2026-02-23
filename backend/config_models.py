"""
AI Model Configuration
Contains hyperparameters, paths, and settings for ML and DL models
"""

import os

# Base directory for models
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'data', 'models')

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# Model file paths
INTRADAY_MODEL_PATH = os.path.join(MODELS_DIR, 'intraday_model.pkl')
SWING_MODEL_PATH = os.path.join(MODELS_DIR, 'swing_model.pkl')
LONGTERM_MODEL_PATH = os.path.join(MODELS_DIR, 'longterm_model.pkl')

# Legacy paths (keep for reference or remove if sure)
FEATURE_SCALER_PATH = os.path.join(MODELS_DIR, 'feature_scaler.pkl')
LSTM_MODEL_PATH = os.path.join(MODELS_DIR, 'lstm_model.keras')

# Training data configuration
TRAINING_CONFIG = {
    'lookback_period': '2y',  # Historical data to fetch
    'test_size': 0.2,  # 80-20 train-test split
    'random_state': 42,
    
    # Label thresholds (for auto-labeling)
    'bullish_threshold': 0.03,  # 3% gain = BULLISH
    'bearish_threshold': -0.01,  # -1% loss = BEARISH
    'forward_days': 5,  # Look 5 days ahead for labeling
}

# XGBoost hyperparameters
XGBOOST_PARAMS = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'objective': 'multi:softmax',
    'num_class': 3,  # BULLISH, NEUTRAL, BEARISH
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'eval_metric': 'mlogloss',
}

# Random Forest hyperparameters
RANDOM_FOREST_PARAMS = {
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1,  # Use all CPU cores
}

# LSTM hyperparameters
LSTM_PARAMS = {
    'sequence_length': 60,  # Use 60 time steps for prediction
    'units': [128, 64, 32],  # 3-layer LSTM
    'dropout': 0.3,
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 0.001,
    'validation_split': 0.2,
}

# Feature names (extracted from technical indicators)
FEATURE_NAMES = [
    # Price features
    'close', 'open', 'high', 'low', 'volume',
    
    # Moving averages
    'ema_9', 'ema_21', 'ema_50', 'ema_200',
    'sma_20',
    
    # Momentum indicators
    'rsi', 'macd', 'macd_signal', 'macd_hist', 'momentum',
    
    # Volatility
    'atr', 'bb_upper', 'bb_lower', 'bb_middle',
    
    # Volume
    'volume_ratio',
    
    # Derived features
    'price_to_ema9', 'price_to_ema21', 'price_to_sma20',
    'ema9_to_ema21', 'ema21_to_ema50',
]

# Label mapping
LABEL_MAP = {
    0: 'BEARISH',
    1: 'NEUTRAL',
    2: 'BULLISH'
}

REVERSE_LABEL_MAP = {
    'BEARISH': 0,
    'NEUTRAL': 1,
    'BULLISH': 2
}

# Model selection
DEFAULT_MODEL = 'ml'  # Changed from 'xgboost' to 'ml' (generic)
USE_ENSEMBLE = True  # Combine multiple models if available
