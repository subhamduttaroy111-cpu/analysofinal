"""
Machine Learning Model Trainer
Trains XGBoost and Random Forest models for stock signal prediction
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

from config import STOCKS
from config_models import (
    XGBOOST_MODEL_PATH, RANDOM_FOREST_MODEL_PATH, FEATURE_SCALER_PATH,
    TRAINING_CONFIG, XGBOOST_PARAMS, RANDOM_FOREST_PARAMS,
    LABEL_MAP, REVERSE_LABEL_MAP
)
from indicators import add_indicators


def create_features(df):
    """Extract features from dataframe with technical indicators"""
    features = []
    
    for i in range(len(df)):
        row = df.iloc[i]
        
        feature_vector = [
            row['Close'], row['Open'], row['High'], row['Low'], row['Volume'],
            row['EMA_9'], row['EMA_21'], row['EMA_50'], row['EMA_200'],
            row['SMA_20'],
            row['RSI'], row['MACD'], row['MACD_Signal'], row['MACD_Hist'], row['Momentum'],
            row['ATR'], row['BB_Upper'], row['BB_Lower'], row['BB_Middle'],
            row['Volume_Ratio'],
            # Derived features
            row['Close'] / row['EMA_9'] if row['EMA_9'] > 0 else 1,
            row['Close'] / row['EMA_21'] if row['EMA_21'] > 0 else 1,
            row['Close'] / row['SMA_20'] if row['SMA_20'] > 0 else 1,
            row['EMA_9'] / row['EMA_21'] if row['EMA_21'] > 0 else 1,
            row['EMA_21'] / row['EMA_50'] if row['EMA_50'] > 0 else 1,
        ]
        
        features.append(feature_vector)
    
    return np.array(features)


def create_labels(df, forward_days=5, bullish_threshold=0.03, bearish_threshold=-0.01):
    """
    Create labels based on forward returns
    
    Args:
        df: DataFrame with stock data
        forward_days: Number of days to look ahead
        bullish_threshold: Return threshold for BULLISH label (e.g., 0.03 = 3%)
        bearish_threshold: Return threshold for BEARISH label (e.g., -0.01 = -1%)
    
    Returns:
        Array of labels (0=BEARISH, 1=NEUTRAL, 2=BULLISH)
    """
    labels = []
    
    for i in range(len(df)):
        if i + forward_days >= len(df):
            # Not enough future data, skip this row
            labels.append(None)
            continue
        
        current_price = df.iloc[i]['Close']
        future_price = df.iloc[i + forward_days]['Close']
        
        forward_return = (future_price - current_price) / current_price
        
        if forward_return >= bullish_threshold:
            labels.append(REVERSE_LABEL_MAP['BULLISH'])
        elif forward_return <= bearish_threshold:
            labels.append(REVERSE_LABEL_MAP['BEARISH'])
        else:
            labels.append(REVERSE_LABEL_MAP['NEUTRAL'])
    
    return np.array(labels)


def prepare_training_data(stocks, period='2y', interval='1d'):
    """
    Download historical data and prepare features and labels
    
    Args:
        stocks: List of stock symbols
        period: Historical period to download
        interval: Data interval
    
    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    print(f"📊 Downloading historical data for {len(stocks)} stocks...")
    
    all_features = []
    all_labels = []
    
    for symbol in stocks:
        try:
            print(f"  Processing {symbol}...")
            
            # Download data
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if len(df) < 100:
                print(f"  ⚠️  Skipping {symbol} - insufficient data ({len(df)} rows)")
                continue
            
            # Add technical indicators
            df = add_indicators(df)
            
            # Create features
            features = create_features(df)
            
            # Create labels
            labels = create_labels(
                df,
                forward_days=TRAINING_CONFIG['forward_days'],
                bullish_threshold=TRAINING_CONFIG['bullish_threshold'],
                bearish_threshold=TRAINING_CONFIG['bearish_threshold']
            )
            
            # Remove rows without labels (last few rows)
            valid_indices = [i for i, label in enumerate(labels) if label is not None]
            
            if len(valid_indices) > 0:
                all_features.append(features[valid_indices])
                all_labels.append(labels[valid_indices])
                print(f"  ✅ {symbol}: {len(valid_indices)} samples")
        
        except Exception as e:
            print(f"  ❌ Error processing {symbol}: {e}")
            continue
    
    if len(all_features) == 0:
        raise ValueError("No training data collected! Check your stock symbols.")
    
    # Combine all stocks
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    
    print(f"\n📈 Total samples: {len(X)}")
    print(f"   BULLISH: {np.sum(y == 2)} ({np.sum(y == 2) / len(y) * 100:.1f}%)")
    print(f"   NEUTRAL: {np.sum(y == 1)} ({np.sum(y == 1) / len(y) * 100:.1f}%)")
    print(f"   BEARISH: {np.sum(y == 0)} ({np.sum(y == 0) / len(y) * 100:.1f}%)")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=TRAINING_CONFIG['test_size'],
        random_state=TRAINING_CONFIG['random_state'],
        stratify=y
    )
    
    print(f"\n🔀 Train/Test Split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Testing:  {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test, scaler


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost classifier"""
    print("\n🚀 Training XGBoost model...")
    
    model = xgb.XGBClassifier(**XGBOOST_PARAMS)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"✅ XGBoost Training Accuracy: {train_acc:.4f}")
    print(f"✅ XGBoost Testing Accuracy:  {test_acc:.4f}")
    
    print("\n📊 Classification Report:")
    print(classification_report(y_test, test_pred, target_names=['BEARISH', 'NEUTRAL', 'BULLISH']))
    
    return model


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest classifier"""
    print("\n🌲 Training Random Forest model...")
    
    model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
    model.fit(X_train, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"✅ Random Forest Training Accuracy: {train_acc:.4f}")
    print(f"✅ Random Forest Testing Accuracy:  {test_acc:.4f}")
    
    print("\n📊 Classification Report:")
    print(classification_report(y_test, test_pred, target_names=['BEARISH', 'NEUTRAL', 'BULLISH']))
    
    return model


def train_all_models():
    """Main training pipeline"""
    print("=" * 60)
    print("🤖 AI MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_training_data(
        STOCKS,
        period=TRAINING_CONFIG['lookback_period']
    )
    
    # Train XGBoost
    xgb_model = train_xgboost(X_train, y_train, X_test, y_test)
    
    # Train Random Forest
    rf_model = train_random_forest(X_train, y_train, X_test, y_test)
    
    # Save models
    print("\n💾 Saving models...")
    joblib.dump(xgb_model, XGBOOST_MODEL_PATH)
    joblib.dump(rf_model, RANDOM_FOREST_MODEL_PATH)
    joblib.dump(scaler, FEATURE_SCALER_PATH)
    
    print(f"✅ XGBoost saved to: {XGBOOST_MODEL_PATH}")
    print(f"✅ Random Forest saved to: {RANDOM_FOREST_MODEL_PATH}")
    print(f"✅ Scaler saved to: {FEATURE_SCALER_PATH}")
    
    print("\n" + "=" * 60)
    print("🎉 Training complete!")
    print("=" * 60)


if __name__ == '__main__':
    train_all_models()
