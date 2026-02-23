"""
LSTM Deep Learning Model Trainer
Trains LSTM neural network for time series stock prediction
"""

import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

from config import STOCKS
from config_models import (
    LSTM_MODEL_PATH, SEQUENCE_SCALER_PATH,
    TRAINING_CONFIG, LSTM_PARAMS,
    REVERSE_LABEL_MAP
)
from indicators import add_indicators


def prepare_sequences(df, sequence_length=60):
    """
    Prepare sequences for LSTM training
    
    Args:
        df: DataFrame with technical indicators
        sequence_length: Number of time steps in each sequence
    
    Returns:
        sequences: 3D array of shape (n_samples, sequence_length, n_features)
        labels: 1D array of labels
    """
    # Select features for LSTM
    feature_cols = ['Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 
                   'EMA_9', 'EMA_21', 'EMA_50', 'Volume_Ratio', 'ATR']
    
    data = df[feature_cols].values
    
    sequences = []
    labels = []
    
    # Create sequences
    for i in range(len(data) - sequence_length - TRAINING_CONFIG['forward_days']):
        # Get sequence of features
        seq = data[i:i + sequence_length]
        
        # Get future price for labeling
        current_price = df.iloc[i + sequence_length]['Close']
        future_price = df.iloc[i + sequence_length + TRAINING_CONFIG['forward_days']]['Close']
        
        forward_return = (future_price - current_price) / current_price
        
        # Create label
        if forward_return >= TRAINING_CONFIG['bullish_threshold']:
            label = REVERSE_LABEL_MAP['BULLISH']
        elif forward_return <= TRAINING_CONFIG['bearish_threshold']:
            label = REVERSE_LABEL_MAP['BEARISH']
        else:
            label = REVERSE_LABEL_MAP['NEUTRAL']
        
        sequences.append(seq)
        labels.append(label)
    
    return np.array(sequences), np.array(labels)


def prepare_lstm_data(stocks, period='2y', interval='1d', sequence_length=60):
    """
    Download and prepare data for LSTM training
    
    Args:
        stocks: List of stock symbols
        period: Historical period
        interval: Data interval
        sequence_length: LSTM sequence length
    
    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    print(f"📊 Preparing LSTM data for {len(stocks)} stocks...")
    
    all_sequences = []
    all_labels = []
    
    for symbol in stocks:
        try:
            print(f"  Processing {symbol}...")
            
            # Download data
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if len(df) < sequence_length + 50:
                print(f"  ⚠️  Skipping {symbol} - insufficient data")
                continue
            
            # Add indicators
            df = add_indicators(df)
            
            # Create sequences
            sequences, labels = prepare_sequences(df, sequence_length)
            
            if len(sequences) > 0:
                all_sequences.append(sequences)
                all_labels.append(labels)
                print(f"  ✅ {symbol}: {len(sequences)} sequences")
        
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
    
    if len(all_sequences) == 0:
        raise ValueError("No LSTM training data collected!")
    
    # Combine all stocks
    X = np.vstack(all_sequences)
    y = np.concatenate(all_labels)
    
    print(f"\n📈 Total sequences: {len(X)}")
    print(f"   Shape: {X.shape}")
    print(f"   BULLISH: {np.sum(y == 2)} ({np.sum(y == 2) / len(y) * 100:.1f}%)")
    print(f"   NEUTRAL: {np.sum(y == 1)} ({np.sum(y == 1) / len(y) * 100:.1f}%)")
    print(f"   BEARISH: {np.sum(y == 0)} ({np.sum(y == 0) / len(y) * 100:.1f}%)")
    
    # Normalize features (scale each feature across all sequences)
    n_samples, n_timesteps, n_features = X.shape
    
    # Reshape for scaling
    X_reshaped = X.reshape(-1, n_features)
    
    # Fit scaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_reshaped)
    
    # Reshape back to sequences
    X_scaled = X_scaled.reshape(n_samples, n_timesteps, n_features)
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=TRAINING_CONFIG['test_size'],
        random_state=TRAINING_CONFIG['random_state']
    )
    
    print(f"\n🔀 Train/Test Split:")
    print(f"   Training: {len(X_train)} sequences")
    print(f"   Testing:  {len(X_test)} sequences")
    
    return X_train, X_test, y_train, y_test, scaler


def build_lstm_model(input_shape, units=[128, 64, 32], dropout=0.3):
    """
    Build LSTM model architecture
    
    Args:
        input_shape: (sequence_length, n_features)
        units: List of LSTM units for each layer
        dropout: Dropout rate
    
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        # First LSTM layer
        LSTM(units[0], return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(dropout),
        
        # Second LSTM layer
        LSTM(units[1], return_sequences=True),
        BatchNormalization(),
        Dropout(dropout),
        
        # Third LSTM layer
        LSTM(units[2], return_sequences=False),
        BatchNormalization(),
        Dropout(dropout),
        
        # Dense layers
        Dense(64, activation='relu'),
        Dropout(dropout),
        Dense(32, activation='relu'),
        
        # Output layer (3 classes: BEARISH, NEUTRAL, BULLISH)
        Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LSTM_PARAMS['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_lstm_model(X_train, y_train, X_test, y_test):
    """Train LSTM model"""
    print("\n🧠 Building LSTM model...")
    
    # Get input shape
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # Build model
    model = build_lstm_model(
        input_shape=input_shape,
        units=LSTM_PARAMS['units'],
        dropout=LSTM_PARAMS['dropout']
    )
    
    print("\n📋 Model Architecture:")
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            LSTM_MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    print("\n🚀 Training LSTM...")
    
    history = model.fit(
        X_train, y_train,
        epochs=LSTM_PARAMS['epochs'],
        batch_size=LSTM_PARAMS['batch_size'],
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\n✅ LSTM Training Accuracy: {train_acc:.4f}")
    print(f"✅ LSTM Testing Accuracy:  {test_acc:.4f}")
    
    return model, history


def train_lstm():
    """Main LSTM training pipeline"""
    print("=" * 60)
    print("🧠 LSTM MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_lstm_data(
        STOCKS,
        period=TRAINING_CONFIG['lookback_period'],
        sequence_length=LSTM_PARAMS['sequence_length']
    )
    
    # Train model
    model, history = train_lstm_model(X_train, y_train, X_test, y_test)
    
    # Save scaler
    print(f"\n💾 Saving scaler...")
    joblib.dump(scaler, SEQUENCE_SCALER_PATH)
    print(f"✅ Scaler saved to: {SEQUENCE_SCALER_PATH}")
    
    print("\n" + "=" * 60)
    print("🎉 LSTM Training complete!")
    print("=" * 60)


if __name__ == '__main__':
    train_lstm()
