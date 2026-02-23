"""
Enhanced LSTM Trainer with Portfolio Data
Uses pre-existing portfolio CSV data and improved LSTM methodology
Based on lstm.txt approach with 60-step sequences and price prediction
"""

import pandas as pd
import numpy as np
import os
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib

from config_models import LSTM_MODEL_PATH, SEQUENCE_SCALER_PATH
from config import STOCKS

# Portfolio CSV path
PORTFOLIO_CSV = os.path.join(os.path.dirname(__file__), '..', 'portfolio_data.csv')


def load_portfolio_data():
    """Load and prepare portfolio data from CSV"""
    print("📊 Loading portfolio data from CSV...")
    
    if os.path.exists(PORTFOLIO_CSV):
        df = pd.read_csv(PORTFOLIO_CSV)
        print(f"✅ Loaded portfolio data: {len(df)} rows")
        print(f"   Stocks in portfolio: {list(df.columns[1:])}")
        return df
    else:
        print(f"⚠️  Portfolio CSV not found at {PORTFOLIO_CSV}")
        return None


def prepare_stock_data_from_portfolio(df, stock_ticker):
    """
    Prepare data for a single stock from portfolio CSV
    Args:
        df: Portfolio DataFrame
        stock_ticker: Ticker symbol (e.g., 'AMZN')
    Returns:
        Prepared dataset for LSTM training
    """
    if stock_ticker not in df.columns:
        print(f"⚠️  {stock_ticker} not found in portfolio data")
        return None
    
    # Create a dataframe with Date and Close price
    stock_data = pd.DataFrame({
        'Date': pd.to_datetime(df['Date']),
        'Close': df[stock_ticker]
    })
    
    # Remove any NaN values
    stock_data = stock_data.dropna()
    
    # Set Date as index
    stock_data.set_index('Date', inplace=True)
    
    print(f"  📈 {stock_ticker}: {len(stock_data)} data points from {stock_data.index[0].date()} to {stock_data.index[-1].date()}")
    
    return stock_data


def prepare_stock_data_from_yfinance(ticker, period='2y'):
    """
    Fallback: Download data from yfinance for stocks not in portfolio
    """
    print(f"  📥 Downloading {ticker} from yfinance...")
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        if len(df) < 100:
            print(f"  ⚠️  Insufficient data for {ticker}")
            return None
        
        # Keep only Close column
        stock_data = df[['Close']]
        print(f"  ✅ {ticker}: {len(stock_data)} data points")
        return stock_data
    
    except Exception as e:
        print(f"  ❌ Error downloading {ticker}: {e}")
        return None


def create_sequences_price_prediction(dataset, sequence_length=60):
    """
    Create sequences for LSTM training (price prediction approach)
    Using proven method from lstm.txt
    
    Args:
        dataset: numpy array of prices
        sequence_length: number of time steps (default: 60)
    
    Returns:
        X_train, y_train: sequences and targets
    """
    X, y = [], []
    
    for i in range(sequence_length, len(dataset)):
        # X: Previous 60 days of prices
        X.append(dataset[i-sequence_length:i, 0])
        # y: Next day's price
        y.append(dataset[i, 0])
    
    X, y = np.array(X), np.array(y)
    
    # Reshape X for LSTM: (samples, timesteps, features)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y


def train_enhanced_lstm():
    """
    Main LSTM training pipeline using portfolio data + yfinance fallback
    """
    print("=" * 70)
    print("🧠 ENHANCED LSTM TRAINING WITH PORTFOLIO DATA")
    print("=" * 70)
    
    # Configuration (matching lstm.txt approach)
    SEQUENCE_LENGTH = 60
    TRAIN_SPLIT = 0.95  # 95% training, 5% testing
    BATCH_SIZE = 1
    EPOCHS = 50
    
    # Try to load portfolio data first
    portfolio_df = load_portfolio_data()
    
    all_X_train, all_y_train = [], []
    all_X_test, all_y_test = [], []
    
    # Prepare data for each stock
    print("\n📊 Preparing training data...")
    
    for ticker in STOCKS:
        clean_ticker = ticker.replace('.NS', '')
        
        # Try portfolio data first
        stock_data = None
        if portfolio_df is not None:
            stock_data = prepare_stock_data_from_portfolio(portfolio_df, clean_ticker)
        
        # Fallback to yfinance
        if stock_data is None:
            stock_data = prepare_stock_data_from_yfinance(ticker, period='2y')
        
        if stock_data is None:
            continue
        
        try:
            # Convert to numpy array
            close_prices = stock_data[['Close']].values
            
            # Scale data to (0, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(close_prices)
            
            # Calculate 95% split point
            training_data_len = int(np.ceil(len(scaled_data) * TRAIN_SPLIT))
            
            # Split into train and test
            train_data = scaled_data[0:training_data_len, :]
            test_data = scaled_data[training_data_len - SEQUENCE_LENGTH:, :]
            
            # Create training sequences
            X_train, y_train = create_sequences_price_prediction(train_data, SEQUENCE_LENGTH)
            
            # Create testing sequences
            X_test, y_test = create_sequences_price_prediction(test_data, SEQUENCE_LENGTH)
            
            if len(X_train) > 0:
                all_X_train.append(X_train)
                all_y_train.append(y_train)
                print(f"  ✅ {clean_ticker}: {len(X_train)} training sequences, {len(X_test)} test sequences")
            
            if len(X_test) > 0:
                all_X_test.append(X_test)
                all_y_test.append(y_test)
        
        except Exception as e:
            print(f"  ❌ Error preparing {ticker}: {e}")
            continue
    
    if len(all_X_train) == 0:
        print("\n❌ No training data collected!")
        return
    
    # Combine all stocks
    X_train = np.vstack(all_X_train)
    y_train = np.concatenate(all_y_train)
    X_test = np.vstack(all_X_test) if len(all_X_test) > 0 else None
    y_test = np.concatenate(all_y_test) if len(all_y_test) > 0 else None
    
    print(f"\n📈 Total training sequences: {len(X_train)}")
    print(f"   Shape: {X_train.shape}")
    if X_test is not None:
        print(f"   Total test sequences: {len(X_test)}")
    
    # Build LSTM model (matching lstm.txt architecture)
    print("\n🏗️  Building LSTM model...")
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    
    print("\n📋 Model Architecture:")
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            LSTM_MODEL_PATH,
            monitor='loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train the model
    print(f"\n🚀 Training LSTM (epochs={EPOCHS}, batch_size={BATCH_SIZE})...")
    
    validation_data = (X_test, y_test) if X_test is not None else None
    
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        validation_data=validation_data,
        verbose=1
    )
    
    # Evaluate
    train_loss, train_mae = model.evaluate(X_train, y_train, verbose=0)
    print(f"\n✅ Training Loss: {train_loss:.6f}")
    print(f"✅ Training MAE: {train_mae:.6f}")
    
    if X_test is not None:
        test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
        print(f"✅ Test Loss: {test_loss:.6f}")
        print(f"✅ Test MAE: {test_mae:.6f}")
    
    # Save scaler (use a generic one for consistency)
    print(f"\n💾 Saving scaler...")
    generic_scaler = MinMaxScaler(feature_range=(0, 1))
    joblib.dump(generic_scaler, SEQUENCE_SCALER_PATH)
    print(f"✅ Scaler saved to: {SEQUENCE_SCALER_PATH}")
    
    print("\n" + "=" * 70)
    print("🎉 ENHANCED LSTM TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Model saved to: {LSTM_MODEL_PATH}")
    print(f"📊 Trained on {len(all_X_train)} stocks with {len(X_train)} total sequences")
    print("\n💡 This model uses the proven 60-step sequence approach for price prediction")


if __name__ == '__main__':
    train_enhanced_lstm()
