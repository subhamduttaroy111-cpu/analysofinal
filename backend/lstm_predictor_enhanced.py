"""
Enhanced LSTM Predictor
Works with the enhanced LSTM model trained on portfolio data
Uses 60-step sequence approach for price prediction
"""

import numpy as np
import pandas as pd
import joblib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

from config_models import LSTM_MODEL_PATH, SEQUENCE_SCALER_PATH, LABEL_MAP


class EnhancedLSTMPredictor:
    """Enhanced LSTM predictor for price-based predictions"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_loaded = False
        self.sequence_length = 60  # Matching lstm.txt approach
        
        self._load_model()
    
    def _load_model(self):
        """Load trained LSTM model"""
        try:
            if os.path.exists(LSTM_MODEL_PATH):
                self.model = load_model(LSTM_MODEL_PATH)
                self.model_loaded = True
                print("✅ Enhanced LSTM model loaded")
                
                # Create scaler
                self.scaler = MinMaxScaler(feature_range=(0, 1))
            else:
                print("⚠️  Enhanced LSTM model not found. Train using lstm_trainer_enhanced.py")
        except Exception as e:
            print(f"❌ Error loading Enhanced LSTM model: {e}")
            self.model_loaded = False
    
    def prepare_sequence(self, df):
        """
        Prepare 60-step sequence from dataframe for prediction
        
        Args:
            df: DataFrame with Close prices
        
        Returns:
            Scaled sequence ready for prediction
        """
        if len(df) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} data points, got {len(df)}")
        
        # Get last 60 closing prices
        close_prices = df['Close'].values[-self.sequence_length:]
        close_prices = close_prices.reshape(-1, 1)
        
        # Fit scaler on these prices
        self.scaler.fit(close_prices)
        
        # Scale data
        scaled_data = self.scaler.transform(close_prices)
        
        # Reshape for LSTM: (1, 60, 1)
        sequence = scaled_data.reshape(1, self.sequence_length, 1)
        
        return sequence, close_prices[-1][0]  # Return sequence and current price
    
    def predict_price(self, df):
        """
        Predict next day's price
        
        Args:
            df: DataFrame with Close prices
        
        Returns:
            dict with predicted price and signal
        """
        if not self.model_loaded:
            return {
                'available': False,
                'error': 'Enhanced LSTM model not trained yet'
            }
        
        try:
            # Prepare sequence
            sequence, current_price = self.prepare_sequence(df)
            
            # Predict scaled price
            predicted_scaled = self.model.predict(sequence, verbose=0)
            
            # Inverse transform to get actual price
            predicted_price = self.scaler.inverse_transform(predicted_scaled)[0][0]
            
            # Calculate price change
            price_change = predicted_price - current_price
            price_change_pct = (price_change / current_price) * 100
            
            # Determine signal
            if price_change_pct >= 2:
                signal = 'BULLISH'
                confidence = min(0.9, 0.6 + (price_change_pct - 2) * 0.05)
            elif price_change_pct <= -1:
                signal = 'BEARISH'
                confidence = min(0.9, 0.6 + abs(price_change_pct + 1) * 0.05)
            else:
                signal = 'NEUTRAL'
                confidence = 0.5 + abs(price_change_pct) * 0.05
            
            # Calculate probabilities
            if signal == 'BULLISH':
                bullish_prob = confidence
                bearish_prob = (1 - confidence) * 0.3
                neutral_prob = 1 - bullish_prob - bearish_prob
            elif signal == 'BEARISH':
                bearish_prob = confidence
                bullish_prob = (1 - confidence) * 0.3
                neutral_prob = 1 - bearish_prob - bullish_prob
            else:
                neutral_prob = confidence
                bullish_prob = (1 - confidence) * 0.5
                bearish_prob = 1 - neutral_prob - bullish_prob
            
            return {
                'available': True,
                'signal': signal,
                'confidence': confidence,
                'predicted_price': round(predicted_price, 2),
                'current_price': round(current_price, 2),
                'price_change': round(price_change, 2),
                'price_change_pct': round(price_change_pct, 2),
                'probabilities': {
                    'BEARISH': float(bearish_prob),
                    'NEUTRAL': float(neutral_prob),
                    'BULLISH': float(bullish_prob)
                },
                'model': 'Enhanced LSTM (Price Prediction)'
            }
        
        except Exception as e:
            return {
                'available': False,
                'error': str(e)
            }
    
    # Alias for compatibility with existing code
    def predict(self, df):
        """Compatibility wrapper for predict_price"""
        return self.predict_price(df)


# Global predictor instance
_enhanced_lstm_predictor = None

def get_enhanced_lstm_predictor():
    """Get or create global enhanced LSTM predictor instance"""
    global _enhanced_lstm_predictor
    if _enhanced_lstm_predictor is None:
        _enhanced_lstm_predictor = EnhancedLSTMPredictor()
    return _enhanced_lstm_predictor
