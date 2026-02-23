"""
Model Manager
Centralized interface for accessing AI models
"""

from ml_predictor import get_predictor as get_ml_predictor
from config_models import USE_ENSEMBLE, DEFAULT_MODEL

# Try to import LSTM predictors (optional dependency)
try:
    # Try to use enhanced LSTM first, fallback to original
    try:
        from lstm_predictor_enhanced import get_enhanced_lstm_predictor
        USE_ENHANCED_LSTM = True
    except ImportError:
        from lstm_predictor import get_lstm_predictor
        USE_ENHANCED_LSTM = False
    LSTM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  LSTM models unavailable (missing dependency): {e}")
    LSTM_AVAILABLE = False


class ModelManager:
    """Unified interface for all AI models"""
    
    def __init__(self):
        self.ml_predictor = get_ml_predictor()
        self.lstm_predictor = None
        
        # Initialize LSTM only if available
        if LSTM_AVAILABLE:
            try:
                if USE_ENHANCED_LSTM:
                    self.lstm_predictor = get_enhanced_lstm_predictor()
                    print("  🚀 Using Enhanced LSTM (Price Prediction)")
                else:
                    self.lstm_predictor = get_lstm_predictor()
                    print("  📊 Using Standard LSTM")
            except Exception as e:
                print(f"⚠️  Failed to initialize LSTM predictor: {e}")
                self.lstm_predictor = None

    
    def predict_ensemble(self, df, strategy_mode='INTRADAY'):
        """
        Ensemble prediction combining ML and LSTM models
        """
        results = {
            'available': False,
            'models_used': []
        }
        
        # Get ML prediction
        ml_result = self.ml_predictor.predict(df, model_type='ensemble', strategy_mode=strategy_mode)
        
        # Get LSTM prediction (only if available)
        lstm_result = {'available': False}
        if self.lstm_predictor:
            try:
                lstm_result = self.lstm_predictor.predict(df)
            except Exception as e:
                print(f"⚠️  LSTM prediction error: {e}")
        
        # Check if models are available
        if ml_result.get('available'):
            results['models_used'].append('ML (XGBoost + Random Forest)')
        
        if lstm_result.get('available'):
            results['models_used'].append('LSTM')
        
        # If no models available, return error
        if not results['models_used']:
            results['error'] = 'No trained models available'
            return results
        
        # Combine predictions
        if ml_result.get('available') and lstm_result.get('available'):
            # Both models available - ensemble
            results['available'] = True
            
            # Average probabilities
            combined_proba = {
                'BEARISH': (ml_result['probabilities']['BEARISH'] + 
                           lstm_result['probabilities']['BEARISH']) / 2,
                'NEUTRAL': (ml_result['probabilities']['NEUTRAL'] + 
                           lstm_result['probabilities']['NEUTRAL']) / 2,
                'BULLISH': (ml_result['probabilities']['BULLISH'] + 
                           lstm_result['probabilities']['BULLISH']) / 2,
            }
            
            # Get final signal
            final_signal = max(combined_proba, key=combined_proba.get)
            final_confidence = combined_proba[final_signal]
            
            results['signal'] = final_signal
            results['confidence'] = final_confidence
            results['probabilities'] = combined_proba
            results['ml_prediction'] = ml_result['signal']
            results['lstm_prediction'] = lstm_result['signal']
            results['model'] = 'Ensemble (ML + LSTM)'
        
        elif ml_result.get('available'):
            # Only ML available
            results = ml_result
            results['available'] = True
        
        elif lstm_result.get('available'):
            # Only LSTM available
            results = lstm_result
            results['available'] = True
        
        return results
    
    def predict(self, df, model_type='auto', strategy_mode='INTRADAY'):
        """
        Make prediction with specified model type
        """
        try:
            if model_type == 'auto':
                # Force ensemble if requested, but handle fallback internally
                return self.predict_ensemble(df, strategy_mode)
            
            elif model_type == 'ml':
                return self.ml_predictor.predict(df, model_type='ensemble', strategy_mode=strategy_mode)
            
            elif model_type == 'lstm':
                if self.lstm_predictor:
                    return self.lstm_predictor.predict(df)
                else:
                    return {'available': False, 'error': 'LSTM not available'}
            
            elif model_type == 'ensemble':
                return self.predict_ensemble(df, strategy_mode)
            
            else:
                return {'available': False, 'error': f'Unknown model type: {model_type}'}
                
        except Exception as e:
            return {'available': False, 'error': str(e)}
    
    def is_available(self):
        """Check if any models are available"""
        ml_ok = self.ml_predictor.models_loaded
        lstm_ok = self.lstm_predictor and self.lstm_predictor.model_loaded
        return ml_ok or lstm_ok
    
    def get_status(self):
        """Get status of all models"""
        return {
            'ml_available': self.ml_predictor.models_loaded,
            'lstm_available': self.lstm_predictor is not None and self.lstm_predictor.model_loaded,
            'any_available': self.is_available()
        }


# Global model manager instance
_model_manager = None

def get_model_manager():
    """Get or create global model manager"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
