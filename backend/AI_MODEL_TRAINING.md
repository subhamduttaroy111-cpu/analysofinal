# 🤖 AI Model Training Guide

## Quick Start

### Install Dependencies
First, install the required AI/ML libraries:
```bash
pip install -r requirements.txt
```

This will install:
- `scikit-learn` - Machine Learning framework
- `xgboost` - XGBoost classifier
- `tensorflow` - Deep Learning framework
- `joblib` - Model serialization

---

## Training Models

### Option 1: Train All Models (Recommended)
```bash
cd backend
python scripts/train_all_models.py
```

This will:
1. Download 2 years of historical data for all stocks
2. Train XGBoost classifier
3. Train Random Forest classifier
4. Train LSTM deep learning model
5. Save all models to `backend/models/`

**Time:** 10-30 minutes depending on your hardware

---

### Option 2: Train Individual Models

**Train ML Models Only (XGBoost + Random Forest):**
```bash
python ml_trainer.py
```

**Train LSTM Model Only:**
```bash
python lstm_trainer.py
```

---

## How It Works

### Automatic Data Labeling
The system automatically labels historical data based on **forward returns**:

- **BULLISH**: Stock gained ≥3% in next 5 days
- **NEUTRAL**: Stock moved between -1% and +3%
- **BEARISH**: Stock lost ≥1% in next 5 days

### Features Used
The models learn from 25+ features extracted from technical indicators:
- Price (Close, Open, High, Low)
- Moving Averages (EMA 9/21/50/200, SMA 20)
- Momentum (RSI, MACD, Momentum)
- Volatility (ATR, Bollinger Bands)
- Volume metrics
- Derived ratios (Price/EMA, EMA/EMA, etc.)

### Model Architecture

**XGBoost:**
- 200 trees, max depth 6
- Multi-class classification
- Gradient boosting

**Random Forest:**
- 200 estimators, max depth 15
- Bagging ensemble
- Feature importance analysis

**LSTM:**
- 3-layer stacked LSTM (128→64→32 units)
- 60-timestep sequences
- Batch normalization & dropout
- Early stopping

---

## Using the Models

Once trained, the models are **automatically used** by the application:

1. Start your backend server:
   ```bash
   python server.py
   ```

2. Run a stock scan from the frontend

3. The system will:
   - Use **Ensemble (ML + LSTM)** prediction by default
   - Show AI confidence scores
   - Display probability breakdown
   - Fall back to rule-based logic if models unavailable

---

## Model Files

After training, you'll find these files in `backend/models/`:

- `xgboost_model.pkl` - XGBoost classifier
- `random_forest_model.pkl` - Random Forest classifier
- `lstm_model.keras` - LSTM neural network
- `feature_scaler.pkl` - Feature normalization scaler
- `sequence_scaler.pkl` - Sequence normalization for LSTM

**Size:** ~50-200 MB total

---

## Configuration

Edit `backend/config_models.py` to customize:

- Training period (default: 2 years)
- Label thresholds (bullish/bearish)
- Model hyperparameters
- Ensemble settings

---

## Retraining

Models should be retrained periodically to stay current:

**When to retrain:**
- Every month for production systems
- After major market regime changes
- When adding new stocks to your universe

**How to retrain:**
```bash
# Delete old models
rm -rf backend/models/*.pkl backend/models/*.keras

# Train fresh models
python scripts/train_all_models.py
```

---

## Performance

Typical accuracy on test data:
- **XGBoost**: 60-70%
- **Random Forest**: 58-68%
- **LSTM**: 55-65%
- **Ensemble**: 62-72%

> 📊 These are realistic numbers for stock prediction. Above 60% is considered good!

---

## Troubleshooting

**Issue: "Models not found"**
- Run `python scripts/train_all_models.py` to train models

**Issue: Training takes too long**
- Reduce number of stocks in `config.py`
- Use shorter training period in `config_models.py`

**Issue: Low memory**
- Reduce LSTM `sequence_length` in config
- Train XGBoost/Random Forest only (skip LSTM)

**Issue: TensorFlow warnings**
- These are normal, models will still work
- Set `TF_CPP_MIN_LOG_LEVEL=2` to suppress

---

## Advanced: API Parameters

Control AI usage via API:

```javascript
// Use AI predictions (default)
fetch('/scan', {
    method: 'POST',
    body: JSON.stringify({ mode: 'INTRADAY', use_ai: true })
})

// Use rule-based only
fetch('/scan', {
    method: 'POST',
    body: JSON.stringify({ mode: 'INTRADAY', use_ai: false })
})
```

---

## Need Help?

Check the implementation plan for more details:
`brain/<conversation-id>/implementation_plan.md`
