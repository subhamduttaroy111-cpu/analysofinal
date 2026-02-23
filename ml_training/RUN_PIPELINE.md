# ML Pipeline Execution Guide

Complete step-by-step guide to run the ML training pipeline.

## Prerequisites

### System Requirements
- Python 3.8 or higher
- ~1 GB free disk space
- Stable internet connection (for data download)
- 4+ GB RAM recommended

### Software Installation

```bash
# Install Python dependencies
pip install -r requirements.txt
```

**Note**: If you encounter issues with installation:
- Use `pip install --upgrade pip` first
- For XGBoost on Windows, you may need Visual C++ Build Tools
- TA-Lib is optional; the pipeline works without it

## Step-by-Step Execution

### Step 1: Verify Configuration

Check that the stock list is loaded correctly:

```bash
python -c "from ml_training.config import STOCK_LIST; print(f'Loaded {len(STOCK_LIST)} stocks')"
```

Expected output: `Loaded 200 stocks`

### Step 2: Download Data (~2-3 hours)

```bash
python ml_training/1_download_data.py
```

**What it does**:
- Downloads 5-minute, 1-hour, daily, and weekly data for all 200 stocks
- Uses sequential downloads with 0.5s delay to avoid rate limits
- Saves data to `data/raw/` subdirectories
- Logs errors to `ml_training/logs/download_errors.log`

**Progress indicators**:
```
[1/200] Processing: RELIANCE.NS
✓ Intraday: RELIANCE.NS (4532 rows)
✓ Swing 1h: RELIANCE.NS (8760 rows)
✓ Swing 1d: RELIANCE.NS (504 rows)
✓ Long-term 1d: RELIANCE.NS (1260 rows)
✓ Long-term 1wk: RELIANCE.NS (260 rows)
Status: [Intraday ✓ | Swing ✓ | Long-term ✓] - Progress: 0.5%
Estimated remaining time: 142.3 minutes
```

**Expected output**:
```
DOWNLOAD COMPLETE
Results:
  ✓ Intraday:   195/200 stocks (97.5%)
  ✓ Swing:      198/200 stocks (99.0%)
  ✓ Long-term:  200/200 stocks (100.0%)

Execution time: 137.2 minutes
```

**Troubleshooting**:
- If many downloads fail, check internet connection
- Some stocks may not have 5-minute data (yfinance limitation)
- Failed downloads are logged in `ml_training/logs/download_failures.txt`

### Step 3: Generate Training Data (~45-60 minutes)

```bash
python ml_training/2_generate_training_data.py
```

**What it does**:
- Calculates technical indicators (RSI, MACD, EMAs, ATR, etc.)
- Detects SMC patterns (order blocks, liquidity zones, FVGs)
- Labels outcomes based on target/SL hits
- Balances datasets to 55-60% success rate
- Saves to `data/processed/`

**Progress indicators**:
```
GENERATING INTRADAY TRAINING DATA
Processing intraday files: 100%|████████████████| 195/195

Generated 12,437 intraday examples
Success rate: 58.3%
Intraday - Before balancing: 7,254 successes, 5,183 failures
Intraday - After balancing: 6,841 successes, 5,183 failures (56.9% success rate)
✓ Saved intraday training data: 12,024 examples
```

**Expected output**:
```
TRAINING DATA GENERATION COMPLETE

INTRADAY:
  Total signals: 12,437
  Success: 7,254 (58.3%)
  Failure: 5,183 (41.7%)

SWING:
  Total signals: 18,942
  Success: 11,365 (60.0%)
  Failure: 7,577 (40.0%)

LONG-TERM:
  Total signals: 24,128
  Success: 14,477 (60.0%)
  Failure: 9,651 (40.0%)
```

### Step 4: Train Models (~30-45 minutes)

```bash
python ml_training/3_train_models.py
```

**What it does**:
- Trains RandomForest, XGBoost, and GradientBoosting models
- Performs hyperparameter tuning with GridSearchCV
- Evaluates on test sets
- Generates visualizations (confusion matrices, feature importance)
- Saves models and metadata to `data/models/`

**Progress indicators**:
```
TRAINING INTRADAY MODEL

Loaded data: 12,024 examples, 17 features
Class distribution: {1: 6,841, 0: 5,183}
Split sizes - Train: 7,214, Val: 1,806, Test: 3,004

Training random_forest model...
Fitting 3 folds for each of 27 candidates, totalling 81 fits
Best parameters: {'max_depth': 15, 'min_samples_split': 10, 'n_estimators': 100}
Validation accuracy: 0.7183

INTRADAY Model Evaluation:
  Accuracy:  0.7242
  Precision: 0.7531
  Recall:    0.6892
  F1-Score:  0.7197
  ROC-AUC:   0.7856

  Top 5 Features:
    order_block_distance: 0.1842
    time_of_day: 0.1534
    rsi: 0.1247
    volatility: 0.0984
    trend_strength: 0.0891

✓ Saved model: data/models/intraday_model.pkl
✓ Saved metadata: data/models/intraday_model_info.json
```

**Expected output**:
```
ALL MODELS TRAINED SUCCESSFULLY!

MODEL PERFORMANCE REPORT

=== INTRADAY MODEL ===
Accuracy:  72.4%
Precision: 75.3%
Recall:    68.9%
F1-Score:  71.9%
ROC-AUC:   0.786

Top 5 Features:
  - order_block_distance: 0.1842
  - time_of_day: 0.1534
  - rsi: 0.1247
  - volatility: 0.0984
  - trend_strength: 0.0891

=== SWING MODEL ===
Accuracy:  76.8%
Precision: 78.2%
Recall:    74.9%
F1-Score:  76.5%
ROC-AUC:   0.832

=== LONG-TERM MODEL ===
Accuracy:  79.6%
Precision: 80.4%
Recall:    78.2%
F1-Score:  79.3%
ROC-AUC:   0.861

Models saved in: data/models/
```

## Verification

After completion, verify all files exist:

```bash
# Check models
ls data/models/

# Expected output:
# intraday_model.pkl
# intraday_model_info.json
# intraday_confusion_matrix.png
# intraday_feature_importance.png
# swing_model.pkl
# swing_model_info.json
# swing_confusion_matrix.png
# swing_feature_importance.png
# longterm_model.pkl
# longterm_model_info.json
# longterm_confusion_matrix.png
# longterm_feature_importance.png
```

## Quick Test Run (Optional)

To test the pipeline with fewer stocks first:

1. Edit `ml_training/stocks.csv` and keep only the first 5 stocks
2. Run the pipeline
3. Verify it works (~10-15 minutes total)
4. Restore the full stock list and run again

## Total Execution Time

- **Download**: ~2-3 hours (depends on internet speed)
- **Generation**: ~45-60 minutes
- **Training**: ~30-45 minutes  
- **Total**: ~3.5-4.5 hours

## Outputs

### Files Created

```
data/
├── raw/                           (~800 MB)
│   ├── intraday/                 (195 CSV files)
│   ├── swing/                    (396 CSV files: 1h + 1d)
│   └── longterm/                 (400 CSV files: 1d + 1wk)
├── processed/                     (~100 MB)
│   ├── intraday_training.csv
│   ├── swing_training.csv
│   └── longterm_training.csv
└── models/                        (~50 MB)
    ├── *_model.pkl (3 files)
    ├── *_model_info.json (3 files)
    ├── *_confusion_matrix.png (3 files)
    └── *_feature_importance.png (3 files)

ml_training/
├── logs/
│   ├── download_errors.log
│   ├── download_failures.txt
│   ├── training_data_generation.log
│   └── model_training.log
└── model_performance_report.txt
```

## Troubleshooting

### Download Issues

**Problem**: Many stocks failing to download  
**Solution**: 
- Check internet connection
- Increase `DOWNLOAD_CONFIG['retry_delay']` in `config.py`
- Reduce `DOWNLOAD_CONFIG['request_delay']` if comfortable with faster requests

**Problem**: "Rate limit exceeded" errors  
**Solution**: 
- Increase `request_delay` to 1.0 seconds in `config.py`
- Resume from where it left off (script skips existing files)

### Training Data Issues

**Problem**: "Insufficient data" warnings  
**Solution**:
- Normal for some stocks - ok to have 190-195 successful instead of 200
- Adjust `order_block_lookback` if too strict

**Problem**: Very low number of training examples (<5000)  
**Solution**:
- Relax signal detection conditions in `timeframe_strategies.py`
- Check downloaded data quality

### Model Training Issues

**Problem**: "XGBoost not available" warning  
**Solution**:
- Normal - pipeline will use RandomForest and GradientBoosting
- Install XGBoost if desired: `pip install xgboost`

**Problem**: Low accuracy (<60%)  
**Solution**:
- Check data quality in `data/raw/`
- Review feature engineering in `utils.py`
- Adjust hyperparameters in `config.py`

## Next Steps

After successful execution:
1. Review `ml_training/model_performance_report.txt`
2. Check confusion matrices in `data/models/`
3. Integrate models into your application (see `INTEGRATION.md`)
4. Backtest on live data

## Retraining

To retrain with fresh data:

```bash
# Option 1: Full pipeline (delete old data first)
rm -rf data/raw/*
python ml_training/1_download_data.py
python ml_training/2_generate_training_data.py
python ml_training/3_train_models.py

# Option 2: Just retrain models (keep existing data)
python ml_training/3_train_models.py
```

Recommended retraining schedule:
- **Intraday**: Every 1-2 months
- **Swing**: Every 2-3 months
- **Long-term**: Every 4-6 months
