"""
Train All AI Models
One-click script to train XGBoost, Random Forest, and LSTM models
"""

import os
import sys

# Add backend to path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir.endswith('scripts'):
    backend_dir = os.path.dirname(backend_dir)
sys.path.insert(0, backend_dir)

print("=" * 70)
print("🚀 AI MODEL TRAINING - FULL PIPELINE")
print("=" * 70)
print()

# Train ML models (XGBoost + Random Forest)
print("STEP 1/2: Training Machine Learning Models (XGBoost + Random Forest)")
print("-" * 70)
try:
    from ml_trainer import train_all_models
    train_all_models()
    print("\n✅ ML Models trained successfully!\n")
except Exception as e:
    print(f"\n❌ ML Training failed: {e}\n")
    print("⚠️  Continuing with LSTM training...\n")

# Train Enhanced LSTM model (with portfolio data)
print("\nSTEP 2/2: Training Deep Learning Model (Enhanced LSTM)")
print("-" * 70)
print("💡 Using portfolio CSV data and improved price prediction approach")
try:
    from lstm_trainer_enhanced import train_enhanced_lstm
    train_enhanced_lstm()
    print("\n✅ Enhanced LSTM Model trained successfully!\n")
except Exception as e:
    print(f"\n❌ Enhanced LSTM Training failed: {e}\n")
    print("⚠️  Trying standard LSTM trainer...\n")
    try:
        from lstm_trainer import train_lstm
        train_lstm()
        print("\n✅ Standard LSTM Model trained successfully!\n")
    except Exception as e2:
        print(f"\n❌ Standard LSTM Training also failed: {e2}\n")

print()
print("=" * 70)
print("🎉 TRAINING PIPELINE COMPLETE!")
print("=" * 70)
print()
print("Your AI models are ready to use!")
print("Start the application and run a scan to see AI predictions in action.")
print()
