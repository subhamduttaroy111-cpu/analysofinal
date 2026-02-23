"""
Model Training Script - Trains three separate ML models for all timeframes.
Uses RandomForest and XGBoost with hyperparameter tuning and comprehensive evaluation.
"""

import pandas as pd
import numpy as np
import os
import logging
import json
import joblib
from typing import Dict, Tuple, Any
import sys
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import XGBoost (optional)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("XGBoost not available, will use RandomForest and GradientBoosting only")

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_training.config import MODEL_CONFIG, PATHS

# Setup logging
os.makedirs(PATHS['logs'], exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PATHS['logs'], 'model_training.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains and evaluates ML models for all timeframes."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.feature_importance = {}
    
    def load_data(self, filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load training data and split into features and labels.
        
        Args:
            filepath: Path to training CSV file
            
        Returns:
            Tuple of (features DataFrame, labels Series)
        """
        try:
            df = pd.read_csv(filepath)
            
            if df.empty:
                logger.error(f"Empty dataset: {filepath}")
                return None, None
            
            # Separate features and labels
            if 'label' not in df.columns:
                logger.error(f"'label' column missing in {filepath}")
                return None, None
            
            # Remove non-feature columns
            exclude_cols = ['label', 'signal_type']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            X = df[feature_cols]
            y = df['label']
            
            # Handle any remaining NaN values
            X = X.fillna(0)
            
            logger.info(f"Loaded data: {len(df)} examples, {len(feature_cols)} features")
            logger.info(f"Class distribution: {y.value_counts().to_dict()}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error loading data from {filepath}: {str(e)}")
            return None, None
    
    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.15,
        val_size: float = 0.15
    ) -> Dict[str, Any]:
        """
        Split data into train, validation, and test sets (time-based).
        
        Args:
            X: Features
            y: Labels
            test_size: Test set proportion
            val_size: Validation set proportion
            
        Returns:
            Dictionary with train/val/test splits
        """
        # Calculate split indices
        total_size = len(X)
        test_idx = int(total_size * (1 - test_size))
        val_idx = int(test_idx * (1 - val_size))
        
        # Time-based split (no shuffling to prevent data leakage)
        X_train = X[:val_idx]
        X_val = X[val_idx:test_idx]
        X_test = X[test_idx:]
        
        y_train = y[:val_idx]
        y_val = y[val_idx:test_idx]
        y_test = y[test_idx:]
        
        logger.info(f"Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }
    
    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        model_type: str,
        config: Dict
    ) -> Any:
        """
        Train a model with hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            model_type: 'random_forest', 'xgboost', or 'gradient_boosting'
            config: Model configuration
            
        Returns:
            Trained model
        """
        logger.info(f"Training {model_type} model...")
        
        if model_type == 'random_forest':
            base_model = RandomForestClassifier(**MODEL_CONFIG['random_forest'])
            param_grid = {
                'n_estimators': [150],
                'max_depth': [20],
                'min_samples_split': [5]
            }
        
        elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
            base_model = XGBClassifier(**MODEL_CONFIG['xgboost'])
            param_grid = {
                'n_estimators': [150],  # Use 50 for Intraday optimization if needed, but 150 generally better
                'max_depth': [10],
                'learning_rate': [0.05]
            }
        
        elif model_type == 'gradient_boosting':
            base_model = GradientBoostingClassifier(**MODEL_CONFIG['gradient_boosting'])
            param_grid = {
                'n_estimators': [150],
                'max_depth': [10],
                'learning_rate': [0.1]
            }
        
        else:
            logger.error(f"Unknown model type: {model_type}")
            return None
        
        # Use TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=tscv,
            scoring='f1',
            n_jobs=2,  # Limited to 2 cores to prevent overheating
            verbose=1
        )
        
        # Fit model
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        logger.info(f"Best parameters: {best_params}")
        
        # Validate on validation set
        val_score = best_model.score(X_val, y_val)
        logger.info(f"Validation accuracy: {val_score:.4f}")
        
        return best_model
    
    def evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        timeframe: str
    ) -> Dict[str, Any]:
        """
        Evaluate model on test set.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            timeframe: Timeframe identifier
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(set(y_test)) > 1 else 0.0,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        logger.info(f"\n{timeframe.upper()} Model Evaluation:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
        logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_test.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            metrics['top_features'] = feature_importance.head(5).to_dict('records')
            self.feature_importance[timeframe] = feature_importance
            
            logger.info(f"\n  Top 5 Features:")
            for idx, row in feature_importance.head(5).iterrows():
                logger.info(f"    {row['feature']}: {row['importance']:.4f}")
        
        return metrics
    
    def plot_confusion_matrix(self, cm: list, timeframe: str):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Failure', 'Success'],
            yticklabels=['Failure', 'Success']
        )
        plt.title(f'{timeframe.capitalize()} Model - Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        output_path = os.path.join(PATHS['models'], f'{timeframe}_confusion_matrix.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved confusion matrix: {output_path}")
    
    def plot_feature_importance(self, timeframe: str):
        """Plot feature importance."""
        if timeframe not in self.feature_importance:
            return
        
        df_importance = self.feature_importance[timeframe].head(10)
        
        plt.figure(figsize=(10, 6))
        plt.barh(df_importance['feature'], df_importance['importance'])
        plt.xlabel('Importance')
        plt.title(f'{timeframe.capitalize()} Model - Top 10 Features')
        plt.gca().invert_yaxis()
        
        output_path = os.path.join(PATHS['models'], f'{timeframe}_feature_importance.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved feature importance plot: {output_path}")
    
    def train_timeframe_model(self, timeframe: str, filepath: str):
        """
        Train model for a specific timeframe.
        
        Args:
            timeframe: 'intraday', 'swing', or 'longterm'
            filepath: Path to training data CSV
        """
        print(f"\n{'='*70}")
        print(f"TRAINING {timeframe.upper()} MODEL")
        print(f"{'='*70}\n")
        
        # Load data
        X, y = self.load_data(filepath)
        if X is None or y is None:
            logger.error(f"Failed to load data for {timeframe}")
            return
        
        # Split data
        splits = self.split_data(X, y)
        
        # Train models and select best
        best_model = None
        best_score = 0
        best_model_type = None
        
        # Try different model types
        model_types = ['random_forest']
        if XGBOOST_AVAILABLE:
            model_types.append('xgboost')
        model_types.append('gradient_boosting')
        
        for model_type in model_types:
            try:
                model = self.train_model(
                    splits['X_train'],
                    splits['y_train'],
                    splits['X_val'],
                    splits['y_val'],
                    model_type,
                    MODEL_CONFIG
                )
                
                if model is None:
                    continue
                
                # Evaluate on validation set
                val_score = f1_score(splits['y_val'], model.predict(splits['X_val']), zero_division=0)
                
                if val_score > best_score:
                    best_score = val_score
                    best_model = model
                    best_model_type = model_type
                
            except Exception as e:
                logger.error(f"Error training {model_type}: {str(e)}")
                continue
        
        if best_model is None:
            logger.error(f"Failed to train any model for {timeframe}")
            return
        
        logger.info(f"\nBest model: {best_model_type} (F1: {best_score:.4f})")
        
        # Final evaluation on test set
        metrics = self.evaluate_model(
            best_model,
            splits['X_test'],
            splits['y_test'],
            timeframe
        )
        
        # Save model
        model_path = os.path.join(PATHS['models'], f'{timeframe}_model.pkl')
        joblib.dump(best_model, model_path)
        logger.info(f"✓ Saved model: {model_path}")
        
        # Save metadata
        metadata = {
            'timeframe': timeframe,
            'model_type': best_model_type,
            'features': X.columns.tolist(),
            'num_features': len(X.columns),
            'training_samples': len(splits['X_train']),
            'test_samples': len(splits['X_test']),
            'metrics': metrics,
            'best_params': best_model.get_params() if hasattr(best_model, 'get_params') else {}
        }
        
        metadata_path = os.path.join(PATHS['models'], f'{timeframe}_model_info.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"✓ Saved metadata: {metadata_path}")
        
        # Plot visualizations
        self.plot_confusion_matrix(metrics['confusion_matrix'], timeframe)
        self.plot_feature_importance(timeframe)
        
        # Store results
        self.results[timeframe] = metrics
        self.models[timeframe] = best_model
    
    def generate_report(self):
        """Generate comprehensive performance report."""
        report_path = os.path.join(os.path.dirname(PATHS['models']), '..', 'ml_training', 'model_performance_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("MODEL PERFORMANCE REPORT\n")
            f.write("="*70 + "\n\n")
            
            for timeframe in ['intraday', 'swing', 'longterm']:
                if timeframe not in self.results:
                    continue
                
                metrics = self.results[timeframe]
                
                f.write(f"=== {timeframe.upper()} MODEL ===\n")
                f.write(f"Accuracy:  {metrics['accuracy']*100:.1f}%\n")
                f.write(f"Precision: {metrics['precision']*100:.1f}%\n")
                f.write(f"Recall:    {metrics['recall']*100:.1f}%\n")
                f.write(f"F1-Score:  {metrics['f1_score']*100:.1f}%\n")
                f.write(f"ROC-AUC:   {metrics['roc_auc']:.3f}\n")
                
                if 'top_features' in metrics:
                    f.write(f"\nTop 5 Features:\n")
                    for feat in metrics['top_features']:
                        f.write(f"  - {feat['feature']}: {feat['importance']:.4f}\n")
                
                f.write("\n" + "-"*70 + "\n\n")
            
            f.write("="*70 + "\n")
            f.write("Models saved in: " + PATHS['models'] + "\n")
            f.write("="*70 + "\n")
        
        logger.info(f"✓ Performance report saved: {report_path}")
        
        # Also print to console
        with open(report_path, 'r') as f:
            print(f"\n{f.read()}")


def main():
    """Main execution function."""
    # Create output directory
    os.makedirs(PATHS['models'], exist_ok=True)
    
    # Create trainer
    trainer = ModelTrainer()
    
    # Train all models
    trainer.train_timeframe_model(
        'intraday',
        os.path.join(PATHS['processed'], 'intraday_training.csv')
    )
    
    trainer.train_timeframe_model(
        'swing',
        os.path.join(PATHS['processed'], 'swing_training.csv')
    )
    
    trainer.train_timeframe_model(
        'longterm',
        os.path.join(PATHS['processed'], 'longterm_training.csv')
    )
    
    # Generate report
    trainer.generate_report()
    
    print(f"\n{'='*70}")
    print("ALL MODELS TRAINED SUCCESSFULLY!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
