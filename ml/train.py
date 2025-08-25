#!/usr/bin/env python3
"""
Main training script for the Hospital Readmission Risk Predictor.
Orchestrates the entire ML pipeline: data loading, feature engineering, model training, and evaluation.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from ml.data.loader import DataLoader, create_sample_data
from ml.features.engineering import FeatureEngineer
from ml.models.trainer import ModelTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Hospital Readmission Risk Predictor')
    parser.add_argument('--data-path', type=str, default='data/',
                       help='Path to data directory')
    parser.add_argument('--output-dir', type=str, default='models/',
                       help='Output directory for models')
    parser.add_argument('--n-samples', type=int, default=2000,
                       help='Number of synthetic samples to generate')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random state for reproducibility')
    parser.add_argument('--use-synthetic', action='store_true',
                       help='Use synthetic data instead of real data')
    parser.add_argument('--cross-validate', action='store_true',
                       help='Perform cross-validation')
    
    args = parser.parse_args()
    
    logger.info("Starting Hospital Readmission Risk Predictor training pipeline")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Step 1: Data Loading
        logger.info("Step 1: Loading and preparing data")
        data_loader = DataLoader(args.data_path)
        
        if args.use_synthetic:
            logger.info("Generating synthetic data")
            df = create_sample_data(n_samples=args.n_samples, random_state=args.random_state)
            
            # Save synthetic data
            data_loader.data_path.mkdir(exist_ok=True)
            df.to_csv(data_loader.data_path / "synthetic_data.csv", index=False)
            logger.info(f"Saved synthetic data to {data_loader.data_path / 'synthetic_data.csv'}")
        else:
            # Try to load existing data
            data_files = list(data_loader.data_path.glob("*.csv")) + list(data_loader.data_path.glob("*.json"))
            if not data_files:
                logger.warning("No data files found, generating synthetic data")
                df = create_sample_data(n_samples=args.n_samples, random_state=args.random_state)
            else:
                # Load the first available data file
                data_file = data_files[0]
                if data_file.suffix == '.csv':
                    df = data_loader.load_csv(str(data_file))
                else:
                    df = data_loader.load_json(str(data_file))
        
        # Validate and clean data
        df_clean, validation_errors = data_loader.validate_and_clean(df)
        
        if validation_errors:
            logger.warning(f"Found {len(validation_errors)} validation errors")
            for error in validation_errors[:5]:  # Show first 5 errors
                logger.warning(f"  - {error}")
        
        logger.info(f"Data loaded: {len(df_clean)} records, {len(df_clean.columns)} columns")
        logger.info(f"Target distribution: {df_clean['readmitted_30d'].value_counts().to_dict()}")
        
        # Step 2: Data Splitting
        logger.info("Step 2: Splitting data into train/validation/test sets")
        
        # Check if we have temporal data for temporal split
        if 'discharge_datetime' in df_clean.columns:
            try:
                train_df, val_df, test_df = data_loader.temporal_split(
                    df_clean, 
                    date_column='discharge_datetime',
                    test_size=0.2,
                    val_size=0.2
                )
                logger.info("Used temporal split based on discharge date")
            except Exception as e:
                logger.warning(f"Temporal split failed: {e}, using random split")
                train_df, val_df, test_df = data_loader.split_data(
                    df_clean, 
                    test_size=0.2, 
                    val_size=0.2, 
                    random_state=args.random_state
                )
        else:
            train_df, val_df, test_df = data_loader.split_data(
                df_clean, 
                test_size=0.2, 
                val_size=0.2, 
                random_state=args.random_state
            )
        
        # Save split data
        data_loader.save_split_data(train_df, val_df, test_df)
        
        # Step 3: Feature Engineering
        logger.info("Step 3: Feature engineering")
        feature_engineer = FeatureEngineer(random_state=args.random_state)
        
        # Fit and transform training data
        X_train = feature_engineer.fit_transform(train_df)
        y_train = X_train['readmitted_30d']
        X_train = X_train.drop('readmitted_30d', axis=1)
        
        # Transform validation and test data
        X_val = feature_engineer.transform(val_df)
        y_val = X_val['readmitted_30d']
        X_val = X_val.drop('readmitted_30d', axis=1)
        
        X_test = feature_engineer.transform(test_df)
        y_test = X_test['readmitted_30d']
        X_test = X_test.drop('readmitted_30d', axis=1)
        
        logger.info(f"Feature engineering completed: {len(X_train.columns)} features")
        logger.info(f"Training set: {X_train.shape}")
        logger.info(f"Validation set: {X_val.shape}")
        logger.info(f"Test set: {X_test.shape}")
        
        # Log feature summary
        feature_summary = feature_engineer.get_feature_summary()
        logger.info(f"Feature summary: {feature_summary}")
        
        # Step 4: Model Training
        logger.info("Step 4: Training models")
        model_trainer = ModelTrainer(
            experiment_name="readmission_prediction",
            random_state=args.random_state
        )
        
        # Train all models
        training_results = model_trainer.train_models(
            X_train, y_train, X_val, y_val, feature_engineer
        )
        
        # Log training results
        logger.info("Training results:")
        for model_name, results in training_results.items():
            if 'error' not in results:
                logger.info(f"  {model_name}:")
                for metric, value in results.items():
                    logger.info(f"    {metric}: {value:.4f}")
            else:
                logger.error(f"  {model_name}: {results['error']}")
        
        # Step 5: Cross-validation (optional)
        if args.cross_validate and model_trainer.best_model is not None:
            logger.info("Step 5: Cross-validation")
            cv_results = model_trainer.cross_validate(X_train, y_train)
            
            # Log CV results
            logger.info("Cross-validation results:")
            for metric, scores in cv_results.items():
                logger.info(f"  {metric}: {np.mean(scores):.4f} (+/- {np.std(scores) * 2:.4f})")
        
        # Step 6: Final Evaluation on Test Set
        logger.info("Step 6: Final evaluation on test set")
        if model_trainer.best_model is not None:
            # Get predictions on test set
            y_test_pred_proba = model_trainer.predict(X_test)
            y_test_pred = (y_test_pred_proba > 0.5).astype(int)
            
            # Calculate test metrics
            from sklearn.metrics import (
                roc_auc_score, average_precision_score, brier_score_loss,
                classification_report, confusion_matrix
            )
            
            test_metrics = {
                'auroc': roc_auc_score(y_test, y_test_pred_proba),
                'auprc': average_precision_score(y_test, y_test_pred_proba),
                'brier_score': brier_score_loss(y_test, y_test_pred_proba)
            }
            
            logger.info("Test set performance:")
            for metric, value in test_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
            
            # Classification report
            logger.info("Classification report:")
            logger.info(classification_report(y_test, y_test_pred))
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_test_pred)
            logger.info("Confusion matrix:")
            logger.info(cm)
            
            # Check acceptance criteria
            logger.info("Checking acceptance criteria:")
            
            # AUPRC improvement over baseline
            baseline_auprc = training_results.get('logreg_calibrated', {}).get('auprc', 0)
            if 'auprc' in test_metrics and baseline_auprc > 0:
                improvement = (test_metrics['auprc'] - baseline_auprc) / baseline_auprc
                logger.info(f"  AUPRC improvement over baseline: {improvement:.2%}")
                if improvement >= 0.20:
                    logger.info("  ✅ AUPRC improvement >= 20% (ACCEPTED)")
                else:
                    logger.warning(f"  ❌ AUPRC improvement {improvement:.2%} < 20% (NOT ACCEPTED)")
            
            # Brier score
            if 'brier_score' in test_metrics:
                if test_metrics['brier_score'] <= 0.18:
                    logger.info("  ✅ Brier Score <= 0.18 (ACCEPTED)")
                else:
                    logger.warning(f"  ❌ Brier Score {test_metrics['brier_score']:.4f} > 0.18 (NOT ACCEPTED)")
        
        # Step 7: Save Models
        logger.info("Step 7: Saving models")
        model_trainer.save_models(args.output_dir)
        
        # Step 8: Generate Training Summary
        logger.info("Step 8: Generating training summary")
        summary = {
            'training_timestamp': datetime.now().isoformat(),
            'data_info': {
                'total_records': len(df_clean),
                'train_size': len(train_df),
                'val_size': len(val_df),
                'test_size': len(test_df),
                'target_distribution': df_clean['readmitted_30d'].value_counts().to_dict()
            },
            'feature_info': feature_summary,
            'model_info': model_trainer.get_model_summary(),
            'training_results': training_results,
            'test_metrics': test_metrics if 'test_metrics' in locals() else None
        }
        
        # Save summary
        import json
        summary_file = Path(args.output_dir) / "training_summary.json"
        summary_file.parent.mkdir(exist_ok=True)
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Training summary saved to {summary_file}")
        logger.info("Training pipeline completed successfully!")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

