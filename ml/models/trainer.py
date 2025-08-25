"""
Model training for the Hospital Readmission Risk Predictor.
Implements the specified models with MLflow tracking and calibration.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import logging
from datetime import datetime
import joblib
import os
from pathlib import Path

# ML libraries
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    classification_report, confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.utils.class_weight import compute_class_weight

# Gradient boosting
import xgboost as xgb
import catboost as cb

# MLflow
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.catboost

from ..features.engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Model trainer for hospital readmission prediction."""
    
    def __init__(self, experiment_name: str = "readmission_prediction", 
                 random_state: int = 42):
        self.experiment_name = experiment_name
        self.random_state = random_state
        self.models = {}
        self.feature_engineer = None
        self.best_model = None
        self.best_model_name = None
        self.calibration_models = {}
        
        # Set up MLflow
        mlflow.set_experiment(experiment_name)
        
        # Model configurations as specified
        self.model_configs = {
            'logreg_calibrated': {
                'type': 'LogisticRegression',
                'params': {
                    'penalty': 'l2',
                    'max_iter': 500,
                    'class_weight': 'balanced',
                    'random_state': random_state
                },
                'calibration': 'isotonic'
            },
            'xgboost_main': {
                'type': 'XGBClassifier',
                'params': {
                    'n_estimators': 1000,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': random_state,
                    'n_jobs': -1
                }
            },
            'catboost_alt': {
                'type': 'CatBoostClassifier',
                'params': {
                    'iterations': 800,
                    'depth': 6,
                    'learning_rate': 0.05,
                    'loss_function': 'Logloss',
                    'eval_metric': 'AUC',
                    'random_seed': random_state,
                    'verbose': False
                }
            }
        }
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                     X_val: pd.DataFrame, y_val: pd.Series,
                     feature_engineer: FeatureEngineer) -> Dict[str, Any]:
        """
        Train all specified models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            feature_engineer: Fitted feature engineer
            
        Returns:
            Dictionary with training results
        """
        self.feature_engineer = feature_engineer
        results = {}
        
        logger.info("Starting model training pipeline")
        
        # Calculate class weights for imbalanced data
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        
        # Train each model
        for model_name, config in self.model_configs.items():
            logger.info(f"Training {model_name}...")
            
            try:
                with mlflow.start_run(run_name=f"train_{model_name}"):
                    # Log parameters
                    mlflow.log_params(config['params'])
                    mlflow.log_param("model_type", config['type'])
                    mlflow.log_param("calibration", config.get('calibration', 'none'))
                    
                    # Train model
                    model, metrics = self._train_single_model(
                        model_name, config, X_train, y_train, X_val, y_val, class_weight_dict
                    )
                    
                    # Store model and results
                    self.models[model_name] = model
                    results[model_name] = metrics
                    
                    # Log metrics
                    for metric_name, metric_value in metrics.items():
                        mlflow.log_metric(metric_name, metric_value)
                    
                    # Log model
                    self._log_model(model, model_name)
                    
                    logger.info(f"Completed training {model_name}")
                    
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        # Select best model based on AUPRC
        self._select_best_model(results)
        
        # Calibrate best model
        if self.best_model is not None:
            self._calibrate_best_model(X_val, y_val)
        
        return results
    
    def _train_single_model(self, model_name: str, config: Dict[str, Any],
                           X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame, y_val: pd.Series,
                           class_weight_dict: Dict[int, float]) -> Tuple[Any, Dict[str, float]]:
        """Train a single model."""
        
        if config['type'] == 'LogisticRegression':
            model = LogisticRegression(**config['params'])
            model.fit(X_train, y_train)
            
        elif config['type'] == 'XGBClassifier':
            # Update class weights for XGBoost
            params = config['params'].copy()
            params['scale_pos_weight'] = class_weight_dict[1] / class_weight_dict[0]
            
            model = xgb.XGBClassifier(**params)
            
            # Use early stopping with eval_set
            eval_set = [(X_val, y_val)]
            model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
            
        elif config['type'] == 'CatBoostClassifier':
            # Update class weights for CatBoost
            params = config['params'].copy()
            params['class_weights'] = class_weight_dict
            
            model = cb.CatBoostClassifier(**params)
            model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
            
        else:
            raise ValueError(f"Unknown model type: {config['type']}")
        
        # Evaluate model
        metrics = self._evaluate_model(model, X_val, y_val)
        
        return model, metrics
    
    def _evaluate_model(self, model: Any, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
        """Evaluate a trained model."""
        # Predictions
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Metrics
        metrics = {
            'auroc': roc_auc_score(y_val, y_pred_proba),
            'auprc': average_precision_score(y_val, y_pred_proba),
            'brier_score': brier_score_loss(y_val, y_pred_proba),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred, zero_division=0),
            'f1': f1_score(y_val, y_pred, zero_division=0)
        }
        
        return metrics
    
    def _select_best_model(self, results: Dict[str, Any]):
        """Select the best model based on AUPRC."""
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not valid_results:
            logger.warning("No models trained successfully")
            return
        
        # Find best model by AUPRC
        best_model_name = max(valid_results.keys(), 
                             key=lambda x: valid_results[x]['auprc'])
        
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]
        
        logger.info(f"Best model selected: {best_model_name}")
        logger.info(f"Best AUPRC: {valid_results[best_model_name]['auprc']:.4f}")
    
    def _calibrate_best_model(self, X_val: pd.DataFrame, y_val: pd.Series):
        """Calibrate the best model using isotonic regression."""
        if self.best_model is None:
            return
        
        logger.info("Calibrating best model...")
        
        # Create calibrated classifier
        calibrated_model = CalibratedClassifierCV(
            self.best_model,
            cv='prefit',
            method='isotonic'
        )
        
        # Fit calibration on validation set
        calibrated_model.fit(X_val, y_val)
        
        # Store calibrated model
        self.calibration_models[self.best_model_name] = calibrated_model
        
        # Evaluate calibrated model
        calibrated_metrics = self._evaluate_model(calibrated_model, X_val, y_val)
        
        logger.info(f"Calibrated model AUPRC: {calibrated_metrics['auprc']:.4f}")
        logger.info(f"Calibrated model Brier Score: {calibrated_metrics['brier_score']:.4f}")
    
    def _log_model(self, model: Any, model_name: str):
        """Log model to MLflow."""
        try:
            if hasattr(model, 'feature_importances_'):
                # Tree-based model
                if 'xgboost' in model_name:
                    mlflow.xgboost.log_model(model, model_name)
                elif 'catboost' in model_name:
                    mlflow.catboost.log_model(model, model_name)
                else:
                    mlflow.sklearn.log_model(model, model_name)
            else:
                # Linear model
                mlflow.sklearn.log_model(model, model_name)
                
        except Exception as e:
            logger.warning(f"Could not log model {model_name}: {e}")
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, 
                      cv_folds: int = 5) -> Dict[str, List[float]]:
        """Perform cross-validation on the best model."""
        if self.best_model is None:
            logger.warning("No best model available for cross-validation")
            return {}
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        cv_scores = {
            'auroc': cross_val_score(self.best_model, X, y, cv=cv, scoring='roc_auc'),
            'auprc': cross_val_score(self.best_model, X, y, cv=cv, scoring='average_precision'),
            'precision': cross_val_score(self.best_model, X, y, cv=cv, scoring='precision'),
            'recall': cross_val_score(self.best_model, X, y, cv=cv, scoring='recall'),
            'f1': cross_val_score(self.best_model, X, y, cv=cv, scoring='f1')
        }
        
        # Log CV results
        for metric, scores in cv_scores.items():
            logger.info(f"CV {metric}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return cv_scores
    
    def save_models(self, output_dir: str = "models"):
        """Save trained models to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save feature engineer
        if self.feature_engineer is not None:
            joblib.dump(self.feature_engineer, output_path / "feature_engineer.pkl")
            logger.info("Saved feature engineer")
        
        # Save best model
        if self.best_model is not None:
            if self.best_model_name in self.calibration_models:
                # Save calibrated model
                model_to_save = self.calibration_models[self.best_model_name]
            else:
                model_to_save = self.best_model
            
            joblib.dump(model_to_save, output_path / "best_model.pkl")
            logger.info("Saved best model")
        
        # Save all models
        for model_name, model in self.models.items():
            joblib.dump(model, output_path / f"{model_name}.pkl")
        
        # Save metadata
        metadata = {
            'best_model_name': self.best_model_name,
            'training_timestamp': datetime.now().isoformat(),
            'feature_engineer_info': self.feature_engineer.get_feature_summary() if self.feature_engineer else None
        }
        
        joblib.dump(metadata, output_path / "training_metadata.pkl")
        logger.info("Saved training metadata")
    
    def load_models(self, model_dir: str = "models"):
        """Load trained models from disk."""
        model_path = Path(model_dir)
        
        if not model_path.exists():
            logger.warning(f"Model directory {model_dir} does not exist")
            return False
        
        try:
            # Load feature engineer
            self.feature_engineer = joblib.load(model_path / "feature_engineer.pkl")
            
            # Load best model
            self.best_model = joblib.load(model_path / "best_model.pkl")
            
            # Load metadata
            metadata = joblib.load(model_path / "training_metadata.pkl")
            self.best_model_name = metadata['best_model_name']
            
            logger.info("Models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def predict(self, X: pd.DataFrame, use_calibrated: bool = True) -> np.ndarray:
        """Make predictions using the best model."""
        if self.best_model is None:
            raise ValueError("No trained model available")
        
        if use_calibrated and self.best_model_name in self.calibration_models:
            model = self.calibration_models[self.best_model_name]
        else:
            model = self.best_model
        
        return model.predict_proba(X)[:, 1]
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of trained models."""
        return {
            'best_model_name': self.best_model_name,
            'total_models': len(self.models),
            'model_names': list(self.models.keys()),
            'calibrated_models': list(self.calibration_models.keys()),
            'feature_engineer_info': self.feature_engineer.get_feature_summary() if self.feature_engineer else None
        }


# Import missing functions
from sklearn.metrics import precision_score, recall_score, f1_score
