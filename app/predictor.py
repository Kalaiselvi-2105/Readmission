#!/usr/bin/env python3
"""
ReadmissionPredictor service for the Hospital Readmission Risk Predictor.
Handles model loading, predictions, and explanations.
"""

import os
import logging
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
import shap

logger = logging.getLogger(__name__)


class ReadmissionPredictor:
    """Main predictor class for hospital readmission risk."""
    
    def __init__(self, models_dir: str = "models/"):
        """
        Initialize the predictor with trained models.
        
        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = Path(models_dir)
        self.models = {}
        self.feature_engineer = None
        self.best_model = None
        self.feature_names = []
        self.explainer = None
        
        # Load models
        self._load_models()
        
    def _load_models(self):
        """Load all trained models and feature engineer."""
        try:
            # Load feature engineer
            feature_engineer_path = self.models_dir / "feature_engineer.pkl"
            if feature_engineer_path.exists():
                self.feature_engineer = joblib.load(feature_engineer_path)
                self.feature_names = self.feature_engineer.feature_names
                logger.info(f"Loaded feature engineer with {len(self.feature_names)} features")
            
            # Load best model
            best_model_path = self.models_dir / "best_model.pkl"
            if best_model_path.exists():
                self.best_model = joblib.load(best_model_path)
                logger.info(f"Loaded best model: {type(self.best_model).__name__}")
            
            # Load individual models for explanations
            model_files = {
                'logistic_regression': 'logreg_calibrated.pkl',
                'xgboost': 'xgboost_main.pkl',
                'catboost': 'catboost_alt.pkl'
            }
            
            for model_name, filename in model_files.items():
                model_path = self.models_dir / filename
                if model_path.exists():
                    self.models[model_name] = joblib.load(model_path)
                    logger.info(f"Loaded {model_name} model")
            
            # Initialize SHAP explainer for the best model
            if self.best_model is not None:
                self._initialize_explainer()
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def _initialize_explainer(self):
        """Initialize SHAP explainer for the best model."""
        try:
            if hasattr(self.best_model, 'feature_importances_'):
                # Tree-based model
                self.explainer = shap.TreeExplainer(self.best_model)
            else:
                # Linear model - use background data
                self.explainer = shap.LinearExplainer(self.best_model, self._get_background_data())
            logger.info("Initialized SHAP explainer")
        except Exception as e:
            logger.warning(f"Could not initialize SHAP explainer: {e}")
            self.explainer = None
    
    def _get_background_data(self):
        """Get background data for SHAP explainer."""
        # For now, return a small sample of zeros
        # In production, you'd want to use actual training data
        return np.zeros((100, len(self.feature_names)))
    
    def predict(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict readmission risk for a single patient.
        
        Args:
            patient_data: Patient data dictionary
            
        Returns:
            Prediction results with risk score and metadata
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame([patient_data])
            
            # Transform features
            if self.feature_engineer is None:
                raise ValueError("Feature engineer not loaded")
            
            df_transformed = self.feature_engineer.transform(df)
            
            # Extract features (excluding target if present)
            feature_cols = [col for col in self.feature_names if col in df_transformed.columns]
            X = df_transformed[feature_cols]
            
            # Ensure all features are present and numeric
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                # Fill missing features with 0
                for feature in missing_features:
                    X[feature] = 0
            
            # Reorder columns to match training order
            X = X[self.feature_names]
            
            # Make prediction
            if self.best_model is None:
                raise ValueError("Best model not loaded")
            
            # Get raw prediction probabilities
            if hasattr(self.best_model, 'predict_proba'):
                proba = self.best_model.predict_proba(X)[0]
                risk_score = proba[1]  # Probability of readmission
            else:
                # Fallback to decision function
                decision = self.best_model.decision_function(X)[0]
                risk_score = 1 / (1 + np.exp(-decision))  # Convert to probability
            
            # Determine risk category
            risk_category = self._categorize_risk(risk_score)
            
            # Get confidence level
            confidence = self._get_confidence_level(risk_score)
            
            return {
                'patient_id': patient_data.get('patient_id', 'unknown'),
                'risk_score': float(risk_score),
                'risk_category': risk_category,
                'confidence': confidence,
                'prediction_timestamp': pd.Timestamp.now().isoformat(),
                'model_version': '1.0.0',
                'features_used': len(self.feature_names)
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    def batch_predict(self, patient_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Predict readmission risk for multiple patients.
        
        Args:
            patient_data_list: List of patient data dictionaries
            
        Returns:
            List of prediction results
        """
        results = []
        for patient_data in patient_data_list:
            try:
                result = self.predict(patient_data)
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting for patient {patient_data.get('patient_id', 'unknown')}: {e}")
                results.append({
                    'patient_id': patient_data.get('patient_id', 'unknown'),
                    'error': str(e),
                    'risk_score': None,
                    'risk_category': 'Error',
                    'confidence': 'Low'
                })
        return results
    
    def explain_prediction(self, patient_data: Dict[str, Any], 
                          top_features: int = 10) -> Dict[str, Any]:
        """
        Explain the prediction using SHAP values.
        
        Args:
            patient_data: Patient data dictionary
            top_features: Number of top features to return
            
        Returns:
            Explanation with feature contributions
        """
        try:
            # Get prediction first
            prediction = self.predict(patient_data)
            
            # Transform features
            df = pd.DataFrame([patient_data])
            df_transformed = self.feature_engineer.transform(df)
            
            # Extract features
            feature_cols = [col for col in self.feature_names if col in df_transformed.columns]
            X = df_transformed[feature_cols]
            
            # Fill missing features
            for feature in self.feature_names:
                if feature not in X.columns:
                    X[feature] = 0
            
            X = X[self.feature_names]
            
            # Get SHAP values
            if self.explainer is not None:
                shap_values = self.explainer.shap_values(X)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # For binary classification, get positive class
                
                # Create feature contribution list
                feature_contributions = []
                for i, feature in enumerate(self.feature_names):
                    contribution = float(shap_values[0, i])
                    feature_contributions.append({
                        'feature_name': feature,
                        'contribution': contribution,
                        'abs_contribution': abs(contribution)
                    })
                
                # Sort by absolute contribution
                feature_contributions.sort(key=lambda x: x['abs_contribution'], reverse=True)
                
                # Get top features
                top_contributions = feature_contributions[:top_features]
                
                return {
                    'patient_id': patient_data.get('patient_id', 'unknown'),
                    'risk_score': prediction['risk_score'],
                    'risk_category': prediction['risk_category'],
                    'feature_contributions': top_contributions,
                    'explanation_timestamp': pd.Timestamp.now().isoformat()
                }
            else:
                # Fallback explanation without SHAP
                return {
                    'patient_id': patient_data.get('patient_id', 'unknown'),
                    'risk_score': prediction['risk_score'],
                    'risk_category': prediction['risk_category'],
                    'feature_contributions': [],
                    'explanation_timestamp': pd.Timestamp.now().isoformat(),
                    'note': 'SHAP explainer not available'
                }
                
        except Exception as e:
            logger.error(f"Error explaining prediction: {e}")
            raise
    
    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize risk score into risk levels."""
        if risk_score < 0.2:
            return 'Low'
        elif risk_score < 0.4:
            return 'Medium-Low'
        elif risk_score < 0.6:
            return 'Medium'
        elif risk_score < 0.8:
            return 'Medium-High'
        else:
            return 'High'
    
    def _get_confidence_level(self, risk_score: float) -> str:
        """Get confidence level based on risk score distance from decision boundary."""
        distance_from_boundary = abs(risk_score - 0.5)
        if distance_from_boundary > 0.3:
            return 'High'
        elif distance_from_boundary > 0.15:
            return 'Medium'
        else:
            return 'Low'
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            'feature_count': len(self.feature_names),
            'feature_names': self.feature_names[:10],  # First 10 features
            'models_loaded': list(self.models.keys()),
            'best_model_type': type(self.best_model).__name__ if self.best_model else None,
            'shap_explainer_available': self.explainer is not None,
            'last_updated': pd.Timestamp.now().isoformat()
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Health check for the predictor service."""
        return {
            'status': 'healthy' if self.best_model is not None else 'unhealthy',
            'models_loaded': len(self.models),
            'feature_engineer_loaded': self.feature_engineer is not None,
            'best_model_loaded': self.best_model is not None,
            'feature_count': len(self.feature_names),
            'timestamp': pd.Timestamp.now().isoformat()
        }

