"""
Feature engineering for the Hospital Readmission Risk Predictor.
Implements temporal, utilization, and trend features as specified.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import category_encoders as ce

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering pipeline for hospital readmission data."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.target_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names = []
        self.categorical_features = []
        self.numeric_features = []
        
    def fit_transform(self, df: pd.DataFrame, target_col: str = 'readmitted_30d') -> pd.DataFrame:
        """
        Fit the feature engineering pipeline and transform the data.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            
        Returns:
            Transformed feature matrix
        """
        logger.info("Starting feature engineering pipeline")
        
        # Store original column names
        original_columns = df.columns.tolist()
        
        # Create features
        df_features = self._create_temporal_features(df)
        df_features = self._create_utilization_features(df_features)
        df_features = self._create_trend_features(df_features)
        df_features = self._create_demographic_features(df_features)
        df_features = self._create_clinical_features(df_features)
        
        # Handle missing values
        df_features = self._handle_missing_values(df_features)
        
        # Encode categorical variables
        df_features = self._encode_categorical_features(df_features, target_col)
        
        # Scale numeric features
        df_features = self._scale_numeric_features(df_features)
        
        # Store feature information
        self.feature_names = [col for col in df_features.columns if col != target_col]
        
        # Exclude non-predictive features consistently
        exclude_features = ['patient_id', 'admission_id', 'admit_datetime', 'discharge_datetime']
        self.feature_names = [col for col in self.feature_names if col not in exclude_features]
        
        # Ensure all remaining features are numeric
        for col in self.feature_names:
            if col in df_features.columns:
                if df_features[col].dtype == 'object':
                    # Convert any remaining object columns to numeric
                    df_features[col] = pd.to_numeric(df_features[col], errors='coerce').fillna(0)
                elif df_features[col].dtype == 'category':
                    # Convert category columns to numeric
                    df_features[col] = df_features[col].cat.codes
                elif df_features[col].dtype == 'datetime64[ns]':
                    # Convert datetime columns to numeric (days since epoch)
                    df_features[col] = (df_features[col] - pd.Timestamp('1970-01-01')).dt.days
                elif df_features[col].dtype == 'bool':
                    # Convert boolean columns to int
                    df_features[col] = df_features[col].astype(int)
        
        # Final check: ensure all features are numeric and select only feature columns
        final_features = df_features[self.feature_names].copy()
        for col in final_features.columns:
            if final_features[col].dtype not in ['int64', 'float64']:
                final_features[col] = pd.to_numeric(final_features[col], errors='coerce').fillna(0)
        
        # Add target column back
        final_features[target_col] = df_features[target_col]
        
        self.categorical_features = [col for col in self.categorical_features if col in final_features.columns]
        self.numeric_features = [col for col in self.numeric_features if col in final_features.columns]
        
        logger.info(f"Feature engineering completed. Final feature count: {len(self.feature_names)}")
        logger.info(f"Feature names: {self.feature_names[:10]}...")  # Show first 10
        logger.info(f"Final data types: {final_features[self.feature_names].dtypes.value_counts()}")
        
        return final_features
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted pipeline.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed feature matrix
        """
        logger.info("Transforming new data with fitted pipeline")
        
        # Store target column if it exists
        target_col = None
        if 'readmitted_30d' in df.columns:
            target_col = df['readmitted_30d']
        
        # Create features
        df_features = self._create_temporal_features(df)
        df_features = self._create_utilization_features(df_features)
        df_features = self._create_trend_features(df_features)
        df_features = self._create_demographic_features(df_features)
        df_features = self._create_clinical_features(df_features)
        
        # Handle missing values
        df_features = self._handle_missing_values_transform(df_features)
        
        # Encode categorical features
        df_features = self._encode_categorical_features_transform(df_features)
        
        # Scale numeric features
        df_features = self._scale_numeric_features_transform(df_features)
        
        # Ensure all features are numeric
        for col in df_features.columns:
            if col != 'readmitted_30d':  # Use string comparison instead of variable comparison
                if df_features[col].dtype == 'object':
                    # Convert any remaining object columns to numeric
                    df_features[col] = pd.to_numeric(df_features[col], errors='coerce').fillna(0)
                elif df_features[col].dtype == 'category':
                    # Convert category columns to numeric
                    df_features[col] = df_features[col].cat.codes
                elif df_features[col].dtype == 'datetime64[ns]':
                    # Convert datetime columns to numeric (days since epoch)
                    df_features[col] = (df_features[col] - pd.Timestamp('1970-01-01')).dt.days
                elif df_features[col].dtype == 'bool':
                    # Convert boolean columns to int
                    df_features[col] = df_features[col].astype(int)
        
        # Final check: ensure all features are numeric
        final_features = df_features.copy()
        for col in final_features.columns:
            if col != 'readmitted_30d' and final_features[col].dtype not in ['int64', 'float64']:
                final_features[col] = pd.to_numeric(final_features[col], errors='coerce').fillna(0)
        
        # Select only feature columns (excluding target and ID columns)
        feature_cols = [col for col in final_features.columns if col not in ['readmitted_30d', 'patient_id', 'admission_id', 'admit_datetime', 'discharge_datetime']]
        final_features = final_features[feature_cols]
        
        # Add target column back if it existed
        if target_col is not None:
            final_features['readmitted_30d'] = target_col
        
        logger.info(f"Transform completed. Final data types: {final_features.dtypes.value_counts()}")
        logger.info(f"Transform completed. Feature count: {len(feature_cols)}")
        
        return final_features
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features as specified."""
        df_features = df.copy()
        
        # Age buckets
        df_features['age_bucket'] = pd.cut(
            df_features['age'], 
            bins=[0, 18, 35, 50, 65, 80, 120], 
            labels=['0-18', '19-35', '36-50', '51-65', '66-80', '80+']
        )
        
        # Week of year discharge
        df_features['week_of_year_discharge'] = pd.to_datetime(
            df_features['discharge_datetime']
        ).dt.isocalendar().week
        
        # Discharge hour (one-hot encoded)
        discharge_hour = pd.to_datetime(df_features['discharge_datetime']).dt.hour
        for hour in range(24):
            df_features[f'discharge_hour_{hour}'] = (discharge_hour == hour).astype(int)
        
        # Season
        discharge_month = pd.to_datetime(df_features['discharge_datetime']).dt.month
        df_features['season'] = pd.cut(
            discharge_month,
            bins=[0, 3, 6, 9, 12],
            labels=['Winter', 'Spring', 'Summer', 'Fall']
        )
        
        # Day of week
        df_features['day_of_week'] = pd.to_datetime(
            df_features['discharge_datetime']
        ).dt.day_name()
        
        # Weekend flag
        df_features['is_weekend'] = pd.to_datetime(
            df_features['discharge_datetime']
        ).dt.weekday >= 5
        
        logger.info("Created temporal features")
        return df_features
    
    def _create_utilization_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create utilization features as specified."""
        df_features = df.copy()
        
        # Readmission rate features
        df_features['prior_readmit_rate_30d'] = df_features['prior_readmit_30d'] / 1.0
        df_features['prior_readmit_rate_365d'] = df_features['prior_readmit_365d'] / 365.0
        
        # ED visit rate
        df_features['ed_visit_rate_180d'] = df_features['ed_visits_180d'] / 180.0
        
        # Length of stay categories
        df_features['los_category'] = pd.cut(
            df_features['los_days'],
            bins=[0, 1, 3, 7, 14, float('inf')],
            labels=['0-1', '2-3', '4-7', '8-14', '14+']
        )
        
        # ICU stay impact
        df_features['icu_los_ratio'] = np.where(
            df_features['icu_stay'],
            df_features['los_days'] * 1.5,  # ICU stays count more
            df_features['los_days']
        )
        
        # Procedure intensity
        df_features['procedure_intensity'] = df_features['procedures_count'] / df_features['los_days']
        
        # Medication intensity
        df_features['medication_intensity'] = df_features['meds_count'] / df_features['los_days']
        
        logger.info("Created utilization features")
        return df_features
    
    def _create_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create trend features as specified."""
        df_features = df.copy()
        
        # Lab delta features (if we had multiple measurements)
        # For now, create stability indices based on single values
        
        # Vitals instability index
        vitals_cols = [col for col in df_features.columns if col.startswith('vitals_')]
        if vitals_cols:
            # Calculate coefficient of variation for vitals
            vitals_data = df_features[vitals_cols].fillna(0)
            df_features['vitals_instability_index'] = vitals_data.std(axis=1) / (vitals_data.mean(axis=1) + 1e-8)
        
        # Lab abnormality flags
        lab_cols = [col for col in df_features.columns if col.startswith('labs_')]
        if lab_cols:
            # Create abnormality flags based on clinical ranges
            lab_ranges = {
                'labs_hgb': (12.0, 16.0),      # Hemoglobin
                'labs_sodium': (135, 145),      # Sodium
                'labs_potassium': (3.5, 5.0),  # Potassium
                'labs_creatinine': (0.6, 1.2), # Creatinine
                'labs_glucose': (70, 140)       # Glucose
            }
            
            for lab_col, (low, high) in lab_ranges.items():
                if lab_col in df_features.columns:
                    df_features[f'{lab_col}_abnormal'] = (
                        (df_features[lab_col] < low) | (df_features[lab_col] > high)
                    ).astype(int)
            
            # Count abnormal labs
            abnormal_cols = [col for col in df_features.columns if col.endswith('_abnormal')]
            if abnormal_cols:
                df_features['abnormal_labs_count'] = df_features[abnormal_cols].sum(axis=1)
        
        # Trend indicators
        df_features['age_los_interaction'] = df_features['age'] * df_features['los_days']
        df_features['comorbidity_los_interaction'] = df_features['charlson_index'] * df_features['los_days']
        
        logger.info("Created trend features")
        return df_features
    
    def _create_demographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create demographic and social features."""
        df_features = df.copy()
        
        # Age-related features
        df_features['age_squared'] = df_features['age'] ** 2
        df_features['elderly_flag'] = (df_features['age'] >= 65).astype(int)
        df_features['young_adult_flag'] = (df_features['age'] >= 18) & (df_features['age'] <= 35)
        
        # Insurance categories
        df_features['medicare_flag'] = (df_features['insurance'] == 'Medicare').astype(int)
        df_features['medicaid_flag'] = (df_features['insurance'] == 'Medicaid').astype(int)
        df_features['private_insurance_flag'] = (df_features['insurance'] == 'Private').astype(int)
        
        # Deprivation index categories
        df_features['deprivation_quartile'] = pd.qcut(
            df_features['deprivation_index'],
            q=4,
            labels=['Low', 'Medium-Low', 'Medium-High', 'High']
        )
        
        # Geographic features (simplified)
        df_features['zip_region'] = df_features['zip_code'].str[:2]
        
        logger.info("Created demographic features")
        return df_features
    
    def _create_clinical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create clinical and medical features."""
        df_features = df.copy()
        
        # Charlson index categories
        df_features['charlson_category'] = pd.cut(
            df_features['charlson_index'],
            bins=[0, 1, 2, 4, float('inf')],
            labels=['None', 'Mild', 'Moderate', 'Severe']
        )
        
        # Risk flags
        df_features['high_risk_patient'] = (
            (df_features['charlson_index'] > 3) |
            (df_features['prior_readmit_30d'] > 0) |
            (df_features['age'] > 75)
        ).astype(int)
        
        # Follow-up urgency
        df_features['followup_urgency'] = np.where(
            df_features['followup_scheduled'],
            np.where(df_features['days_to_followup'] <= 7, 'Urgent', 'Standard'),
            'None'
        )
        
        # Discharge complexity
        discharge_complexity = (
            df_features['procedures_count'] +
            df_features['meds_count'] * 0.5 +
            df_features['charlson_index'] * 2
        )
        df_features['discharge_complexity_score'] = discharge_complexity
        
        # Medication risk
        df_features['medication_risk_score'] = (
            df_features['meds_count'] * 0.1 +
            df_features['high_risk_meds_flag'].astype(int) * 2
        )
        
        logger.info("Created clinical features")
        return df_features
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the feature matrix."""
        df_features = df.copy()
        
        # Identify numeric and categorical columns
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        categorical_cols = df_features.select_dtypes(include=['object', 'category']).columns
        
        # Store for later use
        self.numeric_features = numeric_cols.tolist()
        self.categorical_features = categorical_cols.tolist()
        
        # Fill numeric missing values with median
        if len(numeric_cols) > 0:
            df_features[numeric_cols] = self.imputer.fit_transform(df_features[numeric_cols])
        
        # Fill categorical missing values with mode
        for col in categorical_cols:
            if df_features[col].isnull().any():
                mode_val = df_features[col].mode().iloc[0] if not df_features[col].mode().empty else 'Unknown'
                df_features[col] = df_features[col].fillna(mode_val)
        
        logger.info("Handled missing values")
        return df_features
    
    def _handle_missing_values_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the feature matrix during transform."""
        df_features = df.copy()
        
        # Identify numeric and categorical columns
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        categorical_cols = df_features.select_dtypes(include=['object', 'category']).columns
        
        # Store for later use
        self.numeric_features = numeric_cols.tolist()
        self.categorical_features = categorical_cols.tolist()
        
        # Fill numeric missing values with median
        if len(numeric_cols) > 0:
            df_features[numeric_cols] = self.imputer.transform(df_features[numeric_cols])
        
        # Fill categorical missing values with mode
        for col in categorical_cols:
            if df_features[col].isnull().any():
                mode_val = df_features[col].mode().iloc[0] if not df_features[col].mode().empty else 'Unknown'
                df_features[col] = df_features[col].fillna(mode_val)
        
        logger.info("Handled missing values during transform")
        return df_features
    
    def _encode_categorical_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Encode categorical features using target encoding and label encoding."""
        df_features = df.copy()
        
        # Get all categorical columns (object type)
        categorical_cols = df_features.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target column if it's categorical
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)
        
        # Target encoding for high cardinality categorical variables
        high_cardinality_cols = ['patient_id', 'admission_id', 'zip_code']
        for col in high_cardinality_cols:
            if col in categorical_cols:
                # Target encoding
                target_encoder = ce.TargetEncoder(cols=[col])
                df_features = target_encoder.fit_transform(df_features, df_features[target_col])
                self.target_encoders[col] = target_encoder
                categorical_cols.remove(col)
        
        # Label encoding for other categorical variables
        for col in categorical_cols:
            if col in df_features.columns:
                # Fill missing values with a default category
                df_features[col] = df_features[col].fillna('Unknown')
                label_encoder = LabelEncoder()
                df_features[col] = label_encoder.fit_transform(df_features[col].astype(str))
                self.label_encoders[col] = label_encoder
        
        # Convert boolean columns to int
        boolean_cols = df_features.select_dtypes(include=['bool']).columns
        for col in boolean_cols:
            df_features[col] = df_features[col].astype(int)
        
        logger.info("Encoded categorical features")
        return df_features
    
    def _encode_categorical_features_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical features using fitted encoders."""
        df_features = df.copy()
        
        # Apply target encoding
        for col, encoder in self.target_encoders.items():
            if col in df_features.columns:
                df_features = encoder.transform(df_features)
        
        # Apply label encoding
        for col, encoder in self.label_encoders.items():
            if col in df_features.columns:
                # Handle unseen categories
                df_features[col] = df_features[col].fillna('Unknown')
                df_features[col] = df_features[col].astype(str)
                df_features[col] = df_features[col].map(
                    lambda x: x if x in encoder.classes_ else encoder.classes_[0]
                )
                df_features[col] = encoder.transform(df_features[col])
        
        # Convert boolean columns to int
        boolean_cols = df_features.select_dtypes(include=['bool']).columns
        for col in boolean_cols:
            df_features[col] = df_features[col].astype(int)
        
        return df_features
    
    def _scale_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numeric features using StandardScaler."""
        df_features = df.copy()
        
        # Get numeric columns (excluding target and encoded categorical)
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['readmitted_30d']]
        
        if len(numeric_cols) > 0:
            df_features[numeric_cols] = self.scaler.fit_transform(df_features[numeric_cols])
        
        logger.info("Scaled numeric features")
        return df_features
    
    def _scale_numeric_features_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform numeric features using fitted scaler."""
        df_features = df.copy()
        
        # Get numeric columns (excluding target and encoded categorical)
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['readmitted_30d']]
        
        if len(numeric_cols) > 0:
            df_features[numeric_cols] = self.scaler.transform(df_features[numeric_cols])
        
        return df_features
    
    def get_feature_importance_ranking(self, model) -> List[Tuple[str, float]]:
        """
        Get feature importance ranking from a fitted model.
        
        Args:
            model: Fitted model with feature_importances_ attribute
            
        Returns:
            List of (feature_name, importance) tuples sorted by importance
        """
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute")
            return []
        
        feature_importance = list(zip(self.feature_names, model.feature_importances_))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        return feature_importance
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """Get summary of engineered features."""
        return {
            'total_features': len(self.feature_names),
            'numeric_features': len(self.numeric_features),
            'categorical_features': len(self.categorical_features),
            'feature_names': self.feature_names,
            'categorical_features_list': self.categorical_features,
            'numeric_features_list': self.numeric_features
        }
