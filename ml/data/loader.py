"""
Data loading and validation utilities for the Hospital Readmission Risk Predictor.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import logging
from datetime import datetime, timedelta
import json

from .schema import PatientData, validate_dataframe, LabResults, VitalSigns

logger = logging.getLogger(__name__)


class DataLoader:
    """Data loader for hospital readmission data."""
    
    def __init__(self, data_path: str = "data/"):
        self.data_path = Path(data_path)
        # Only create directory if it's a directory path, not a file path
        if not self.data_path.suffix:  # No file extension means it's a directory
            self.data_path.mkdir(exist_ok=True)
        
    def load_csv(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Loaded DataFrame
        """
        try:
            # If file_path is relative, make it relative to data_path
            if not Path(file_path).is_absolute():
                full_path = self.data_path / file_path
            else:
                full_path = Path(file_path)
            
            df = pd.read_csv(full_path)
            logger.info(f"Loaded {len(df)} records from {full_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV file {file_path}: {e}")
            raise
    
    def load_json(self, file_path: str) -> pd.DataFrame:
        """
        Load data from JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Loaded DataFrame
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame([data])
                
            logger.info(f"Loaded {len(df)} records from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {e}")
            raise
    
    def validate_and_clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
        """
        Validate and clean the loaded data.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Tuple of (cleaned DataFrame, validation errors)
        """
        errors = validate_dataframe(df)
        
        if errors:
            logger.warning(f"Found {len(errors)} validation errors")
            for error in errors:
                logger.warning(f"  - {error}")
        
        # Basic cleaning
        df_clean = df.copy()
        
        # Handle missing values
        df_clean = self._handle_missing_values(df_clean)
        
        # Convert datetime columns
        df_clean = self._convert_datetime_columns(df_clean)
        
        # Handle nested structures (labs, vitals, icd_codes)
        df_clean = self._flatten_nested_columns(df_clean)
        
        # Ensure numeric columns are numeric
        df_clean = self._ensure_numeric_columns(df_clean)
        
        return df_clean, errors
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the DataFrame."""
        df_clean = df.copy()
        
        # Fill missing values with appropriate defaults
        numeric_columns = ['age', 'charlson_index', 'prior_readmit_30d', 'prior_readmit_365d',
                          'ed_visits_180d', 'los_days', 'procedures_count', 'meds_count',
                          'days_to_followup', 'deprivation_index']
        
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(0)
        
        # Fill categorical columns
        categorical_columns = ['sex', 'insurance', 'discharge_disposition']
        for col in categorical_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna('Unknown')
        
        # Fill boolean columns
        boolean_columns = ['icu_stay', 'high_risk_meds_flag', 'followup_scheduled']
        for col in boolean_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(False)
        
        return df_clean
    
    def _convert_datetime_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert datetime columns to proper datetime type."""
        df_clean = df.copy()
        
        datetime_columns = ['admit_datetime', 'discharge_datetime']
        for col in datetime_columns:
            if col in df_clean.columns:
                try:
                    df_clean[col] = pd.to_datetime(df_clean[col])
                except Exception as e:
                    logger.warning(f"Could not convert {col} to datetime: {e}")
        
        return df_clean
    
    def _flatten_nested_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flatten nested columns like labs, vitals, and icd_codes."""
        df_clean = df.copy()
        
        # Handle labs
        if 'labs' in df_clean.columns:
            labs_df = pd.json_normalize(df_clean['labs'])
            for col in labs_df.columns:
                df_clean[f'labs_{col}'] = labs_df[col]
            df_clean = df_clean.drop('labs', axis=1)
        
        # Handle vitals
        if 'vitals' in df_clean.columns:
            vitals_df = pd.json_normalize(df_clean['vitals'])
            for col in vitals_df.columns:
                df_clean[f'vitals_{col}'] = vitals_df[col]
            df_clean = df_clean.drop('vitals', axis=1)
        
        # Handle icd_codes (count and create binary flags for common ones)
        if 'icd_codes' in df_clean.columns:
            df_clean['icd_count'] = df_clean['icd_codes'].apply(
                lambda x: len(x) if isinstance(x, list) else 1
            )
            
            # Create binary flags for common ICD categories
            common_icd_prefixes = ['I50', 'E11', 'I25', 'I10', 'N18']
            for prefix in common_icd_prefixes:
                df_clean[f'icd_{prefix}'] = df_clean['icd_codes'].apply(
                    lambda x: any(code.startswith(prefix) for code in x) if isinstance(x, list) else False
                )
            
            df_clean = df_clean.drop('icd_codes', axis=1)
        
        return df_clean
    
    def _ensure_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure numeric columns are properly typed."""
        df_clean = df.copy()
        
        numeric_columns = [
            'age', 'charlson_index', 'prior_readmit_30d', 'prior_readmit_365d',
            'ed_visits_180d', 'los_days', 'procedures_count', 'meds_count',
            'days_to_followup', 'deprivation_index'
        ]
        
        # Add lab and vital columns
        lab_columns = [col for col in df_clean.columns if col.startswith('labs_')]
        vital_columns = [col for col in df_clean.columns if col.startswith('vitals_')]
        numeric_columns.extend(lab_columns + vital_columns)
        
        for col in numeric_columns:
            if col in df_clean.columns:
                try:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                except Exception as e:
                    logger.warning(f"Could not convert {col} to numeric: {e}")
        
        return df_clean
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, 
                   val_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            df: Input DataFrame
            test_size: Proportion for test set
            val_size: Proportion for validation set (from remaining data)
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # First split: test vs rest
        test_idx = int(len(df) * (1 - test_size))
        df_rest = df.iloc[:test_idx].copy()
        df_test = df.iloc[test_idx:].copy()
        
        # Second split: validation vs train
        val_idx = int(len(df_rest) * (1 - val_size))
        df_train = df_rest.iloc[:val_idx].copy()
        df_val = df_rest.iloc[val_idx:].copy()
        
        logger.info(f"Data split: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")
        
        return df_train, df_val, df_test
    
    def temporal_split(self, df: pd.DataFrame, date_column: str = 'discharge_datetime',
                      test_size: float = 0.2, val_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data temporally based on discharge date.
        
        Args:
            df: Input DataFrame
            date_column: Column containing discharge dates
            test_size: Proportion for test set (most recent)
            val_size: Proportion for validation set (second most recent)
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        df_sorted = df.sort_values(date_column).copy()
        
        # Calculate split indices
        test_idx = int(len(df_sorted) * (1 - test_size))
        val_idx = int(len(df_sorted) * (1 - test_size - val_size))
        
        df_train = df_sorted.iloc[:val_idx].copy()
        df_val = df_sorted.iloc[val_idx:test_idx].copy()
        df_test = df_sorted.iloc[test_idx:].copy()
        
        logger.info(f"Temporal split: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")
        logger.info(f"Train period: {df_train[date_column].min()} to {df_train[date_column].max()}")
        logger.info(f"Val period: {df_val[date_column].min()} to {df_val[date_column].max()}")
        logger.info(f"Test period: {df_test[date_column].min()} to {df_test[date_column].max()}")
        
        return df_train, df_val, df_test
    
    def save_split_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                       test_df: pd.DataFrame, output_dir: str = "data/processed"):
        """
        Save split data to files.
        
        Args:
            train_df: Training data
            val_df: Validation data
            test_df: Test data
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Save to CSV
        train_df.to_csv(output_path / "train.csv", index=False)
        val_df.to_csv(output_path / "val.csv", index=False)
        test_df.to_csv(output_path / "test.csv", index=False)
        
        # Save metadata
        metadata = {
            "split_info": {
                "train_size": len(train_df),
                "val_size": len(val_df),
                "test_size": len(test_df),
                "total_size": len(train_df) + len(val_df) + len(test_df),
                "split_timestamp": datetime.now().isoformat()
            },
            "target_distribution": {
                "train": train_df.get('readmitted_30d', pd.Series()).value_counts().to_dict(),
                "val": val_df.get('readmitted_30d', pd.Series()).value_counts().to_dict(),
                "test": test_df.get('readmitted_30d', pd.Series()).value_counts().to_dict()
            }
        }
        
        with open(output_path / "split_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Split data saved to {output_path}")
    
    def load_split_data(self, data_dir: str = "data/processed") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load previously split data.
        
        Args:
            data_dir: Directory containing split data
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        data_path = Path(data_dir)
        
        train_df = pd.read_csv(data_path / "train.csv")
        val_df = pd.read_csv(data_path / "val.csv")
        test_df = pd.read_csv(data_path / "test.csv")
        
        logger.info(f"Loaded split data: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        return train_df, val_df, test_df


def create_sample_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Create sample data for development and testing.
    
    Args:
        n_samples: Number of samples to generate
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with sample hospital readmission data
    """
    np.random.seed(random_state)
    
    # Generate sample data
    data = []
    
    for i in range(n_samples):
        # Basic patient info
        patient_id = f"P{i:04d}"
        admission_id = f"A{i:04d}"
        
        # Random dates within last year
        base_date = datetime.now() - timedelta(days=365)
        admit_date = base_date + timedelta(days=np.random.randint(0, 365))
        los_days = np.random.exponential(5) + 1  # Exponential distribution for LOS
        discharge_date = admit_date + timedelta(days=los_days)
        
        # Demographics
        age = np.random.randint(18, 95)
        sex = np.random.choice(['M', 'F', 'Other'], p=[0.45, 0.45, 0.1])
        insurance = np.random.choice(['Medicare', 'Medicaid', 'Private', 'Self-Pay', 'Other'], 
                                   p=[0.4, 0.2, 0.3, 0.05, 0.05])
        
        # Clinical indicators
        charlson_index = np.random.exponential(2) + 0.5
        prior_readmit_30d = np.random.poisson(0.3)
        prior_readmit_365d = np.random.poisson(1.2)
        ed_visits_180d = np.random.poisson(0.8)
        
        # Hospital stay
        icu_stay = np.random.choice([True, False], p=[0.2, 0.8])
        procedures_count = np.random.poisson(2)
        
        # Labs (with realistic ranges)
        labs = {
            'hgb': np.random.normal(13.5, 2.0),  # Hemoglobin
            'sodium': np.random.normal(140, 5),   # Sodium
            'potassium': np.random.normal(4.0, 0.5),  # Potassium
            'creatinine': np.random.exponential(1.0) + 0.5,  # Creatinine
            'glucose': np.random.normal(120, 30)  # Glucose
        }
        
        # Vitals
        vitals = {
            'sbp': np.random.normal(140, 20),    # Systolic BP
            'dbp': np.random.normal(85, 15),     # Diastolic BP
            'hr': np.random.normal(80, 15),      # Heart rate
            'rr': np.random.normal(18, 4),       # Respiratory rate
            'spo2': np.random.normal(98, 2)      # Oxygen saturation
        }
        
        # Medications and follow-up
        meds_count = np.random.poisson(5) + 1
        high_risk_meds_flag = np.random.choice([True, False], p=[0.3, 0.7])
        discharge_disposition = np.random.choice(
            ['Home', 'Home with Services', 'SNF', 'Rehab', 'Hospice', 'Expired', 'Other'],
            p=[0.6, 0.15, 0.1, 0.1, 0.02, 0.01, 0.02]
        )
        followup_scheduled = np.random.choice([True, False], p=[0.8, 0.2])
        days_to_followup = np.random.randint(1, 30) if followup_scheduled else None
        
        # Demographics
        zip_code = f"{np.random.randint(10000, 99999)}"
        deprivation_index = np.random.uniform(0, 1)
        
        # Target variable (30-day readmission)
        # Higher risk for older patients, longer LOS, more comorbidities
        readmission_risk = (
            0.1 +  # Base risk
            0.02 * (age - 50) / 50 +  # Age effect
            0.1 * (charlson_index - 1) / 3 +  # Comorbidity effect
            0.05 * (los_days - 3) / 5 +  # LOS effect
            0.1 * prior_readmit_30d +  # Prior readmission effect
            0.05 * prior_readmit_365d / 2  # Historical readmission effect
        )
        readmitted_30d = np.random.choice([True, False], p=[readmission_risk, 1-readmission_risk])
        
        # ICD codes (simplified)
        icd_codes = []
        if charlson_index > 2:
            icd_codes.append('I50.9')  # Heart failure
        if age > 65:
            icd_codes.append('E11.9')  # Type 2 diabetes
        if not icd_codes:
            icd_codes.append('Z51.11')  # Encounter for chemotherapy
        
        patient_data = {
            'patient_id': patient_id,
            'admission_id': admission_id,
            'admit_datetime': admit_date,
            'discharge_datetime': discharge_date,
            'age': age,
            'sex': sex,
            'insurance': insurance,
            'icd_codes': icd_codes,
            'charlson_index': charlson_index,
            'prior_readmit_30d': prior_readmit_30d,
            'prior_readmit_365d': prior_readmit_365d,
            'ed_visits_180d': ed_visits_180d,
            'los_days': los_days,
            'icu_stay': icu_stay,
            'procedures_count': procedures_count,
            'labs': labs,
            'vitals': vitals,
            'meds_count': meds_count,
            'high_risk_meds_flag': high_risk_meds_flag,
            'discharge_disposition': discharge_disposition,
            'followup_scheduled': followup_scheduled,
            'days_to_followup': days_to_followup,
            'zip_code': zip_code,
            'deprivation_index': deprivation_index,
            'readmitted_30d': readmitted_30d
        }
        
        data.append(patient_data)
    
    df = pd.DataFrame(data)
    logger.info(f"Generated {len(df)} sample records")
    logger.info(f"Readmission rate: {df['readmitted_30d'].mean():.2%}")
    
    return df
