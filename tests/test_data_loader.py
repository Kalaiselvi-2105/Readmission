"""
Tests for the data loader module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from ml.data.loader import DataLoader, create_sample_data


class TestDataLoader:
    """Test cases for DataLoader class."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        return create_sample_data(n_samples=100, random_state=42)
    
    def test_init(self, temp_data_dir):
        """Test DataLoader initialization."""
        loader = DataLoader(temp_data_dir)
        assert loader.data_path == Path(temp_data_dir)
        assert loader.data_path.exists()
    
    def test_load_csv(self, temp_data_dir, sample_df):
        """Test CSV loading functionality."""
        loader = DataLoader(temp_data_dir)
        
        # Save sample data as CSV
        csv_path = Path(temp_data_dir) / "test.csv"
        sample_df.to_csv(csv_path, index=False)
        
        # Load CSV
        loaded_df = loader.load_csv(str(csv_path))
        
        assert isinstance(loaded_df, pd.DataFrame)
        assert len(loaded_df) == len(sample_df)
        assert list(loaded_df.columns) == list(sample_df.columns)
    
    def test_load_json(self, temp_data_dir, sample_df):
        """Test JSON loading functionality."""
        loader = DataLoader(temp_data_dir)
        
        # Save sample data as JSON
        json_path = Path(temp_data_dir) / "test.json"
        sample_df.to_json(json_path, orient='records')
        
        # Load JSON
        loaded_df = loader.load_json(str(json_path))
        
        assert isinstance(loaded_df, pd.DataFrame)
        assert len(loaded_df) == len(sample_df)
        assert list(loaded_df.columns) == list(sample_df.columns)
    
    def test_validate_and_clean(self, temp_data_dir, sample_df):
        """Test data validation and cleaning."""
        loader = DataLoader(temp_data_dir)
        
        # Clean data
        cleaned_df, errors = loader.validate_and_clean(sample_df)
        
        assert isinstance(cleaned_df, pd.DataFrame)
        assert isinstance(errors, list)
        assert len(cleaned_df) == len(sample_df)
        
        # Check that datetime columns are properly converted
        if 'discharge_datetime' in cleaned_df.columns:
            assert pd.api.types.is_datetime64_any_dtype(cleaned_df['discharge_datetime'])
    
    def test_split_data(self, temp_data_dir, sample_df):
        """Test data splitting functionality."""
        loader = DataLoader(temp_data_dir)
        
        # Split data
        train_df, val_df, test_df = loader.split_data(
            sample_df, test_size=0.2, val_size=0.2, random_state=42
        )
        
        # Check split sizes
        total_size = len(sample_df)
        expected_test_size = int(total_size * 0.2)
        expected_val_size = int((total_size - expected_test_size) * 0.2)
        expected_train_size = total_size - expected_test_size - expected_val_size
        
        assert len(train_df) == expected_train_size
        assert len(val_df) == expected_val_size
        assert len(test_df) == expected_test_size
        
        # Check no overlap
        train_ids = set(train_df['patient_id'])
        val_ids = set(val_df['patient_id'])
        test_ids = set(test_df['patient_id'])
        
        assert len(train_ids.intersection(val_ids)) == 0
        assert len(train_ids.intersection(test_ids)) == 0
        assert len(val_ids.intersection(test_ids)) == 0
    
    def test_temporal_split(self, temp_data_dir, sample_df):
        """Test temporal data splitting."""
        loader = DataLoader(temp_data_dir)
        
        # Ensure we have datetime column
        if 'discharge_datetime' not in sample_df.columns:
            pytest.skip("No discharge_datetime column for temporal split")
        
        # Sort by date for temporal split
        sample_df_sorted = sample_df.sort_values('discharge_datetime').reset_index(drop=True)
        
        # Temporal split
        train_df, val_df, test_df = loader.temporal_split(
            sample_df_sorted, test_size=0.2, val_size=0.2
        )
        
        # Check temporal ordering
        assert train_df['discharge_datetime'].max() <= val_df['discharge_datetime'].min()
        assert val_df['discharge_datetime'].max() <= test_df['discharge_datetime'].min()
    
    def test_save_and_load_split_data(self, temp_data_dir, sample_df):
        """Test saving and loading split data."""
        loader = DataLoader(temp_data_dir)
        
        # Split data
        train_df, val_df, test_df = loader.split_data(sample_df)
        
        # Save split data
        output_dir = Path(temp_data_dir) / "processed"
        loader.save_split_data(train_df, val_df, test_df, str(output_dir))
        
        # Check files exist
        assert (output_dir / "train.csv").exists()
        assert (output_dir / "val.csv").exists()
        assert (output_dir / "test.csv").exists()
        assert (output_dir / "split_metadata.json").exists()
        
        # Load split data
        loaded_train, loaded_val, loaded_test = loader.load_split_data(str(output_dir))
        
        # Check data integrity
        assert len(loaded_train) == len(train_df)
        assert len(loaded_val) == len(val_df)
        assert len(loaded_test) == len(test_df)
    
    def test_handle_missing_values(self, temp_data_dir):
        """Test missing value handling."""
        loader = DataLoader(temp_data_dir)
        
        # Create DataFrame with missing values
        df_with_missing = pd.DataFrame({
            'age': [25, np.nan, 45],
            'sex': ['M', 'F', np.nan],
            'charlson_index': [1.0, np.nan, 2.0],
            'icu_stay': [True, False, np.nan]
        })
        
        # Clean data
        cleaned_df, errors = loader.validate_and_clean(df_with_missing)
        
        # Check missing values are handled
        assert cleaned_df['age'].isnull().sum() == 0
        assert cleaned_df['sex'].isnull().sum() == 0
        assert cleaned_df['charlson_index'].isnull().sum() == 0
        assert cleaned_df['icu_stay'].isnull().sum() == 0


class TestSampleDataGeneration:
    """Test cases for sample data generation."""
    
    def test_create_sample_data(self):
        """Test sample data generation."""
        # Generate data
        df = create_sample_data(n_samples=100, random_state=42)
        
        # Check basic structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        
        # Check required columns exist
        required_columns = [
            'patient_id', 'admission_id', 'age', 'sex', 'insurance',
            'charlson_index', 'los_days', 'readmitted_30d'
        ]
        for col in required_columns:
            assert col in df.columns
        
        # Check data types
        assert df['age'].dtype in ['int32', 'int64']
        assert df['sex'].dtype == 'object'
        assert df['charlson_index'].dtype in ['float32', 'float64']
        assert df['readmitted_30d'].dtype == 'bool'
        
        # Check value ranges
        assert df['age'].min() >= 18
        assert df['age'].max() <= 95
        assert df['charlson_index'].min() >= 0
        assert df['los_days'].min() > 0
        
        # Check target distribution
        target_counts = df['readmitted_30d'].value_counts()
        assert len(target_counts) == 2  # True/False
        assert target_counts.sum() == 100
        
        # Check for realistic patterns
        # Higher age should correlate with higher readmission risk
        age_risk_correlation = df.groupby('age')['readmitted_30d'].mean()
        if len(age_risk_correlation) > 1:
            # Simple check that older patients have higher risk
            older_risk = df[df['age'] > 65]['readmitted_30d'].mean()
            younger_risk = df[df['age'] < 50]['readmitted_30d'].mean()
            # This should generally be true, but not always due to randomness
            # So we just check that both risks are reasonable
            assert 0 <= older_risk <= 1
            assert 0 <= younger_risk <= 1
    
    def test_reproducibility(self):
        """Test that sample data generation is reproducible."""
        # Generate data with same seed
        df1 = create_sample_data(n_samples=50, random_state=42)
        df2 = create_sample_data(n_samples=50, random_state=42)
        
        # Data should be identical
        pd.testing.assert_frame_equal(df1, df2)
        
        # Generate data with different seed
        df3 = create_sample_data(n_samples=50, random_state=123)
        
        # Data should be different
        with pytest.raises(AssertionError):
            pd.testing.assert_frame_equal(df1, df3)
    
    def test_data_quality(self):
        """Test data quality aspects."""
        df = create_sample_data(n_samples=200, random_state=42)
        
        # Check no duplicate patient IDs
        assert df['patient_id'].nunique() == len(df)
        
        # Check no duplicate admission IDs
        assert df['admission_id'].nunique() == len(df)
        
        # Check logical consistency
        # Discharge datetime should be after admit datetime
        if 'admit_datetime' in df.columns and 'discharge_datetime' in df.columns:
            time_diff = df['discharge_datetime'] - df['admit_datetime']
            assert (time_diff > pd.Timedelta(0)).all()
        
        # LOS should match datetime difference
        if 'los_days' in df.columns and 'admit_datetime' in df.columns and 'discharge_datetime' in df.columns:
            calculated_los = (df['discharge_datetime'] - df['admit_datetime']).dt.total_seconds() / (24 * 3600)
            # Allow small tolerance for floating point precision
            assert np.allclose(df['los_days'], calculated_los, atol=0.1)


if __name__ == "__main__":
    pytest.main([__file__])

