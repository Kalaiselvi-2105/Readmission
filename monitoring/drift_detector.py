#!/usr/bin/env python3
"""
Data drift detection and model monitoring service for the Hospital Readmission Risk Predictor.
Uses Evidently AI to monitor data quality and model performance.
"""

import os
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric
from evidently.metrics.data_quality import DataQualityMetrics
from evidently.metrics.data_drift import DataDriftMetrics
from evidently.metrics.regression_performance import RegressionMetrics
from evidently.metrics.classification_performance import ClassificationMetrics
from evidently.test_suite import TestSuite
from evidently.test_preset import DataStabilityTestPreset, NoTargetPerformanceTestPreset
from evidently.tests import TestNumberOfMissingValues, TestNumberOfColumns, TestColumnType
from evidently.tests import TestColumnDrift, TestDatasetDrift, TestColumnValueRange

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DriftDetector:
    """Data drift detection and monitoring service."""
    
    def __init__(self, reference_data_path: str, models_dir: str = "models/"):
        """
        Initialize the drift detector.
        
        Args:
            reference_data_path: Path to reference dataset
            models_dir: Directory containing trained models
        """
        self.reference_data_path = reference_data_path
        self.models_dir = models_dir
        self.reference_data = None
        self.column_mapping = None
        self.reports_dir = "monitoring/reports"
        
        # Create reports directory
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Load reference data
        self._load_reference_data()
        self._setup_column_mapping()
        
    def _load_reference_data(self):
        """Load reference dataset for comparison."""
        try:
            self.reference_data = pd.read_csv(self.reference_data_path)
            logger.info(f"Loaded reference data: {self.reference_data.shape}")
            
            # Clean and prepare reference data
            self.reference_data = self._prepare_data(self.reference_data)
            
        except Exception as e:
            logger.error(f"Failed to load reference data: {e}")
            raise
    
    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for drift detection."""
        # Remove target column if present
        if 'readmitted_30d' in data.columns:
            data = data.drop('readmitted_30d', axis=1)
        
        # Convert categorical columns
        categorical_columns = data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            data[col] = data[col].astype('category')
        
        # Handle missing values
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())
        
        categorical_columns = data.select_dtypes(include=['category']).columns
        data[categorical_columns] = data[categorical_columns].fillna(data[categorical_columns].mode().iloc[0])
        
        return data
    
    def _setup_column_mapping(self):
        """Setup column mapping for Evidently."""
        # Define feature columns
        feature_columns = [
            'age', 'charlson_index', 'prior_readmit_30d', 'prior_readmit_365d',
            'ed_visits_180d', 'los_days', 'procedures_count', 'meds_count',
            'days_to_followup', 'deprivation_index', 'icd_count'
        ]
        
        # Define categorical columns
        categorical_columns = [
            'sex', 'insurance', 'discharge_disposition', 'icu_stay',
            'high_risk_meds_flag', 'followup_scheduled', 'icd_I50',
            'icd_E11', 'icd_I25', 'icd_I10', 'icd_N18'
        ]
        
        # Define numerical columns
        numerical_columns = [col for col in feature_columns if col not in categorical_columns]
        
        self.column_mapping = ColumnMapping(
            target=None,  # No target for drift detection
            numerical_features=numerical_columns,
            categorical_features=categorical_columns,
            datetime_features=None,
            id_column=None,
            prediction=None
        )
    
    def detect_data_drift(self, current_data: pd.DataFrame, 
                         drift_threshold: float = 0.05) -> Dict[str, Any]:
        """
        Detect data drift between reference and current data.
        
        Args:
            current_data: Current dataset to compare
            drift_threshold: Threshold for drift detection
            
        Returns:
            Dictionary containing drift detection results
        """
        try:
            # Prepare current data
            current_data = self._prepare_data(current_data)
            
            # Create drift report
            drift_report = Report(metrics=[
                DataDriftPreset(),
                DataQualityPreset()
            ])
            
            drift_report.run(
                reference_data=self.reference_data,
                current_data=current_data,
                column_mapping=self.column_mapping
            )
            
            # Extract drift metrics
            drift_metrics = drift_report.metrics
            
            # Check for significant drift
            drift_detected = False
            drifted_columns = []
            
            for metric in drift_metrics:
                if hasattr(metric, 'result'):
                    result = metric.result
                    if hasattr(result, 'drift_detected'):
                        if result.drift_detected:
                            drift_detected = True
                            if hasattr(result, 'column_name'):
                                drifted_columns.append(result.column_name)
            
            # Generate detailed report
            report_data = {
                'timestamp': datetime.now().isoformat(),
                'drift_detected': drift_detected,
                'drifted_columns': drifted_columns,
                'drift_threshold': drift_threshold,
                'reference_data_shape': self.reference_data.shape,
                'current_data_shape': current_data.shape,
                'metrics': drift_report.json()
            }
            
            # Save report
            self._save_report('drift_detection', report_data)
            
            logger.info(f"Data drift detection completed. Drift detected: {drift_detected}")
            return report_data
            
        except Exception as e:
            logger.error(f"Error in drift detection: {e}")
            raise
    
    def check_data_quality(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Check data quality metrics.
        
        Args:
            current_data: Current dataset to analyze
            
        Returns:
            Dictionary containing data quality metrics
        """
        try:
            # Prepare current data
            current_data = self._prepare_data(current_data)
            
            # Create data quality report
            quality_report = Report(metrics=[
                DataQualityPreset()
            ])
            
            quality_report.run(
                reference_data=self.reference_data,
                current_data=current_data,
                column_mapping=self.column_mapping
            )
            
            # Extract quality metrics
            quality_metrics = quality_report.metrics
            
            # Generate report data
            report_data = {
                'timestamp': datetime.now().isoformat(),
                'data_quality_metrics': quality_report.json(),
                'current_data_shape': current_data.shape,
                'missing_values': current_data.isnull().sum().to_dict(),
                'data_types': current_data.dtypes.to_dict()
            }
            
            # Save report
            self._save_report('data_quality', report_data)
            
            logger.info("Data quality check completed")
            return report_data
            
        except Exception as e:
            logger.error(f"Error in data quality check: {e}")
            raise
    
    def run_stability_tests(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run data stability tests.
        
        Args:
            current_data: Current dataset to test
            
        Returns:
            Dictionary containing test results
        """
        try:
            # Prepare current data
            current_data = self._prepare_data(current_data)
            
            # Create test suite
            test_suite = TestSuite(tests=[
                DataStabilityTestPreset(),
                NoTargetPerformanceTestPreset()
            ])
            
            test_suite.run(
                reference_data=self.reference_data,
                current_data=current_data,
                column_mapping=self.column_mapping
            )
            
            # Extract test results
            test_results = test_suite.json()
            
            # Generate report data
            report_data = {
                'timestamp': datetime.now().isoformat(),
                'test_results': test_results,
                'tests_passed': test_suite.as_dict()['summary']['all_passed'],
                'total_tests': len(test_suite.as_dict()['tests']),
                'passed_tests': sum(1 for test in test_suite.as_dict()['tests'] if test['status'] == 'SUCCESS'),
                'failed_tests': sum(1 for test in test_suite.as_dict()['tests'] if test['status'] == 'FAIL')
            }
            
            # Save report
            self._save_report('stability_tests', report_data)
            
            logger.info(f"Stability tests completed. Tests passed: {report_data['tests_passed']}")
            return report_data
            
        except Exception as e:
            logger.error(f"Error in stability tests: {e}")
            raise
    
    def _save_report(self, report_type: str, report_data: Dict[str, Any]):
        """Save monitoring report to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{report_type}_{timestamp}.json"
        filepath = os.path.join(self.reports_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            logger.info(f"Report saved: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
    
    def generate_monitoring_summary(self) -> Dict[str, Any]:
        """Generate a summary of all monitoring activities."""
        try:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'reference_data_info': {
                    'shape': self.reference_data.shape,
                    'columns': list(self.reference_data.columns),
                    'data_types': self.reference_data.dtypes.to_dict()
                },
                'monitoring_config': {
                    'reports_directory': self.reports_dir,
                    'column_mapping': {
                        'numerical_features': self.column_mapping.numerical_features,
                        'categorical_features': self.column_mapping.categorical_features
                    }
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating monitoring summary: {e}")
            raise


def main():
    """Main function to run drift detection."""
    try:
        # Initialize drift detector
        reference_data_path = os.getenv("REFERENCE_DATA_PATH", "data/processed/hospital_readmissions_processed.csv")
        
        if not os.path.exists(reference_data_path):
            logger.error(f"Reference data not found: {reference_data_path}")
            return
        
        detector = DriftDetector(reference_data_path)
        
        # Generate monitoring summary
        summary = detector.generate_monitoring_summary()
        logger.info("Monitoring summary generated")
        
        # Example: Run drift detection on sample data
        # This would typically be run on incoming data streams
        logger.info("Drift detection service initialized successfully")
        logger.info("Ready to monitor data streams...")
        
        # Keep service running
        while True:
            time.sleep(60)  # Check every minute
            
    except Exception as e:
        logger.error(f"Monitoring service failed: {e}")
        raise


if __name__ == "__main__":
    main()
