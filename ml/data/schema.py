"""
Data schema definitions for the Hospital Readmission Risk Predictor.
Matches the specified schema with Pydantic models for validation.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator
import pandas as pd


class LabResults(BaseModel):
    """Laboratory test results."""
    hgb: Optional[float] = Field(None, description="Hemoglobin (g/dL)")
    sodium: Optional[float] = Field(None, description="Sodium (mEq/L)")
    potassium: Optional[float] = Field(None, description="Potassium (mEq/L)")
    creatinine: Optional[float] = Field(None, description="Creatinine (mg/dL)")
    glucose: Optional[float] = Field(None, description="Glucose (mg/dL)")


class VitalSigns(BaseModel):
    """Vital signs measurements."""
    sbp: Optional[float] = Field(None, description="Systolic Blood Pressure (mmHg)")
    dbp: Optional[float] = Field(None, description="Diastolic Blood Pressure (mmHg)")
    hr: Optional[float] = Field(None, description="Heart Rate (bpm)")
    rr: Optional[float] = Field(None, description="Respiratory Rate (breaths/min)")
    spo2: Optional[float] = Field(None, description="Oxygen Saturation (%)")


class PatientData(BaseModel):
    """Complete patient data schema for training and prediction."""
    patient_id: str = Field(..., description="Unique patient identifier")
    admission_id: str = Field(..., description="Unique admission identifier")
    admit_datetime: datetime = Field(..., description="Admission date and time")
    discharge_datetime: datetime = Field(..., description="Discharge date and time")
    age: int = Field(..., ge=0, le=120, description="Patient age in years")
    sex: str = Field(..., description="Patient sex (M/F/Other)")
    insurance: str = Field(..., description="Insurance type")
    icd_codes: List[str] = Field(default_factory=list, description="ICD diagnosis codes")
    charlson_index: float = Field(..., description="Charlson Comorbidity Index")
    prior_readmit_30d: int = Field(..., ge=0, description="Prior 30-day readmissions")
    prior_readmit_365d: int = Field(..., ge=0, description="Prior 365-day readmissions")
    ed_visits_180d: int = Field(..., ge=0, description="ED visits in last 180 days")
    los_days: float = Field(..., gt=0, description="Length of stay in days")
    icu_stay: bool = Field(False, description="ICU stay during admission")
    procedures_count: int = Field(..., ge=0, description="Number of procedures")
    labs: LabResults = Field(default_factory=LabResults, description="Laboratory results")
    vitals: VitalSigns = Field(default_factory=VitalSigns, description="Vital signs")
    meds_count: int = Field(..., ge=0, description="Number of medications")
    high_risk_meds_flag: bool = Field(False, description="High-risk medications flag")
    discharge_disposition: str = Field(..., description="Discharge disposition")
    followup_scheduled: bool = Field(False, description="Follow-up appointment scheduled")
    days_to_followup: Optional[int] = Field(None, ge=0, description="Days to follow-up")
    zip_code: str = Field(..., description="Patient zip code")
    deprivation_index: float = Field(..., description="Area deprivation index")
    readmitted_30d: Optional[bool] = Field(None, description="30-day readmission (TARGET)")

    @validator('sex')
    def validate_sex(cls, v):
        valid_sexes = ['M', 'F', 'Other']
        if v not in valid_sexes:
            raise ValueError(f'sex must be one of {valid_sexes}')
        return v

    @validator('discharge_disposition')
    def validate_discharge_disposition(cls, v):
        valid_dispositions = [
            'Home', 'Home with Services', 'SNF', 'Rehab', 'Hospice', 'Expired', 'Other'
        ]
        if v not in valid_dispositions:
            raise ValueError(f'discharge_disposition must be one of {valid_dispositions}')
        return v

    @validator('insurance')
    def validate_insurance(cls, v):
        valid_insurances = ['Medicare', 'Medicaid', 'Private', 'Self-Pay', 'Other']
        if v not in valid_insurances:
            raise ValueError(f'insurance must be one of {valid_insurances}')
        return v

    class Config:
        schema_extra = {
            "example": {
                "patient_id": "P001",
                "admission_id": "A001",
                "admit_datetime": "2023-01-01T10:00:00",
                "discharge_datetime": "2023-01-05T14:00:00",
                "age": 65,
                "sex": "M",
                "insurance": "Medicare",
                "icd_codes": ["I50.9", "E11.9"],
                "charlson_index": 3.0,
                "prior_readmit_30d": 1,
                "prior_readmit_365d": 2,
                "ed_visits_180d": 1,
                "los_days": 4.5,
                "icu_stay": False,
                "procedures_count": 2,
                "labs": {
                    "hgb": 12.5,
                    "sodium": 140.0,
                    "potassium": 4.0,
                    "creatinine": 1.2,
                    "glucose": 120.0
                },
                "vitals": {
                    "sbp": 140.0,
                    "dbp": 85.0,
                    "hr": 75.0,
                    "rr": 16.0,
                    "spo2": 98.0
                },
                "meds_count": 5,
                "high_risk_meds_flag": True,
                "discharge_disposition": "Home",
                "followup_scheduled": True,
                "days_to_followup": 7,
                "zip_code": "12345",
                "deprivation_index": 0.6,
                "readmitted_30d": False
            }
        }


class PredictionRequest(BaseModel):
    """Request schema for prediction endpoint (excludes target variable)."""
    patient: PatientData = Field(..., description="Patient data for prediction")

    @validator('patient')
    def validate_no_target(cls, v):
        if v.readmitted_30d is not None:
            raise ValueError("Target variable 'readmitted_30d' should not be included in prediction requests")
        return v


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint."""
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Predicted readmission risk (0-1)")
    readmit_pred: bool = Field(..., description="Binary prediction (True if risk > threshold)")
    threshold: float = Field(..., description="Threshold used for binary prediction")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    model_version: str = Field(..., description="Model version used for prediction")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Prediction timestamp")


class ExplanationRequest(BaseModel):
    """Request schema for explanation endpoint."""
    patient: PatientData = Field(..., description="Patient data for explanation")


class FeatureContribution(BaseModel):
    """Individual feature contribution to prediction."""
    name: str = Field(..., description="Feature name")
    contribution: float = Field(..., description="SHAP contribution value")
    importance_rank: int = Field(..., description="Feature importance rank")


class ExplanationResponse(BaseModel):
    """Response schema for explanation endpoint."""
    top_features: List[FeatureContribution] = Field(..., description="Top contributing features")
    shap_values: List[float] = Field(..., description="Raw SHAP values for all features")
    base_value: float = Field(..., description="Base prediction value")
    model_version: str = Field(..., description="Model version used for explanation")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Explanation timestamp")


def validate_dataframe(df: pd.DataFrame) -> List[str]:
    """
    Validate DataFrame against schema and return list of validation errors.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        List of validation error messages
    """
    errors = []
    
    # Check required columns
    required_columns = PatientData.__fields__.keys()
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
    
    # Check data types and ranges
    for idx, row in df.iterrows():
        try:
            # Convert row to dict and validate
            row_dict = row.to_dict()
            PatientData(**row_dict)
        except Exception as e:
            errors.append(f"Row {idx}: {str(e)}")
    
    return errors


def create_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create feature matrix from raw patient data.
    
    Args:
        df: Raw patient data DataFrame
        
    Returns:
        Feature matrix ready for ML models
    """
    # This will be implemented in the feature engineering module
    # For now, return the original DataFrame
    return df.copy()

