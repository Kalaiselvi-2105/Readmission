#!/usr/bin/env python3
"""
Pydantic schemas for the Hospital Readmission Risk Predictor API.
Defines request and response models for all endpoints.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator
import pandas as pd


# Authentication schemas
class UserLogin(BaseModel):
    """User login request."""
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")


class UserResponse(BaseModel):
    """User response model."""
    username: str
    email: str
    full_name: str
    role: str
    permissions: List[str]
    is_active: bool


class TokenResponse(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


# Patient data schemas
class LabResults(BaseModel):
    """Laboratory test results."""
    hgb: Optional[float] = Field(None, description="Hemoglobin (g/dL)")
    sodium: Optional[float] = Field(None, description="Sodium (mEq/L)")
    potassium: Optional[float] = Field(None, description="Potassium (mEq/L)")
    creatinine: Optional[float] = Field(None, description="Creatinine (mg/dL)")
    glucose: Optional[float] = Field(None, description="Glucose (mg/dL)")


class VitalSigns(BaseModel):
    """Vital signs measurements."""
    sbp: Optional[float] = Field(None, description="Systolic blood pressure (mmHg)")
    dbp: Optional[float] = Field(None, description="Diastolic blood pressure (mmHg)")
    hr: Optional[float] = Field(None, description="Heart rate (bpm)")
    rr: Optional[float] = Field(None, description="Respiratory rate (breaths/min)")
    spo2: Optional[float] = Field(None, description="Oxygen saturation (%)")


class PatientData(BaseModel):
    """Patient data for prediction."""
    patient_id: str = Field(..., description="Unique patient identifier")
    admission_id: str = Field(..., description="Unique admission identifier")
    age: int = Field(..., ge=0, le=120, description="Patient age in years")
    sex: str = Field(..., description="Patient sex (M/F/Other)")
    insurance: str = Field(..., description="Insurance type")
    charlson_index: int = Field(..., ge=0, le=10, description="Charlson comorbidity index")
    prior_readmit_30d: int = Field(0, ge=0, description="Prior readmissions in last 30 days")
    prior_readmit_365d: int = Field(0, ge=0, description="Prior readmissions in last 365 days")
    ed_visits_180d: int = Field(0, ge=0, description="ED visits in last 180 days")
    los_days: int = Field(..., ge=0, description="Length of stay in days")
    icu_stay: bool = Field(False, description="Whether patient had ICU stay")
    procedures_count: int = Field(0, ge=0, description="Number of procedures")
    meds_count: int = Field(0, ge=0, description="Number of medications")
    high_risk_meds_flag: bool = Field(False, description="Flag for high-risk medications")
    discharge_disposition: str = Field(..., description="Discharge disposition")
    followup_scheduled: bool = Field(False, description="Whether follow-up is scheduled")
    days_to_followup: int = Field(0, ge=0, description="Days until follow-up appointment")
    zip_code: str = Field(..., description="Patient zip code")
    deprivation_index: float = Field(0.0, description="Area deprivation index")
    labs: Optional[LabResults] = Field(None, description="Laboratory results")
    vitals: Optional[VitalSigns] = Field(None, description="Vital signs")
    icd_count: int = Field(0, ge=0, description="Number of ICD diagnosis codes")
    icd_I50: bool = Field(False, description="ICD-10 I50 (Heart failure)")
    icd_E11: bool = Field(False, description="ICD-10 E11 (Type 2 diabetes)")
    icd_I25: bool = Field(False, description="ICD-10 I25 (Chronic ischemic heart disease)")
    icd_I10: bool = Field(False, description="ICD-10 I10 (Essential hypertension)")
    icd_N18: bool = Field(False, description="ICD-10 N18 (Chronic kidney disease)")
    
    @validator('sex')
    def validate_sex(cls, v):
        valid_sexes = ['M', 'F', 'Other']
        if v not in valid_sexes:
            raise ValueError(f'sex must be one of {valid_sexes}')
        return v
    
    @validator('discharge_disposition')
    def validate_discharge_disposition(cls, v):
        valid_dispositions = ['Home', 'SNF', 'Rehab', 'Hospice', 'Other']
        if v not in valid_dispositions:
            raise ValueError(f'discharge_disposition must be one of {valid_dispositions}')
        return v


class PredictionRequest(BaseModel):
    """Request for single patient prediction."""
    patient_data: PatientData


class PredictionResponse(BaseModel):
    """Response for single patient prediction."""
    patient_id: str
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Probability of readmission")
    risk_category: str = Field(..., description="Risk category (Low/Medium-Low/Medium/Medium-High/High)")
    confidence: str = Field(..., description="Confidence level (Low/Medium/High)")
    prediction_timestamp: str = Field(..., description="ISO timestamp of prediction")
    model_version: str = Field(..., description="Model version used")
    features_used: int = Field(..., description="Number of features used")


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions."""
    patients: List[PatientData] = Field(..., description="List of patients for prediction")
    batch_id: Optional[str] = Field(None, description="Optional batch identifier")


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    batch_id: Optional[str]
    predictions: List[PredictionResponse]
    total_patients: int
    processing_timestamp: str
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="Any errors encountered")


class ExplanationRequest(BaseModel):
    """Request for prediction explanation."""
    patient_data: PatientData
    top_features: int = Field(10, ge=1, le=50, description="Number of top features to return")


class FeatureContribution(BaseModel):
    """Feature contribution to prediction."""
    feature_name: str = Field(..., description="Feature name")
    contribution: float = Field(..., description="SHAP contribution value")
    abs_contribution: float = Field(..., description="Absolute contribution value")


class ExplanationResponse(BaseModel):
    """Response for prediction explanation."""
    patient_id: str
    risk_score: float
    risk_category: str
    feature_contributions: List[FeatureContribution]
    explanation_timestamp: str
    note: Optional[str] = Field(None, description="Additional notes about explanation")


# Health and monitoring schemas
class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="ISO timestamp")
    version: str = Field(..., description="API version")
    uptime: float = Field(..., description="Service uptime in seconds")
    model_status: Dict[str, Any] = Field(..., description="Model loading status")


class MetricsResponse(BaseModel):
    """Model performance metrics response."""
    model_info: Dict[str, Any] = Field(..., description="Model information")
    performance_metrics: Dict[str, float] = Field(..., description="Performance metrics")
    feature_importance: List[Dict[str, Any]] = Field(..., description="Top feature importance")
    last_updated: str = Field(..., description="Last model update timestamp")


class RateLimitInfo(BaseModel):
    """Rate limit information response."""
    current_tokens: float = Field(..., description="Current available tokens")
    max_tokens: int = Field(..., description="Maximum token capacity")
    refill_rate_per_minute: float = Field(..., description="Token refill rate per minute")
    last_refill: str = Field(..., description="Last token refill timestamp")


# Admin schemas
class UserCreateRequest(BaseModel):
    """Request to create a new user."""
    username: str = Field(..., description="Username")
    email: str = Field(..., description="Email address")
    full_name: str = Field(..., description="Full name")
    role: str = Field(..., description="User role")
    password: str = Field(..., description="Password")


class UserUpdateRequest(BaseModel):
    """Request to update user permissions."""
    username: str = Field(..., description="Username")
    permissions: List[str] = Field(..., description="New permissions list")


class AdminStatsResponse(BaseModel):
    """Admin statistics response."""
    total_users: int = Field(..., description="Total number of users")
    active_users: int = Field(..., description="Number of active users")
    inactive_users: int = Field(..., description="Number of inactive users")
    role_distribution: Dict[str, int] = Field(..., description="Users per role")
    timestamp: str = Field(..., description="Statistics timestamp")


# Error response schemas
class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    timestamp: str = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier for tracking")


class ValidationErrorResponse(BaseModel):
    """Validation error response."""
    error: str = Field("Validation Error", description="Error type")
    field_errors: List[Dict[str, Any]] = Field(..., description="Field-specific validation errors")
    timestamp: str = Field(..., description="Error timestamp")


# Utility schemas
class PaginationParams(BaseModel):
    """Pagination parameters."""
    page: int = Field(1, ge=1, description="Page number")
    size: int = Field(10, ge=1, le=100, description="Page size")


class SearchParams(BaseModel):
    """Search parameters."""
    query: Optional[str] = Field(None, description="Search query")
    filters: Optional[Dict[str, Any]] = Field(None, description="Search filters")
    sort_by: Optional[str] = Field(None, description="Sort field")
    sort_order: Optional[str] = Field("asc", description="Sort order (asc/desc)")


# Data validation schemas
class DataValidationRequest(BaseModel):
    """Request for data validation."""
    data: List[Dict[str, Any]] = Field(..., description="Data to validate")
    schema_version: str = Field(..., description="Schema version to validate against")


class DataValidationResponse(BaseModel):
    """Response for data validation."""
    valid: bool = Field(..., description="Whether data is valid")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="Validation errors")
    warnings: List[Dict[str, Any]] = Field(default_factory=list, description="Validation warnings")
    validated_count: int = Field(..., description="Number of records validated")
    timestamp: str = Field(..., description="Validation timestamp")


# Model management schemas
class ModelInfo(BaseModel):
    """Model information."""
    model_id: str = Field(..., description="Model identifier")
    model_type: str = Field(..., description="Model type")
    version: str = Field(..., description="Model version")
    training_date: str = Field(..., description="Training date")
    performance_metrics: Dict[str, float] = Field(..., description="Performance metrics")
    feature_count: int = Field(..., description="Number of features")
    is_active: bool = Field(..., description="Whether model is active")


class ModelDeploymentRequest(BaseModel):
    """Request to deploy a model."""
    model_id: str = Field(..., description="Model to deploy")
    environment: str = Field(..., description="Deployment environment")
    force: bool = Field(False, description="Force deployment even if validation fails")


class ModelDeploymentResponse(BaseModel):
    """Response for model deployment."""
    success: bool = Field(..., description="Whether deployment succeeded")
    model_id: str = Field(..., description="Deployed model ID")
    environment: str = Field(..., description="Deployment environment")
    deployment_timestamp: str = Field(..., description="Deployment timestamp")
    message: str = Field(..., description="Deployment message")

