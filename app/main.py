#!/usr/bin/env python3
"""
Main FastAPI application for the Hospital Readmission Risk Predictor.
Provides REST API endpoints for predictions, explanations, and system management.
"""

import os
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np

from .schemas import (
    PredictionRequest, PredictionResponse, ExplanationRequest,
    ExplanationResponse, BatchPredictionRequest, BatchPredictionResponse,
    HealthResponse, MetricsResponse, UserLogin, TokenResponse,
    UserResponse, AdminStatsResponse, RateLimitInfo
)
from .auth import (
    get_current_user, User, authenticate_user, create_access_token,
    can_predict, can_explain, can_admin, can_monitor
)
from .rate_limiter import (
    check_rate_limit_dependency, get_rate_limit_info_dependency,
    rate_limiter
)
from .predictor import ReadmissionPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Hospital Readmission Risk Predictor",
    description="AI-powered system for predicting 30-day hospital readmission risk",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
startup_time = time.time()
predictor: Optional[ReadmissionPredictor] = None
security = HTTPBearer()


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    global predictor
    try:
        logger.info("Starting Hospital Readmission Risk Predictor...")
        
        # Load models
        models_dir = os.getenv("MODELS_DIR", "models/")
        predictor = ReadmissionPredictor(models_dir)
        
        logger.info("✅ Application started successfully")
        logger.info(f"Models loaded from: {models_dir}")
        
    except Exception as e:
        logger.error(f"❌ Failed to start application: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Shutting down Hospital Readmission Risk Predictor...")


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        model_status = predictor.health_check() if predictor else {"status": "unhealthy"}
        
        return HealthResponse(
            status="healthy" if predictor and model_status["status"] == "healthy" else "unhealthy",
            timestamp=datetime.utcnow().isoformat(),
            version="1.0.0",
            uptime=time.time() - startup_time,
            model_status=model_status
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.utcnow().isoformat(),
            version="1.0.0",
            uptime=time.time() - startup_time,
            model_status={"error": str(e)}
        )


# Authentication endpoints
@app.post("/auth/login", response_model=TokenResponse)
async def login(login_data: UserLogin):
    """User login endpoint."""
    try:
        user = authenticate_user(login_data.username, login_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password"
            )
        
        # Create access token
        access_token_expires = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30")) * 60
        access_token = create_access_token(
            data={"sub": user.username, "role": user.role, "permissions": user.permissions},
            expires_delta=datetime.timedelta(minutes=access_token_expires)
        )
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=access_token_expires,
            user=UserResponse(
                username=user.username,
                email=user.email,
                full_name=user.full_name,
                role=user.role,
                permissions=user.permissions,
                is_active=user.is_active
            )
        )
        
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@app.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information."""
    return UserResponse(
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.full_name,
        role=current_user.role,
        permissions=current_user.permissions,
        is_active=current_user.is_active
    )


# Prediction endpoints
@app.post("/predict", response_model=PredictionResponse)
async def predict_readmission(
    request: PredictionRequest,
    current_user: User = Depends(can_predict),
    rate_limit: bool = Depends(check_rate_limit_dependency)
):
    """Predict readmission risk for a single patient."""
    try:
        if not predictor:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction service not available"
            )
        
        # Convert patient data to dict
        patient_dict = request.patient_data.dict()
        
        # Make prediction
        prediction = predictor.predict(patient_dict)
        
        return PredictionResponse(**prediction)
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict_readmission(
    request: BatchPredictionRequest,
    current_user: User = Depends(can_predict),
    rate_limit: bool = Depends(check_rate_limit_dependency)
):
    """Predict readmission risk for multiple patients."""
    try:
        if not predictor:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction service not available"
            )
        
        # Convert patient data to list of dicts
        patients_dict = [patient.dict() for patient in request.patients]
        
        # Make batch predictions
        predictions = predictor.batch_predict(patients_dict)
        
        # Convert to response format
        prediction_responses = []
        errors = []
        
        for pred in predictions:
            if "error" in pred:
                errors.append(pred)
            else:
                prediction_responses.append(PredictionResponse(**pred))
        
        return BatchPredictionResponse(
            batch_id=request.batch_id,
            predictions=prediction_responses,
            total_patients=len(request.patients),
            processing_timestamp=datetime.utcnow().isoformat(),
            errors=errors
        )
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


# Explanation endpoints
@app.post("/explain", response_model=ExplanationResponse)
async def explain_prediction(
    request: ExplanationRequest,
    current_user: User = Depends(can_explain),
    rate_limit: bool = Depends(check_rate_limit_dependency)
):
    """Explain prediction using SHAP values."""
    try:
        if not predictor:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction service not available"
            )
        
        # Convert patient data to dict
        patient_dict = request.patient_data.dict()
        
        # Get explanation
        explanation = predictor.explain_prediction(
            patient_dict, 
            top_features=request.top_features
        )
        
        return ExplanationResponse(**explanation)
        
    except Exception as e:
        logger.error(f"Explanation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explanation failed: {str(e)}"
        )


# Metrics and monitoring endpoints
@app.get("/metrics", response_model=MetricsResponse)
async def get_model_metrics(current_user: User = Depends(can_monitor)):
    """Get model performance metrics."""
    try:
        if not predictor:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction service not available"
            )
        
        # Get model info
        model_info = predictor.get_model_info()
        
        # Load training summary if available
        training_summary = {}
        try:
            import json
            summary_path = Path("models/training_summary.json")
            if summary_path.exists():
                with open(summary_path, 'r') as f:
                    training_summary = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load training summary: {e}")
        
        # Extract performance metrics
        performance_metrics = {}
        if "test_metrics" in training_summary:
            test_metrics = training_summary["test_metrics"]
            if "best_model" in test_metrics:
                best_model_metrics = test_metrics["best_model"]
                performance_metrics = {
                    "auroc": best_model_metrics.get("auroc", 0.0),
                    "auprc": best_model_metrics.get("auprc", 0.0),
                    "precision": best_model_metrics.get("precision", 0.0),
                    "recall": best_model_metrics.get("recall", 0.0),
                    "f1_score": best_model_metrics.get("f1_score", 0.0),
                    "brier_score": best_model_metrics.get("brier_score", 0.0)
                }
        
        # Get feature importance
        feature_importance = []
        if predictor.best_model and hasattr(predictor.best_model, 'feature_importances_'):
            feature_importance = predictor.feature_engineer.get_feature_importance_ranking(
                predictor.best_model
            )[:10]  # Top 10 features
        
        return MetricsResponse(
            model_info=model_info,
            performance_metrics=performance_metrics,
            feature_importance=[
                {"feature": feat, "importance": imp} 
                for feat, imp in feature_importance
            ],
            last_updated=training_summary.get("training_timestamp", datetime.utcnow().isoformat())
        )
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get metrics: {str(e)}"
        )


@app.get("/rate_limit_info", response_model=RateLimitInfo)
async def get_rate_limit_info(
    current_user: User = Depends(get_current_user),
    rate_limit_info: Dict = Depends(get_rate_limit_info_dependency)
):
    """Get rate limit information for current user."""
    if "error" in rate_limit_info:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=rate_limit_info["error"]
        )
    
    return RateLimitInfo(**rate_limit_info)


# Admin endpoints
@app.get("/admin/users", response_model=List[UserResponse])
async def get_all_users(current_user: User = Depends(can_admin)):
    """Get all users (admin only)."""
    try:
        from .auth import get_all_users as get_users
        users = get_users()
        return [
            UserResponse(
                username=user.username,
                email=user.email,
                full_name=user.full_name,
                role=user.role,
                permissions=user.permissions,
                is_active=user.is_active
            )
            for user in users
        ]
    except Exception as e:
        logger.error(f"Failed to get users: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get users: {str(e)}"
        )


@app.get("/admin/stats", response_model=AdminStatsResponse)
async def get_admin_stats(current_user: User = Depends(can_admin)):
    """Get admin statistics (admin only)."""
    try:
        from .auth import get_user_stats
        stats = get_user_stats()
        return AdminStatsResponse(**stats)
    except Exception as e:
        logger.error(f"Failed to get admin stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get admin stats: {str(e)}"
        )


# Utility endpoints
@app.get("/model_info")
async def get_model_info(current_user: User = Depends(get_current_user)):
    """Get information about loaded models."""
    try:
        if not predictor:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction service not available"
            )
        
        return predictor.get_model_info()
        
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Hospital Readmission Risk Predictor API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
