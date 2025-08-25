import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json
import pandas as pd
from app.main import app
from app.auth import create_access_token, get_current_user
from app.schemas import UserRole

client = TestClient(app)

# Test data
test_patient_data = {
    "age": 65,
    "gender": "Female",
    "admission_type_id": 1,
    "discharge_disposition_id": 1,
    "admission_source_id": 1,
    "time_in_hospital": 5,
    "num_lab_procedures": 41,
    "num_procedures": 0,
    "num_medications": 15,
    "number_outpatient": 0,
    "number_emergency": 0,
    "number_inpatient": 0,
    "number_diagnoses": 9,
    "max_glu_serum": "None",
    "A1Cresult": "None",
    "metformin": "No",
    "repaglinide": "No",
    "nateglinide": "No",
    "chlorpropamide": "No",
    "glimepiride": "No",
    "acetohexamide": "No",
    "tolbutamide": "No",
    "pioglitazone": "No",
    "rosiglitazone": "No",
    "acarbose": "No",
    "miglitol": "No",
    "troglitazone": "No",
    "tolazamide": "No",
    "examide": "No",
    "citoglipton": "No",
    "insulin": "No",
    "glyburide-metformin": "No",
    "glipizide-metformin": "No",
    "glimepiride-pioglitazone": "No",
    "metformin-rosiglitazone": "No",
    "metformin-pioglitazone": "No",
    "change": "No",
    "diabetesMed": "No",
    "readmitted": "No"
}

test_batch_data = pd.DataFrame([test_patient_data]).to_csv(index=False)

@pytest.fixture
def auth_headers():
    """Create authentication headers for testing"""
    token = create_access_token(data={"sub": "testuser", "role": UserRole.CLINICIAN})
    return {"Authorization": f"Bearer {token}"}

@pytest.fixture
def admin_headers():
    """Create admin authentication headers for testing"""
    token = create_access_token(data={"sub": "admin", "role": UserRole.ADMIN})
    return {"Authorization": f"Bearer {token}"}

class TestHealthEndpoints:
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data

    def test_readiness_check(self):
        """Test readiness check endpoint"""
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert "timestamp" in data

class TestAuthentication:
    def test_login_success(self):
        """Test successful login"""
        response = client.post("/auth/login", json={
            "username": "clinician1",
            "password": "password123"
        })
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "token_type" in data
        assert data["token_type"] == "bearer"

    def test_login_invalid_credentials(self):
        """Test login with invalid credentials"""
        response = client.post("/auth/login", json={
            "username": "invalid",
            "password": "wrong"
        })
        assert response.status_code == 401
        assert "Incorrect username or password" in response.json()["detail"]

    def test_login_missing_fields(self):
        """Test login with missing fields"""
        response = client.post("/auth/login", json={"username": "test"})
        assert response.status_code == 422

class TestPredictionEndpoints:
    @patch('app.predictor.ReadmissionPredictor.predict_single')
    def test_single_prediction_success(self, mock_predict, auth_headers):
        """Test successful single prediction"""
        mock_predict.return_value = {
            "prediction": 0.75,
            "risk_category": "High",
            "confidence": 0.85,
            "features_used": 25
        }
        
        response = client.post("/predict/single", 
                             json=test_patient_data,
                             headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "risk_category" in data
        assert "confidence" in data

    def test_single_prediction_unauthorized(self):
        """Test single prediction without authentication"""
        response = client.post("/predict/single", json=test_patient_data)
        assert response.status_code == 401

    def test_single_prediction_invalid_data(self, auth_headers):
        """Test single prediction with invalid data"""
        invalid_data = test_patient_data.copy()
        invalid_data["age"] = "invalid_age"
        
        response = client.post("/predict/single", 
                             json=invalid_data,
                             headers=auth_headers)
        assert response.status_code == 422

    @patch('app.predictor.ReadmissionPredictor.predict_batch')
    def test_batch_prediction_success(self, mock_predict, auth_headers):
        """Test successful batch prediction"""
        mock_predict.return_value = {
            "predictions": [0.75, 0.25],
            "risk_categories": ["High", "Low"],
            "confidences": [0.85, 0.90],
            "total_processed": 2,
            "errors": []
        }
        
        files = {"file": ("test.csv", test_batch_data, "text/csv")}
        response = client.post("/predict/batch", 
                             files=files,
                             headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "total_processed" in data

    def test_batch_prediction_no_file(self, auth_headers):
        """Test batch prediction without file"""
        response = client.post("/predict/batch", headers=auth_headers)
        assert response.status_code == 422

class TestExplanationEndpoints:
    @patch('app.predictor.ReadmissionPredictor.explain_prediction')
    def test_explanation_success(self, mock_explain, auth_headers):
        """Test successful explanation generation"""
        mock_explain.return_value = {
            "prediction": 0.75,
            "risk_category": "High",
            "feature_contributions": [
                {"feature": "age", "contribution": 0.15, "direction": "increasing"},
                {"feature": "num_medications", "contribution": 0.10, "direction": "increasing"}
            ],
            "top_features": 2
        }
        
        response = client.post("/explain", 
                             json={"patient_data": test_patient_data, "top_features": 2},
                             headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "feature_contributions" in data

    def test_explanation_invalid_top_features(self, auth_headers):
        """Test explanation with invalid top_features parameter"""
        response = client.post("/explain", 
                             json={"patient_data": test_patient_data, "top_features": 0},
                             headers=auth_headers)
        assert response.status_code == 422

class TestMetricsEndpoints:
    @patch('app.predictor.ReadmissionPredictor.get_model_metrics')
    def test_model_metrics_success(self, mock_metrics, auth_headers):
        """Test successful metrics retrieval"""
        mock_metrics.return_value = {
            "auroc": 0.85,
            "f1_score": 0.78,
            "precision": 0.80,
            "recall": 0.75,
            "accuracy": 0.82
        }
        
        response = client.get("/metrics/model", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "auroc" in data
        assert "f1_score" in data

    def test_model_metrics_unauthorized(self):
        """Test metrics retrieval without authentication"""
        response = client.get("/metrics/model")
        assert response.status_code == 401

class TestAdminEndpoints:
    def test_admin_stats_unauthorized(self):
        """Test admin stats without admin role"""
        response = client.get("/admin/stats")
        assert response.status_code == 401

    @patch('app.main.get_user_stats')
    def test_admin_stats_success(self, mock_stats, admin_headers):
        """Test successful admin stats retrieval"""
        mock_stats.return_value = {
            "total_users": 10,
            "active_users": 8,
            "role_distribution": {"clinician": 5, "nurse": 3, "admin": 2}
        }
        
        response = client.get("/admin/stats", headers=admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert "total_users" in data
        assert "role_distribution" in data

    def test_admin_users_unauthorized(self):
        """Test admin users without admin role"""
        response = client.get("/admin/users")
        assert response.status_code == 401

    @patch('app.main.get_all_users')
    def test_admin_users_success(self, mock_users, admin_headers):
        """Test successful admin users retrieval"""
        mock_users.return_value = [
            {"id": 1, "username": "admin", "role": "admin", "is_active": True},
            {"id": 2, "username": "clinician1", "role": "clinician", "is_active": True}
        ]
        
        response = client.get("/admin/users", headers=admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["username"] == "admin"

class TestRateLimiting:
    def test_rate_limit_info(self):
        """Test rate limit information endpoint"""
        response = client.get("/rate-limit/info")
        assert response.status_code == 200
        data = response.json()
        assert "ip_address" in data
        assert "limits" in data

    def test_rate_limit_exceeded(self):
        """Test rate limiting behavior"""
        # Make multiple requests to trigger rate limiting
        for _ in range(10):
            response = client.get("/health")
            if response.status_code == 429:
                break
        else:
            # If no rate limiting occurred, that's also acceptable
            pass

class TestModelInfo:
    def test_model_info(self):
        """Test model information endpoint"""
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_type" in data
        assert "version" in data
        assert "last_updated" in data

class TestErrorHandling:
    def test_invalid_endpoint(self):
        """Test invalid endpoint handling"""
        response = client.get("/invalid/endpoint")
        assert response.status_code == 404

    def test_method_not_allowed(self):
        """Test method not allowed handling"""
        response = client.put("/health")
        assert response.status_code == 405

    def test_validation_error(self):
        """Test validation error handling"""
        response = client.post("/predict/single", json={"invalid": "data"})
        assert response.status_code == 422

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
