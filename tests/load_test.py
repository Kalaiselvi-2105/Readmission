"""
Load Testing Script for Hospital Readmission Prediction System

This script uses Locust to perform load testing on the API endpoints
to ensure the system can handle expected traffic loads.
"""

import json
import random
from locust import HttpUser, task, between, events
from typing import Dict, Any

class HospitalReadmissionUser(HttpUser):
    """Simulates a user interacting with the hospital readmission prediction system"""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Login and get authentication token"""
        try:
            # Login to get token
            login_data = {
                "username": "clinician1",
                "password": "password123"
            }
            
            response = self.client.post("/auth/login", json=login_data)
            if response.status_code == 200:
                data = response.json()
                self.token = data.get("access_token")
                self.headers = {"Authorization": f"Bearer {self.token}"}
                print(f"✅ User logged in successfully: {self.token[:20]}...")
            else:
                print(f"❌ Login failed: {response.status_code}")
                self.token = None
                self.headers = {}
                
        except Exception as e:
            print(f"❌ Error during login: {e}")
            self.token = None
            self.headers = {}
    
    def get_random_patient_data(self) -> Dict[str, Any]:
        """Generate random patient data for testing"""
        return {
            "age": random.randint(18, 95),
            "gender": random.choice(["Female", "Male"]),
            "admission_type_id": random.randint(1, 8),
            "discharge_disposition_id": random.randint(1, 29),
            "admission_source_id": random.randint(1, 25),
            "time_in_hospital": random.randint(1, 14),
            "num_lab_procedures": random.randint(0, 132),
            "num_procedures": random.randint(0, 6),
            "num_medications": random.randint(0, 81),
            "number_outpatient": random.randint(0, 38),
            "number_emergency": random.randint(0, 76),
            "number_inpatient": random.randint(0, 21),
            "number_diagnoses": random.randint(1, 16),
            "max_glu_serum": random.choice(["None", ">200", ">300", "Norm"]),
            "A1Cresult": random.choice(["None", ">7", ">8", "Norm"]),
            "metformin": random.choice(["No", "Up", "Down", "Steady"]),
            "repaglinide": random.choice(["No", "Up", "Down", "Steady"]),
            "nateglinide": random.choice(["No", "Up", "Down", "Steady"]),
            "chlorpropamide": random.choice(["No", "Up", "Down", "Steady"]),
            "glimepiride": random.choice(["No", "Up", "Down", "Steady"]),
            "acetohexamide": random.choice(["No", "Up", "Down", "Steady"]),
            "tolbutamide": random.choice(["No", "Up", "Down", "Steady"]),
            "pioglitazone": random.choice(["No", "Up", "Down", "Steady"]),
            "rosiglitazone": random.choice(["No", "Up", "Down", "Steady"]),
            "acarbose": random.choice(["No", "Up", "Down", "Steady"]),
            "miglitol": random.choice(["No", "Up", "Down", "Steady"]),
            "troglitazone": random.choice(["No", "Up", "Down", "Steady"]),
            "tolazamide": random.choice(["No", "Up", "Down", "Steady"]),
            "examide": random.choice(["No", "Up", "Down", "Steady"]),
            "citoglipton": random.choice(["No", "Up", "Down", "Steady"]),
            "insulin": random.choice(["No", "Up", "Down", "Steady"]),
            "glyburide-metformin": random.choice(["No", "Up", "Down", "Steady"]),
            "glipizide-metformin": random.choice(["No", "Up", "Down", "Steady"]),
            "glimepiride-pioglitazone": random.choice(["No", "Up", "Down", "Steady"]),
            "metformin-rosiglitazone": random.choice(["No", "Up", "Down", "Steady"]),
            "metformin-pioglitazone": random.choice(["No", "Up", "Down", "Steady"]),
            "change": random.choice(["No", "Ch"]),
            "diabetesMed": random.choice(["No", "Yes"])
        }
    
    def get_random_batch_data(self, size: int = 5) -> str:
        """Generate random batch data for testing"""
        import pandas as pd
        
        batch_data = []
        for _ in range(size):
            patient = self.get_random_patient_data()
            batch_data.append(patient)
        
        df = pd.DataFrame(batch_data)
        return df.to_csv(index=False)
    
    @task(3)
    def health_check(self):
        """Check system health (high frequency)"""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")
    
    @task(2)
    def readiness_check(self):
        """Check system readiness"""
        with self.client.get("/ready", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Readiness check failed: {response.status_code}")
    
    @task(2)
    def single_prediction(self):
        """Make single patient prediction (authenticated)"""
        if not self.token:
            return
        
        patient_data = self.get_random_patient_data()
        
        with self.client.post(
            "/predict/single", 
            json=patient_data,
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "prediction" in data and "risk_category" in data:
                    response.success()
                else:
                    response.failure("Invalid prediction response format")
            else:
                response.failure(f"Prediction failed: {response.status_code}")
    
    @task(1)
    def batch_prediction(self):
        """Make batch predictions (authenticated)"""
        if not self.token:
            return
        
        batch_data = self.get_random_batch_data(random.randint(3, 10))
        
        files = {"file": ("test_batch.csv", batch_data, "text/csv")}
        
        with self.client.post(
            "/predict/batch",
            files=files,
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "predictions" in data and "total_processed" in data:
                    response.success()
                else:
                    response.failure("Invalid batch prediction response format")
            else:
                response.failure(f"Batch prediction failed: {response.status_code}")
    
    @task(1)
    def explanation_request(self):
        """Request prediction explanation (authenticated)"""
        if not self.token:
            return
        
        patient_data = self.get_random_patient_data()
        explanation_request = {
            "patient_data": patient_data,
            "top_features": random.randint(3, 10)
        }
        
        with self.client.post(
            "/explain",
            json=explanation_request,
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "feature_contributions" in data:
                    response.success()
                else:
                    response.failure("Invalid explanation response format")
            else:
                response.failure(f"Explanation request failed: {response.status_code}")
    
    @task(1)
    def get_model_metrics(self):
        """Get model performance metrics (authenticated)"""
        if not self.token:
            return
        
        with self.client.get(
            "/metrics/model",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "auroc" in data and "f1_score" in data:
                    response.success()
                else:
                    response.failure("Invalid metrics response format")
            else:
                response.failure(f"Metrics request failed: {response.status_code}")
    
    @task(1)
    def get_model_info(self):
        """Get model information (public endpoint)"""
        with self.client.get("/model/info", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "model_type" in data and "version" in data:
                    response.success()
                else:
                    response.failure("Invalid model info response format")
            else:
                response.failure(f"Model info request failed: {response.status_code}")
    
    @task(1)
    def rate_limit_info(self):
        """Get rate limit information (public endpoint)"""
        with self.client.get("/rate-limit/info", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "ip_address" in data and "limits" in data:
                    response.success()
                else:
                    response.failure("Invalid rate limit info response format")
            else:
                response.failure(f"Rate limit info request failed: {response.status_code}")

class AdminUser(HttpUser):
    """Simulates an admin user accessing administrative endpoints"""
    
    wait_time = between(2, 5)  # Admins make fewer requests
    
    def on_start(self):
        """Login as admin"""
        try:
            login_data = {
                "username": "admin",
                "password": "admin123"
            }
            
            response = self.client.post("/auth/login", json=login_data)
            if response.status_code == 200:
                data = response.json()
                self.token = data.get("access_token")
                self.headers = {"Authorization": f"Bearer {self.token}"}
                print(f"✅ Admin logged in successfully: {self.token[:20]}...")
            else:
                print(f"❌ Admin login failed: {response.status_code}")
                self.token = None
                self.headers = {}
                
        except Exception as e:
            print(f"❌ Error during admin login: {e}")
            self.token = None
            self.headers = {}
    
    @task(1)
    def get_admin_stats(self):
        """Get administrative statistics"""
        if not self.token:
            return
        
        with self.client.get(
            "/admin/stats",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "total_users" in data and "role_distribution" in data:
                    response.success()
                else:
                    response.failure("Invalid admin stats response format")
            else:
                response.failure(f"Admin stats request failed: {response.status_code}")
    
    @task(1)
    def get_all_users(self):
        """Get all user information"""
        if not self.token:
            return
        
        with self.client.get(
            "/admin/users",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    response.success()
                else:
                    response.failure("Invalid users response format")
            else:
                response.failure(f"Users request failed: {response.status_code}")

# Event handlers for monitoring
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when the test starts"""
    print("🚀 Load testing started!")
    print(f"Target host: {environment.host}")
    print(f"Number of users: {environment.runner.user_count}")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when the test stops"""
    print("🏁 Load testing completed!")
    
    # Print summary statistics
    stats = environment.stats
    print(f"\n📊 Test Summary:")
    print(f"Total requests: {stats.total.num_requests}")
    print(f"Failed requests: {stats.total.num_failures}")
    print(f"Average response time: {stats.total.avg_response_time:.2f}ms")
    print(f"Min response time: {stats.total.min_response_time:.2f}ms")
    print(f"Max response time: {stats.total.max_response_time:.2f}ms")
    print(f"Requests per second: {stats.total.current_rps:.2f}")

@events.request.add_listener
def on_request(request_type, name, response_time, response_length, response, context, exception, start_time, url, **kwargs):
    """Called for each request"""
    if exception:
        print(f"❌ Request failed: {name} - {exception}")
    elif response.status_code >= 400:
        print(f"⚠️  Request error: {name} - {response.status_code}")

# Configuration for different load scenarios
def get_load_scenarios():
    """Define different load testing scenarios"""
    return {
        "smoke": {
            "users": 5,
            "spawn_rate": 1,
            "run_time": "30s"
        },
        "load": {
            "users": 20,
            "spawn_rate": 2,
            "run_time": "2m"
        },
        "stress": {
            "users": 50,
            "spawn_rate": 5,
            "run_time": "5m"
        },
        "spike": {
            "users": 100,
            "spawn_rate": 20,
            "run_time": "1m"
        },
        "endurance": {
            "users": 30,
            "spawn_rate": 3,
            "run_time": "10m"
        }
    }

if __name__ == "__main__":
    # This allows running the script directly for testing
    import subprocess
    import sys
    
    print("🏥 Hospital Readmission Prediction System - Load Testing")
    print("=" * 60)
    
    scenarios = get_load_scenarios()
    print("Available load scenarios:")
    for name, config in scenarios.items():
        print(f"  {name}: {config['users']} users, {config['spawn_rate']} spawn rate, {config['run_time']} duration")
    
    print("\nTo run a specific scenario:")
    print("  locust -f tests/load_test.py --host=http://localhost:8000")
    print("\nOr run with specific parameters:")
    print("  locust -f tests/load_test.py --host=http://localhost:8000 -u 20 -r 2 --run-time 2m")
