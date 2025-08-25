# Hospital Readmission Risk Predictor

An AI-powered system for predicting 30-day hospital readmission risk using machine learning and explainable AI techniques.

## 🏥 Project Overview

This system helps healthcare providers identify patients at high risk of readmission within 30 days of discharge, enabling proactive interventions to improve patient outcomes and reduce healthcare costs.

## ✨ Features

- **AI-Powered Predictions**: Machine learning models trained on comprehensive patient data
- **Real-time Risk Assessment**: Instant readmission risk predictions for individual patients
- **Batch Processing**: Handle multiple patients simultaneously via CSV upload
- **Explainable AI**: SHAP-based explanations for model predictions
- **Role-Based Access Control**: Secure access for different healthcare roles
- **Comprehensive Monitoring**: Data drift detection and model performance tracking
- **Modern Web Interface**: React-based dashboard with intuitive UX
- **RESTful API**: FastAPI backend with comprehensive documentation
- **Docker Support**: Containerized deployment for easy scaling

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Frontend│    │  FastAPI Backend│    │  MLflow Tracking│
│                 │    │                 │    │                 │
│  - Dashboard    │◄──►│  - Predictions  │◄──►│  - Model Registry│
│  - Predictions  │    │  - Explanations │    │  - Experiments  │
│  - Analytics    │    │  - Auth & RBAC  │    │  - Artifacts    │
│  - Admin Panel  │    │  - Rate Limiting│    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │   Monitoring    │    │   Docker Compose│
│                 │    │                 │    │                 │
│  - User Mgmt    │    │  - Drift Detection│   │  - Orchestration│
│  - Predictions  │    │  - Data Quality │    │  - Scaling      │
│  - Audit Logs   │    │  - Performance  │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker & Docker Compose
- PostgreSQL 15+

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Readmission
```

### 2. Environment Setup

```bash
# Copy environment file
cp env.example .env

# Edit environment variables
nano .env
```

### 3. Start with Docker

```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f api
```

### 4. Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **MLflow**: http://localhost:5000

### 5. Default Login Credentials

| Role | Username | Password | Permissions |
|------|----------|----------|-------------|
| Admin | `admin.user` | `admin123` | Full access |
| Doctor | `doctor.smith` | `password123` | Predict, Explain |
| Nurse | `nurse.jones` | `password123` | Predict only |

## 🛠️ Development Setup

### Backend Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL="postgresql://postgres:password@localhost:5432/readmission"
export SECRET_KEY="your-secret-key"

# Run the application
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Development

```bash
# Navigate to UI directory
cd ui

# Install dependencies
npm install

# Start development server
npm start
```

### Database Setup

```bash
# Start PostgreSQL
docker run -d \
  --name postgres \
  -e POSTGRES_DB=readmission \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=password \
  -p 5432:5432 \
  postgres:15

# Initialize database
psql -h localhost -U postgres -d readmission -f infra/init.sql
```

## 📊 Model Training

### 1. Data Preparation

```bash
# Preprocess raw data
python ml/preprocess_dataset.py

# Feature engineering
python ml/train.py --config configs/training_config.yaml
```

### 2. Model Training

```bash
# Train models with MLflow tracking
python ml/train.py \
  --experiment-name "hospital_readmission" \
  --data-path "data/processed/hospital_readmissions_processed.csv" \
  --models-dir "models/"
```

### 3. Model Evaluation

```bash
# Evaluate model performance
python ml/evaluate.py \
  --model-path "models/best_model.pkl" \
  --test-data "data/processed/test.csv"
```

## 🔧 Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RATE_LIMIT=100

# Database
DATABASE_URL=postgresql://postgres:password@localhost:5432/readmission

# JWT Security
SECRET_KEY=your-super-secret-jwt-key
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Model Configuration
MODELS_DIR=models/
MODEL_VERSION=1.0.0

# Monitoring
DRIFT_DETECTION_ENABLED=true
MONITORING_INTERVAL=300
```

### Model Configuration

```yaml
# configs/training_config.yaml
data:
  target_column: "readmitted_30d"
  categorical_features:
    - "sex"
    - "insurance"
    - "discharge_disposition"
  numerical_features:
    - "age"
    - "charlson_index"
    - "los_days"

training:
  test_size: 0.2
  random_state: 42
  cv_folds: 5
  
models:
  - "logistic_regression"
  - "random_forest"
  - "xgboost"
  - "catboost"
```

## 📈 API Usage

### Authentication

```bash
# Login to get JWT token
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "doctor.smith", "password": "password123"}'
```

### Single Prediction

```bash
# Make prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Authorization: Bearer <your-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_data": {
      "patient_id": "P001",
      "admission_id": "A001",
      "age": 65,
      "sex": "M",
      "charlson_index": 2,
      "los_days": 5
    }
  }'
```

### Batch Prediction

```bash
# Upload CSV for batch processing
curl -X POST "http://localhost:8000/batch_predict" \
  -H "Authorization: Bearer <your-token>" \
  -F "file=@patients.csv"
```

### Get Explanation

```bash
# Get SHAP explanation
curl -X POST "http://localhost:8000/explain" \
  -H "Authorization: Bearer <your-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_data": {...},
    "top_features": 10
  }'
```

## 🔍 Monitoring & Observability

### Health Checks

```bash
# System health
curl "http://localhost:8000/health"

# Model metrics
curl "http://localhost:8000/metrics"
```

### Data Drift Detection

```bash
# Run drift detection
python monitoring/drift_detector.py \
  --reference-data "data/processed/hospital_readmissions_processed.csv" \
  --current-data "data/current/patients.csv"
```

### MLflow Tracking

- **Experiments**: Track model training runs
- **Metrics**: Monitor performance over time
- **Artifacts**: Store models and datasets
- **Registry**: Version and deploy models

## 🧪 Testing

### Backend Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_predictor.py

# Run with coverage
pytest --cov=app tests/
```

### Frontend Tests

```bash
cd ui

# Run tests
npm test

# Run with coverage
npm run test:coverage

# Lint code
npm run lint
```

### Integration Tests

```bash
# Test API endpoints
pytest tests/integration/

# Test database operations
pytest tests/test_database.py
```

## 🚀 Deployment

### Production Deployment

```bash
# Build production images
docker-compose -f docker-compose.prod.yml build

# Deploy with production config
docker-compose -f docker-compose.prod.yml up -d

# Scale services
docker-compose -f docker-compose.prod.yml up -d --scale api=3
```

### Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n readmission

# Scale deployment
kubectl scale deployment readmission-api --replicas=5
```

### Environment-Specific Configs

```bash
# Development
docker-compose -f docker-compose.dev.yml up

# Staging
docker-compose -f docker-compose.staging.yml up

# Production
docker-compose -f docker-compose.prod.yml up
```

## 📚 Documentation

### API Documentation

- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

### Code Documentation

```bash
# Generate documentation
pdoc --html app/

# View in browser
open html/app/index.html
```

## 🤝 Contributing

### Development Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Add tests for new functionality
5. Run the test suite: `pytest tests/`
6. Commit your changes: `git commit -m 'Add amazing feature'`
7. Push to the branch: `git push origin feature/amazing-feature`
8. Open a Pull Request

### Code Style

```bash
# Backend formatting
black app/
isort app/
flake8 app/

# Frontend formatting
cd ui
npm run lint:fix
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Healthcare Data**: Sample data for demonstration purposes
- **Open Source Libraries**: FastAPI, React, MLflow, Evidently
- **Research Community**: For insights into hospital readmission prediction

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Documentation**: [Project Wiki](https://github.com/your-repo/wiki)

## 🔮 Roadmap

- [ ] Real-time streaming predictions
- [ ] Advanced fairness monitoring
- [ ] Multi-language support
- [ ] Mobile application
- [ ] Integration with EHR systems
- [ ] Advanced analytics dashboard
- [ ] Automated model retraining
- [ ] Cloud-native deployment options

---

**Built with ❤️ for better healthcare outcomes**

