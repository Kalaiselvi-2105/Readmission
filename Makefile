.PHONY: help setup train evaluate serve batch_score test clean docker-build docker-run

help: ## Show this help message
	@echo "Hospital Readmission Risk Predictor - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Install Python dependencies
	pip install -r requirements.txt
	@echo "✅ Dependencies installed"

generate-data: ## Generate synthetic data for development
	python ml/generate_data.py
	@echo "✅ Synthetic data generated"

train: ## Train ML models
	python ml/train.py
	@echo "✅ Models trained and registered"

evaluate: ## Evaluate model performance
	python ml/evaluate.py
	@echo "✅ Model evaluation completed"

explain: ## Generate SHAP explanations
	python ml/explain.py
	@echo "✅ SHAP explanations generated"

serve: ## Start FastAPI server
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

batch_score: ## Run batch scoring pipeline
	python ml/batch_score.py
	@echo "✅ Batch scoring completed"

test: ## Run all tests
	pytest tests/ -v
	@echo "✅ All tests passed"

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

test-integration: ## Run integration tests only
	pytest tests/integration/ -v

test-data: ## Run data validation tests
	pytest tests/data/ -v

lint: ## Run code linting
	black app/ ml/ tests/
	flake8 app/ ml/ tests/
	mypy app/ ml/

clean: ## Clean generated files
	rm -rf mlflow/
	rm -rf models/
	rm -rf data/processed/
	rm -rf data/raw/
	@echo "✅ Cleaned generated files"

docker-build: ## Build Docker image
	docker build -t readmission-predictor .
	@echo "✅ Docker image built"

docker-run: ## Run with Docker Compose
	docker-compose up -d
	@echo "✅ Services started with Docker Compose"

docker-stop: ## Stop Docker services
	docker-compose down
	@echo "✅ Docker services stopped"

setup-ui: ## Setup React UI dependencies
	cd ui && npm install
	@echo "✅ UI dependencies installed"

start-ui: ## Start React development server
	cd ui && npm start

build-ui: ## Build React app for production
	cd ui && npm run build
	@echo "✅ UI built for production"

monitoring: ## Start monitoring services
	mlflow server --host 0.0.0.0 --port 5000 &
	@echo "✅ MLflow server started on port 5000"

full-setup: setup generate-data train ## Complete setup pipeline
	@echo "✅ Full setup completed - ready to serve!"

dev: setup generate-data train monitoring serve ## Development mode
	@echo "✅ Development environment ready!"

