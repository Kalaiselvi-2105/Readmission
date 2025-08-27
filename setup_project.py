#!/usr/bin/env python3
"""
Comprehensive project setup script for the Hospital Readmission Risk Predictor.
This script sets up the entire project structure, installs dependencies, and runs initial setup.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
import argparse
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(command, description, check=True):
    """Run a shell command with logging."""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {command}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=check, 
            capture_output=True, 
            text=True
        )
        
        if result.stdout:
            logger.info(f"Output: {result.stdout}")
        
        if result.stderr:
            logger.warning(f"Stderr: {result.stderr}")
        
        return result.returncode == 0
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        if check:
            raise
        return False


def create_directories():
    """Create necessary project directories."""
    logger.info("Creating project directories...")
    
    directories = [
        "data",
        "data/raw",
        "data/processed",
        "models",
        "monitoring",
        "monitoring/reports",
        "logs",
        "ui/src",
        "ui/public",
        "infra",
        "tests/unit",
        "tests/integration",
        "tests/data",
        "docs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def setup_python_environment():
    """Set up Python virtual environment and install dependencies."""
    logger.info("Setting up Python environment...")
    
    # Check if virtual environment exists
    if not Path("venv").exists():
        logger.info("Creating virtual environment...")
        if not run_command("python -m venv venv", "Creating virtual environment"):
            logger.error("Failed to create virtual environment")
            return False
    
    # Activate virtual environment and install dependencies
    if os.name == 'nt':  # Windows
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
    else:  # Unix/Linux/macOS
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
    
    # Install dependencies
    if not run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip"):
        logger.warning("Failed to upgrade pip, continuing...")
    
    if not run_command(f"{pip_cmd} install -r requirements.txt", "Installing Python dependencies"):
        logger.error("Failed to install Python dependencies")
        return False
    
    logger.info("Python environment setup completed")
    return True


def setup_node_environment():
    """Set up Node.js environment and install UI dependencies."""
    logger.info("Setting up Node.js environment...")
    
    # Check if Node.js is installed
    if not run_command("node --version", "Checking Node.js installation", check=False):
        logger.warning("Node.js not found. Please install Node.js 18+ to use the UI.")
        return False
    
    # Check if npm is available
    if not run_command("npm --version", "Checking npm installation", check=False):
        logger.warning("npm not found. Please install npm to use the UI.")
        return False
    
    # Install UI dependencies
    os.chdir("ui")
    if not run_command("npm install", "Installing UI dependencies"):
        logger.error("Failed to install UI dependencies")
        os.chdir("..")
        return False
    
    os.chdir("..")
    logger.info("Node.js environment setup completed")
    return True


def generate_initial_data():
    """Generate initial synthetic data for development."""
    logger.info("Generating initial synthetic data...")
    
    # Activate virtual environment for Python commands
    if os.name == 'nt':  # Windows
        python_cmd = "venv\\Scripts\\python"
    else:  # Unix/Linux/macOS
        python_cmd = "venv/bin/python"
    
    if not run_command(f"{python_cmd} ml/generate_data.py --n-samples 2000 --validate", "Generating synthetic data"):
        logger.error("Failed to generate synthetic data")
        return False
    
    logger.info("Initial data generation completed")
    return True


def run_initial_training():
    """Run initial model training."""
    logger.info("Running initial model training...")
    
    # Activate virtual environment for Python commands
    if os.name == 'nt':  # Windows
        python_cmd = "venv\\Scripts\\python"
    else:  # Unix/Linux/macOS
        python_cmd = "venv/bin/python"
    
    if not run_command(f"{python_cmd} ml/train.py --use-synthetic --cross-validate", "Training initial models"):
        logger.error("Failed to train initial models")
        return False
    
    logger.info("Initial model training completed")
    return True


def create_config_files():
    """Create configuration files."""
    logger.info("Creating configuration files...")
    
    # Create .env file from template
    if not Path(".env").exists() and Path("env.example").exists():
        import shutil
        shutil.copy("env.example", ".env")
        logger.info("Created .env file from template")
        logger.warning("Please review and update .env file with your configuration")
    
    # Create MLflow configuration
    mlflow_config = {
        "tracking_uri": "http://localhost:5000",
        "registry_uri": "http://localhost:5000",
        "experiment_name": "readmission_prediction"
    }
    
    with open("mlflow_config.json", "w") as f:
        json.dump(mlflow_config, f, indent=2)
    
    logger.info("Created MLflow configuration file")
    
    # Create database initialization script
    db_init_sql = """
-- Database initialization script for Hospital Readmission Risk Predictor
CREATE DATABASE IF NOT EXISTS readmission;

-- Create tables for storing predictions and audit logs
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    patient_id VARCHAR(255) NOT NULL,
    admission_id VARCHAR(255) NOT NULL,
    risk_score DECIMAL(5,4) NOT NULL,
    prediction BOOLEAN NOT NULL,
    threshold DECIMAL(3,2) NOT NULL,
    model_version VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_id VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100),
    resource_id VARCHAR(255),
    details JSONB,
    ip_address INET,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_predictions_patient_id ON predictions(patient_id);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit_logs(created_at);

-- Insert sample users for development
INSERT INTO audit_logs (user_id, action, resource_type, details) VALUES
('admin', 'system_startup', 'system', '{"message": "System initialized"}')
ON CONFLICT DO NOTHING;
"""
    
    infra_dir = Path("infra")
    infra_dir.mkdir(exist_ok=True)
    
    with open(infra_dir / "init.sql", "w") as f:
        f.write(db_init_sql)
    
    logger.info("Created database initialization script")


def create_documentation():
    """Create initial documentation."""
    logger.info("Creating documentation...")
    
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    
    # API documentation
    api_docs = """# API Documentation

## Endpoints

### Health Check
- **GET** `/health` - Service health and model status

### Predictions
- **POST** `/predict` - Single patient prediction
- **POST** `/batch_predict` - Batch predictions for multiple patients

### Explanations
- **POST** `/explain` - SHAP feature explanations

### Monitoring
- **GET** `/metrics` - Model performance metrics and drift detection

## Authentication
All endpoints require JWT authentication via Bearer token in Authorization header.

## Rate Limiting
API is rate-limited to 100 requests per minute per user.

## Request/Response Examples
See the interactive API documentation at `/docs` when running in development mode.
"""
    
    with open(docs_dir / "API.md", "w") as f:
        f.write(api_docs)
    
    # Development guide
    dev_guide = """# Development Guide

## Setup
1. Install Python 3.11+ and Node.js 18+
2. Run `python setup_project.py` to set up the project
3. Activate virtual environment: `source venv/bin/activate` (Unix) or `venv\\Scripts\\activate` (Windows)

## Development Workflow
1. **Data**: Use `python ml/generate_data.py` to create synthetic data
2. **Training**: Use `python ml/train.py` to train models
3. **API**: Use `uvicorn app.main:app --reload` to start development server
4. **UI**: Use `cd ui && npm start` to start React development server

## Testing
- Run all tests: `pytest tests/`
- Run specific test categories: `pytest tests/unit/`, `pytest tests/integration/`

## Docker
- Build: `docker build -t readmission-predictor .`
- Run: `docker-compose up -d`

## MLflow
- Start tracking server: `mlflow server --host 0.0.0.0 --port 5000`
- View experiments: http://localhost:5000
"""
    
    with open(docs_dir / "DEVELOPMENT.md", "w") as f:
        f.write(dev_guide)
    
    logger.info("Created documentation files")


def run_tests():
    """Run the test suite."""
    logger.info("Running test suite...")
    
    # Activate virtual environment for Python commands
    if os.name == 'nt':  # Windows
        python_cmd = "venv\\Scripts\\python"
    else:  # Unix/Linux/macOS
        python_cmd = "venv/bin/python"
    
    if not run_command(f"{python_cmd} -m pytest tests/ -v", "Running tests"):
        logger.warning("Some tests failed. This is normal for initial setup.")
        return False
    
    logger.info("Test suite completed")
    return True


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description='Setup Hospital Readmission Risk Predictor project')
    parser.add_argument('--skip-ui', action='store_true', help='Skip UI setup')
    parser.add_argument('--skip-training', action='store_true', help='Skip initial model training')
    parser.add_argument('--skip-tests', action='store_true', help='Skip test execution')
    parser.add_argument('--data-samples', type=int, default=2000, help='Number of synthetic data samples')
    
    args = parser.parse_args()
    
    logger.info("Starting Hospital Readmission Risk Predictor project setup...")
    
    try:
        # Step 1: Create directories
        create_directories()
        
        # Step 2: Setup Python environment
        if not setup_python_environment():
            logger.error("Python environment setup failed")
            return 1
        
        # Step 3: Setup Node.js environment (optional)
        if not args.skip_ui:
            setup_node_environment()
        
        # Step 4: Create configuration files
        create_config_files()
        
        # Step 5: Create documentation
        create_documentation()
        
        # Step 6: Generate initial data
        if not generate_initial_data():
            logger.error("Initial data generation failed")
            return 1
        
        # Step 7: Run initial training (optional)
        if not args.skip_training:
            if not run_initial_training():
                logger.error("Initial training failed")
                return 1
        
        # Step 8: Run tests (optional)
        if not args.skip_tests:
            run_tests()
        
        # Step 9: Final setup instructions
        logger.info("\n" + "="*60)
        logger.info("PROJECT SETUP COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info("\nNext steps:")
        logger.info("1. Review and update .env file with your configuration")
        logger.info("2. Activate virtual environment:")
        if os.name == 'nt':
            logger.info("   venv\\Scripts\\activate")
        else:
            logger.info("   source venv/bin/activate")
        logger.info("3. Start MLflow tracking server: mlflow server --host 0.0.0.0 --port 5000")
        logger.info("4. Start API server: uvicorn app.main:app --reload")
        if not args.skip_ui:
            logger.info("5. Start UI: cd ui && npm start")
        logger.info("\nDocumentation available in docs/ directory")
        logger.info("API documentation available at http://localhost:8000/docs when running")
        
        return 0
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)






