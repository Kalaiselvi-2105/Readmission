# Testing Guide

This document provides comprehensive guidance for testing the Hospital Readmission Prediction System.

## Table of Contents

1. [Overview](#overview)
2. [Test Structure](#test-structure)
3. [Running Tests](#running-tests)
4. [Test Categories](#test-categories)
5. [Test Configuration](#test-configuration)
6. [Coverage Reports](#coverage-reports)
7. [Load Testing](#load-testing)
8. [Continuous Integration](#continuous-integration)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

## Overview

The testing suite is designed to ensure the reliability, performance, and security of the Hospital Readmission Prediction System. It covers:

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing
- **API Tests**: REST API endpoint testing
- **ML Tests**: Machine learning component testing
- **Frontend Tests**: React component testing
- **Load Tests**: Performance and stress testing
- **Security Tests**: Vulnerability scanning

## Test Structure

```
tests/
├── __init__.py
├── test_api.py              # API endpoint tests
├── test_ml_components.py    # ML pipeline tests
├── test_frontend.py         # Frontend component tests
├── load_test.py             # Load testing with Locust
├── conftest.py              # Shared test fixtures
└── fixtures/                # Test data and fixtures
    ├── sample_data.csv
    └── test_models/
```

## Running Tests

### Prerequisites

Install testing dependencies:

```bash
pip install -r requirements-test.txt
```

### Basic Test Execution

```bash
# Run all tests
python -m pytest

# Run with verbose output
python -m pytest -v

# Run specific test file
python -m pytest tests/test_api.py

# Run specific test class
python -m pytest tests/test_api.py::TestAuthentication

# Run specific test method
python -m pytest tests/test_api.py::TestAuthentication::test_login_success
```

### Using the Test Runner Script

```bash
# Run all tests
python run_tests.py --all

# Run specific test types
python run_tests.py --api
python run_tests.py --ml
python run_tests.py --frontend

# Run with coverage
python run_tests.py --coverage

# Run in parallel
python run_tests.py --parallel

# Run smoke tests
python run_tests.py --smoke

# Check dependencies
python run_tests.py --check-deps

# Clean test artifacts
python run_tests.py --clean
```

### Docker-based Testing

```bash
# Run tests in CI/CD environment
docker-compose -f docker-compose.ci.yml up ci-test-runner

# Run specific test services
docker-compose -f docker-compose.ci.yml up ci-api ci-db ci-redis
```

## Test Categories

### 1. Unit Tests (`@pytest.mark.unit`)

Test individual functions and methods in isolation.

```bash
python -m pytest tests/ -m unit
```

**Examples:**
- Data validation functions
- Utility functions
- Individual ML model methods

### 2. Integration Tests (`@pytest.mark.integration`)

Test component interactions and workflows.

```bash
python -m pytest tests/ -m integration
```

**Examples:**
- End-to-end prediction pipeline
- Database operations
- API authentication flow

### 3. API Tests (`@pytest.mark.api`)

Test REST API endpoints and responses.

```bash
python -m pytest tests/ -m api
```

**Examples:**
- HTTP status codes
- Response formats
- Authentication requirements
- Rate limiting

### 4. ML Tests (`@pytest.mark.ml`)

Test machine learning components.

```bash
python -m pytest tests/ -m ml
```

**Examples:**
- Model loading
- Prediction accuracy
- Feature engineering
- Data preprocessing

### 5. Frontend Tests (`@pytest.mark.frontend`)

Test React components and UI logic.

```bash
python -m pytest tests/ -m frontend
```

**Examples:**
- Component rendering
- State management
- User interactions
- API integration

### 6. Performance Tests (`@pytest.mark.benchmark`)

Test system performance characteristics.

```bash
python -m pytest tests/ -m benchmark
```

**Examples:**
- Response times
- Throughput
- Memory usage
- CPU utilization

### 7. Smoke Tests (`@pytest.mark.smoke`)

Quick tests to verify basic functionality.

```bash
python -m pytest tests/ -m smoke
```

**Examples:**
- Health checks
- Basic authentication
- Core prediction endpoint

## Test Configuration

### pytest.ini

The main pytest configuration file defines:

- Test discovery patterns
- Markers and their descriptions
- Coverage settings
- Output formats
- Warning filters

### Environment Variables

```bash
# Testing environment
export TESTING=true
export ENVIRONMENT=test

# Database configuration
export DATABASE_URL=postgresql://test_user:test_pass@localhost:5432/test_db
export REDIS_URL=redis://localhost:6379

# MLflow configuration
export MLFLOW_TRACKING_URI=http://localhost:5000

# Logging
export LOG_LEVEL=DEBUG
```

### Test Fixtures

Common test fixtures are defined in `conftest.py`:

```python
@pytest.fixture
def test_client():
    """FastAPI test client"""
    return TestClient(app)

@pytest.fixture
def auth_headers():
    """Authentication headers for testing"""
    token = create_access_token(data={"sub": "testuser", "role": "clinician"})
    return {"Authorization": f"Bearer {token}"}

@pytest.fixture
def sample_patient_data():
    """Sample patient data for testing"""
    return {
        "age": 65,
        "gender": "Female",
        # ... other fields
    }
```

## Coverage Reports

### Running Coverage

```bash
# Run tests with coverage
python -m pytest --cov=app --cov=ml --cov-report=html --cov-report=term-missing

# Generate coverage report
coverage run -m pytest
coverage report
coverage html
```

### Coverage Configuration

```ini
# .coveragerc
[run]
source = app,ml
omit = 
    */tests/*
    */venv/*
    */__pycache__/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
```

### Coverage Reports

- **Terminal**: `--cov-report=term-missing`
- **HTML**: `--cov-report=html:htmlcov/`
- **XML**: `--cov-report=xml:coverage.xml`
- **JSON**: `--cov-report=json:coverage.json`

## Load Testing

### Using Locust

```bash
# Install Locust
pip install locust

# Run load test
locust -f tests/load_test.py --host=http://localhost:8000

# Run with specific parameters
locust -f tests/load_test.py --host=http://localhost:8000 -u 20 -r 2 --run-time 2m

# Run headless
locust -f tests/load_test.py --host=http://localhost:8000 --headless -u 10 -r 1 --run-time 30s
```

### Load Test Scenarios

```python
# Available scenarios in load_test.py
scenarios = {
    "smoke": {"users": 5, "spawn_rate": 1, "run_time": "30s"},
    "load": {"users": 20, "spawn_rate": 2, "run_time": "2m"},
    "stress": {"users": 50, "spawn_rate": 5, "run_time": "5m"},
    "spike": {"users": 100, "spawn_rate": 20, "run_time": "1m"},
    "endurance": {"users": 30, "spawn_rate": 3, "run_time": "10m"}
}
```

### Performance Metrics

- **Response Time**: Average, min, max
- **Throughput**: Requests per second
- **Error Rate**: Failed requests percentage
- **Resource Usage**: CPU, memory, network

## Continuous Integration

### GitHub Actions

The CI/CD pipeline automatically runs:

1. **Code Quality**: Linting, formatting, type checking
2. **Unit Tests**: All test categories
3. **Security Scanning**: Bandit, npm audit, Snyk
4. **Integration Tests**: Database, Redis, MLflow
5. **Performance Tests**: Load testing, benchmarks
6. **Documentation**: API docs, Sphinx build

### Local CI Simulation

```bash
# Run CI pipeline locally
docker-compose -f docker-compose.ci.yml up --build

# Run specific CI stages
docker-compose -f docker-compose.ci.yml up ci-code-quality
docker-compose -f docker-compose.ci.yml up ci-security
docker-compose -f docker-compose.ci.yml up ci-test-runner
```

## Best Practices

### 1. Test Organization

- Group related tests in classes
- Use descriptive test names
- Follow AAA pattern (Arrange, Act, Assert)
- Keep tests independent and isolated

### 2. Test Data Management

- Use fixtures for common data
- Generate test data dynamically
- Clean up test data after tests
- Use realistic but minimal test data

### 3. Mocking and Stubbing

- Mock external dependencies
- Use dependency injection for testability
- Avoid mocking implementation details
- Test the interface, not the implementation

### 4. Assertions

- Use specific assertions
- Test both positive and negative cases
- Verify error conditions
- Check edge cases and boundaries

### 5. Performance Testing

- Test realistic load scenarios
- Monitor resource usage
- Set performance baselines
- Test failure scenarios

### 6. Security Testing

- Test authentication and authorization
- Validate input sanitization
- Check for common vulnerabilities
- Test rate limiting and DDoS protection

## Troubleshooting

### Common Issues

#### 1. Import Errors

```bash
# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or use pytest with proper path
python -m pytest --import-mode=importlib
```

#### 2. Database Connection Issues

```bash
# Check database service
docker-compose ps

# Verify connection string
echo $DATABASE_URL

# Test connection manually
python -c "import psycopg2; print(psycopg2.connect('$DATABASE_URL'))"
```

#### 3. Test Timeouts

```bash
# Increase timeout
pytest --timeout=300

# Run specific slow tests
pytest -m "not slow"

# Use pytest-xdist for parallel execution
pytest -n auto
```

#### 4. Coverage Issues

```bash
# Check coverage configuration
cat .coveragerc

# Verify source paths
coverage report --show-missing

# Debug coverage collection
coverage debug data
```

### Debug Mode

```bash
# Run with debug output
pytest -v -s --tb=long

# Use pdb for debugging
pytest --pdb

# Run single test with debug
pytest tests/test_api.py::test_login_success -v -s --pdb
```

### Test Isolation

```bash
# Run tests in isolation
pytest --dist=no

# Use temporary directories
pytest --basetemp=/tmp/pytest

# Clean between tests
pytest --cache-clear
```

## Test Maintenance

### Regular Tasks

1. **Update Test Data**: Keep test data current with schema changes
2. **Review Coverage**: Ensure new code is covered by tests
3. **Update Dependencies**: Keep testing libraries up to date
4. **Performance Monitoring**: Track test execution times
5. **Documentation**: Update test documentation as needed

### Test Review Checklist

- [ ] Tests cover all critical paths
- [ ] Edge cases are tested
- [ ] Error conditions are handled
- [ ] Tests are fast and reliable
- [ ] Mocking is appropriate
- [ ] Assertions are specific
- [ ] Test data is realistic
- [ ] Tests are independent

### Continuous Improvement

- Monitor test execution metrics
- Identify slow or flaky tests
- Refactor tests for better maintainability
- Add tests for bug fixes
- Improve test coverage over time

## Support

For testing-related issues:

1. Check this documentation
2. Review test logs and error messages
3. Consult pytest documentation
4. Check GitHub Issues for known problems
5. Contact the development team

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [Locust Documentation](https://docs.locust.io/)
- [Testing Best Practices](https://realpython.com/python-testing/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)
