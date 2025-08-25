#!/usr/bin/env python3
"""
Test Runner Script for Hospital Readmission Prediction System

This script provides a convenient way to run different types of tests
with various options and configurations.
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=False,
            text=True
        )
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ {description} completed successfully in {elapsed_time:.2f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ {description} failed with error: {e}")
        return False

def check_dependencies():
    """Check if required testing dependencies are installed"""
    required_packages = [
        'pytest',
        'pytest-cov',
        'pytest-html',
        'pytest-xdist'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing required testing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All required testing packages are installed")
    return True

def run_unit_tests():
    """Run unit tests"""
    return run_command(
        "python -m pytest tests/ -m unit -v --tb=short",
        "Unit Tests"
    )

def run_integration_tests():
    """Run integration tests"""
    return run_command(
        "python -m pytest tests/ -m integration -v --tb=short",
        "Integration Tests"
    )

def run_api_tests():
    """Run API tests"""
    return run_command(
        "python -m pytest tests/test_api.py -v --tb=short",
        "API Tests"
    )

def run_ml_tests():
    """Run machine learning tests"""
    return run_command(
        "python -m pytest tests/test_ml_components.py -v --tb=short",
        "Machine Learning Tests"
    )

def run_frontend_tests():
    """Run frontend tests"""
    return run_command(
        "python -m pytest tests/test_frontend.py -v --tb=short",
        "Frontend Tests"
    )

def run_all_tests():
    """Run all tests"""
    return run_command(
        "python -m pytest tests/ -v --tb=short",
        "All Tests"
    )

def run_tests_with_coverage():
    """Run tests with coverage reporting"""
    return run_command(
        "python -m pytest tests/ -v --cov=app --cov=ml --cov-report=term-missing --cov-report=html:htmlcov",
        "Tests with Coverage"
    )

def run_tests_parallel():
    """Run tests in parallel"""
    return run_command(
        "python -m pytest tests/ -v -n auto --tb=short",
        "Parallel Tests"
    )

def run_smoke_tests():
    """Run smoke tests"""
    return run_command(
        "python -m pytest tests/ -m smoke -v --tb=short",
        "Smoke Tests"
    )

def run_slow_tests():
    """Run slow tests"""
    return run_command(
        "python -m pytest tests/ -m slow -v --tb=short",
        "Slow Tests"
    )

def generate_test_report():
    """Generate comprehensive test report"""
    return run_command(
        "python -m pytest tests/ -v --html=test-report.html --self-contained-html --junitxml=junit.xml",
        "Test Report Generation"
    )

def clean_test_artifacts():
    """Clean up test artifacts"""
    artifacts = [
        'htmlcov/',
        'coverage.xml',
        'junit.xml',
        'test-report.html',
        '.coverage',
        '.pytest_cache/',
        '__pycache__/'
    ]
    
    print("\n🧹 Cleaning up test artifacts...")
    
    for artifact in artifacts:
        if os.path.exists(artifact):
            if os.path.isdir(artifact):
                import shutil
                shutil.rmtree(artifact)
            else:
                os.remove(artifact)
            print(f"  Removed: {artifact}")
    
    print("✅ Test artifacts cleaned up")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Test Runner for Hospital Readmission Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py --all                    # Run all tests
  python run_tests.py --unit                   # Run only unit tests
  python run_tests.py --api --coverage         # Run API tests with coverage
  python run_tests.py --parallel               # Run tests in parallel
  python run_tests.py --smoke                  # Run smoke tests
  python run_tests.py --clean                  # Clean test artifacts
        """
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all tests'
    )
    
    parser.add_argument(
        '--unit',
        action='store_true',
        help='Run unit tests only'
    )
    
    parser.add_argument(
        '--integration',
        action='store_true',
        help='Run integration tests only'
    )
    
    parser.add_argument(
        '--api',
        action='store_true',
        help='Run API tests only'
    )
    
    parser.add_argument(
        '--ml',
        action='store_true',
        help='Run machine learning tests only'
    )
    
    parser.add_argument(
        '--frontend',
        action='store_true',
        help='Run frontend tests only'
    )
    
    parser.add_argument(
        '--coverage',
        action='store_true',
        help='Run tests with coverage reporting'
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run tests in parallel'
    )
    
    parser.add_argument(
        '--smoke',
        action='store_true',
        help='Run smoke tests only'
    )
    
    parser.add_argument(
        '--slow',
        action='store_true',
        help='Run slow tests only'
    )
    
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate comprehensive test report'
    )
    
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Clean up test artifacts'
    )
    
    parser.add_argument(
        '--check-deps',
        action='store_true',
        help='Check testing dependencies'
    )
    
    args = parser.parse_args()
    
    # Check if we're in the right directory
    if not os.path.exists('tests/'):
        print("❌ Error: 'tests/' directory not found. Please run this script from the project root.")
        sys.exit(1)
    
    print("🏥 Hospital Readmission Prediction System - Test Runner")
    print("=" * 60)
    
    # Check dependencies if requested
    if args.check_deps:
        if not check_dependencies():
            sys.exit(1)
        return
    
    # Clean artifacts if requested
    if args.clean:
        clean_test_artifacts()
        return
    
    # Check dependencies before running tests
    if not check_dependencies():
        print("\n❌ Please install missing dependencies before running tests.")
        sys.exit(1)
    
    # Run tests based on arguments
    success = True
    
    if args.all:
        success = run_all_tests()
    elif args.unit:
        success = run_unit_tests()
    elif args.integration:
        success = run_integration_tests()
    elif args.api:
        success = run_api_tests()
    elif args.ml:
        success = run_ml_tests()
    elif args.frontend:
        success = run_frontend_tests()
    elif args.smoke:
        success = run_smoke_tests()
    elif args.slow:
        success = run_slow_tests()
    elif args.coverage:
        success = run_tests_with_coverage()
    elif args.parallel:
        success = run_tests_parallel()
    elif args.report:
        success = generate_test_report()
    else:
        # Default: run all tests
        success = run_all_tests()
    
    # Final summary
    print(f"\n{'='*60}")
    if success:
        print("🎉 All requested tests completed successfully!")
    else:
        print("💥 Some tests failed. Please check the output above.")
        sys.exit(1)
    
    print("=" * 60)

if __name__ == "__main__":
    main()
