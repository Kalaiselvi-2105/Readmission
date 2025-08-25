#!/usr/bin/env python3
"""
Startup script for the Hospital Readmission Risk Predictor.
This script helps users quickly start the system and check dependencies.
"""

import os
import sys
import subprocess
import platform
import time
from pathlib import Path

def print_banner():
    """Print the startup banner."""
    print("🏥" + "="*60 + "🏥")
    print("🏥  Hospital Readmission Risk Predictor - Startup Script  🏥")
    print("🏥" + "="*60 + "🏥")
    print()

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 11):
        print(f"❌ Python 3.11+ required. Current version: {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True

def check_docker():
    """Check if Docker is available."""
    print("🐳 Checking Docker...")
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Docker available: {result.stdout.strip()}")
            return True
        else:
            print("❌ Docker not available")
            return False
    except FileNotFoundError:
        print("❌ Docker not found. Please install Docker Desktop")
        return False

def check_docker_compose():
    """Check if Docker Compose is available."""
    print("🐳 Checking Docker Compose...")
    try:
        result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Docker Compose available: {result.stdout.strip()}")
            return True
        else:
            print("❌ Docker Compose not available")
            return False
    except FileNotFoundError:
        print("❌ Docker Compose not found. Please install Docker Compose")
        return False

def check_node():
    """Check if Node.js is available."""
    print("🟢 Checking Node.js...")
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.strip()
            major_version = int(version.split('.')[0].replace('v', ''))
            if major_version >= 18:
                print(f"✅ Node.js {version} - Compatible")
                return True
            else:
                print(f"❌ Node.js 18+ required. Current version: {version}")
                return False
        else:
            print("❌ Node.js not available")
            return False
    except FileNotFoundError:
        print("❌ Node.js not found. Please install Node.js 18+")
        return False

def check_dependencies():
    """Check all system dependencies."""
    print("🔍 Checking system dependencies...")
    print()
    
    checks = [
        check_python_version,
        check_docker,
        check_docker_compose,
        check_node
    ]
    
    all_passed = True
    for check in checks:
        if not check():
            all_passed = False
        print()
    
    return all_passed

def setup_environment():
    """Setup environment configuration."""
    print("⚙️  Setting up environment...")
    
    env_file = Path(".env")
    env_example = Path("env.example")
    
    if not env_file.exists() and env_example.exists():
        print("📝 Creating .env file from template...")
        try:
            with open(env_example, 'r') as f:
                content = f.read()
            
            with open(env_file, 'w') as f:
                f.write(content)
            
            print("✅ .env file created. Please edit it with your configuration.")
        except Exception as e:
            print(f"❌ Failed to create .env file: {e}")
            return False
    elif env_file.exists():
        print("✅ .env file already exists")
    else:
        print("⚠️  No environment template found")
    
    return True

def install_python_dependencies():
    """Install Python dependencies."""
    print("📦 Installing Python dependencies...")
    
    if not Path("venv").exists():
        print("🐍 Creating virtual environment...")
        try:
            subprocess.run([sys.executable, '-m', 'venv', 'venv'], check=True)
            print("✅ Virtual environment created")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create virtual environment: {e}")
            return False
    
    # Activate virtual environment and install dependencies
    if platform.system() == "Windows":
        pip_path = "venv\\Scripts\\pip"
        activate_script = "venv\\Scripts\\activate"
    else:
        pip_path = "venv/bin/pip"
        activate_script = "venv/bin/activate"
    
    print("📥 Installing requirements...")
    try:
        subprocess.run([pip_path, 'install', '-r', 'requirements.txt'], check=True)
        print("✅ Python dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Python dependencies: {e}")
        return False

def install_node_dependencies():
    """Install Node.js dependencies."""
    print("📦 Installing Node.js dependencies...")
    
    ui_dir = Path("ui")
    if not ui_dir.exists():
        print("❌ UI directory not found")
        return False
    
    try:
        subprocess.run(['npm', 'install'], cwd=ui_dir, check=True)
        print("✅ Node.js dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Node.js dependencies: {e}")
        return False

def start_services():
    """Start the services using Docker Compose."""
    print("🚀 Starting services with Docker Compose...")
    
    try:
        # Start services in background
        subprocess.run(['docker-compose', 'up', '-d'], check=True)
        print("✅ Services started successfully")
        
        # Wait a moment for services to initialize
        print("⏳ Waiting for services to initialize...")
        time.sleep(10)
        
        # Check service status
        print("📊 Checking service status...")
        subprocess.run(['docker-compose', 'ps'])
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start services: {e}")
        return False

def show_access_info():
    """Show access information for the services."""
    print("\n" + "="*60)
    print("🎉 System is ready! Here's how to access the services:")
    print("="*60)
    print()
    print("🌐 Frontend (React):     http://localhost:3000")
    print("🔌 Backend API:          http://localhost:8000")
    print("📚 API Documentation:    http://localhost:8000/docs")
    print("📊 MLflow Tracking:      http://localhost:5000")
    print("🗄️  PostgreSQL:          localhost:5432")
    print()
    print("🔑 Default Login Credentials:")
    print("   Admin:    admin.user / admin123")
    print("   Doctor:   doctor.smith / password123")
    print("   Nurse:    nurse.jones / password123")
    print()
    print("📖 For more information, see README.md")
    print("="*60)

def main():
    """Main startup function."""
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Some dependencies are missing. Please install them and try again.")
        return
    
    # Setup environment
    if not setup_environment():
        print("❌ Failed to setup environment.")
        return
    
    # Ask user what they want to do
    print("🚀 What would you like to do?")
    print("1. Quick start with Docker (recommended)")
    print("2. Install dependencies manually")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        print("\n🚀 Quick start with Docker...")
        if start_services():
            show_access_info()
        else:
            print("❌ Failed to start services. Check the logs above.")
    
    elif choice == "2":
        print("\n📦 Manual dependency installation...")
        if install_python_dependencies() and install_node_dependencies():
            print("✅ All dependencies installed successfully")
            print("📖 Please see README.md for next steps")
        else:
            print("❌ Failed to install some dependencies")
    
    elif choice == "3":
        print("👋 Goodbye!")
    
    else:
        print("❌ Invalid choice. Please run the script again.")

if __name__ == "__main__":
    main()
