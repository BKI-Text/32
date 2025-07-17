#!/usr/bin/env python3
"""
Beverly Knits AI Supply Chain Planner - Project Initialization Script

This script initializes the Beverly Knits AI Supply Chain Planner project
by creating necessary directories, generating sample data, and setting up
the development environment.
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary project directories"""
    directories = [
        "data/input",
        "data/output", 
        "data/backup",
        "data/demo",
        "data/sample",
        "logs",
        "models/cache",
        "models/ml_models",
        "temp/ml_processing",
        "config",
        "docs"
    ]
    
    logger.info("Creating project directories...")
    
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    # Create empty __init__.py files where needed
    init_files = [
        "src/core/interfaces/__init__.py",
        "src/core/use_cases/__init__.py",
        "models/__init__.py"
    ]
    
    for init_file in init_files:
        path = Path(init_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text("")
            logger.info(f"Created __init__.py: {init_file}")

def check_python_version():
    """Check if Python version is compatible"""
    import sys
    
    version = sys.version_info
    logger.info(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 12):
        logger.error("Python 3.12 or higher is required")
        return False
    
    return True

def install_dependencies():
    """Install project dependencies"""
    logger.info("Installing project dependencies...")
    
    try:
        # Install main dependencies
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        
        logger.info("Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False

def generate_sample_data():
    """Generate sample data for testing"""
    logger.info("Generating sample data...")
    
    try:
        # Add src to path
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        
        from src.utils.sample_data_generator import generate_sample_data
        
        # Generate sample data
        dataset = generate_sample_data(save_csv=True, output_dir="data/sample/")
        
        logger.info("Sample data generated successfully")
        logger.info(f"Sample data location: data/sample/")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to generate sample data: {e}")
        return False

def create_config_files():
    """Create configuration files if they don't exist"""
    logger.info("Setting up configuration files...")
    
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        template_file = Path("config/.env.template")
        if template_file.exists():
            env_file.write_text(template_file.read_text())
            logger.info("Created .env file from template")
        else:
            logger.warning("Template .env file not found")
    
    # Ensure config directory exists
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    logger.info("Configuration files ready")

def run_initial_tests():
    """Run initial tests to verify setup"""
    logger.info("Running initial tests...")
    
    try:
        # Run a subset of tests to verify setup
        subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/test_domain_entities.py", 
            "-v", "--tb=short"
        ], check=True)
        
        logger.info("Initial tests passed")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Initial tests failed: {e}")
        return False

def print_setup_summary():
    """Print setup summary and next steps"""
    print("\n" + "="*60)
    print("🧶 Beverly Knits AI Supply Chain Planner")
    print("="*60)
    print("✅ Project initialization completed successfully!")
    print("\n📁 Project Structure:")
    print("   - src/          → Core application code")
    print("   - tests/        → Comprehensive test suite")
    print("   - data/         → Data files and samples")
    print("   - config/       → Configuration files")
    print("   - models/       → ML models and cache")
    print("   - logs/         → Application logs")
    print("\n🚀 Next Steps:")
    print("   1. Run the demo:           python demo.py")
    print("   2. Launch web interface:   streamlit run main.py")
    print("   3. Run full tests:         python tests/run_tests.py")
    print("   4. Generate more data:     python -m src.utils.sample_data_generator")
    print("\n📊 Sample Data:")
    print("   - Location: data/sample/")
    print("   - Ready for testing and demonstration")
    print("\n🔧 Configuration:")
    print("   - Main config: config/app_config.json")
    print("   - Environment: .env (created from template)")
    print("   - Customize settings as needed")
    print("\n📚 Documentation:")
    print("   - README.md     → Complete usage guide")
    print("   - demo.py       → Interactive demonstration")
    print("   - tests/        → Code examples and validation")
    print(f"\n🎯 Ready to optimize your supply chain with AI!")
    print("="*60)

def main():
    """Main initialization function"""
    print("🧶 Initializing Beverly Knits AI Supply Chain Planner...")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create project directories
    create_directories()
    
    # Create configuration files
    create_config_files()
    
    # Install dependencies
    if not install_dependencies():
        logger.error("Failed to install dependencies. Please install manually.")
        sys.exit(1)
    
    # Generate sample data
    if not generate_sample_data():
        logger.warning("Failed to generate sample data. You can generate it later.")
    
    # Run initial tests
    if not run_initial_tests():
        logger.warning("Initial tests failed. Please check the installation.")
    
    # Print setup summary
    print_setup_summary()
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)