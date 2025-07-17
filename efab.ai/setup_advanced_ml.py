#!/usr/bin/env python3
"""
Setup script for advanced ML dependencies
Creates a virtual environment and installs required packages
"""

import subprocess
import sys
import os
from pathlib import Path

def setup_virtual_environment():
    """Set up virtual environment and install advanced ML packages"""
    
    venv_path = Path("venv_ml")
    
    # Create virtual environment
    if not venv_path.exists():
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", str(venv_path)])
    
    # Activate virtual environment and install packages
    if os.name == 'nt':  # Windows
        pip_path = venv_path / "Scripts" / "pip"
        python_path = venv_path / "Scripts" / "python"
    else:  # Unix/Linux
        pip_path = venv_path / "bin" / "pip"
        python_path = venv_path / "bin" / "python"
    
    # Install packages
    packages = [
        "statsmodels",
        "prophet",
        "tensorflow",
        "scikit-learn",
        "pandas",
        "numpy",
        "plotly",
        "streamlit"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.run([str(pip_path), "install", package], check=True)
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
    
    # Create activation script
    activation_script = """#!/bin/bash
# Beverly Knits ML Environment Activation Script
source venv_ml/bin/activate
echo "🧶 Beverly Knits ML Environment Activated"
echo "Available packages: statsmodels, prophet, tensorflow, scikit-learn"
"""
    
    with open("activate_ml_env.sh", "w") as f:
        f.write(activation_script)
    
    os.chmod("activate_ml_env.sh", 0o755)
    
    print("\n🎉 Virtual environment setup complete!")
    print("To activate: source activate_ml_env.sh")
    print("To run with ML packages: venv_ml/bin/python your_script.py")

if __name__ == "__main__":
    setup_virtual_environment()