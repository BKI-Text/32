#!/usr/bin/env python3
"""
Test runner for Beverly Knits AI Supply Chain Planner
"""

import sys
import subprocess
import os
from pathlib import Path

def run_tests():
    """Run all tests with pytest"""
    
    # Add project root to Python path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    # Change to project directory
    os.chdir(project_root)
    
    # Run pytest with coverage
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "--cov=src",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov"
    ]
    
    print("Running Beverly Knits AI Supply Chain Planner Tests...")
    print("=" * 60)
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("📊 Coverage report generated in htmlcov/index.html")
    else:
        print("\n" + "=" * 60)
        print("❌ Some tests failed!")
        print("📝 Check the output above for details")
    
    return result.returncode

if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)