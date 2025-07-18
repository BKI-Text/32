#!/usr/bin/env python3
"""
Run Performance Monitoring Dashboard
Beverly Knits AI Supply Chain Planner
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function to run the dashboard"""
    logger.info("🚀 Starting ML Performance Monitoring Dashboard")
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    dashboard_path = script_dir / "src" / "monitoring" / "streamlit_dashboard.py"
    
    # Check if dashboard file exists
    if not dashboard_path.exists():
        logger.error(f"Dashboard file not found: {dashboard_path}")
        sys.exit(1)
    
    # Check if streamlit is available
    try:
        import streamlit
        logger.info("✅ Streamlit is available")
    except ImportError:
        logger.error("❌ Streamlit is not installed. Please install it with: pip install streamlit")
        sys.exit(1)
    
    # Run the Streamlit dashboard
    try:
        logger.info(f"Starting dashboard at: {dashboard_path}")
        logger.info("Dashboard will be available at: http://localhost:8501")
        logger.info("Press Ctrl+C to stop the dashboard")
        
        # Change to the script directory
        os.chdir(script_dir)
        
        # Run streamlit
        cmd = ["streamlit", "run", str(dashboard_path), "--server.port=8501", "--server.headless=false"]
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")
    except Exception as e:
        logger.error(f"Error running dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()