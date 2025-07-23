#!/usr/bin/env python3
"""
ERP Setup Script
Beverly Knits AI Supply Chain Planner
"""

import sys
import os
import logging
from datetime import datetime
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.integrations.erp_manager import erp_manager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_erp():
    """Set up ERP connection"""
    logger.info("🚀 Setting up ERP connection")
    logger.info("="*60)
    
    # Set credentials
    logger.info("📝 Setting ERP credentials")
    erp_manager.set_credentials(username='psytz', password='big$cat')
    
    # Test connection
    logger.info("🔗 Testing ERP connection")
    connection_test = erp_manager.test_connection()
    
    if connection_test['success']:
        logger.info("✅ ERP connection test passed")
        
        # Perform initial data sync
        logger.info("🔄 Performing initial data sync")
        sync_results = erp_manager.sync_data()
        
        if sync_results['success']:
            logger.info("✅ Initial data sync completed")
            
            # Get domain objects
            logger.info("🏗️ Converting to domain objects")
            domain_objects = erp_manager.get_domain_objects()
            
            # Show summary
            logger.info("📊 ERP Integration Summary:")
            logger.info(f"  Materials: {len(domain_objects['materials'])}")
            logger.info(f"  Suppliers: {len(domain_objects['suppliers'])}")
            logger.info(f"  Forecasts: {len(domain_objects['forecasts'])}")
            logger.info(f"  BOMs: {len(domain_objects['boms'])}")
            
            # Save configuration
            status = erp_manager.get_status()
            config_file = f"erp_setup_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(config_file, 'w') as f:
                json.dump(status, f, indent=2, default=str)
            
            logger.info(f"✅ ERP setup completed successfully")
            logger.info(f"📄 Configuration saved to: {config_file}")
            
            return True
        else:
            logger.error("❌ Initial data sync failed")
            return False
    else:
        logger.error("❌ ERP connection test failed")
        return False

def main():
    """Main setup function"""
    logger.info("🎯 Beverly Knits AI Supply Chain Planner - ERP Setup")
    logger.info(f"⏰ Started at: {datetime.now().isoformat()}")
    
    try:
        success = setup_erp()
        
        if success:
            logger.info("\n🎉 ERP SETUP COMPLETE!")
            logger.info("✅ Your Efab ERP system is now connected")
            logger.info("✅ Data synchronization is working")
            logger.info("✅ Beverly Knits AI can now access your ERP data")
            
            logger.info("\n📋 What's Next:")
            logger.info("1. Run the main application: streamlit run main.py")
            logger.info("2. Your ERP data will be automatically loaded")
            logger.info("3. AI-powered recommendations will use your live data")
            
            return True
        else:
            logger.error("\n💥 ERP SETUP FAILED!")
            logger.error("Please check the error messages above")
            logger.error("Contact support if issues persist")
            return False
            
    except Exception as e:
        logger.error(f"❌ Setup failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)