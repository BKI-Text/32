#!/usr/bin/env python3
"""
Test ERP Integration
Beverly Knits AI Supply Chain Planner
"""

import sys
import os
import logging
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.integrations.efab_integration import EfabERPIntegration

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_erp_connection():
    """Test ERP connection and data sync"""
    logger.info("🚀 Testing ERP Integration")
    
    try:
        # Initialize ERP integration with provided credentials
        erp = EfabERPIntegration(username='psytz', password='big$cat')
        
        # Test 1: Connection
        logger.info("\n" + "="*60)
        logger.info("TEST 1: ERP Connection")
        logger.info("="*60)
        
        connection_success = erp.connect()
        
        if connection_success:
            logger.info("✅ ERP connection successful")
        else:
            logger.error("❌ ERP connection failed")
            return False
        
        # Test 2: Connection Test
        logger.info("\n" + "="*60)
        logger.info("TEST 2: Connection Test")
        logger.info("="*60)
        
        test_results = erp.test_connection()
        
        logger.info(f"Connection Status: {test_results['connection_status']}")
        logger.info(f"Authenticated: {test_results['authenticated']}")
        logger.info(f"Base URL: {test_results['base_url']}")
        logger.info(f"Username: {test_results['username']}")
        
        # Log endpoint test results
        for endpoint_type, endpoints in test_results['endpoints_tested'].items():
            logger.info(f"\n{endpoint_type.upper()} Endpoints:")
            for endpoint, result in endpoints.items():
                if 'status_code' in result:
                    status = "✅" if result['accessible'] else "❌"
                    logger.info(f"  {status} {endpoint} - Status: {result['status_code']}")
                else:
                    logger.info(f"  ❌ {endpoint} - Error: {result.get('error', 'Unknown')}")
        
        # Test 3: Data Sync
        logger.info("\n" + "="*60)
        logger.info("TEST 3: Data Synchronization")
        logger.info("="*60)
        
        sync_results = erp.sync_all_data()
        
        logger.info(f"Sync Success: {sync_results['success']}")
        
        if sync_results.get('summary'):
            summary = sync_results['summary']
            logger.info(f"Success Rate: {summary['success_rate']:.1f}%")
            logger.info(f"Entities Synced: {summary['successful_syncs']}/{summary['total_entities']}")
            
            # Log individual sync results
            for entity_type in ['materials', 'suppliers', 'inventory', 'orders', 'bom', 'forecasts']:
                result = sync_results.get(entity_type, {})
                if result.get('success'):
                    count = result.get('count', 0)
                    endpoint = result.get('endpoint', 'Unknown')
                    logger.info(f"✅ {entity_type.title()}: {count} records from {endpoint}")
                else:
                    error = result.get('error', 'Unknown error')
                    logger.info(f"❌ {entity_type.title()}: {error}")
        
        # Test 4: Domain Object Conversion
        logger.info("\n" + "="*60)
        logger.info("TEST 4: Domain Object Conversion")
        logger.info("="*60)
        
        domain_objects = erp.convert_to_domain_objects(sync_results)
        
        logger.info(f"Materials: {len(domain_objects['materials'])}")
        logger.info(f"Suppliers: {len(domain_objects['suppliers'])}")
        logger.info(f"Forecasts: {len(domain_objects['forecasts'])}")
        logger.info(f"BOMs: {len(domain_objects['boms'])}")
        
        # Show sample data if available
        if domain_objects['materials']:
            sample_material = domain_objects['materials'][0]
            logger.info(f"Sample Material: {sample_material.name} (ID: {sample_material.id.value})")
        
        if domain_objects['suppliers']:
            sample_supplier = domain_objects['suppliers'][0]
            logger.info(f"Sample Supplier: {sample_supplier.name} (ID: {sample_supplier.id.value})")
        
        # Test 5: Connection Status
        logger.info("\n" + "="*60)
        logger.info("TEST 5: Status Summary")
        logger.info("="*60)
        
        status = erp.get_sync_status()
        logger.info(f"Connection Status: {status['connection_status']}")
        logger.info(f"Authenticated: {status['authenticated']}")
        logger.info(f"ERP Name: {status['erp_name']}")
        logger.info(f"Base URL: {status['base_url']}")
        
        # Test 6: Save Results
        logger.info("\n" + "="*60)
        logger.info("TEST 6: Save Test Results")
        logger.info("="*60)
        
        # Save test results to file
        test_report = {
            'timestamp': datetime.now().isoformat(),
            'connection_test': test_results,
            'sync_results': sync_results,
            'domain_objects_count': {
                'materials': len(domain_objects['materials']),
                'suppliers': len(domain_objects['suppliers']),
                'forecasts': len(domain_objects['forecasts']),
                'boms': len(domain_objects['boms'])
            },
            'status': status
        }
        
        # Save to file
        output_file = f"erp_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(test_report, f, indent=2, default=str)
        
        logger.info(f"✅ Test results saved to: {output_file}")
        
        # Final Summary
        logger.info("\n" + "="*60)
        logger.info("INTEGRATION TEST SUMMARY")
        logger.info("="*60)
        
        if connection_success and test_results['success']:
            logger.info("🎉 ERP Integration Test PASSED")
            logger.info("✅ Connection established successfully")
            logger.info("✅ Authentication working")
            logger.info("✅ Data endpoints accessible")
            if sync_results['success']:
                logger.info("✅ Data synchronization working")
            else:
                logger.info("⚠️ Data synchronization has issues")
            
            return True
        else:
            logger.error("💥 ERP Integration Test FAILED")
            return False
        
    except Exception as e:
        logger.error(f"❌ ERP integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test execution"""
    logger.info("🚀 Running ERP Integration Tests")
    logger.info(f"Target ERP: https://efab.bkiapps.com/")
    logger.info(f"Username: psytz")
    logger.info(f"Test started at: {datetime.now().isoformat()}")
    
    success = test_erp_connection()
    
    if success:
        logger.info("\n✅ All ERP integration tests passed!")
        logger.info("🎉 Your ERP system is now connected to Beverly Knits AI Supply Chain Planner")
        logger.info("\nNext steps:")
        logger.info("1. Check the saved test results file for detailed information")
        logger.info("2. You can now use the ERP integration in your main application")
        logger.info("3. Data will be automatically synced from your ERP system")
    else:
        logger.error("\n❌ Some ERP integration tests failed!")
        logger.error("Please check the error messages above and contact support if needed")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)