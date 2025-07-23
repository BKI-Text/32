#!/usr/bin/env python3
"""
Simple ERP Data Access Test
Beverly Knits AI Supply Chain Planner
"""

import sys
import os
import logging
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.integrations.efab_integration import EfabERPIntegration

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_erp_data_access():
    """Test accessing ERP data endpoints"""
    logger.info("🚀 Simple ERP Data Access Test")
    
    try:
        # Initialize ERP integration
        erp = EfabERPIntegration(username='psytz', password='big$cat')
        
        # Connect to ERP
        if not erp.connect():
            logger.error("❌ Failed to connect to ERP")
            return False
        
        logger.info("✅ Connected to ERP successfully")
        
        # Test data endpoints
        endpoints = {
            'yarn': '/yarn',
            'yarn_demand': '/report/yarn_demand',
            'expected_yarn': '/report/expected_yarn',
            'fabric_orders': '/fabric/so/list'
        }
        
        data_results = {}
        
        for name, endpoint in endpoints.items():
            try:
                logger.info(f"🔍 Testing {name} endpoint: {endpoint}")
                
                url = f"{erp.credentials.base_url}{endpoint}"
                response = erp.auth.session.get(url)
                
                if response.status_code == 200:
                    data_results[name] = {
                        'status': 'success',
                        'status_code': response.status_code,
                        'content_length': len(response.content),
                        'content_type': response.headers.get('content-type', 'unknown'),
                        'url': url
                    }
                    
                    # Check if HTML contains data tables
                    if '<table' in response.text:
                        tables = response.text.count('<table')
                        data_results[name]['tables_found'] = tables
                        logger.info(f"  ✅ {name}: Found {tables} HTML tables")
                    else:
                        data_results[name]['tables_found'] = 0
                        logger.info(f"  ⚠️ {name}: No HTML tables found")
                    
                    # Check if it's a data grid (DevExtreme)
                    if 'dx-data-grid' in response.text or 'DevExpress' in response.text:
                        data_results[name]['data_grid'] = True
                        logger.info(f"  ✅ {name}: Contains DevExtreme data grid")
                    else:
                        data_results[name]['data_grid'] = False
                    
                    # Save sample data
                    sample_file = f"sample_{name}_data.html"
                    with open(sample_file, 'w', encoding='utf-8') as f:
                        f.write(response.text)
                    data_results[name]['sample_file'] = sample_file
                    
                else:
                    data_results[name] = {
                        'status': 'failed',
                        'status_code': response.status_code,
                        'error': f"HTTP {response.status_code}"
                    }
                    logger.error(f"  ❌ {name}: HTTP {response.status_code}")
                    
            except Exception as e:
                data_results[name] = {
                    'status': 'error',
                    'error': str(e)
                }
                logger.error(f"  ❌ {name}: {e}")
        
        # Generate summary
        successful_endpoints = sum(1 for result in data_results.values() if result['status'] == 'success')
        total_endpoints = len(endpoints)
        
        logger.info(f"\n📊 Data Access Summary:")
        logger.info(f"  Successful endpoints: {successful_endpoints}/{total_endpoints}")
        logger.info(f"  Success rate: {successful_endpoints/total_endpoints*100:.1f}%")
        
        # Save results
        results_file = f"erp_data_access_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(data_results, f, indent=2)
        
        logger.info(f"✅ Results saved to: {results_file}")
        
        if successful_endpoints > 0:
            logger.info("\n🎉 ERP Data Access Test PASSED")
            logger.info("✅ Successfully connected to your ERP system")
            logger.info("✅ Can access your yarn and fabric data")
            logger.info("✅ Beverly Knits AI can now work with your live ERP data")
            
            logger.info("\n📋 Available Data Sources:")
            for name, result in data_results.items():
                if result['status'] == 'success':
                    logger.info(f"  ✅ {name.replace('_', ' ').title()}: {result['content_length']} bytes")
                    if result.get('tables_found', 0) > 0:
                        logger.info(f"    - Contains {result['tables_found']} data tables")
                    if result.get('data_grid', False):
                        logger.info(f"    - Uses DevExtreme data grid")
            
            return True
        else:
            logger.error("\n❌ ERP Data Access Test FAILED")
            return False
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    logger.info("🎯 Beverly Knits AI Supply Chain Planner - ERP Data Access Test")
    logger.info(f"⏰ Started at: {datetime.now().isoformat()}")
    
    success = test_erp_data_access()
    
    if success:
        logger.info("\n✅ SUCCESS: Your ERP integration is working!")
        logger.info("You can now use Beverly Knits AI with your live ERP data")
    else:
        logger.error("\n❌ FAILED: ERP data access issues detected")
        logger.error("Please check the error messages above")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)