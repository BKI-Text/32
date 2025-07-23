#!/usr/bin/env python3
"""
Beverly Knits AI System - Post-Fix Validation Test
Tests critical components after dependency and security fixes
"""

import sys
import traceback
from datetime import datetime

def test_pydantic_v2_compatibility():
    """Test Pydantic v2 compatibility"""
    try:
        from src.config.settings import settings
        from src.core.domain.entities import Material, MaterialType, Supplier, RiskLevel
        from src.core.domain.value_objects import MaterialId, SupplierId, LeadTime
        
        # Test settings
        assert settings.environment is not None
        print("✅ Settings module loaded successfully")
        
        # Test domain entities
        material = Material(
            id=MaterialId(value='TEST001'),
            name='Test Material',
            type=MaterialType.YARN,
            is_critical=True
        )
        
        supplier = Supplier(
            id=SupplierId(value='SUP001'),
            name='Test Supplier',
            lead_time=LeadTime(days=14),
            reliability_score=0.95,
            risk_level=RiskLevel.LOW
        )
        
        print("✅ Domain entities working with Pydantic v2")
        return True
    except Exception as e:
        print(f"❌ Pydantic v2 compatibility test failed: {e}")
        traceback.print_exc()
        return False

def test_erp_integration():
    """Test ERP integration still works"""
    try:
        from src.integrations.efab_integration import EfabERPIntegration
        
        erp = EfabERPIntegration(username='test', password='test')
        assert erp.credentials.base_url == 'https://efab.bkiapps.com'
        
        print("✅ ERP integration module working")
        return True
    except Exception as e:
        print(f"❌ ERP integration test failed: {e}")
        traceback.print_exc()
        return False

def test_ml_imports():
    """Test ML components import correctly"""
    try:
        from src.engine.planning_engine import PlanningEngine
        from src.engine.ml_model_manager import MLModelManager
        from src.engine.forecasting.arima_forecaster import ARIMAForecaster
        
        print("✅ ML components import successfully")
        return True
    except Exception as e:
        print(f"❌ ML imports test failed: {e}")
        traceback.print_exc()
        return False

def test_api_imports():
    """Test API components import correctly"""
    try:
        from api.main import app
        from api.routers.auth import router as auth_router
        
        print("✅ API components import successfully")
        return True
    except Exception as e:
        print(f"❌ API imports test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all validation tests"""
    print("🧪 Beverly Knits AI System - Post-Fix Validation")
    print("=" * 60)
    print(f"Test run time: {datetime.now()}")
    print()
    
    tests = [
        ("Pydantic v2 Compatibility", test_pydantic_v2_compatibility),
        ("ERP Integration", test_erp_integration),
        ("ML Components", test_ml_imports),
        ("API Components", test_api_imports),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Running: {test_name}")
        result = test_func()
        results.append((test_name, result))
        print()
    
    # Summary
    print("=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print()
    print(f"Tests passed: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - System is ready for next phase!")
        return True
    else:
        print("⚠️ Some tests failed - please review errors above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)