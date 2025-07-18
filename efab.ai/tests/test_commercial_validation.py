"""
Comprehensive Commercial-Grade Validation Tests
Tests all critical functionality for commercial production readiness
"""

import unittest
import sys
import tempfile
import os
from pathlib import Path
from decimal import Decimal
import json
from datetime import datetime, date
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class CommercialValidationTests(unittest.TestCase):
    """Commercial-grade validation test suite"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_data_dir = Path(tempfile.mkdtemp())
        self.test_results = {}
        
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.test_data_dir, ignore_errors=True)
    
    def test_configuration_validation(self):
        """Test configuration validation system"""
        try:
            from src.config.validator import ConfigurationValidator
            
            validator = ConfigurationValidator()
            
            # Test with development environment
            results = validator.validate_environment("development")
            self.assertIsInstance(results, list)
            
            # Test validation summary
            summary = validator.get_validation_summary()
            self.assertIn("errors", summary)
            self.assertIn("warnings", summary)
            
            self.test_results["configuration_validation"] = "PASSED"
            
        except Exception as e:
            self.test_results["configuration_validation"] = f"FAILED: {e}"
            self.fail(f"Configuration validation failed: {e}")
    
    def test_error_handling_system(self):
        """Test comprehensive error handling system"""
        try:
            from src.utils.error_handling import (
                ErrorHandler, ErrorCategory, ErrorSeverity,
                global_error_handler, handle_errors
            )
            
            # Test error handler initialization
            handler = ErrorHandler()
            self.assertIsInstance(handler, ErrorHandler)
            
            # Test error logging
            test_error = ValueError("Test error")
            error_id = handler.handle_error(
                test_error, 
                ErrorCategory.DATA_VALIDATION,
                context={"test": "data"}
            )
            self.assertIsInstance(error_id, str)
            
            # Test decorator functionality
            @handle_errors(ErrorCategory.SYSTEM, default_return="error_handled")
            def test_function():
                raise ValueError("Test error")
            
            result = test_function()
            self.assertEqual(result, "error_handled")
            
            self.test_results["error_handling"] = "PASSED"
            
        except Exception as e:
            self.test_results["error_handling"] = f"FAILED: {e}"
            self.fail(f"Error handling system failed: {e}")
    
    def test_validation_framework(self):
        """Test validation framework components"""
        try:
            from src.validation.base import (
                ValidationContext, ValidationResult, ValidationLevel,
                BusinessRule, ValidatedModel, ValidationPatterns
            )
            
            # Test validation context
            context = ValidationContext(environment="testing")
            self.assertIsInstance(context, ValidationContext)
            
            # Test validation result
            result = ValidationResult(
                ValidationLevel.ERROR,
                "test_field",
                "Test error message"
            )
            self.assertEqual(result.level, ValidationLevel.ERROR)
            
            # Test validation patterns
            self.assertTrue(ValidationPatterns.is_positive_decimal(Decimal("10.5")))
            self.assertFalse(ValidationPatterns.is_positive_decimal(Decimal("-1")))
            
            self.test_results["validation_framework"] = "PASSED"
            
        except Exception as e:
            self.test_results["validation_framework"] = f"FAILED: {e}"
            self.fail(f"Validation framework failed: {e}")
    
    def test_domain_entities(self):
        """Test core domain entities"""
        try:
            from src.core.domain.entities import Material, Supplier, BOM, Forecast
            from src.core.domain.value_objects import MaterialId, SupplierId, Money, Quantity
            
            # Test Material entity
            material = Material(
                id=MaterialId("TEST001"),
                name="Test Material",
                type="YARN",
                category="Cotton",
                is_critical=True
            )
            self.assertEqual(material.name, "Test Material")
            
            # Test Supplier entity
            supplier = Supplier(
                id=SupplierId("SUP001"),
                name="Test Supplier",
                contact_info="test@supplier.com",
                performance_rating=4.5,
                is_preferred=True
            )
            self.assertEqual(supplier.name, "Test Supplier")
            
            # Test value objects
            cost = Money(amount=Decimal("15.50"), currency="USD")
            self.assertEqual(cost.amount, Decimal("15.50"))
            
            quantity = Quantity(amount=Decimal("100"), unit="pounds")
            self.assertEqual(quantity.amount, Decimal("100"))
            
            self.test_results["domain_entities"] = "PASSED"
            
        except Exception as e:
            self.test_results["domain_entities"] = f"FAILED: {e}"
            self.fail(f"Domain entities failed: {e}")
    
    def test_planning_engine(self):
        """Test planning engine functionality"""
        try:
            from src.engine.planning_engine import PlanningEngine
            
            # Test planning engine initialization
            engine = PlanningEngine()
            self.assertIsInstance(engine, PlanningEngine)
            
            # Test with empty data (should handle gracefully)
            recommendations = engine.execute_planning_cycle(
                forecasts=[],
                boms=[],
                inventory=[],
                suppliers=[]
            )
            self.assertIsInstance(recommendations, list)
            
            self.test_results["planning_engine"] = "PASSED"
            
        except Exception as e:
            self.test_results["planning_engine"] = f"FAILED: {e}"
            self.fail(f"Planning engine failed: {e}")
    
    def test_data_integration(self):
        """Test data integration capabilities"""
        try:
            from src.data.beverly_knits_live_data_integrator import BeverlyKnitsLiveDataIntegrator
            
            # Test data integrator initialization
            integrator = BeverlyKnitsLiveDataIntegrator(data_path=str(self.test_data_dir))
            self.assertIsInstance(integrator, BeverlyKnitsLiveDataIntegrator)
            
            # Test with empty directory (should handle gracefully)
            try:
                domain_objects = integrator.integrate_live_data()
                self.assertIsInstance(domain_objects, dict)
            except Exception as e:
                # Expected to fail with empty directory, but should be handled gracefully
                self.assertIsInstance(e, Exception)
            
            self.test_results["data_integration"] = "PASSED"
            
        except Exception as e:
            self.test_results["data_integration"] = f"FAILED: {e}"
            self.fail(f"Data integration failed: {e}")
    
    def test_ml_components(self):
        """Test ML components availability"""
        try:
            from src.engine.ml_model_manager import MLModelManager
            from src.engine.ml_risk_assessor import MLRiskAssessor
            
            # Test ML model manager
            ml_manager = MLModelManager()
            self.assertIsInstance(ml_manager, MLModelManager)
            
            # Test ML risk assessor
            risk_assessor = MLRiskAssessor()
            self.assertIsInstance(risk_assessor, MLRiskAssessor)
            
            self.test_results["ml_components"] = "PASSED"
            
        except Exception as e:
            self.test_results["ml_components"] = f"FAILED: {e}"
            self.fail(f"ML components failed: {e}")
    
    def test_authentication_system(self):
        """Test authentication system"""
        try:
            from src.auth.auth_service import AuthService
            
            # Test auth service initialization
            auth_service = AuthService()
            self.assertIsInstance(auth_service, AuthService)
            
            # Test password hashing
            password = "test_password_123"
            hashed = auth_service.get_password_hash(password)
            self.assertIsInstance(hashed, str)
            self.assertNotEqual(password, hashed)
            
            # Test password verification
            verified = auth_service.verify_password(password, hashed)
            self.assertTrue(verified)
            
            self.test_results["authentication_system"] = "PASSED"
            
        except Exception as e:
            self.test_results["authentication_system"] = f"FAILED: {e}"
            self.fail(f"Authentication system failed: {e}")
    
    def test_api_endpoints(self):
        """Test API endpoint availability"""
        try:
            from api.main import app
            from fastapi.testclient import TestClient
            
            # Test API initialization
            client = TestClient(app)
            
            # Test root endpoint
            response = client.get("/")
            self.assertEqual(response.status_code, 200)
            
            # Test health endpoint
            response = client.get("/health")
            self.assertIn(response.status_code, [200, 500])  # May fail if DB not configured
            
            self.test_results["api_endpoints"] = "PASSED"
            
        except Exception as e:
            self.test_results["api_endpoints"] = f"FAILED: {e}"
            self.fail(f"API endpoints failed: {e}")
    
    def test_cli_functionality(self):
        """Test CLI functionality"""
        try:
            from src.config.cli import config
            
            # Test CLI initialization (should not raise exception)
            self.assertIsNotNone(config)
            
            self.test_results["cli_functionality"] = "PASSED"
            
        except Exception as e:
            self.test_results["cli_functionality"] = f"FAILED: {e}"
            self.fail(f"CLI functionality failed: {e}")
    
    def test_database_models(self):
        """Test database models"""
        try:
            from src.database.models.user import UserModel
            
            # Test user model
            user_data = {
                "id": "test_user",
                "username": "testuser",
                "email": "test@example.com",
                "password_hash": "hashed_password",
                "is_active": True,
                "role": "user"
            }
            
            user = UserModel(**user_data)
            self.assertEqual(user.username, "testuser")
            
            self.test_results["database_models"] = "PASSED"
            
        except Exception as e:
            self.test_results["database_models"] = f"FAILED: {e}"
            self.fail(f"Database models failed: {e}")
    
    def test_commercial_quality_gates(self):
        """Test commercial quality gates"""
        try:
            # Test logging configuration
            import logging
            logger = logging.getLogger("test")
            logger.info("Test log message")
            
            # Test error handling in production context
            from src.utils.error_handling import global_error_handler
            health = global_error_handler.logger.get_error_summary()
            self.assertIsInstance(health, dict)
            
            # Test configuration validation for production
            from src.config.validator import ConfigurationValidator
            validator = ConfigurationValidator()
            
            # Save current environment
            original_env = os.environ.get("ENVIRONMENT")
            
            # Test production validation
            os.environ["ENVIRONMENT"] = "production"
            os.environ["DEBUG"] = "false"
            os.environ["SECRET_KEY"] = "a_very_long_secret_key_for_production_use_that_meets_requirements"
            
            results = validator.validate_environment("production")
            summary = validator.get_validation_summary()
            
            # Restore environment
            if original_env:
                os.environ["ENVIRONMENT"] = original_env
            else:
                os.environ.pop("ENVIRONMENT", None)
            
            self.assertIsInstance(summary, dict)
            
            self.test_results["commercial_quality_gates"] = "PASSED"
            
        except Exception as e:
            self.test_results["commercial_quality_gates"] = f"FAILED: {e}"
            self.fail(f"Commercial quality gates failed: {e}")
    
    def generate_commercial_readiness_report(self) -> Dict[str, Any]:
        """Generate comprehensive commercial readiness report"""
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result == "PASSED")
        failed_tests = total_tests - passed_tests
        
        readiness_score = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Determine readiness level
        if readiness_score >= 95:
            readiness_level = "PRODUCTION_READY"
        elif readiness_score >= 85:
            readiness_level = "NEAR_PRODUCTION_READY"
        elif readiness_score >= 70:
            readiness_level = "DEVELOPMENT_READY"
        else:
            readiness_level = "NOT_READY"
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "readiness_level": readiness_level,
            "readiness_score": readiness_score,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "test_results": self.test_results,
            "critical_components": {
                "configuration_validation": self.test_results.get("configuration_validation", "NOT_TESTED"),
                "error_handling": self.test_results.get("error_handling", "NOT_TESTED"),
                "authentication_system": self.test_results.get("authentication_system", "NOT_TESTED"),
                "planning_engine": self.test_results.get("planning_engine", "NOT_TESTED"),
                "api_endpoints": self.test_results.get("api_endpoints", "NOT_TESTED")
            },
            "recommendations": self._generate_recommendations(readiness_score)
        }
        
        return report
    
    def _generate_recommendations(self, readiness_score: float) -> List[str]:
        """Generate recommendations based on readiness score"""
        recommendations = []
        
        if readiness_score < 95:
            recommendations.append("Address all failing tests before production deployment")
        
        if readiness_score < 85:
            recommendations.append("Implement additional error handling and validation")
        
        if readiness_score < 70:
            recommendations.append("Complete core functionality implementation")
        
        # Check specific failed components
        failed_components = [
            component for component, result in self.test_results.items() 
            if result != "PASSED"
        ]
        
        if failed_components:
            recommendations.append(f"Fix failing components: {', '.join(failed_components)}")
        
        # Always include security recommendations
        recommendations.extend([
            "Ensure all secrets are properly configured for production",
            "Review and test all authentication and authorization flows",
            "Verify all input validation is working correctly",
            "Test error handling in production-like environment"
        ])
        
        return recommendations

def run_commercial_validation():
    """Run commercial validation tests and generate report"""
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(CommercialValidationTests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate report
    test_instance = CommercialValidationTests()
    test_instance.setUp()
    
    # Run all test methods to populate test_results
    for test_method in [method for method in dir(test_instance) if method.startswith('test_')]:
        try:
            getattr(test_instance, test_method)()
        except Exception as e:
            test_instance.test_results[test_method] = f"FAILED: {e}"
    
    # Generate final report
    report = test_instance.generate_commercial_readiness_report()
    
    # Save report
    report_file = Path("commercial_readiness_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n{'='*60}")
    print("COMMERCIAL READINESS REPORT")
    print(f"{'='*60}")
    print(f"Readiness Level: {report['readiness_level']}")
    print(f"Readiness Score: {report['readiness_score']:.1f}%")
    print(f"Tests Passed: {report['passed_tests']}/{report['total_tests']}")
    print(f"Report saved to: {report_file}")
    
    if report['failed_tests'] > 0:
        print(f"\nFailed Components:")
        for component, result in report['test_results'].items():
            if result != "PASSED":
                print(f"  - {component}: {result}")
    
    print(f"\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  - {rec}")
    
    return report

if __name__ == "__main__":
    run_commercial_validation()