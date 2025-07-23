"""Configuration Validator for Beverly Knits AI Supply Chain Planner"""

import os
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

@dataclass
class ValidationResult:
    """Result of a configuration validation"""
    level: ValidationLevel
    category: str
    key: str
    message: str
    current_value: Optional[str] = None
    suggested_value: Optional[str] = None

class ConfigurationValidator:
    """Validates configuration settings for different environments"""
    
    def __init__(self):
        self.results: List[ValidationResult] = []
        self.required_settings = self._get_required_settings()
        self.validation_rules = self._get_validation_rules()
    
    def validate_environment(self, environment: str) -> List[ValidationResult]:
        """Validate configuration for a specific environment"""
        self.results.clear()
        
        # Check required settings
        self._validate_required_settings(environment)
        
        # Check validation rules
        self._validate_rules(environment)
        
        # Environment-specific validations
        self._validate_environment_specific(environment)
        
        return self.results
    
    def _get_required_settings(self) -> Dict[str, List[str]]:
        """Get required settings by category"""
        return {
            "basic": [
                "ENVIRONMENT",
                "DEBUG",
                "SECRET_KEY"
            ],
            "database": [
                "DATABASE_TYPE",
                "DATABASE_NAME"
            ],
            "api": [
                "API_HOST",
                "API_PORT"
            ],
            "logging": [
                "LOG_LEVEL",
                "LOG_TO_CONSOLE"
            ],
            "security": [
                "SECRET_KEY",
                "CORS_ALLOWED_ORIGINS"
            ]
        }
    
    def _get_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Get validation rules for settings"""
        return {
            "DATABASE_PORT": {
                "type": int,
                "min": 1,
                "max": 65535,
                "default": 5432
            },
            "API_PORT": {
                "type": int,
                "min": 1,
                "max": 65535,
                "default": 8000
            },
            "LOG_LEVEL": {
                "type": str,
                "choices": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                "default": "INFO"
            },
            "DEBUG": {
                "type": bool,
                "default": False
            },
            "SECRET_KEY": {
                "type": str,
                "min_length": 32,
                "pattern": r"^[A-Za-z0-9!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>?]+$"
            },
            "ENVIRONMENT": {
                "type": str,
                "choices": ["development", "testing", "staging", "production"],
                "default": "development"
            }
        }
    
    def _validate_required_settings(self, environment: str):
        """Validate required settings are present"""
        for category, settings in self.required_settings.items():
            for setting in settings:
                value = os.getenv(setting)
                if not value:
                    self.results.append(ValidationResult(
                        level=ValidationLevel.ERROR,
                        category=category,
                        key=setting,
                        message=f"Required setting '{setting}' is not set",
                        suggested_value=self._get_default_value(setting)
                    ))
    
    def _validate_rules(self, environment: str):
        """Validate settings against rules"""
        for setting, rules in self.validation_rules.items():
            value = os.getenv(setting)
            if value:
                self._validate_setting_value(setting, value, rules)
    
    def _validate_setting_value(self, setting: str, value: str, rules: Dict[str, Any]):
        """Validate a specific setting value"""
        try:
            # Type validation
            if "type" in rules:
                expected_type = rules["type"]
                if expected_type == int:
                    typed_value = int(value)
                elif expected_type == bool:
                    typed_value = value.lower() in ("true", "1", "yes", "on")
                elif expected_type == float:
                    typed_value = float(value)
                else:
                    typed_value = value
                
                # Range validation
                if "min" in rules and typed_value < rules["min"]:
                    self.results.append(ValidationResult(
                        level=ValidationLevel.ERROR,
                        category="validation",
                        key=setting,
                        message=f"Value {typed_value} is below minimum {rules['min']}",
                        current_value=value,
                        suggested_value=str(rules["min"])
                    ))
                
                if "max" in rules and typed_value > rules["max"]:
                    self.results.append(ValidationResult(
                        level=ValidationLevel.ERROR,
                        category="validation",
                        key=setting,
                        message=f"Value {typed_value} is above maximum {rules['max']}",
                        current_value=value,
                        suggested_value=str(rules["max"])
                    ))
                
                # Choice validation
                if "choices" in rules and typed_value not in rules["choices"]:
                    self.results.append(ValidationResult(
                        level=ValidationLevel.ERROR,
                        category="validation",
                        key=setting,
                        message=f"Value '{typed_value}' is not in allowed choices: {rules['choices']}",
                        current_value=value,
                        suggested_value=rules["choices"][0]
                    ))
                
                # Length validation
                if "min_length" in rules and len(str(typed_value)) < rules["min_length"]:
                    self.results.append(ValidationResult(
                        level=ValidationLevel.ERROR,
                        category="validation",
                        key=setting,
                        message=f"Value length {len(str(typed_value))} is below minimum {rules['min_length']}",
                        current_value=value
                    ))
                
                # Pattern validation
                if "pattern" in rules:
                    import re
                    if not re.match(rules["pattern"], str(typed_value)):
                        self.results.append(ValidationResult(
                            level=ValidationLevel.ERROR,
                            category="validation",
                            key=setting,
                            message=f"Value does not match required pattern",
                            current_value=value
                        ))
        
        except ValueError as e:
            self.results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                category="validation",
                key=setting,
                message=f"Invalid value type: {str(e)}",
                current_value=value
            ))
    
    def _validate_environment_specific(self, environment: str):
        """Validate environment-specific settings"""
        if environment == "production":
            self._validate_production_settings()
        elif environment == "development":
            self._validate_development_settings()
        elif environment == "testing":
            self._validate_testing_settings()
    
    def _validate_production_settings(self):
        """Validate production-specific settings"""
        # Debug should be False in production
        debug = os.getenv("DEBUG", "false").lower()
        if debug in ("true", "1", "yes", "on"):
            self.results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                category="security",
                key="DEBUG",
                message="Debug mode should be disabled in production",
                current_value=debug,
                suggested_value="false"
            ))
        
        # Secret key should be strong
        secret_key = os.getenv("SECRET_KEY", "")
        if len(secret_key) < 32:
            self.results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                category="security",
                key="SECRET_KEY",
                message="Secret key must be at least 32 characters in production",
                current_value="***"
            ))
        
        # CORS should be restricted
        cors_origins = os.getenv("CORS_ALLOWED_ORIGINS", "*")
        if cors_origins == "*":
            self.results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                category="security",
                key="CORS_ALLOWED_ORIGINS",
                message="CORS should be restricted in production",
                current_value=cors_origins,
                suggested_value="https://yourdomain.com"
            ))
    
    def _validate_development_settings(self):
        """Validate development-specific settings"""
        # Check for development-friendly settings
        log_level = os.getenv("LOG_LEVEL", "INFO")
        if log_level not in ("DEBUG", "INFO"):
            self.results.append(ValidationResult(
                level=ValidationLevel.INFO,
                category="development",
                key="LOG_LEVEL",
                message="Consider using DEBUG or INFO log level in development",
                current_value=log_level,
                suggested_value="DEBUG"
            ))
    
    def _validate_testing_settings(self):
        """Validate testing-specific settings"""
        # Database should be test database
        db_name = os.getenv("DATABASE_NAME", "")
        if not db_name.endswith("_test"):
            self.results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                category="testing",
                key="DATABASE_NAME",
                message="Test database name should end with '_test'",
                current_value=db_name,
                suggested_value=f"{db_name}_test"
            ))
    
    def _get_default_value(self, setting: str) -> Optional[str]:
        """Get default value for a setting"""
        defaults = {
            "ENVIRONMENT": "development",
            "DEBUG": "false",
            "SECRET_KEY": "change-this-secret-key-in-production",
            "DATABASE_TYPE": "sqlite",
            "DATABASE_NAME": "beverlyknits.db",
            "API_HOST": "0.0.0.0",
            "API_PORT": "8000",
            "LOG_LEVEL": "INFO",
            "LOG_TO_CONSOLE": "true",
            "CORS_ALLOWED_ORIGINS": "*"
        }
        return defaults.get(setting)
    
    def print_results(self, show_info: bool = True):
        """Print validation results"""
        if not self.results:
            print("✅ No validation issues found")
            return
        
        for result in self.results:
            if not show_info and result.level == ValidationLevel.INFO:
                continue
            
            icon = "❌" if result.level == ValidationLevel.ERROR else "⚠️" if result.level == ValidationLevel.WARNING else "ℹ️"
            print(f"{icon} {result.category.upper()}: {result.key}")
            print(f"   {result.message}")
            if result.current_value:
                print(f"   Current: {result.current_value}")
            if result.suggested_value:
                print(f"   Suggested: {result.suggested_value}")
            print()
    
    def get_validation_summary(self) -> Dict[str, int]:
        """Get validation summary"""
        summary = {
            "errors": len([r for r in self.results if r.level == ValidationLevel.ERROR]),
            "warnings": len([r for r in self.results if r.level == ValidationLevel.WARNING]),
            "info": len([r for r in self.results if r.level == ValidationLevel.INFO])
        }
        return summary

def validate_current_environment() -> List[ValidationResult]:
    """Validate the current environment configuration"""
    validator = ConfigurationValidator()
    current_env = os.getenv("ENVIRONMENT", "development")
    return validator.validate_environment(current_env)

if __name__ == "__main__":
    # Command line interface
    import sys
    
    if len(sys.argv) > 1:
        environment = sys.argv[1]
    else:
        environment = os.getenv("ENVIRONMENT", "development")
    
    validator = ConfigurationValidator()
    results = validator.validate_environment(environment)
    
    print(f"Validating configuration for environment: {environment}")
    validator.print_results()
    
    summary = validator.get_validation_summary()
    if summary['errors'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)