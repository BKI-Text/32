"""
Environment Configuration Manager
Provides centralized configuration management with environment variable support.
"""

import os
import json
from typing import Dict, Any, Optional, Union
from pathlib import Path
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)

class EnvironmentConfigManager:
    """
    Centralized configuration manager that supports:
    - Environment variables with fallback to JSON config
    - Type conversion and validation
    - Configuration inheritance and overrides
    - Runtime configuration updates
    """
    
    def __init__(self, config_file: str = "config/app_config.json", env_prefix: str = "BEVERLY_"):
        self.config_file = Path(config_file)
        self.env_prefix = env_prefix
        self._config_cache = {}
        self._load_configuration()
    
    def _load_configuration(self):
        """Load configuration from JSON file and environment variables."""
        
        # Load base configuration from JSON
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                self._config_cache = json.load(f)
        else:
            logger.warning(f"Configuration file not found: {self.config_file}")
            self._config_cache = {}
        
        # Override with environment variables
        self._apply_environment_overrides()
        
        logger.info("Configuration loaded successfully")
    
    def _apply_environment_overrides(self):
        """Apply environment variable overrides to configuration."""
        
        env_mappings = {
            # Database configuration
            f"{self.env_prefix}DB_HOST": "database.host",
            f"{self.env_prefix}DB_PORT": "database.port",
            f"{self.env_prefix}DB_NAME": "database.database",
            f"{self.env_prefix}DB_USER": "database.username",
            f"{self.env_prefix}DB_PASSWORD": "database.password",
            
            # Application configuration
            f"{self.env_prefix}ENVIRONMENT": "environment",
            f"{self.env_prefix}DEBUG": "debug",
            f"{self.env_prefix}LOG_LEVEL": "logging.level",
            
            # Data paths
            f"{self.env_prefix}LIVE_DATA_PATH": "data_integration.live_data_path",
            f"{self.env_prefix}INPUT_DATA_PATH": "data_integration.input_data_path",
            f"{self.env_prefix}OUTPUT_DATA_PATH": "data_integration.output_data_path",
            f"{self.env_prefix}BACKUP_DATA_PATH": "data_integration.backup_data_path",
            
            # Planning configuration
            f"{self.env_prefix}PLANNING_HORIZON_DAYS": "planning.planning_horizon_days",
            f"{self.env_prefix}SAFETY_STOCK_PERCENTAGE": "planning.safety_stock_percentage",
            f"{self.env_prefix}COST_WEIGHT": "planning.cost_weight",
            f"{self.env_prefix}RELIABILITY_WEIGHT": "planning.reliability_weight",
            
            # AI/ML configuration
            f"{self.env_prefix}ENABLE_AI": "ai.enable_ai_integration",
            f"{self.env_prefix}MODEL_CACHE_PATH": "ai.model_cache_path",
            
            # Streamlit configuration
            f"{self.env_prefix}STREAMLIT_PORT": "streamlit.port",
            f"{self.env_prefix}STREAMLIT_HOST": "streamlit.host",
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                self._set_nested_config(config_path, self._convert_env_value(env_value))
                logger.debug(f"Applied environment override: {env_var} -> {config_path}")
    
    def _set_nested_config(self, path: str, value: Any):
        """Set a nested configuration value using dot notation."""
        
        keys = path.split('.')
        current = self._config_cache
        
        # Navigate to the parent of the final key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the final value
        current[keys[-1]] = value
    
    def _convert_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Convert environment variable string to appropriate Python type."""
        
        # Boolean conversion
        if value.lower() in ('true', '1', 'yes', 'on'):
            return True
        elif value.lower() in ('false', '0', 'no', 'off'):
            return False
        
        # Number conversion
        try:
            # Try integer first
            if '.' not in value:
                return int(value)
            else:
                return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'database.host')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        
        keys = key.split('.')
        current = self._config_cache
        
        try:
            for k in keys:
                current = current[k]
            return current
        except (KeyError, TypeError):
            return default
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.
        
        Args:
            section: Section name (e.g., 'database', 'planning')
            
        Returns:
            Configuration section as dictionary
        """
        
        return self.get(section, {})
    
    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation
            value: Value to set
        """
        
        self._set_nested_config(key, value)
    
    def update_from_dict(self, config_dict: Dict[str, Any], prefix: str = ""):
        """
        Update configuration from dictionary.
        
        Args:
            config_dict: Dictionary of configuration values
            prefix: Optional prefix for keys
        """
        
        for key, value in config_dict.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                self.update_from_dict(value, full_key)
            else:
                self.set(full_key, value)
    
    def save_to_file(self, file_path: Optional[str] = None):
        """
        Save current configuration to JSON file.
        
        Args:
            file_path: Optional file path, defaults to original config file
        """
        
        output_path = Path(file_path) if file_path else self.config_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self._config_cache, f, indent=2, default=str)
        
        logger.info(f"Configuration saved to {output_path}")
    
    def get_environment_info(self) -> Dict[str, Any]:
        """
        Get information about current environment configuration.
        
        Returns:
            Dictionary with environment information
        """
        
        env_vars_found = []
        for env_var in os.environ:
            if env_var.startswith(self.env_prefix):
                env_vars_found.append({
                    'name': env_var,
                    'value': os.environ[env_var]
                })
        
        return {
            'environment': self.get('environment', 'development'),
            'debug_mode': self.get('debug', False),
            'config_file': str(self.config_file),
            'env_prefix': self.env_prefix,
            'environment_variables_found': len(env_vars_found),
            'environment_variables': env_vars_found
        }
    
    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate current configuration and return validation report.
        
        Returns:
            Validation report with issues and warnings
        """
        
        issues = []
        warnings = []
        
        # Validate required configuration
        required_paths = [
            'data_integration.live_data_path',
            'planning.planning_horizon_days',
            'planning.safety_stock_percentage'
        ]
        
        for path in required_paths:
            if self.get(path) is None:
                issues.append(f"Required configuration missing: {path}")
        
        # Validate data types and ranges
        numeric_ranges = {
            'planning.safety_stock_percentage': (0.0, 1.0),
            'planning.cost_weight': (0.0, 1.0),
            'planning.reliability_weight': (0.0, 1.0),
            'planning.planning_horizon_days': (1, 365)
        }
        
        for path, (min_val, max_val) in numeric_ranges.items():
            value = self.get(path)
            if value is not None:
                try:
                    numeric_value = float(value)
                    if not (min_val <= numeric_value <= max_val):
                        warnings.append(f"Value out of range for {path}: {value} (should be {min_val}-{max_val})")
                except (ValueError, TypeError):
                    issues.append(f"Invalid numeric value for {path}: {value}")
        
        # Validate path existence
        path_configs = [
            'data_integration.live_data_path',
            'data_integration.input_data_path',
            'data_integration.output_data_path'
        ]
        
        for path_config in path_configs:
            path_value = self.get(path_config)
            if path_value and not Path(path_value).exists():
                warnings.append(f"Path does not exist: {path_config} = {path_value}")
        
        return {
            'validation_timestamp': logger.info.__name__,
            'is_valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'total_issues': len(issues),
            'total_warnings': len(warnings)
        }

# Global configuration instance
config_manager = EnvironmentConfigManager()

def get_config() -> EnvironmentConfigManager:
    """Get the global configuration manager instance."""
    return config_manager

def reload_config():
    """Reload configuration from files and environment."""
    global config_manager
    config_manager._load_configuration()
    logger.info("Configuration reloaded")

# Convenience functions for common configuration access
def get_planning_config() -> Dict[str, Any]:
    """Get planning configuration section."""
    return config_manager.get_section('planning')

def get_data_integration_config() -> Dict[str, Any]:
    """Get data integration configuration section."""
    return config_manager.get_section('data_integration')

def get_database_config() -> Dict[str, Any]:
    """Get database configuration section."""
    return config_manager.get_section('database')

def get_ai_config() -> Dict[str, Any]:
    """Get AI/ML configuration section."""
    return config_manager.get_section('ai')

def get_streamlit_config() -> Dict[str, Any]:
    """Get Streamlit configuration section."""
    return config_manager.get_section('streamlit')