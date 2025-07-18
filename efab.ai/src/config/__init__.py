"""Configuration module for Beverly Knits AI Supply Chain Planner"""

from .config_manager import ConfigManager, config_manager
from .settings import settings, Settings, PLANNING_CONFIG, PlanningConfig
from .environment_loader import EnvironmentLoader, load_environment_config
from .validator import ConfigurationValidator, validate_current_environment

# Legacy compatibility imports
from .environment_config import (
    EnvironmentConfigManager, 
    get_config as get_env_config,
    get_planning_config,
    get_data_integration_config,
    get_database_config,
    get_ai_config,
    get_streamlit_config
)

def get_config():
    """Legacy compatibility function"""
    return config_manager.config

__all__ = [
    # New configuration system
    "ConfigManager",
    "config_manager", 
    "settings",
    "Settings",
    "PLANNING_CONFIG",
    "PlanningConfig",
    "EnvironmentLoader",
    "load_environment_config",
    "ConfigurationValidator",
    "validate_current_environment",
    
    # Legacy compatibility
    "get_config",
    "EnvironmentConfigManager",
    "get_env_config",
    "get_planning_config",
    "get_data_integration_config", 
    "get_database_config",
    "get_ai_config",
    "get_streamlit_config"
]