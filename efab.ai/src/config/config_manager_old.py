import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict, field
from decimal import Decimal
import logging

from .environment_config import EnvironmentConfigManager

logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    host: str = "localhost"
    port: int = 5432
    database: str = "beverly_knits"
    username: str = "postgres"
    password: str = ""
    pool_size: int = 5
    max_overflow: int = 10

@dataclass
class LoggingConfig:
    """Logging configuration settings"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_bytes: int = 10485760  # 10MB
    backup_count: int = 5

@dataclass
class PlanningConfig:
    """Planning engine configuration"""
    source_weights: Dict[str, float] = field(default_factory=lambda: {
        "sales_order": 1.0,
        "prod_plan": 0.9,
        "projection": 0.7,
        "sales_history": 0.8
    })
    
    safety_stock_percentage: float = 0.15
    planning_horizon_days: int = 90
    forecast_lookback_days: int = 30
    
    cost_weight: float = 0.6
    reliability_weight: float = 0.4
    max_suppliers_per_material: int = 3
    
    enable_eoq_optimization: bool = True
    enable_multi_supplier: bool = True
    enable_risk_assessment: bool = True
    
    default_ordering_cost: float = 100.0
    default_holding_cost_rate: float = 0.25
    
    risk_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "high": 0.7,
        "medium": 0.85,
        "low": 1.0
    })

@dataclass
class DataIntegrationConfig:
    """Data integration configuration"""
    input_data_path: str = "data/input/"
    output_data_path: str = "data/output/"
    backup_data_path: str = "data/backup/"
    
    auto_fix_negative_inventory: bool = True
    auto_fix_bom_percentages: bool = True
    auto_clean_cost_data: bool = True
    auto_remove_invalid_suppliers: bool = True
    
    validation_rules: Dict[str, Any] = field(default_factory=lambda: {
        "min_reliability_score": 0.0,
        "max_reliability_score": 1.0,
        "min_cost_per_unit": 0.01,
        "max_lead_time_days": 365,
        "required_bom_sum_tolerance": 0.01
    })

@dataclass
class StreamlitConfig:
    """Streamlit application configuration"""
    port: int = 8501
    host: str = "localhost"
    title: str = "Beverly Knits AI Supply Chain Planner"
    page_icon: str = "🧶"
    layout: str = "wide"
    
    theme: Dict[str, str] = field(default_factory=lambda: {
        "primary_color": "#2E86AB",
        "background_color": "#FFFFFF",
        "secondary_background_color": "#F0F2F6",
        "text_color": "#262730"
    })

@dataclass
class AIConfig:
    """AI/ML configuration"""
    enable_ai_integration: bool = False
    zen_mcp_server_config: str = "config/zen_ml_config.json"
    model_cache_path: str = "models/cache/"
    
    forecasting_models: Dict[str, bool] = field(default_factory=lambda: {
        "arima": True,
        "prophet": True,
        "xgboost": True,
        "lstm": False  # Requires additional setup
    })
    
    ml_features: Dict[str, bool] = field(default_factory=lambda: {
        "demand_forecasting": True,
        "supplier_risk_scoring": True,
        "price_prediction": False,
        "quality_prediction": False
    })

@dataclass
class BeverlyKnitsConfig:
    """Main application configuration"""
    app_name: str = "Beverly Knits AI Supply Chain Planner"
    version: str = "1.0.0"
    environment: str = "development"
    debug: bool = True
    
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    data_integration: DataIntegrationConfig = field(default_factory=DataIntegrationConfig)
    streamlit: StreamlitConfig = field(default_factory=StreamlitConfig)
    ai: AIConfig = field(default_factory=AIConfig)

class ConfigManager:
    """Configuration management system with environment variable support"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "config/app_config.json"
        self.config = BeverlyKnitsConfig()
        self.env_config = EnvironmentConfigManager(self.config_file)
        self._load_config()
    
    def _load_config(self):
        """Load configuration using environment configuration manager"""
        # Use environment config manager for unified loading
        try:
            # Update dataclass config from environment config manager
            self._sync_from_env_config()
            logger.info("Configuration loaded with environment variable support")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
    
    def _sync_from_env_config(self):
        """Sync dataclass config from environment config manager"""
        # Database config
        self.config.database.host = self.env_config.get('database.host', self.config.database.host)
        self.config.database.port = self.env_config.get('database.port', self.config.database.port)
        self.config.database.database = self.env_config.get('database.database', self.config.database.database)
        self.config.database.username = self.env_config.get('database.username', self.config.database.username)
        self.config.database.password = self.env_config.get('database.password', self.config.database.password)
        
        # Planning config
        self.config.planning.safety_stock_percentage = self.env_config.get('planning.safety_stock_percentage', self.config.planning.safety_stock_percentage)
        self.config.planning.planning_horizon_days = self.env_config.get('planning.planning_horizon_days', self.config.planning.planning_horizon_days)
        self.config.planning.cost_weight = self.env_config.get('planning.cost_weight', self.config.planning.cost_weight)
        self.config.planning.reliability_weight = self.env_config.get('planning.reliability_weight', self.config.planning.reliability_weight)
        
        # Data integration config
        self.config.data_integration.input_data_path = self.env_config.get('data_integration.input_data_path', self.config.data_integration.input_data_path)
        self.config.data_integration.output_data_path = self.env_config.get('data_integration.output_data_path', self.config.data_integration.output_data_path)
        self.config.data_integration.backup_data_path = self.env_config.get('data_integration.backup_data_path', self.config.data_integration.backup_data_path)
        
        # Add live data path support
        live_data_path = self.env_config.get('data_integration.live_data_path', 'data/live/')
        if not hasattr(self.config.data_integration, 'live_data_path'):
            # Add live_data_path to data integration config
            setattr(self.config.data_integration, 'live_data_path', live_data_path)
        
        # Streamlit config
        self.config.streamlit.port = self.env_config.get('streamlit.port', self.config.streamlit.port)
        self.config.streamlit.host = self.env_config.get('streamlit.host', self.config.streamlit.host)
        
        # AI config
        self.config.ai.enable_ai_integration = self.env_config.get('ai.enable_ai_integration', self.config.ai.enable_ai_integration)
        self.config.ai.model_cache_path = self.env_config.get('ai.model_cache_path', self.config.ai.model_cache_path)
        
        # General config
        self.config.environment = self.env_config.get('environment', self.config.environment)
        self.config.debug = self.env_config.get('debug', self.config.debug)
    
    def _update_config_from_dict(self, config_data: Dict[str, Any]):
        """Update configuration from dictionary"""
        for section, values in config_data.items():
            if hasattr(self.config, section):
                section_config = getattr(self.config, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
    
    def get_env_config(self) -> EnvironmentConfigManager:
        """Get the environment configuration manager instance"""
        return self.env_config
    
    def save_config(self, config_file: Optional[str] = None):
        """Save current configuration to file"""
        output_file = config_file or self.config_file
        
        # Create directory if it doesn't exist
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            config_dict = asdict(self.config)
            with open(output_file, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
            logger.info(f"Configuration saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save config to {output_file}: {e}")
    
    def get_config(self) -> BeverlyKnitsConfig:
        """Get current configuration"""
        return self.config
    
    def update_config(self, section: str, key: str, value: Any):
        """Update a specific configuration value"""
        if hasattr(self.config, section):
            section_config = getattr(self.config, section)
            if hasattr(section_config, key):
                setattr(section_config, key, value)
                logger.info(f"Updated {section}.{key} = {value}")
            else:
                logger.warning(f"Unknown config key: {section}.{key}")
        else:
            logger.warning(f"Unknown config section: {section}")
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration and return validation results"""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Validate planning configuration
        if not 0 <= self.config.planning.safety_stock_percentage <= 1:
            validation_results["errors"].append(
                "safety_stock_percentage must be between 0 and 1"
            )
            validation_results["valid"] = False
        
        if self.config.planning.planning_horizon_days <= 0:
            validation_results["errors"].append(
                "planning_horizon_days must be positive"
            )
            validation_results["valid"] = False
        
        # Validate weights sum to 1
        weight_sum = self.config.planning.cost_weight + self.config.planning.reliability_weight
        if abs(weight_sum - 1.0) > 0.001:
            validation_results["errors"].append(
                "cost_weight and reliability_weight must sum to 1.0"
            )
            validation_results["valid"] = False
        
        # Validate database configuration
        if not self.config.database.host:
            validation_results["warnings"].append(
                "Database host is empty"
            )
        
        # Validate data paths
        for path_name in ["input_data_path", "output_data_path", "backup_data_path"]:
            path = getattr(self.config.data_integration, path_name)
            if not Path(path).exists():
                validation_results["warnings"].append(
                    f"Data path does not exist: {path}"
                )
        
        return validation_results
    
    def get_environment_template(self) -> str:
        """Generate environment variable template"""
        template = """
# Beverly Knits AI Supply Chain Planner - Environment Variables

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=beverly_knits
DB_USER=postgres
DB_PASSWORD=your_password_here

# Planning Configuration
SAFETY_STOCK_PERCENTAGE=0.15
PLANNING_HORIZON_DAYS=90
COST_WEIGHT=0.6
RELIABILITY_WEIGHT=0.4

# Data Integration
INPUT_DATA_PATH=data/input/
OUTPUT_DATA_PATH=data/output/

# Streamlit Configuration
STREAMLIT_PORT=8501
STREAMLIT_HOST=localhost

# AI Configuration
ENABLE_AI_INTEGRATION=false

# Application Settings
ENVIRONMENT=development
DEBUG=true
"""
        return template

# Global configuration instance
config_manager = ConfigManager()

# Convenience function to get config
def get_config() -> BeverlyKnitsConfig:
    return config_manager.get_config()

# Convenience function to update config
def update_config(section: str, key: str, value: Any):
    config_manager.update_config(section, key, value)

if __name__ == "__main__":
    # Generate configuration template
    config_manager.save_config("config/app_config.json")
    
    # Generate environment template
    with open("config/.env.template", "w") as f:
        f.write(config_manager.get_environment_template())
    
    print("Configuration files generated:")
    print("- config/app_config.json")
    print("- config/.env.template")