"""Settings and Configuration Models for Beverly Knits AI Supply Chain Planner"""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from typing import Optional, List, Dict, Any
from enum import Enum
import os
from pathlib import Path
from decimal import Decimal

class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class DatabaseSettings(BaseSettings):
    """Database configuration settings"""
    
    # Database connection
    database_url: Optional[str] = Field(None, env="DATABASE_URL")
    database_type: str = Field("sqlite", env="DATABASE_TYPE")
    database_name: str = Field("beverlyknits.db", env="DATABASE_NAME")
    database_host: str = Field("localhost", env="DATABASE_HOST")
    database_port: int = Field(5432, env="DATABASE_PORT")
    database_user: str = Field("postgres", env="DATABASE_USER")
    database_password: str = Field("", env="DATABASE_PASSWORD")
    
    # Connection pool settings
    pool_size: int = Field(5, env="DATABASE_POOL_SIZE")
    max_overflow: int = Field(10, env="DATABASE_MAX_OVERFLOW")
    pool_timeout: int = Field(30, env="DATABASE_POOL_TIMEOUT")
    pool_recycle: int = Field(3600, env="DATABASE_POOL_RECYCLE")
    
    # Query settings
    query_timeout: int = Field(30, env="DATABASE_QUERY_TIMEOUT")
    echo_queries: bool = Field(False, env="DATABASE_ECHO_QUERIES")
    
    @field_validator('database_url', mode='before')
    @classmethod
    def build_database_url(cls, v, info):
        """Build database URL if not provided"""
        if v:
            return v
        
        values = info.data if info else {}
        db_type = values.get('database_type', 'sqlite')
        
        if db_type == 'sqlite':
            db_name = values.get('database_name', 'beverlyknits.db')
            return f"sqlite:///{db_name}"
        elif db_type == 'postgresql':
            user = values.get('database_user', 'postgres')
            password = values.get('database_password', '')
            host = values.get('database_host', 'localhost')
            port = values.get('database_port', 5432)
            db_name = values.get('database_name', 'beverlyknits')
            
            if password:
                return f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
            else:
                return f"postgresql://{user}@{host}:{port}/{db_name}"
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

class SecuritySettings(BaseSettings):
    """Security configuration settings"""
    
    # JWT settings
    secret_key: str = Field("your-secret-key-change-this-in-production", env="SECRET_KEY")
    algorithm: str = Field("HS256", env="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    refresh_token_expire_days: int = Field(7, env="REFRESH_TOKEN_EXPIRE_DAYS")
    
    # Password settings
    password_min_length: int = Field(8, env="PASSWORD_MIN_LENGTH")
    password_require_uppercase: bool = Field(True, env="PASSWORD_REQUIRE_UPPERCASE")
    password_require_lowercase: bool = Field(True, env="PASSWORD_REQUIRE_LOWERCASE")
    password_require_numbers: bool = Field(True, env="PASSWORD_REQUIRE_NUMBERS")
    password_require_special: bool = Field(True, env="PASSWORD_REQUIRE_SPECIAL")
    
    # Account lockout settings
    max_failed_attempts: int = Field(5, env="MAX_FAILED_ATTEMPTS")
    lockout_duration_minutes: int = Field(30, env="LOCKOUT_DURATION_MINUTES")
    
    # Session settings
    session_timeout_minutes: int = Field(60, env="SESSION_TIMEOUT_MINUTES")
    
    # Rate limiting
    rate_limit_requests_per_minute: int = Field(100, env="RATE_LIMIT_RPM")
    rate_limit_burst_size: int = Field(20, env="RATE_LIMIT_BURST")
    
    # CORS settings
    cors_allow_origins: List[str] = Field(["*"], env="CORS_ALLOW_ORIGINS")
    cors_allow_credentials: bool = Field(True, env="CORS_ALLOW_CREDENTIALS")
    cors_allow_methods: List[str] = Field(["*"], env="CORS_ALLOW_METHODS")
    cors_allow_headers: List[str] = Field(["*"], env="CORS_ALLOW_HEADERS")
    
    @field_validator('cors_allow_origins', mode='before')
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
    @field_validator('cors_allow_methods', mode='before')
    @classmethod
    def parse_cors_methods(cls, v):
        """Parse CORS methods from string or list"""
        if isinstance(v, str):
            return [method.strip() for method in v.split(',')]
        return v
    
    @field_validator('cors_allow_headers', mode='before')
    @classmethod
    def parse_cors_headers(cls, v):
        """Parse CORS headers from string or list"""
        if isinstance(v, str):
            return [header.strip() for header in v.split(',')]
        return v

class APISettings(BaseSettings):
    """API configuration settings"""
    
    # Server settings
    host: str = Field("0.0.0.0", env="API_HOST")
    port: int = Field(8000, env="API_PORT")
    workers: int = Field(1, env="API_WORKERS")
    
    # API metadata
    title: str = Field("Beverly Knits AI Supply Chain Planner API", env="API_TITLE")
    description: str = Field("Intelligent supply chain optimization API for textile manufacturing", env="API_DESCRIPTION")
    version: str = Field("1.0.0", env="API_VERSION")
    
    # Documentation
    docs_url: str = Field("/docs", env="API_DOCS_URL")
    redoc_url: str = Field("/redoc", env="API_REDOC_URL")
    openapi_url: str = Field("/openapi.json", env="API_OPENAPI_URL")
    
    # Request settings
    max_request_size: int = Field(10 * 1024 * 1024, env="MAX_REQUEST_SIZE")  # 10MB
    request_timeout: int = Field(30, env="REQUEST_TIMEOUT")
    
    # Response settings
    include_request_id: bool = Field(True, env="INCLUDE_REQUEST_ID")
    include_timestamp: bool = Field(True, env="INCLUDE_TIMESTAMP")
    
    # Pagination defaults
    default_page_size: int = Field(20, env="DEFAULT_PAGE_SIZE")
    max_page_size: int = Field(100, env="MAX_PAGE_SIZE")

class MLSettings(BaseSettings):
    """Machine Learning configuration settings"""
    
    # Model paths
    model_directory: str = Field("models", env="ML_MODEL_DIRECTORY")
    model_cache_size: int = Field(10, env="ML_MODEL_CACHE_SIZE")
    model_cache_ttl: int = Field(3600, env="ML_MODEL_CACHE_TTL")  # 1 hour
    
    # Training settings
    enable_training: bool = Field(True, env="ML_ENABLE_TRAINING")
    auto_retrain: bool = Field(True, env="ML_AUTO_RETRAIN")
    retrain_threshold: float = Field(0.1, env="ML_RETRAIN_THRESHOLD")
    
    # Forecasting settings
    forecast_horizon_days: int = Field(30, env="ML_FORECAST_HORIZON")
    forecast_confidence_threshold: float = Field(0.8, env="ML_FORECAST_CONFIDENCE")
    
    # Model types
    enabled_models: List[str] = Field(["arima", "prophet", "lstm", "xgboost"], env="ML_ENABLED_MODELS")
    default_model: str = Field("prophet", env="ML_DEFAULT_MODEL")
    
    # Performance settings
    max_training_time: int = Field(1800, env="ML_MAX_TRAINING_TIME")  # 30 minutes
    prediction_batch_size: int = Field(1000, env="ML_PREDICTION_BATCH_SIZE")
    
    @field_validator('enabled_models', mode='before')
    @classmethod
    def parse_enabled_models(cls, v):
        """Parse enabled models from string or list"""
        if isinstance(v, str):
            return [model.strip() for model in v.split(',')]
        return v

class DataSettings(BaseSettings):
    """Data processing configuration settings"""
    
    # File paths
    data_directory: str = Field("data", env="DATA_DIRECTORY")
    live_data_directory: str = Field("data/live", env="LIVE_DATA_DIRECTORY")
    backup_directory: str = Field("data/backups", env="BACKUP_DIRECTORY")
    
    # Data processing
    enable_data_validation: bool = Field(True, env="ENABLE_DATA_VALIDATION")
    enable_data_quality_fixes: bool = Field(True, env="ENABLE_DATA_QUALITY_FIXES")
    data_refresh_interval: int = Field(3600, env="DATA_REFRESH_INTERVAL")  # 1 hour
    
    # File formats
    supported_formats: List[str] = Field(["csv", "xlsx", "json"], env="SUPPORTED_FORMATS")
    default_encoding: str = Field("utf-8", env="DEFAULT_ENCODING")
    
    # Quality thresholds
    quality_score_threshold: float = Field(0.8, env="QUALITY_SCORE_THRESHOLD")
    completeness_threshold: float = Field(0.9, env="COMPLETENESS_THRESHOLD")
    
    # Batch processing
    batch_size: int = Field(1000, env="DATA_BATCH_SIZE")
    max_batch_time: int = Field(300, env="MAX_BATCH_TIME")  # 5 minutes
    
    @field_validator('supported_formats', mode='before')
    @classmethod
    def parse_supported_formats(cls, v):
        """Parse supported formats from string or list"""
        if isinstance(v, str):
            return [fmt.strip() for fmt in v.split(',')]
        return v

class LoggingSettings(BaseSettings):
    """Logging configuration settings"""
    
    # Log levels
    log_level: LogLevel = Field(LogLevel.INFO, env="LOG_LEVEL")
    sql_log_level: LogLevel = Field(LogLevel.WARNING, env="SQL_LOG_LEVEL")
    
    # Log output
    log_to_console: bool = Field(True, env="LOG_TO_CONSOLE")
    log_to_file: bool = Field(True, env="LOG_TO_FILE")
    log_file_path: str = Field("logs/app.log", env="LOG_FILE_PATH")
    
    # Log rotation
    log_rotation_size: str = Field("10MB", env="LOG_ROTATION_SIZE")
    log_retention_days: int = Field(30, env="LOG_RETENTION_DAYS")
    
    # Log format
    log_format: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    
    # Structured logging
    enable_json_logging: bool = Field(False, env="ENABLE_JSON_LOGGING")
    include_trace_id: bool = Field(True, env="INCLUDE_TRACE_ID")

class PlanningSettings(BaseSettings):
    """Planning engine configuration settings"""
    
    # Planning parameters
    planning_horizon_days: int = Field(90, env="PLANNING_HORIZON_DAYS")
    safety_stock_days: int = Field(7, env="SAFETY_STOCK_DAYS")
    
    # Source weights
    sales_order_weight: float = Field(1.0, env="SALES_ORDER_WEIGHT")
    production_plan_weight: float = Field(0.9, env="PRODUCTION_PLAN_WEIGHT")
    forecast_weight: float = Field(0.7, env="FORECAST_WEIGHT")
    sales_history_weight: float = Field(0.8, env="SALES_HISTORY_WEIGHT")
    
    # Optimization settings
    enable_optimization: bool = Field(True, env="ENABLE_OPTIMIZATION")
    optimization_timeout: int = Field(300, env="OPTIMIZATION_TIMEOUT")  # 5 minutes
    
    # Supplier selection
    cost_weight: float = Field(0.6, env="COST_WEIGHT")
    reliability_weight: float = Field(0.4, env="RELIABILITY_WEIGHT")
    lead_time_weight: float = Field(0.2, env="LEAD_TIME_WEIGHT")
    risk_weight: float = Field(0.1, env="RISK_WEIGHT")
    
    # Constraints
    max_supplier_concentration: float = Field(0.6, env="MAX_SUPPLIER_CONCENTRATION")
    min_safety_stock_coverage: float = Field(0.95, env="MIN_SAFETY_STOCK_COVERAGE")
    
    # EOQ settings
    default_ordering_cost: float = Field(100.0, env="DEFAULT_ORDERING_COST")
    default_holding_cost_rate: float = Field(0.25, env="DEFAULT_HOLDING_COST_RATE")
    
    # Material categories
    material_categories: Dict[str, Dict[str, float]] = Field(
        default={
            "yarn": {"safety_buffer": 0.15, "critical_threshold": 0.2},
            "fabric": {"safety_buffer": 0.10, "critical_threshold": 0.15},
            "thread": {"safety_buffer": 0.20, "critical_threshold": 0.25},
            "accessory": {"safety_buffer": 0.05, "critical_threshold": 0.10},
            "trim": {"safety_buffer": 0.10, "critical_threshold": 0.15}
        },
        env="MATERIAL_CATEGORIES"
    )
    
    # Risk thresholds
    risk_thresholds: Dict[str, float] = Field(
        default={
            "high": 0.7,
            "medium": 0.85,
            "low": 1.0
        },
        env="RISK_THRESHOLDS"
    )
    
    # Seasonal adjustments
    seasonal_adjustments: Dict[str, float] = Field(
        default={
            "Q1": 0.9,
            "Q2": 1.1,
            "Q3": 1.0,
            "Q4": 1.2
        },
        env="SEASONAL_ADJUSTMENTS"
    )

class Settings(BaseSettings):
    """Main application settings"""
    
    # Environment
    environment: Environment = Field(Environment.DEVELOPMENT, env="ENVIRONMENT")
    debug: bool = Field(False, env="DEBUG")
    testing: bool = Field(False, env="TESTING")
    
    # Application info
    app_name: str = Field("Beverly Knits AI Supply Chain Planner", env="APP_NAME")
    app_version: str = Field("1.0.0", env="APP_VERSION")
    
    # Sub-settings
    database: DatabaseSettings = DatabaseSettings()
    security: SecuritySettings = SecuritySettings()
    api: APISettings = APISettings()
    ml: MLSettings = MLSettings()
    data: DataSettings = DataSettings()
    logging: LoggingSettings = LoggingSettings()
    planning: PlanningSettings = PlanningSettings()
    
    model_config = {
        "env_file": ".env",
        "env_nested_delimiter": "__",
        "case_sensitive": False,
        "extra": "ignore"  # Ignore extra fields from environment
    }
    
    @field_validator('environment', mode='before')
    @classmethod
    def validate_environment(cls, v):
        """Validate environment value"""
        if isinstance(v, str):
            return Environment(v.lower())
        return v
    
    @field_validator('debug', mode='before')
    @classmethod
    def set_debug_from_env(cls, v, info):
        """Set debug mode based on environment"""
        values = info.data if info else {}
        if values.get('environment') == Environment.DEVELOPMENT:
            return True
        return v
    
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.environment == Environment.DEVELOPMENT
    
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.environment == Environment.PRODUCTION
    
    def is_testing(self) -> bool:
        """Check if running in testing mode"""
        return self.environment == Environment.TESTING or self.testing
    
    def get_database_url(self) -> str:
        """Get database URL for current environment"""
        return self.database.database_url
    
    def get_log_level(self) -> str:
        """Get log level for current environment"""
        if self.is_development():
            return LogLevel.DEBUG.value
        elif self.is_testing():
            return LogLevel.WARNING.value
        else:
            return self.logging.log_level.value
    
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            self.data.data_directory,
            self.data.live_data_directory,
            self.data.backup_directory,
            self.ml.model_directory,
            Path(self.logging.log_file_path).parent
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

# Global settings instance
settings = Settings()

# Legacy compatibility class
class PlanningConfig:
    """Legacy planning configuration for backwards compatibility"""
    
    @property
    def SOURCE_WEIGHTS(self):
        return {
            "sales_order": settings.planning.sales_order_weight,
            "prod_plan": settings.planning.production_plan_weight,
            "projection": settings.planning.forecast_weight,
            "sales_history": settings.planning.sales_history_weight
        }
    
    @property
    def SAFETY_STOCK_PERCENTAGE(self):
        return 0.15
    
    @property
    def PLANNING_HORIZON_DAYS(self):
        return settings.planning.planning_horizon_days
    
    @property
    def FORECAST_LOOKBACK_DAYS(self):
        return 30
    
    @property
    def COST_WEIGHT(self):
        return settings.planning.cost_weight
    
    @property
    def RELIABILITY_WEIGHT(self):
        return settings.planning.reliability_weight
    
    @property
    def MAX_SUPPLIERS_PER_MATERIAL(self):
        return 3
    
    @property
    def ENABLE_EOQ_OPTIMIZATION(self):
        return settings.planning.enable_optimization
    
    @property
    def ENABLE_MULTI_SUPPLIER(self):
        return True
    
    @property
    def ENABLE_RISK_ASSESSMENT(self):
        return True
    
    @property
    def DEFAULT_ORDERING_COST(self):
        return Decimal(str(settings.planning.default_ordering_cost))
    
    @property
    def DEFAULT_HOLDING_COST_RATE(self):
        return settings.planning.default_holding_cost_rate
    
    @property
    def RISK_THRESHOLDS(self):
        return settings.planning.risk_thresholds
    
    @property
    def MATERIAL_CATEGORIES(self):
        return settings.planning.material_categories
    
    @property
    def SUPPLIER_PERFORMANCE_TIERS(self):
        return {
            "premium": {"reliability_min": 0.95, "cost_multiplier": 1.1},
            "standard": {"reliability_min": 0.85, "cost_multiplier": 1.0},
            "budget": {"reliability_min": 0.70, "cost_multiplier": 0.9}
        }
    
    @property
    def SEASONAL_ADJUSTMENTS(self):
        return settings.planning.seasonal_adjustments
    
    @property
    def UNIT_CONVERSIONS(self):
        return {
            ("lb", "kg"): 0.453592,
            ("kg", "lb"): 2.20462,
            ("yard", "meter"): 0.9144,
            ("meter", "yard"): 1.09361
        }

# Legacy compatibility instance
PLANNING_CONFIG = PlanningConfig()