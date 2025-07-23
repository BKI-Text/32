"""Configuration Manager for Beverly Knits AI Supply Chain Planner"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

from .settings import settings, Settings
from .environment_loader import EnvironmentLoader, setup_logging_from_env
from .validator import ConfigurationValidator

logger = logging.getLogger(__name__)

# Legacy compatibility classes
@dataclass
class DatabaseConfig:
    """Database configuration (legacy compatibility)"""
    host: str = "localhost"
    port: int = 5432
    database: str = "beverlyknits"
    username: str = "postgres"
    password: str = ""
    pool_size: int = 5
    max_overflow: int = 10
    echo: bool = False

@dataclass
class ServerConfig:
    """Server configuration (legacy compatibility)"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    reload: bool = False
    workers: int = 1

@dataclass
class LoggingConfig:
    """Logging configuration (legacy compatibility)"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5

@dataclass
class Config:
    """Main configuration class (legacy compatibility)"""
    environment: str = "development"
    debug: bool = False
    secret_key: str = "dev-secret-key-change-in-production"
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def __post_init__(self):
        """Post-initialization processing"""
        # Set debug mode based on environment
        if self.environment == "development":
            self.debug = True
            self.server.debug = True
            self.server.reload = True
        
        # Ensure data directories exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure required directories exist"""
        directories = [
            "data",
            "data/live",
            "data/backups",
            "logs",
            "models",
            "exports"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "environment": self.environment,
            "debug": self.debug,
            "secret_key": self.secret_key,
            "database": {
                "host": self.database.host,
                "port": self.database.port,
                "database": self.database.database,
                "username": self.database.username,
                "password": "***" if self.database.password else "",
                "pool_size": self.database.pool_size,
                "max_overflow": self.database.max_overflow,
                "echo": self.database.echo
            },
            "server": {
                "host": self.server.host,
                "port": self.server.port,
                "debug": self.server.debug,
                "reload": self.server.reload,
                "workers": self.server.workers
            },
            "logging": {
                "level": self.logging.level,
                "format": self.logging.format,
                "file_path": self.logging.file_path,
                "max_bytes": self.logging.max_bytes,
                "backup_count": self.logging.backup_count
            }
        }

class ConfigManager:
    """Configuration manager for the application"""
    
    def __init__(self, config_file: Optional[str] = None, environment: Optional[str] = None):
        self.config_file = config_file or "config/app_config.json"
        self.environment = environment or os.getenv("ENVIRONMENT", "development")
        
        # Load environment configuration
        self._load_environment()
        
        # Create legacy config object for backward compatibility
        self.config = self._create_legacy_config()
        
        # Setup logging from new settings
        self._setup_logging()
        
        # Validate configuration
        self._validate_config()
    
    def _load_environment(self):
        """Load environment-specific configuration"""
        loader = EnvironmentLoader()
        loader.load_environment(self.environment)
        
        # Create necessary directories
        settings.create_directories()
        
        logger.info(f"Configuration loaded for environment: {self.environment}")
    
    def _create_legacy_config(self) -> Config:
        """Create legacy config object from new settings"""
        config = Config()
        
        # Map new settings to legacy config
        config.environment = settings.environment.value
        config.debug = settings.debug
        config.secret_key = settings.security.secret_key
        
        # Database mapping
        config.database.host = settings.database.database_host
        config.database.port = settings.database.database_port
        config.database.database = settings.database.database_name
        config.database.username = settings.database.database_user
        config.database.password = settings.database.database_password
        config.database.pool_size = settings.database.pool_size
        config.database.max_overflow = settings.database.max_overflow
        config.database.echo = settings.database.echo_queries
        
        # Server mapping
        config.server.host = settings.api.host
        config.server.port = settings.api.port
        config.server.debug = settings.debug
        config.server.reload = settings.debug
        config.server.workers = settings.api.workers
        
        # Logging mapping
        config.logging.level = settings.logging.log_level.value
        config.logging.format = settings.logging.log_format
        config.logging.file_path = settings.logging.log_file_path if settings.logging.log_to_file else None
        config.logging.max_bytes = 10 * 1024 * 1024  # 10MB
        config.logging.backup_count = 5
        
        return config
    
    def _setup_logging(self):
        """Setup logging configuration"""
        setup_logging_from_env()
        logger.info(f"Logging configured for environment: {self.environment}")
    
    def _validate_config(self):
        """Validate configuration settings"""
        validator = ConfigurationValidator()
        results = validator.validate_environment(self.environment)
        
        # Log validation results
        summary = validator.get_validation_summary()
        if summary['errors'] > 0:
            logger.error(f"Configuration validation failed: {summary['errors']} errors")
            for result in results:
                if result.level.value == "error":
                    logger.error(f"Config error - {result.key}: {result.message}")
        elif summary['warnings'] > 0:
            logger.warning(f"Configuration validation warnings: {summary['warnings']} warnings")
    
    def save_config(self, config_file: Optional[str] = None):
        """Save current configuration to file"""
        file_path = config_file or self.config_file
        
        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        with open(file_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        logger.info(f"Configuration saved to {file_path}")
    
    def get_database_url(self) -> str:
        """Get database connection URL"""
        return settings.get_database_url()
    
    def reload_config(self):
        """Reload configuration from file"""
        self._load_environment()
        self.config = self._create_legacy_config()
        self._setup_logging()
        logger.info("Configuration reloaded")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        return {
            "environment": settings.environment.value,
            "debug": settings.debug,
            "database_type": settings.database.database_type,
            "server": {
                "host": settings.api.host,
                "port": settings.api.port,
                "workers": settings.api.workers
            },
            "logging": {
                "level": settings.logging.log_level.value,
                "file_enabled": settings.logging.log_to_file
            },
            "config_file": self.config_file,
            "loaded_at": datetime.now().isoformat()
        }
    
    def get_settings(self) -> Settings:
        """Get the new settings object"""
        return settings
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate current configuration and return results"""
        validator = ConfigurationValidator()
        results = validator.validate_environment(self.environment)
        return {
            "results": results,
            "summary": validator.get_validation_summary()
        }

# Global configuration manager instance
config_manager = ConfigManager()