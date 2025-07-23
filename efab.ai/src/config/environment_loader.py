"""Environment Configuration Loader for Beverly Knits AI Supply Chain Planner"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

class EnvironmentLoader:
    """Loads environment-specific configuration files"""
    
    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path.cwd()
        self.loaded_files = []
    
    def load_environment(self, environment: Optional[str] = None) -> Dict[str, Any]:
        """Load environment-specific configuration"""
        
        # Determine environment
        if environment is None:
            environment = os.getenv("ENVIRONMENT", "development")
        
        logger.info(f"Loading configuration for environment: {environment}")
        
        # Load environment files in order of precedence
        env_files = self._get_env_files(environment)
        
        for env_file in env_files:
            if env_file.exists():
                logger.info(f"Loading environment file: {env_file}")
                load_dotenv(env_file, override=True)
                self.loaded_files.append(str(env_file))
            else:
                logger.warning(f"Environment file not found: {env_file}")
        
        # Return current environment variables
        return dict(os.environ)
    
    def _get_env_files(self, environment: str) -> list[Path]:
        """Get list of environment files in order of precedence"""
        env_files = []
        
        # Base .env file (lowest precedence)
        base_env = self.base_path / ".env"
        if base_env.exists():
            env_files.append(base_env)
        
        # Environment-specific file
        env_specific = self.base_path / f".env.{environment}"
        if env_specific.exists():
            env_files.append(env_specific)
        
        # Local override file (highest precedence)
        local_env = self.base_path / ".env.local"
        if local_env.exists():
            env_files.append(local_env)
        
        return env_files
    
    def get_loaded_files(self) -> list[str]:
        """Get list of loaded configuration files"""
        return self.loaded_files.copy()
    
    def validate_environment(self, environment: str) -> bool:
        """Validate that environment configuration exists"""
        env_file = self.base_path / f".env.{environment}"
        return env_file.exists()
    
    def list_available_environments(self) -> list[str]:
        """List all available environment configurations"""
        env_files = self.base_path.glob(".env.*")
        environments = []
        
        for env_file in env_files:
            if env_file.name.startswith(".env.") and not env_file.name.endswith(".local"):
                env_name = env_file.name[5:]  # Remove ".env." prefix
                environments.append(env_name)
        
        return sorted(environments)
    
    def create_environment_template(self, environment: str, template_vars: Dict[str, Any]) -> Path:
        """Create a new environment configuration template"""
        env_file = self.base_path / f".env.{environment}"
        
        if env_file.exists():
            raise FileExistsError(f"Environment file already exists: {env_file}")
        
        # Create template content
        content = f"# {environment.title()} Environment Configuration\n"
        content += f"ENVIRONMENT={environment}\n\n"
        
        for category, vars_dict in template_vars.items():
            content += f"# {category.title()} Configuration\n"
            for key, value in vars_dict.items():
                content += f"{key}={value}\n"
            content += "\n"
        
        # Write template file
        env_file.write_text(content)
        logger.info(f"Created environment template: {env_file}")
        
        return env_file
    
    def copy_environment(self, source_env: str, target_env: str) -> Path:
        """Copy configuration from one environment to another"""
        source_file = self.base_path / f".env.{source_env}"
        target_file = self.base_path / f".env.{target_env}"
        
        if not source_file.exists():
            raise FileNotFoundError(f"Source environment file not found: {source_file}")
        
        if target_file.exists():
            raise FileExistsError(f"Target environment file already exists: {target_file}")
        
        # Copy content and update environment line
        content = source_file.read_text()
        content = content.replace(f"ENVIRONMENT={source_env}", f"ENVIRONMENT={target_env}")
        
        target_file.write_text(content)
        logger.info(f"Copied environment configuration from {source_env} to {target_env}")
        
        return target_file

def load_environment_config(environment: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to load environment configuration"""
    loader = EnvironmentLoader()
    return loader.load_environment(environment)

def setup_logging_from_env():
    """Setup logging configuration from environment variables"""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_to_console = os.getenv("LOG_TO_CONSOLE", "true").lower() == "true"
    log_to_file = os.getenv("LOG_TO_FILE", "false").lower() == "true"
    log_file_path = os.getenv("LOG_FILE_PATH", "logs/app.log")
    log_format = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Create log directory if needed
    if log_to_file:
        Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    handlers = []
    
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(console_handler)
    
    if log_to_file:
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=handlers,
        force=True
    )
    
    logger.info(f"Logging configured: level={log_level}, console={log_to_console}, file={log_to_file}")

def get_environment_info() -> Dict[str, Any]:
    """Get information about the current environment configuration"""
    loader = EnvironmentLoader()
    
    current_env = os.getenv("ENVIRONMENT", "development")
    available_envs = loader.list_available_environments()
    
    return {
        "current_environment": current_env,
        "available_environments": available_envs,
        "loaded_files": loader.get_loaded_files(),
        "environment_variables": {
            key: value for key, value in os.environ.items()
            if key.startswith(("DATABASE_", "API_", "ML_", "LOG_", "SECRET_", "CORS_"))
        }
    }

if __name__ == "__main__":
    # Command line interface for environment management
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python environment_loader.py <command> [args]")
        print("Commands:")
        print("  list - List available environments")
        print("  info - Show current environment info")
        print("  load <env> - Load specific environment")
        print("  validate <env> - Validate environment configuration")
        sys.exit(1)
    
    command = sys.argv[1]
    loader = EnvironmentLoader()
    
    if command == "list":
        environments = loader.list_available_environments()
        print("Available environments:")
        for env in environments:
            print(f"  - {env}")
    
    elif command == "info":
        info = get_environment_info()
        print(f"Current environment: {info['current_environment']}")
        print(f"Available environments: {', '.join(info['available_environments'])}")
        print(f"Loaded files: {', '.join(info['loaded_files'])}")
    
    elif command == "load":
        if len(sys.argv) < 3:
            print("Usage: python environment_loader.py load <environment>")
            sys.exit(1)
        
        env = sys.argv[2]
        config = loader.load_environment(env)
        print(f"Loaded configuration for environment: {env}")
        print(f"Loaded files: {', '.join(loader.get_loaded_files())}")
    
    elif command == "validate":
        if len(sys.argv) < 3:
            print("Usage: python environment_loader.py validate <environment>")
            sys.exit(1)
        
        env = sys.argv[2]
        is_valid = loader.validate_environment(env)
        if is_valid:
            print(f"Environment '{env}' is valid")
        else:
            print(f"Environment '{env}' is invalid or missing")
            sys.exit(1)
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)