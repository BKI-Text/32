"""Configuration Management CLI for Beverly Knits AI Supply Chain Planner"""

import click
import os
import json
from pathlib import Path
from typing import Dict, Any

from .environment_loader import EnvironmentLoader
from .validator import ConfigurationValidator
from .settings import settings

@click.group()
def config():
    """Configuration management commands"""
    logger.info("Configuration CLI initialized")
    
    # Ensure required directories exist
    try:
        from ..config.settings import settings
        settings.create_directories()
    except Exception as e:
        logger.warning(f"Failed to create directories during CLI init: {e}")
        
def main():
    """Main entry point for CLI"""
    try:
        config()
    except Exception as e:
        logger.error(f"CLI execution failed: {e}")
        exit(1)

@config.command()
def info():
    """Show current configuration information"""
    click.echo(f"🔧 Configuration Information")
    click.echo(f"Current Environment: {settings.environment.value}")
    click.echo(f"Debug Mode: {'enabled' if settings.debug else 'disabled'}")
    click.echo(f"Database Type: {settings.database.database_type}")
    click.echo(f"Database URL: {settings.database.database_url}")
    click.echo(f"API Host: {settings.api.host}:{settings.api.port}")
    click.echo(f"Log Level: {settings.logging.log_level.value}")
    click.echo(f"Log to File: {'enabled' if settings.logging.log_to_file else 'disabled'}")
    
    if settings.logging.log_to_file:
        click.echo(f"Log File: {settings.logging.log_file_path}")

@config.command()
def environments():
    """List available environments"""
    loader = EnvironmentLoader()
    available_envs = loader.list_available_environments()
    
    current_env = os.getenv("ENVIRONMENT", "development")
    
    click.echo("📦 Available Environments:")
    for env in available_envs:
        marker = "🟢" if env == current_env else "🔵"
        click.echo(f"  {marker} {env}")
    
    if not available_envs:
        click.echo("  No environment files found")

@config.command()
@click.argument('environment')
def validate(environment):
    """Validate configuration for an environment"""
    validator = ConfigurationValidator()
    
    # Load environment first
    loader = EnvironmentLoader()
    if not loader.validate_environment(environment):
        click.echo(f"❌ Environment '{environment}' configuration file not found")
        return
    
    # Load and validate
    loader.load_environment(environment)
    results = validator.validate_environment(environment)
    
    # Print results
    click.echo(f"🔍 Validating configuration for environment: {environment}")
    validator.print_results()
    
    summary = validator.get_validation_summary()
    if summary['errors'] > 0:
        click.echo(f"\n❌ Validation failed with {summary['errors']} errors")
        exit(1)
    elif summary['warnings'] > 0:
        click.echo(f"\n⚠️  Validation completed with {summary['warnings']} warnings")
    else:
        click.echo(f"\n✅ Validation passed successfully")

@config.command()
@click.argument('environment')
@click.option('--output', '-o', help='Output file path')
def export(environment, output):
    """Export environment configuration"""
    loader = EnvironmentLoader()
    
    if not loader.validate_environment(environment):
        click.echo(f"❌ Environment '{environment}' configuration file not found")
        return
    
    # Load environment
    env_vars = loader.load_environment(environment)
    
    # Filter environment variables related to the application
    app_vars = {}
    prefixes = ['DATABASE_', 'API_', 'ML_', 'LOG_', 'SECRET_', 'CORS_', 'ENVIRONMENT', 'DEBUG', 'TESTING']
    
    for key, value in env_vars.items():
        if any(key.startswith(prefix) for prefix in prefixes):
            app_vars[key] = value
    
    # Export to file or stdout
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(app_vars, f, indent=2)
        
        click.echo(f"📄 Configuration exported to {output_path}")
    else:
        click.echo(f"📄 Configuration for environment '{environment}':")
        click.echo(json.dumps(app_vars, indent=2))

@config.command()
@click.argument('source_env')
@click.argument('target_env')
def copy(source_env, target_env):
    """Copy configuration from one environment to another"""
    loader = EnvironmentLoader()
    
    try:
        target_file = loader.copy_environment(source_env, target_env)
        click.echo(f"✅ Configuration copied from '{source_env}' to '{target_env}'")
        click.echo(f"📁 Created: {target_file}")
    except FileNotFoundError as e:
        click.echo(f"❌ {e}")
    except FileExistsError as e:
        click.echo(f"❌ {e}")

@config.command()
@click.argument('environment')
@click.option('--force', is_flag=True, help='Force creation even if file exists')
def create(environment, force):
    """Create a new environment configuration"""
    loader = EnvironmentLoader()
    env_file = Path(f".env.{environment}")
    
    if env_file.exists() and not force:
        click.echo(f"❌ Environment file already exists: {env_file}")
        click.echo("Use --force to overwrite")
        return
    
    # Template variables
    template_vars = {
        "Basic": {
            "ENVIRONMENT": environment,
            "DEBUG": "false",
            "SECRET_KEY": "change-this-secret-key"
        },
        "Database": {
            "DATABASE_TYPE": "sqlite",
            "DATABASE_NAME": f"beverlyknits_{environment}.db",
            "DATABASE_HOST": "localhost",
            "DATABASE_PORT": "5432",
            "DATABASE_USER": "postgres",
            "DATABASE_PASSWORD": ""
        },
        "API": {
            "API_HOST": "0.0.0.0",
            "API_PORT": "8000",
            "API_WORKERS": "1"
        },
        "Logging": {
            "LOG_LEVEL": "INFO",
            "LOG_TO_CONSOLE": "true",
            "LOG_TO_FILE": "false",
            "LOG_FILE_PATH": f"logs/{environment}.log"
        }
    }
    
    if force and env_file.exists():
        env_file.unlink()
    
    try:
        created_file = loader.create_environment_template(environment, template_vars)
        click.echo(f"✅ Environment configuration created: {created_file}")
        click.echo("🔧 Don't forget to update the SECRET_KEY and other sensitive values!")
    except FileExistsError as e:
        click.echo(f"❌ {e}")

@config.command()
@click.option('--environment', '-e', help='Environment to check')
def check(environment):
    """Check configuration health"""
    if environment:
        # Load specific environment
        loader = EnvironmentLoader()
        if not loader.validate_environment(environment):
            click.echo(f"❌ Environment '{environment}' configuration file not found")
            return
        
        loader.load_environment(environment)
        current_env = environment
    else:
        current_env = os.getenv("ENVIRONMENT", "development")
    
    click.echo(f"🔍 Health check for environment: {current_env}")
    
    # Check configuration validity
    validator = ConfigurationValidator()
    results = validator.validate_environment(current_env)
    summary = validator.get_validation_summary()
    
    if summary['errors'] > 0:
        click.echo(f"❌ Configuration health: FAILED ({summary['errors']} errors)")
        validator.print_results(show_info=False)
        return
    
    # Check database connection
    try:
        from ..database.connection import health_check
        db_health = health_check()
        
        if db_health['status'] == 'healthy':
            click.echo(f"✅ Database health: OK ({db_health['database_type']})")
        else:
            click.echo(f"❌ Database health: FAILED ({db_health['error']})")
            return
    except Exception as e:
        click.echo(f"❌ Database health check failed: {e}")
        return
    
    # Check directories
    try:
        settings.create_directories()
        click.echo("✅ Directories: OK")
    except Exception as e:
        click.echo(f"❌ Directory creation failed: {e}")
        return
    
    # Check file permissions
    if settings.logging.log_to_file:
        log_file = Path(settings.logging.log_file_path)
        if log_file.exists():
            if log_file.is_file() and os.access(log_file, os.W_OK):
                click.echo("✅ Log file permissions: OK")
            else:
                click.echo("❌ Log file permissions: FAILED")
                return
    
    if summary['warnings'] > 0:
        click.echo(f"⚠️  Configuration health: OK with {summary['warnings']} warnings")
    else:
        click.echo("✅ Configuration health: EXCELLENT")

@config.command()
@click.option('--key', '-k', help='Specific configuration key to show')
@click.option('--environment', '-e', help='Environment to query')
def get(key, environment):
    """Get configuration value(s)"""
    if environment:
        # Load specific environment
        loader = EnvironmentLoader()
        if not loader.validate_environment(environment):
            click.echo(f"❌ Environment '{environment}' configuration file not found")
            return
        
        loader.load_environment(environment)
    
    if key:
        # Get specific key
        value = os.getenv(key)
        if value is not None:
            click.echo(f"{key}={value}")
        else:
            click.echo(f"❌ Configuration key '{key}' not found")
    else:
        # Show all configuration
        click.echo("🔧 Current Configuration:")
        
        # Group by category
        categories = {
            "Environment": ["ENVIRONMENT", "DEBUG", "TESTING"],
            "Database": [k for k in os.environ.keys() if k.startswith("DATABASE_")],
            "API": [k for k in os.environ.keys() if k.startswith("API_")],
            "Security": [k for k in os.environ.keys() if k.startswith(("SECRET_", "CORS_", "MAX_FAILED", "LOCKOUT_"))],
            "Logging": [k for k in os.environ.keys() if k.startswith("LOG_")],
            "ML": [k for k in os.environ.keys() if k.startswith("ML_")],
            "Data": [k for k in os.environ.keys() if k.startswith("DATA_")],
            "Planning": [k for k in os.environ.keys() if k.startswith(("PLANNING_", "SALES_", "COST_", "ENABLE_"))]
        }
        
        for category, keys in categories.items():
            if keys:
                click.echo(f"\n📂 {category}:")
                for key in sorted(keys):
                    value = os.getenv(key, "")
                    # Hide sensitive values
                    if any(sensitive in key.lower() for sensitive in ["password", "secret", "key"]):
                        value = "***" if value else ""
                    click.echo(f"  {key}={value}")

@config.command()
@click.option('--environment', '-e', help='Environment to use')
def directories(environment):
    """Create required directories"""
    if environment:
        # Load specific environment
        loader = EnvironmentLoader()
        if not loader.validate_environment(environment):
            click.echo(f"❌ Environment '{environment}' configuration file not found")
            return
        
        loader.load_environment(environment)
    
    try:
        settings.create_directories()
        click.echo("✅ All required directories created successfully")
        
        # List created directories
        directories = [
            settings.data.data_directory,
            settings.data.live_data_directory,
            settings.data.backup_directory,
            settings.ml.model_directory,
            Path(settings.logging.log_file_path).parent
        ]
        
        click.echo("\n📁 Directories:")
        for directory in directories:
            path = Path(directory)
            status = "✅" if path.exists() else "❌"
            click.echo(f"  {status} {path}")
            
    except Exception as e:
        click.echo(f"❌ Failed to create directories: {e}")

if __name__ == "__main__":
    config()