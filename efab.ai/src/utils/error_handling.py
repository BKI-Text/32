"""
Comprehensive Error Handling and Logging System
Provides robust error handling, logging, and monitoring for the Beverly Knits AI Supply Chain Planner.
"""

import logging
import traceback
import sys
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from functools import wraps
from pathlib import Path
import json
from enum import Enum
from dataclasses import dataclass, asdict

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for classification."""
    DATA_VALIDATION = "data_validation"
    PLANNING_ENGINE = "planning_engine"
    DATA_INTEGRATION = "data_integration"
    ML_PROCESSING = "ml_processing"
    UI_INTERACTION = "ui_interaction"
    EXTERNAL_SERVICE = "external_service"
    CONFIGURATION = "configuration"
    SYSTEM = "system"

@dataclass
class ErrorReport:
    """Structured error report."""
    timestamp: str
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    details: Dict[str, Any]
    stack_trace: Optional[str]
    context: Dict[str, Any]
    user_action: Optional[str] = None
    resolution_steps: Optional[List[str]] = None

class BeverlyKnitsLogger:
    """Enhanced logging system for Beverly Knits application."""
    
    def __init__(self, name: str = "beverly_knits", log_level: str = "INFO"):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Create logs directory
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        # Setup handlers
        self._setup_handlers()
        
        # Error tracking
        self.error_counts = {category: 0 for category in ErrorCategory}
        self.recent_errors = []
        
    def _setup_handlers(self):
        """Setup logging handlers."""
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_format)
        
        # File handler for all logs
        file_handler = logging.FileHandler(
            self.logs_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_format)
        
        # Error file handler
        error_handler = logging.FileHandler(
            self.logs_dir / f"{self.name}_errors_{datetime.now().strftime('%Y%m%d')}.log"
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_format)
        
        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
    
    def log_error(
        self, 
        error: Exception, 
        category: ErrorCategory,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        user_action: Optional[str] = None
    ) -> str:
        """Log an error with detailed information."""
        
        error_id = f"{category.value}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        error_report = ErrorReport(
            timestamp=datetime.now().isoformat(),
            error_id=error_id,
            category=category,
            severity=severity,
            message=str(error),
            details={
                "error_type": type(error).__name__,
                "error_args": error.args
            },
            stack_trace=traceback.format_exc(),
            context=context or {},
            user_action=user_action,
            resolution_steps=self._get_resolution_steps(category, error)
        )
        
        # Log the error
        log_message = f"[{error_id}] {category.value.upper()}: {error}"
        if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self.logger.error(log_message)
        else:
            self.logger.warning(log_message)
        
        # Track error
        self.error_counts[category] += 1
        self.recent_errors.append(error_report)
        
        # Keep only recent errors (last 100)
        if len(self.recent_errors) > 100:
            self.recent_errors = self.recent_errors[-100:]
        
        # Save detailed error report
        self._save_error_report(error_report)
        
        return error_id
    
    def _get_resolution_steps(self, category: ErrorCategory, error: Exception) -> List[str]:
        """Get suggested resolution steps based on error category and type."""
        
        resolution_map = {
            ErrorCategory.DATA_VALIDATION: [
                "Check data file format and column names",
                "Verify data types and ranges",
                "Run data quality validation",
                "Check for missing or corrupted data"
            ],
            ErrorCategory.PLANNING_ENGINE: [
                "Verify input data completeness",
                "Check planning configuration parameters",
                "Validate supplier and material relationships",
                "Review forecast data quality"
            ],
            ErrorCategory.DATA_INTEGRATION: [
                "Check file permissions and accessibility",
                "Verify file encoding (UTF-8, UTF-8-BOM)",
                "Validate CSV structure and headers",
                "Check for data consistency issues"
            ],
            ErrorCategory.ML_PROCESSING: [
                "Verify ML model availability",
                "Check data preprocessing steps",
                "Validate feature engineering",
                "Review model input requirements"
            ],
            ErrorCategory.UI_INTERACTION: [
                "Refresh the browser page",
                "Clear browser cache",
                "Check network connectivity",
                "Verify user session validity"
            ],
            ErrorCategory.EXTERNAL_SERVICE: [
                "Check external service connectivity",
                "Verify API credentials and permissions",
                "Review service rate limits",
                "Check service status and availability"
            ],
            ErrorCategory.CONFIGURATION: [
                "Review configuration file syntax",
                "Verify environment variables",
                "Check file paths and permissions",
                "Validate configuration parameters"
            ],
            ErrorCategory.SYSTEM: [
                "Check system resources (memory, disk)",
                "Verify Python environment and dependencies",
                "Review system logs",
                "Restart the application if necessary"
            ]
        }
        
        return resolution_map.get(category, ["Contact system administrator for assistance"])
    
    def _save_error_report(self, error_report: ErrorReport):
        """Save detailed error report to file."""
        
        error_file = self.logs_dir / f"error_reports_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        with open(error_file, 'a') as f:
            f.write(json.dumps(asdict(error_report)) + '\n')
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary and statistics."""
        
        total_errors = sum(self.error_counts.values())
        
        return {
            "total_errors": total_errors,
            "errors_by_category": dict(self.error_counts),
            "recent_errors_count": len(self.recent_errors),
            "critical_errors": sum(1 for e in self.recent_errors if e.severity == ErrorSeverity.CRITICAL),
            "high_severity_errors": sum(1 for e in self.recent_errors if e.severity == ErrorSeverity.HIGH),
            "most_recent_error": self.recent_errors[-1].timestamp if self.recent_errors else None
        }

class ErrorHandler:
    """Global error handler for the Beverly Knits application."""
    
    def __init__(self):
        self.logger = BeverlyKnitsLogger()
        self.error_handlers = {}
        self._setup_default_handlers()
    
    def _setup_default_handlers(self):
        """Setup default error handlers for common exceptions."""
        
        self.error_handlers.update({
            FileNotFoundError: self._handle_file_not_found,
            PermissionError: self._handle_permission_error,
            ValueError: self._handle_value_error,
            KeyError: self._handle_key_error,
            ConnectionError: self._handle_connection_error,
            ImportError: self._handle_import_error,
            MemoryError: self._handle_memory_error
        })
    
    def handle_error(
        self,
        error: Exception,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        context: Optional[Dict[str, Any]] = None,
        user_action: Optional[str] = None
    ) -> str:
        """Handle an error with appropriate logging and response."""
        
        # Determine severity
        severity = self._determine_severity(error)
        
        # Use specific handler if available
        error_type = type(error)
        if error_type in self.error_handlers:
            return self.error_handlers[error_type](error, category, context, user_action)
        
        # Generic error handling
        return self.logger.log_error(error, category, severity, context, user_action)
    
    def _determine_severity(self, error: Exception) -> ErrorSeverity:
        """Determine error severity based on exception type."""
        
        critical_errors = [MemoryError, SystemError, OSError]
        high_errors = [ConnectionError, ImportError, FileNotFoundError]
        medium_errors = [ValueError, TypeError, AttributeError]
        
        error_type = type(error)
        
        if any(isinstance(error, err_type) for err_type in critical_errors):
            return ErrorSeverity.CRITICAL
        elif any(isinstance(error, err_type) for err_type in high_errors):
            return ErrorSeverity.HIGH
        elif any(isinstance(error, err_type) for err_type in medium_errors):
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _handle_file_not_found(
        self, 
        error: FileNotFoundError, 
        category: ErrorCategory,
        context: Optional[Dict[str, Any]] = None,
        user_action: Optional[str] = None
    ) -> str:
        """Handle file not found errors."""
        
        enhanced_context = {
            "missing_file": str(error.filename) if error.filename else "unknown",
            "working_directory": str(Path.cwd()),
            **(context or {})
        }
        
        return self.logger.log_error(
            error, 
            category or ErrorCategory.DATA_INTEGRATION, 
            ErrorSeverity.HIGH,
            enhanced_context,
            user_action or "Check file path and ensure file exists"
        )
    
    def _handle_permission_error(
        self, 
        error: PermissionError, 
        category: ErrorCategory,
        context: Optional[Dict[str, Any]] = None,
        user_action: Optional[str] = None
    ) -> str:
        """Handle permission errors."""
        
        enhanced_context = {
            "access_file": str(error.filename) if error.filename else "unknown",
            **(context or {})
        }
        
        return self.logger.log_error(
            error, 
            category or ErrorCategory.SYSTEM, 
            ErrorSeverity.HIGH,
            enhanced_context,
            user_action or "Check file permissions and user access rights"
        )
    
    def _handle_value_error(
        self, 
        error: ValueError, 
        category: ErrorCategory,
        context: Optional[Dict[str, Any]] = None,
        user_action: Optional[str] = None
    ) -> str:
        """Handle value errors."""
        
        return self.logger.log_error(
            error, 
            category or ErrorCategory.DATA_VALIDATION, 
            ErrorSeverity.MEDIUM,
            context,
            user_action or "Verify input data format and values"
        )
    
    def _handle_key_error(
        self, 
        error: KeyError, 
        category: ErrorCategory,
        context: Optional[Dict[str, Any]] = None,
        user_action: Optional[str] = None
    ) -> str:
        """Handle key errors."""
        
        enhanced_context = {
            "missing_key": str(error.args[0]) if error.args else "unknown",
            **(context or {})
        }
        
        return self.logger.log_error(
            error, 
            category or ErrorCategory.DATA_VALIDATION, 
            ErrorSeverity.MEDIUM,
            enhanced_context,
            user_action or "Check data structure and required fields"
        )
    
    def _handle_connection_error(
        self, 
        error: ConnectionError, 
        category: ErrorCategory,
        context: Optional[Dict[str, Any]] = None,
        user_action: Optional[str] = None
    ) -> str:
        """Handle connection errors."""
        
        return self.logger.log_error(
            error, 
            category or ErrorCategory.EXTERNAL_SERVICE, 
            ErrorSeverity.HIGH,
            context,
            user_action or "Check network connectivity and service availability"
        )
    
    def _handle_import_error(
        self, 
        error: ImportError, 
        category: ErrorCategory,
        context: Optional[Dict[str, Any]] = None,
        user_action: Optional[str] = None
    ) -> str:
        """Handle import errors."""
        
        enhanced_context = {
            "missing_module": error.name if hasattr(error, 'name') else "unknown",
            **(context or {})
        }
        
        return self.logger.log_error(
            error, 
            category or ErrorCategory.SYSTEM, 
            ErrorSeverity.HIGH,
            enhanced_context,
            user_action or "Install missing dependencies or check Python environment"
        )
    
    def _handle_memory_error(
        self, 
        error: MemoryError, 
        category: ErrorCategory,
        context: Optional[Dict[str, Any]] = None,
        user_action: Optional[str] = None
    ) -> str:
        """Handle memory errors."""
        
        return self.logger.log_error(
            error, 
            category or ErrorCategory.SYSTEM, 
            ErrorSeverity.CRITICAL,
            context,
            user_action or "Reduce data size or increase available memory"
        )

# Global error handler instance
global_error_handler = ErrorHandler()

def handle_errors(
    category: ErrorCategory = ErrorCategory.SYSTEM,
    reraise: bool = False,
    default_return: Any = None
):
    """Decorator for automatic error handling."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_id = global_error_handler.handle_error(
                    e, 
                    category,
                    context={
                        "function": func.__name__,
                        "args": str(args)[:200],  # Limit context size
                        "kwargs": str(kwargs)[:200]
                    }
                )
                
                if reraise:
                    raise
                
                global_error_handler.logger.logger.warning(
                    f"Function {func.__name__} failed with error {error_id}, returning default value"
                )
                return default_return
        
        return wrapper
    return decorator

def safe_execute(
    func: Callable,
    category: ErrorCategory = ErrorCategory.SYSTEM,
    context: Optional[Dict[str, Any]] = None,
    default_return: Any = None
) -> Any:
    """Safely execute a function with error handling."""
    
    try:
        return func()
    except Exception as e:
        global_error_handler.handle_error(e, category, context)
        return default_return

# Monitoring and health check functions
def get_system_health() -> Dict[str, Any]:
    """Get overall system health status."""
    
    error_summary = global_error_handler.logger.get_error_summary()
    
    # Determine health status
    critical_errors = error_summary["critical_errors"]
    high_errors = error_summary["high_severity_errors"]
    total_errors = error_summary["total_errors"]
    
    if critical_errors > 0:
        health_status = "critical"
    elif high_errors > 5:
        health_status = "degraded"
    elif total_errors > 20:
        health_status = "warning"
    else:
        health_status = "healthy"
    
    return {
        "status": health_status,
        "timestamp": datetime.now().isoformat(),
        "error_summary": error_summary,
        "uptime": "unknown",  # Would need to track application start time
        "system_resources": {
            "memory_usage": "unknown",  # Would need psutil
            "disk_usage": "unknown"
        }
    }

def generate_error_report() -> str:
    """Generate a comprehensive error report."""
    
    error_summary = global_error_handler.logger.get_error_summary()
    recent_errors = global_error_handler.logger.recent_errors[-10:]  # Last 10 errors
    
    report = f"""
ERROR REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

SUMMARY:
Total Errors: {error_summary['total_errors']}
Critical Errors: {error_summary['critical_errors']}
High Severity Errors: {error_summary['high_severity_errors']}

ERRORS BY CATEGORY:
"""
    
    for category, count in error_summary['errors_by_category'].items():
        if count > 0:
            report += f"  {category}: {count}\n"
    
    if recent_errors:
        report += f"\nRECENT ERRORS (Last {len(recent_errors)}):\n"
        for error in recent_errors:
            report += f"  [{error.timestamp}] {error.category.value}: {error.message}\n"
    
    return report