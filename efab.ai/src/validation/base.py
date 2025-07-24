"""Base Validation Framework for Beverly Knits AI Supply Chain Planner"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Type, get_type_hints
from pydantic import BaseModel, Field, field_validator, root_validator
from enum import Enum
from datetime import datetime, date
from decimal import Decimal
import re
import logging

logger = logging.getLogger(__name__)

class ValidationLevel(str, Enum):
    """Validation severity levels"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

class ValidationResult:
    """Result of a validation check"""
    
    def __init__(self, 
                 level: ValidationLevel,
                 field: str,
                 message: str,
                 code: Optional[str] = None,
                 value: Any = None,
                 suggestion: Optional[str] = None):
        self.level = level
        self.field = field
        self.message = message
        self.code = code
        self.value = value
        self.suggestion = suggestion
        self.timestamp = datetime.utcnow()
    
    def __repr__(self):
        return f"ValidationResult(level={self.level}, field={self.field}, message={self.message})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "level": self.level.value,
            "field": self.field,
            "message": self.message,
            "code": self.code,
            "value": self.value,
            "suggestion": self.suggestion,
            "timestamp": self.timestamp.isoformat()
        }

class ValidationContext:
    """Context for validation operations"""
    
    def __init__(self, 
                 user_id: Optional[str] = None,
                 request_id: Optional[str] = None,
                 environment: str = "development",
                 strict_mode: bool = False):
        self.user_id = user_id
        self.request_id = request_id
        self.environment = environment
        self.strict_mode = strict_mode
        self.results: List[ValidationResult] = []
        self.metadata: Dict[str, Any] = {}
    
    def add_result(self, result: ValidationResult):
        """Add validation result"""
        self.results.append(result)
        
        # Log based on severity
        if result.level == ValidationLevel.ERROR:
            logger.error(f"Validation error in {result.field}: {result.message}")
        elif result.level == ValidationLevel.WARNING:
            logger.warning(f"Validation warning in {result.field}: {result.message}")
        else:
            logger.info(f"Validation info in {result.field}: {result.message}")
    
    def add_error(self, field: str, message: str, code: Optional[str] = None, value: Any = None):
        """Add validation error"""
        self.add_result(ValidationResult(ValidationLevel.ERROR, field, message, code, value))
    
    def add_warning(self, field: str, message: str, code: Optional[str] = None, value: Any = None):
        """Add validation warning"""
        self.add_result(ValidationResult(ValidationLevel.WARNING, field, message, code, value))
    
    def add_info(self, field: str, message: str, code: Optional[str] = None, value: Any = None):
        """Add validation info"""
        self.add_result(ValidationResult(ValidationLevel.INFO, field, message, code, value))
    
    def has_errors(self) -> bool:
        """Check if there are validation errors"""
        return any(r.level == ValidationLevel.ERROR for r in self.results)
    
    def has_warnings(self) -> bool:
        """Check if there are validation warnings"""
        return any(r.level == ValidationLevel.WARNING for r in self.results)
    
    def get_errors(self) -> List[ValidationResult]:
        """Get all validation errors"""
        return [r for r in self.results if r.level == ValidationLevel.ERROR]
    
    def get_warnings(self) -> List[ValidationResult]:
        """Get all validation warnings"""
        return [r for r in self.results if r.level == ValidationLevel.WARNING]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary"""
        errors = self.get_errors()
        warnings = self.get_warnings()
        
        return {
            "total_results": len(self.results),
            "error_count": len(errors),
            "warning_count": len(warnings),
            "info_count": len(self.results) - len(errors) - len(warnings),
            "is_valid": len(errors) == 0,
            "errors": [r.to_dict() for r in errors],
            "warnings": [r.to_dict() for r in warnings]
        }

class BusinessRule(ABC):
    """Abstract base class for business rules"""
    
    def __init__(self, name: str, description: str, severity: ValidationLevel = ValidationLevel.ERROR):
        self.name = name
        self.description = description
        self.severity = severity
    
    @abstractmethod
    def validate(self, value: Any, context: ValidationContext) -> bool:
        """Validate the rule against a value"""
        logger.debug(f"Validating rule '{self.name}' against value: {value}")
        result = self._validate_rule(value, context)
        
        if not result:
            logger.warning(f"Rule '{self.name}' failed for value: {value}")
        
        return result
    
    @abstractmethod
    def _validate_rule(self, value: Any, context: ValidationContext) -> bool:
        """Internal validation logic - override in subclasses"""
        raise NotImplementedError("Subclasses must implement _validate_rule")
    
    def get_error_message(self, value: Any) -> str:
        """Get error message for failed validation"""
        return f"Business rule '{self.name}' failed: {self.description}"

class ValidatedModel(BaseModel):
    """Base model with enhanced validation"""
    
    class Config:
        validate_assignment = True
        extra = "forbid"
        use_enum_values = True
        validate_all = True
        allow_population_by_field_name = True
        
    def __init__(self, **data):
        super().__init__(**data)
        self._validation_context = ValidationContext()
        self._apply_business_rules()
    
    @property
    def validation_context(self) -> ValidationContext:
        """Get validation context"""
        return self._validation_context
    
    def _apply_business_rules(self):
        """Apply business rules to the model"""
        rules = self.get_business_rules()
        
        for field_name, field_rules in rules.items():
            if hasattr(self, field_name):
                field_value = getattr(self, field_name)
                
                for rule in field_rules:
                    if not rule.validate(field_value, self._validation_context):
                        self._validation_context.add_result(
                            ValidationResult(
                                level=rule.severity,
                                field=field_name,
                                message=rule.get_error_message(field_value),
                                code=rule.name,
                                value=field_value
                            )
                        )
    
    def get_business_rules(self) -> Dict[str, List[BusinessRule]]:
        """Get business rules for this model - override in subclasses"""
        return {}
    
    def is_valid(self) -> bool:
        """Check if model is valid"""
        return not self._validation_context.has_errors()
    
    def get_validation_errors(self) -> List[ValidationResult]:
        """Get validation errors"""
        return self._validation_context.get_errors()
    
    def get_validation_warnings(self) -> List[ValidationResult]:
        """Get validation warnings"""
        return self._validation_context.get_warnings()
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation summary"""
        return self._validation_context.get_summary()

# Common validation patterns
class ValidationPatterns:
    """Common validation patterns"""
    
    EMAIL = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    PHONE = re.compile(r'^\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$')
    SKU = re.compile(r'^[A-Z0-9]{3,20}$')
    CURRENCY = re.compile(r'^[A-Z]{3}$')
    PASSWORD_STRONG = re.compile(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$')
    
    @staticmethod
    def is_positive_decimal(value: Union[Decimal, float, int]) -> bool:
        """Check if value is a positive decimal"""
        try:
            return Decimal(str(value)) > 0
        except:
            return False
    
    @staticmethod
    def is_non_negative_decimal(value: Union[Decimal, float, int]) -> bool:
        """Check if value is a non-negative decimal"""
        try:
            return Decimal(str(value)) >= 0
        except:
            return False
    
    @staticmethod
    def is_percentage(value: Union[Decimal, float, int]) -> bool:
        """Check if value is a valid percentage (0-100)"""
        try:
            dec_value = Decimal(str(value))
            return 0 <= dec_value <= 100
        except:
            return False
    
    @staticmethod
    def is_ratio(value: Union[Decimal, float, int]) -> bool:
        """Check if value is a valid ratio (0-1)"""
        try:
            dec_value = Decimal(str(value))
            return 0 <= dec_value <= 1
        except:
            return False
    
    @staticmethod
    def is_future_date(value: Union[date, datetime]) -> bool:
        """Check if date is in the future"""
        try:
            if isinstance(value, datetime):
                return value.date() > date.today()
            return value > date.today()
        except:
            return False
    
    @staticmethod
    def is_past_date(value: Union[date, datetime]) -> bool:
        """Check if date is in the past"""
        try:
            if isinstance(value, datetime):
                return value.date() < date.today()
            return value < date.today()
        except:
            return False
    
    @staticmethod
    def is_valid_lead_time(value: int) -> bool:
        """Check if lead time is valid (1-365 days)"""
        return isinstance(value, int) and 1 <= value <= 365
    
    @staticmethod
    def is_valid_quantity(value: Union[Decimal, float, int]) -> bool:
        """Check if quantity is valid (positive number)"""
        try:
            return Decimal(str(value)) > 0
        except:
            return False

# Common business rules
class PositiveNumberRule(BusinessRule):
    """Rule to ensure number is positive"""
    
    def __init__(self, field_name: str = "value"):
        super().__init__(
            name="positive_number",
            description=f"{field_name} must be a positive number",
            severity=ValidationLevel.ERROR
        )
    
    def _validate_rule(self, value: Any, context: ValidationContext) -> bool:
        return ValidationPatterns.is_positive_decimal(value)

class NonNegativeNumberRule(BusinessRule):
    """Rule to ensure number is non-negative"""
    
    def __init__(self, field_name: str = "value"):
        super().__init__(
            name="non_negative_number",
            description=f"{field_name} must be non-negative",
            severity=ValidationLevel.ERROR
        )
    
    def _validate_rule(self, value: Any, context: ValidationContext) -> bool:
        return ValidationPatterns.is_non_negative_decimal(value)

class FutureDateRule(BusinessRule):
    """Rule to ensure date is in the future"""
    
    def __init__(self, field_name: str = "date"):
        super().__init__(
            name="future_date",
            description=f"{field_name} must be in the future",
            severity=ValidationLevel.ERROR
        )
    
    def _validate_rule(self, value: Any, context: ValidationContext) -> bool:
        return ValidationPatterns.is_future_date(value)

class ValidLeadTimeRule(BusinessRule):
    """Rule to ensure lead time is valid"""
    
    def __init__(self):
        super().__init__(
            name="valid_lead_time",
            description="Lead time must be between 1 and 365 days",
            severity=ValidationLevel.ERROR
        )
    
    def _validate_rule(self, value: Any, context: ValidationContext) -> bool:
        return ValidationPatterns.is_valid_lead_time(value)

class ReasonableQuantityRule(BusinessRule):
    """Rule to ensure quantity is reasonable"""
    
    def __init__(self, max_quantity: Decimal = Decimal('1000000')):
        self.max_quantity = max_quantity
        super().__init__(
            name="reasonable_quantity",
            description=f"Quantity should be reasonable (less than {max_quantity})",
            severity=ValidationLevel.WARNING
        )
    
    def _validate_rule(self, value: Any, context: ValidationContext) -> bool:
        try:
            return Decimal(str(value)) <= self.max_quantity
        except:
            return False

class ValidationError(Exception):
    """Custom validation error"""
    
    def __init__(self, message: str, results: List[ValidationResult] = None):
        super().__init__(message)
        self.results = results or []
    
    def get_error_details(self) -> Dict[str, Any]:
        """Get detailed error information"""
        return {
            "message": str(self),
            "results": [r.to_dict() for r in self.results],
            "error_count": len([r for r in self.results if r.level == ValidationLevel.ERROR]),
            "warning_count": len([r for r in self.results if r.level == ValidationLevel.WARNING])
        }

def validate_model(model: ValidatedModel, strict: bool = False) -> ValidationContext:
    """Validate a model and return context"""
    context = model.validation_context
    
    if strict and context.has_warnings():
        raise ValidationError(
            "Model validation failed with warnings in strict mode",
            context.results
        )
    
    if context.has_errors():
        raise ValidationError(
            "Model validation failed with errors",
            context.get_errors()
        )
    
    return context

def create_validation_decorator(rules: List[BusinessRule]):
    """Create a validation decorator for field validation"""
    def decorator(func):
        def wrapper(cls, value, values, **kwargs):
            context = ValidationContext()
            
            for rule in rules:
                if not rule.validate(value, context):
                    context.add_result(
                        ValidationResult(
                            level=rule.severity,
                            field=func.__name__,
                            message=rule.get_error_message(value),
                            code=rule.name,
                            value=value
                        )
                    )
            
            if context.has_errors():
                error_messages = [r.message for r in context.get_errors()]
                raise ValueError("; ".join(error_messages))
            
            return func(cls, value, values, **kwargs)
        return wrapper
    return decorator