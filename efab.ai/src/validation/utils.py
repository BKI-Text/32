"""Validation Utilities for Beverly Knits AI Supply Chain Planner"""

from typing import Dict, Any, List, Optional, Union, Type
from decimal import Decimal, DecimalError
from datetime import datetime, date
import re
import logging

from .base import ValidationContext, ValidationResult, ValidationLevel, ValidationError
from .schemas import (
    MaterialValidationSchema, SupplierValidationSchema, SupplierMaterialValidationSchema,
    InventoryValidationSchema, BOMValidationSchema, ForecastValidationSchema,
    ProcurementRecommendationValidationSchema, UserValidationSchema
)

logger = logging.getLogger(__name__)

class ValidationUtils:
    """Utility functions for validation operations"""
    
    @staticmethod
    def validate_batch_data(
        data_list: List[Dict[str, Any]], 
        schema_class: Type,
        stop_on_first_error: bool = False
    ) -> Dict[str, Any]:
        """Validate a batch of data items
        
        Args:
            data_list: List of data dictionaries to validate
            schema_class: Validation schema class to use
            stop_on_first_error: Whether to stop on first validation error
            
        Returns:
            Dictionary with validation results
        """
        results = {
            "total_items": len(data_list),
            "valid_items": [],
            "invalid_items": [],
            "errors": [],
            "warnings": [],
            "summary": {
                "valid_count": 0,
                "invalid_count": 0,
                "error_count": 0,
                "warning_count": 0
            }
        }
        
        for index, item_data in enumerate(data_list):
            try:
                # Validate individual item
                validated_model = schema_class(**item_data)
                context = validated_model.validation_context
                
                item_result = {
                    "index": index,
                    "data": item_data,
                    "is_valid": not context.has_errors(),
                    "errors": [r.to_dict() for r in context.get_errors()],
                    "warnings": [r.to_dict() for r in context.get_warnings()]
                }
                
                if context.has_errors():
                    results["invalid_items"].append(item_result)
                    results["summary"]["invalid_count"] += 1
                    results["summary"]["error_count"] += len(context.get_errors())
                    
                    if stop_on_first_error:
                        break
                else:
                    results["valid_items"].append(item_result)
                    results["summary"]["valid_count"] += 1
                
                if context.has_warnings():
                    results["summary"]["warning_count"] += len(context.get_warnings())
                
            except Exception as e:
                error_result = {
                    "index": index,
                    "data": item_data,
                    "is_valid": False,
                    "errors": [{
                        "field": "system",
                        "message": f"Validation system error: {str(e)}",
                        "code": "validation_system_error"
                    }],
                    "warnings": []
                }
                
                results["invalid_items"].append(error_result)
                results["summary"]["invalid_count"] += 1
                results["summary"]["error_count"] += 1
                
                if stop_on_first_error:
                    break
        
        return results
    
    @staticmethod
    def validate_csv_data(
        csv_data: List[Dict[str, Any]], 
        schema_class: Type,
        required_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Validate CSV data with additional column checks
        
        Args:
            csv_data: List of CSV row dictionaries
            schema_class: Validation schema class to use
            required_columns: List of required column names
            
        Returns:
            Dictionary with validation results
        """
        results = ValidationUtils.validate_batch_data(csv_data, schema_class)
        
        # Add column validation
        if required_columns and csv_data:
            missing_columns = []
            first_row = csv_data[0]
            
            for column in required_columns:
                if column not in first_row:
                    missing_columns.append(column)
            
            if missing_columns:
                results["column_errors"] = {
                    "missing_columns": missing_columns,
                    "available_columns": list(first_row.keys())
                }
        
        return results
    
    @staticmethod
    def create_validation_summary(validation_results: List[ValidationResult]) -> Dict[str, Any]:
        """Create a summary of validation results
        
        Args:
            validation_results: List of validation results
            
        Returns:
            Dictionary with validation summary
        """
        summary = {
            "total_results": len(validation_results),
            "error_count": 0,
            "warning_count": 0,
            "info_count": 0,
            "errors_by_field": {},
            "warnings_by_field": {},
            "most_common_errors": {},
            "most_common_warnings": {}
        }
        
        for result in validation_results:
            if result.level == ValidationLevel.ERROR:
                summary["error_count"] += 1
                
                # Track errors by field
                if result.field not in summary["errors_by_field"]:
                    summary["errors_by_field"][result.field] = []
                summary["errors_by_field"][result.field].append(result.message)
                
                # Track most common errors
                if result.message not in summary["most_common_errors"]:
                    summary["most_common_errors"][result.message] = 0
                summary["most_common_errors"][result.message] += 1
                
            elif result.level == ValidationLevel.WARNING:
                summary["warning_count"] += 1
                
                # Track warnings by field
                if result.field not in summary["warnings_by_field"]:
                    summary["warnings_by_field"][result.field] = []
                summary["warnings_by_field"][result.field].append(result.message)
                
                # Track most common warnings
                if result.message not in summary["most_common_warnings"]:
                    summary["most_common_warnings"][result.message] = 0
                summary["most_common_warnings"][result.message] += 1
                
            else:
                summary["info_count"] += 1
        
        # Sort most common errors and warnings
        summary["most_common_errors"] = dict(
            sorted(summary["most_common_errors"].items(), key=lambda x: x[1], reverse=True)
        )
        summary["most_common_warnings"] = dict(
            sorted(summary["most_common_warnings"].items(), key=lambda x: x[1], reverse=True)
        )
        
        return summary
    
    @staticmethod
    def format_validation_report(
        validation_results: List[ValidationResult],
        include_warnings: bool = True,
        include_info: bool = False
    ) -> str:
        """Format validation results as a human-readable report
        
        Args:
            validation_results: List of validation results
            include_warnings: Whether to include warnings in the report
            include_info: Whether to include info messages in the report
            
        Returns:
            Formatted validation report string
        """
        if not validation_results:
            return "✅ No validation issues found"
        
        summary = ValidationUtils.create_validation_summary(validation_results)
        
        lines = []
        lines.append("📋 Validation Report")
        lines.append("=" * 50)
        
        # Summary section
        lines.append(f"Total Results: {summary['total_results']}")
        lines.append(f"Errors: {summary['error_count']}")
        if include_warnings:
            lines.append(f"Warnings: {summary['warning_count']}")
        if include_info:
            lines.append(f"Info: {summary['info_count']}")
        lines.append("")
        
        # Errors section
        if summary['error_count'] > 0:
            lines.append("❌ ERRORS:")
            for field, messages in summary['errors_by_field'].items():
                lines.append(f"  • {field}:")
                for message in messages:
                    lines.append(f"    - {message}")
            lines.append("")
        
        # Warnings section
        if include_warnings and summary['warning_count'] > 0:
            lines.append("⚠️  WARNINGS:")
            for field, messages in summary['warnings_by_field'].items():
                lines.append(f"  • {field}:")
                for message in messages:
                    lines.append(f"    - {message}")
            lines.append("")
        
        # Most common issues
        if summary['most_common_errors']:
            lines.append("🔥 Most Common Errors:")
            for error, count in list(summary['most_common_errors'].items())[:5]:
                lines.append(f"  • {error} ({count} occurrences)")
            lines.append("")
        
        if include_warnings and summary['most_common_warnings']:
            lines.append("⚠️  Most Common Warnings:")
            for warning, count in list(summary['most_common_warnings'].items())[:5]:
                lines.append(f"  • {warning} ({count} occurrences)")
            lines.append("")
        
        # Overall status
        if summary['error_count'] == 0:
            if summary['warning_count'] == 0:
                lines.append("✅ Validation Status: PASSED")
            else:
                lines.append("⚠️  Validation Status: PASSED WITH WARNINGS")
        else:
            lines.append("❌ Validation Status: FAILED")
        
        return "\n".join(lines)
    
    @staticmethod
    def sanitize_input_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize input data to prevent common issues
        
        Args:
            data: Input data dictionary
            
        Returns:
            Sanitized data dictionary
        """
        sanitized = {}
        
        for key, value in data.items():
            # Skip None values
            if value is None:
                continue
            
            # Sanitize strings
            if isinstance(value, str):
                # Strip whitespace
                value = value.strip()
                
                # Skip empty strings
                if not value:
                    continue
                
                # Handle common string issues
                if value.lower() in ['null', 'none', 'n/a', 'na', '']:
                    continue
                
                # Convert string numbers to appropriate types
                if key.endswith('_qty') or key.endswith('_amount') or key.endswith('_cost'):
                    try:
                        value = Decimal(value)
                    except (ValueError, TypeError, DecimalError) as e:
                        logger.warning(f"Failed to convert {key}={value} to Decimal: {e}")
                        # Keep original value if conversion fails
                elif key.endswith('_score') or key.endswith('_rate'):
                    try:
                        value = float(value)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Failed to convert {key}={value} to float: {e}")
                        # Keep original value if conversion fails
                elif key.endswith('_days') or key.endswith('_count'):
                    try:
                        value = int(value)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Failed to convert {key}={value} to int: {e}")
                        # Keep original value if conversion fails
            
            # Sanitize numeric values
            elif isinstance(value, (int, float)):
                # Convert to appropriate type
                if key.endswith('_qty') or key.endswith('_amount') or key.endswith('_cost'):
                    value = Decimal(str(value))
                elif key.endswith('_days') or key.endswith('_count'):
                    value = int(value)
                else:
                    value = float(value)
            
            # Handle dates
            elif isinstance(value, str) and key.endswith('_date'):
                try:
                    # Try parsing common date formats
                    for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S']:
                        try:
                            parsed_date = datetime.strptime(value, fmt)
                            value = parsed_date.date()
                            break
                        except ValueError:
                            continue
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to parse date {key}={value}: {e}")
                    # Keep original value if parsing fails
            
            sanitized[key] = value
        
        return sanitized
    
    @staticmethod
    def extract_validation_errors(exception: Exception) -> List[Dict[str, Any]]:
        """Extract validation errors from various exception types
        
        Args:
            exception: The exception to extract errors from
            
        Returns:
            List of error dictionaries
        """
        errors = []
        
        if isinstance(exception, ValidationError):
            # Our custom validation error
            errors.extend([result.to_dict() for result in exception.results])
        
        elif hasattr(exception, 'errors'):
            # Pydantic validation error
            for error in exception.errors():
                errors.append({
                    "field": ".".join(str(loc) for loc in error['loc']),
                    "message": error['msg'],
                    "code": error['type'],
                    "value": error.get('input')
                })
        
        else:
            # Generic exception
            errors.append({
                "field": "unknown",
                "message": str(exception),
                "code": "generic_error",
                "value": None
            })
        
        return errors

class ValidationSchemaRegistry:
    """Registry for validation schemas"""
    
    _schemas = {
        "material": MaterialValidationSchema,
        "supplier": SupplierValidationSchema,
        "supplier_material": SupplierMaterialValidationSchema,
        "inventory": InventoryValidationSchema,
        "bom": BOMValidationSchema,
        "forecast": ForecastValidationSchema,
        "procurement_recommendation": ProcurementRecommendationValidationSchema,
        "user": UserValidationSchema,
    }
    
    @classmethod
    def get_schema(cls, schema_name: str) -> Optional[Type]:
        """Get validation schema by name"""
        return cls._schemas.get(schema_name)
    
    @classmethod
    def list_schemas(cls) -> List[str]:
        """List available schema names"""
        return list(cls._schemas.keys())
    
    @classmethod
    def register_schema(cls, name: str, schema_class: Type):
        """Register a new validation schema"""
        cls._schemas[name] = schema_class
    
    @classmethod
    def validate_data(cls, schema_name: str, data: Dict[str, Any]) -> ValidationContext:
        """Validate data using a registered schema"""
        schema_class = cls.get_schema(schema_name)
        if not schema_class:
            raise ValueError(f"Unknown validation schema: {schema_name}")
        
        context = ValidationContext()
        try:
            validated_model = schema_class(**data)
            context.results.extend(validated_model.validation_context.results)
        except Exception as e:
            context.add_error(
                field="data",
                message=f"Validation failed: {str(e)}",
                code="validation_error"
            )
        
        return context

# Convenience functions for quick validation
def validate_material_data(data: Dict[str, Any]) -> ValidationContext:
    """Quick validation for material data"""
    return ValidationSchemaRegistry.validate_data("material", data)

def validate_supplier_data(data: Dict[str, Any]) -> ValidationContext:
    """Quick validation for supplier data"""
    return ValidationSchemaRegistry.validate_data("supplier", data)

def validate_inventory_data(data: Dict[str, Any]) -> ValidationContext:
    """Quick validation for inventory data"""
    return ValidationSchemaRegistry.validate_data("inventory", data)

def validate_bom_data(data: Dict[str, Any]) -> ValidationContext:
    """Quick validation for BOM data"""
    return ValidationSchemaRegistry.validate_data("bom", data)

def validate_forecast_data(data: Dict[str, Any]) -> ValidationContext:
    """Quick validation for forecast data"""
    return ValidationSchemaRegistry.validate_data("forecast", data)

def create_validation_decorator(schema_name: str):
    """Create a validation decorator for functions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Assume first argument is data to validate
            if args:
                data = args[0]
                context = ValidationSchemaRegistry.validate_data(schema_name, data)
                if context.has_errors():
                    raise ValidationError(
                        f"Validation failed for {schema_name}",
                        context.get_errors()
                    )
            return func(*args, **kwargs)
        return wrapper
    return decorator