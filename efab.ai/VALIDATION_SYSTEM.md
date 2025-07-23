# Beverly Knits AI Supply Chain Planner - Validation System

## Overview

The Beverly Knits AI Supply Chain Planner includes a comprehensive validation system that ensures data integrity and business rule compliance across all system components. The validation system is built using Pydantic and provides multi-layered validation with business rules, field validation, and API request/response validation.

## Architecture

### Core Components

#### 1. Base Validation Framework (`src/validation/base.py`)
- **ValidationLevel**: Enum for validation severity (ERROR, WARNING, INFO)
- **ValidationResult**: Container for validation results with metadata
- **ValidationContext**: Context manager for validation operations
- **BusinessRule**: Abstract base class for business rules
- **ValidatedModel**: Enhanced Pydantic BaseModel with business rule support
- **ValidationPatterns**: Common regex patterns and validation utilities

#### 2. Validation Schemas (`src/validation/schemas.py`)
- **MaterialValidationSchema**: Validates material data
- **SupplierValidationSchema**: Validates supplier data
- **SupplierMaterialValidationSchema**: Validates supplier-material relationships
- **InventoryValidationSchema**: Validates inventory data
- **BOMValidationSchema**: Validates Bill of Materials data
- **ForecastValidationSchema**: Validates demand forecast data
- **ProcurementRecommendationValidationSchema**: Validates procurement recommendations
- **UserValidationSchema**: Validates user data

#### 3. Validation Middleware (`src/validation/middleware.py`)
- **ValidationMiddleware**: FastAPI middleware for automatic request validation
- **ValidationService**: Service for manual validation operations
- Automatic endpoint-schema mapping
- Error response formatting

#### 4. Validation Utilities (`src/validation/utils.py`)
- **ValidationUtils**: Utility functions for batch validation, CSV validation, reporting
- **ValidationSchemaRegistry**: Registry for validation schemas
- Convenience functions for quick validation
- Data sanitization utilities

## Features

### 1. Multi-Level Validation

#### Pydantic Field Validation
```python
class MaterialValidationSchema(ValidatedModel):
    id: str = Field(..., description="Material ID")
    name: str = Field(..., min_length=2, max_length=255, description="Material name")
    type: str = Field(..., description="Material type")
    
    @validator('id')
    def validate_id(cls, v):
        if not ValidationPatterns.SKU.match(v):
            raise ValueError("Material ID must be valid SKU format")
        return v
```

#### Business Rule Validation
```python
class ValidSkuRule(BusinessRule):
    def validate(self, value: str, context: ValidationContext) -> bool:
        return bool(ValidationPatterns.SKU.match(value))

class MaterialValidationSchema(ValidatedModel):
    def get_business_rules(self) -> Dict[str, List[BusinessRule]]:
        return {
            "id": [ValidSkuRule()],
            "name": [ValidSupplierNameRule()],
        }
```

### 2. API Request Validation

The ValidationMiddleware automatically validates incoming API requests:

```python
# Automatically validates POST/PUT requests
POST /api/v1/materials/
{
    "id": "YARN001",
    "name": "Cotton Yarn",
    "type": "yarn"
}

# Returns 422 with validation errors for invalid data
{
    "detail": "Validation failed",
    "errors": [
        {
            "field": "id",
            "message": "Material ID must be valid SKU format",
            "code": "valid_sku",
            "value": "invalid-id"
        }
    ],
    "timestamp": "2025-01-17T18:00:00Z"
}
```

### 3. Validation Patterns

#### Common Patterns
- **SKU Format**: `^[A-Z0-9]{3,20}$`
- **Email Format**: Standard email regex
- **Currency Code**: `^[A-Z]{3}$`
- **Phone Number**: US phone number format

#### Validation Utilities
- **is_positive_decimal()**: Validates positive numbers
- **is_non_negative_decimal()**: Validates non-negative numbers
- **is_percentage()**: Validates percentages (0-100)
- **is_ratio()**: Validates ratios (0-1)
- **is_future_date()**: Validates future dates
- **is_valid_lead_time()**: Validates lead times (1-365 days)

### 4. Business Rules

#### Pre-built Rules
- **PositiveNumberRule**: Ensures numbers are positive
- **NonNegativeNumberRule**: Ensures numbers are non-negative
- **FutureDateRule**: Ensures dates are in the future
- **ValidLeadTimeRule**: Validates lead times
- **ReasonableQuantityRule**: Warns about unusually large quantities
- **ValidSkuRule**: Validates SKU format
- **ValidSupplierNameRule**: Validates supplier names
- **ReasonableCostRule**: Warns about unusually high costs

#### Custom Rules
```python
class CustomBusinessRule(BusinessRule):
    def __init__(self):
        super().__init__(
            name="custom_rule",
            description="Custom business rule",
            severity=ValidationLevel.WARNING
        )
    
    def validate(self, value: Any, context: ValidationContext) -> bool:
        # Custom validation logic
        return True
```

## Usage Examples

### 1. Direct Validation
```python
from src.validation import MaterialValidationSchema

# Create and validate material
material_data = {
    "id": "YARN001",
    "name": "Cotton Yarn",
    "type": "yarn",
    "is_critical": True
}

validated_material = MaterialValidationSchema(**material_data)

# Check validation results
if validated_material.is_valid():
    print("✅ Material data is valid")
else:
    for error in validated_material.get_validation_errors():
        print(f"❌ {error.field}: {error.message}")
```

### 2. Convenience Functions
```python
from src.validation import validate_material_data, validate_supplier_data

# Quick validation
context = validate_material_data(material_data)
if context.has_errors():
    print("Validation failed")
```

### 3. Batch Validation
```python
from src.validation import ValidationUtils

# Validate multiple items
materials = [material1, material2, material3]
results = ValidationUtils.validate_batch_data(
    materials, 
    MaterialValidationSchema,
    stop_on_first_error=False
)

print(f"Valid items: {results['summary']['valid_count']}")
print(f"Invalid items: {results['summary']['invalid_count']}")
```

### 4. CSV Validation
```python
# Validate CSV data
csv_data = [
    {"id": "YARN001", "name": "Cotton Yarn", "type": "yarn"},
    {"id": "YARN002", "name": "Wool Yarn", "type": "yarn"}
]

results = ValidationUtils.validate_csv_data(
    csv_data,
    MaterialValidationSchema,
    required_columns=["id", "name", "type"]
)
```

### 5. Validation Reports
```python
from src.validation import ValidationUtils

# Generate validation report
report = ValidationUtils.format_validation_report(
    validation_results,
    include_warnings=True,
    include_info=False
)
print(report)
```

## Configuration

### API Validation Configuration
```python
# In api/main.py
validation_config = {
    "enabled": True,
    "strict_mode": False  # If True, warnings also cause errors
}
app.add_middleware(ValidationMiddleware, validation_config=validation_config)
```

### Schema Registry
```python
from src.validation import ValidationSchemaRegistry

# Register custom schema
ValidationSchemaRegistry.register_schema("custom", CustomValidationSchema)

# Use registered schema
context = ValidationSchemaRegistry.validate_data("custom", data)
```

## API Integration

### Automatic Validation
The ValidationMiddleware automatically validates:
- POST requests to create endpoints
- PUT requests to update endpoints
- Specific endpoints based on URL patterns

### Skipped Endpoints
- GET requests (read operations)
- Health check endpoints
- Documentation endpoints
- Authentication login endpoint

### Error Responses
```json
{
    "detail": "Validation failed",
    "errors": [
        {
            "field": "field_name",
            "message": "Validation error message",
            "code": "validation_code",
            "value": "invalid_value"
        }
    ],
    "timestamp": "2025-01-17T18:00:00Z"
}
```

## Testing

### Unit Tests
```bash
# Test validation system
python3 test_validation.py
```

### API Validation Tests
```bash
# Start API server
uvicorn api.main:app --reload

# Test validation middleware
python3 test_api_validation.py
```

## Business Entity Validation

### Material Validation
- **ID**: Must be valid SKU format (3-20 alphanumeric characters)
- **Name**: 2-255 characters, no leading/trailing spaces
- **Type**: Must be one of: yarn, fabric, thread, accessory, trim
- **Specifications**: String key-value pairs, limited length

### Supplier Validation
- **ID**: Must be valid SKU format
- **Name**: 2-255 characters
- **Lead Time**: 1-365 days
- **Reliability Score**: 0.0-1.0
- **Risk Level**: low, medium, high

### Inventory Validation
- **Material ID**: Must be valid SKU format
- **Quantities**: Must be non-negative
- **Units**: Must be from allowed list
- **PO Expected Date**: Must be future date if PO quantity > 0

### BOM Validation
- **SKU ID**: Must be valid SKU format
- **Material ID**: Must be valid SKU format
- **Quantity per Unit**: Must be positive
- **Waste Percentage**: 0-50%, warns if > 25%
- **Efficiency Factor**: 0.5-2.0, warns if < 0.5

### Forecast Validation
- **SKU ID**: Must be valid SKU format
- **Forecast Quantity**: Must be positive
- **Forecast Date**: Cannot be more than 30 days in past
- **Confidence Score**: 0.0-1.0
- **Source**: Must be from allowed list

### Procurement Recommendation Validation
- **Material ID**: Must be valid SKU format
- **Supplier ID**: Must be valid SKU format
- **Quantities**: Must be positive
- **Dates**: Required by date must be after recommendation date
- **Urgency Level**: low, medium, high, critical
- **Status**: pending, approved, rejected, ordered

### User Validation
- **Username**: 3-50 characters, alphanumeric with underscores/hyphens
- **Email**: Must be valid email format
- **Role**: admin, manager, user, viewer
- **Permissions**: Must be from predefined list

## Best Practices

### 1. Validation Design
- Use appropriate validation levels (ERROR vs WARNING)
- Provide clear, actionable error messages
- Include suggestions for fixing validation errors
- Use business rules for domain-specific validation

### 2. Performance
- Batch validation for large datasets
- Use stop_on_first_error for early termination
- Cache validation results when appropriate
- Validate at API boundaries

### 3. Error Handling
- Always handle validation errors gracefully
- Log validation failures for monitoring
- Provide user-friendly error messages
- Include field-specific error information

### 4. Testing
- Test both valid and invalid data
- Test edge cases and boundary conditions
- Test business rule validation
- Test API validation middleware

## Extension Points

### Adding New Validation Schemas
1. Create new schema class inheriting from ValidatedModel
2. Add field validators and business rules
3. Register schema in ValidationSchemaRegistry
4. Update middleware endpoint mapping

### Adding New Business Rules
1. Create new rule class inheriting from BusinessRule
2. Implement validate() method
3. Add to schema's get_business_rules() method
4. Test rule functionality

### Custom Validation Middleware
1. Create custom middleware class
2. Implement validation logic
3. Add to FastAPI application
4. Configure validation settings

## Monitoring and Logging

### Validation Metrics
- Validation success/failure rates
- Most common validation errors
- Validation performance metrics
- Business rule violation trends

### Logging
- Validation errors logged at ERROR level
- Validation warnings logged at WARNING level
- Validation info logged at INFO level
- Request-specific validation context

## Security Considerations

### Input Sanitization
- Sanitize input data before validation
- Handle malicious input gracefully
- Validate all user inputs
- Use parameterized queries

### Data Protection
- Don't log sensitive validation data
- Mask sensitive fields in error messages
- Use secure validation patterns
- Validate against injection attacks

## Conclusion

The Beverly Knits AI Supply Chain Planner validation system provides comprehensive, multi-layered validation for all system components. It ensures data integrity, enforces business rules, and provides clear feedback for validation failures. The system is designed to be extensible, performant, and easy to use for both developers and API consumers.