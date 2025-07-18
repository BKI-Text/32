"""Validation Middleware for Beverly Knits AI Supply Chain Planner"""

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Dict, Any, Optional
import logging
import json
from datetime import datetime

from .base import ValidationContext, ValidationResult, ValidationLevel, ValidationError
from .schemas import (
    MaterialValidationSchema, SupplierValidationSchema, SupplierMaterialValidationSchema,
    InventoryValidationSchema, BOMValidationSchema, ForecastValidationSchema,
    ProcurementRecommendationValidationSchema, UserValidationSchema
)

logger = logging.getLogger(__name__)

class ValidationMiddleware(BaseHTTPMiddleware):
    """Middleware to validate incoming requests"""
    
    def __init__(self, app, validation_config: Optional[Dict[str, Any]] = None):
        super().__init__(app)
        self.validation_config = validation_config or {}
        self.enabled = self.validation_config.get("enabled", True)
        self.strict_mode = self.validation_config.get("strict_mode", False)
        
        # Endpoint-specific validation schemas
        self.validation_schemas = {
            "POST /api/v1/materials/": MaterialValidationSchema,
            "PUT /api/v1/materials/{material_id}": MaterialValidationSchema,
            "POST /api/v1/suppliers/": SupplierValidationSchema,
            "PUT /api/v1/suppliers/{supplier_id}": SupplierValidationSchema,
            "POST /api/v1/suppliers/{supplier_id}/materials/": SupplierMaterialValidationSchema,
            "PUT /api/v1/suppliers/{supplier_id}/materials/{material_id}": SupplierMaterialValidationSchema,
            "POST /api/v1/inventory/": InventoryValidationSchema,
            "PUT /api/v1/inventory/{material_id}": InventoryValidationSchema,
            "POST /api/v1/bom/": BOMValidationSchema,
            "PUT /api/v1/bom/{bom_id}": BOMValidationSchema,
            "POST /api/v1/forecasting/forecasts/": ForecastValidationSchema,
            "PUT /api/v1/forecasting/forecasts/{forecast_id}": ForecastValidationSchema,
            "POST /api/v1/planning/recommendations/": ProcurementRecommendationValidationSchema,
            "PUT /api/v1/planning/recommendations/{recommendation_id}": ProcurementRecommendationValidationSchema,
            "POST /api/v1/auth/register": UserValidationSchema,
            "PUT /api/v1/auth/users/{user_id}": UserValidationSchema,
        }
    
    async def dispatch(self, request: Request, call_next):
        """Process the request and apply validation"""
        
        if not self.enabled:
            return await call_next(request)
        
        # Skip validation for certain endpoints
        if self._should_skip_validation(request):
            return await call_next(request)
        
        # Get validation schema for this endpoint
        schema_class = self._get_validation_schema(request)
        if not schema_class:
            return await call_next(request)
        
        try:
            # Read request body
            body = await request.body()
            if not body:
                return await call_next(request)
            
            # Parse JSON body
            try:
                request_data = json.loads(body.decode())
            except json.JSONDecodeError:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"detail": "Invalid JSON in request body"}
                )
            
            # Create validation context
            context = ValidationContext(
                user_id=getattr(request.state, 'user_id', None),
                request_id=getattr(request.state, 'request_id', None),
                strict_mode=self.strict_mode
            )
            
            # Validate the request data
            validation_result = self._validate_request_data(
                request_data, schema_class, context
            )
            
            # Handle validation errors
            if validation_result.has_errors():
                logger.warning(f"Validation failed for {request.method} {request.url.path}")
                return self._create_validation_error_response(validation_result)
            
            # Log validation warnings
            if validation_result.has_warnings():
                logger.warning(f"Validation warnings for {request.method} {request.url.path}")
                for warning in validation_result.get_warnings():
                    logger.warning(f"  - {warning.field}: {warning.message}")
            
            # Store validation context in request state
            request.state.validation_context = context
            
            # Reconstruct request with validated data
            request._body = json.dumps(request_data).encode()
            
        except Exception as e:
            logger.error(f"Validation middleware error: {e}")
            if self.strict_mode:
                return JSONResponse(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content={"detail": "Validation system error"}
                )
        
        return await call_next(request)
    
    def _should_skip_validation(self, request: Request) -> bool:
        """Check if validation should be skipped for this request"""
        skip_paths = [
            "/docs",
            "/redoc",
            "/openapi.json",
            "/health",
            "/metrics",
            "/favicon.ico"
        ]
        
        # Skip GET requests by default
        if request.method == "GET":
            return True
        
        # Skip specific paths
        for path in skip_paths:
            if request.url.path.startswith(path):
                return True
        
        # Skip auth login
        if request.url.path == "/api/v1/auth/login":
            return True
        
        return False
    
    def _get_validation_schema(self, request: Request) -> Optional[type]:
        """Get the validation schema for this endpoint"""
        endpoint_key = f"{request.method} {request.url.path}"
        
        # Try exact match first
        if endpoint_key in self.validation_schemas:
            return self.validation_schemas[endpoint_key]
        
        # Try pattern matching for parameterized endpoints
        for pattern, schema_class in self.validation_schemas.items():
            if self._matches_pattern(request.method, request.url.path, pattern):
                return schema_class
        
        return None
    
    def _matches_pattern(self, method: str, path: str, pattern: str) -> bool:
        """Check if method and path match the pattern"""
        pattern_method, pattern_path = pattern.split(" ", 1)
        
        if method != pattern_method:
            return False
        
        # Simple pattern matching for path parameters
        path_parts = path.split("/")
        pattern_parts = pattern_path.split("/")
        
        if len(path_parts) != len(pattern_parts):
            return False
        
        for path_part, pattern_part in zip(path_parts, pattern_parts):
            if pattern_part.startswith("{") and pattern_part.endswith("}"):
                # This is a path parameter, skip
                continue
            elif path_part != pattern_part:
                return False
        
        return True
    
    def _validate_request_data(
        self, 
        data: Dict[str, Any], 
        schema_class: type, 
        context: ValidationContext
    ) -> ValidationContext:
        """Validate request data against schema"""
        try:
            # Create validated model instance
            validated_model = schema_class(**data)
            
            # Get validation context from model
            model_context = validated_model.validation_context
            
            # Merge results into our context
            context.results.extend(model_context.results)
            
            return context
            
        except ValueError as e:
            # Pydantic validation error
            context.add_error(
                field="request_body",
                message=str(e),
                code="pydantic_validation_error"
            )
            return context
        except Exception as e:
            # Unexpected error
            context.add_error(
                field="request_body",
                message=f"Validation failed: {str(e)}",
                code="validation_system_error"
            )
            return context
    
    def _create_validation_error_response(self, context: ValidationContext) -> JSONResponse:
        """Create error response from validation context"""
        errors = []
        
        for error in context.get_errors():
            errors.append({
                "field": error.field,
                "message": error.message,
                "code": error.code,
                "value": error.value
            })
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "detail": "Validation failed",
                "errors": errors,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

class ValidationService:
    """Service for manual validation operations"""
    
    @staticmethod
    def validate_material(data: Dict[str, Any]) -> ValidationContext:
        """Validate material data"""
        return ValidationService._validate_with_schema(data, MaterialValidationSchema)
    
    @staticmethod
    def validate_supplier(data: Dict[str, Any]) -> ValidationContext:
        """Validate supplier data"""
        return ValidationService._validate_with_schema(data, SupplierValidationSchema)
    
    @staticmethod
    def validate_supplier_material(data: Dict[str, Any]) -> ValidationContext:
        """Validate supplier material data"""
        return ValidationService._validate_with_schema(data, SupplierMaterialValidationSchema)
    
    @staticmethod
    def validate_inventory(data: Dict[str, Any]) -> ValidationContext:
        """Validate inventory data"""
        return ValidationService._validate_with_schema(data, InventoryValidationSchema)
    
    @staticmethod
    def validate_bom(data: Dict[str, Any]) -> ValidationContext:
        """Validate BOM data"""
        return ValidationService._validate_with_schema(data, BOMValidationSchema)
    
    @staticmethod
    def validate_forecast(data: Dict[str, Any]) -> ValidationContext:
        """Validate forecast data"""
        return ValidationService._validate_with_schema(data, ForecastValidationSchema)
    
    @staticmethod
    def validate_procurement_recommendation(data: Dict[str, Any]) -> ValidationContext:
        """Validate procurement recommendation data"""
        return ValidationService._validate_with_schema(data, ProcurementRecommendationValidationSchema)
    
    @staticmethod
    def validate_user(data: Dict[str, Any]) -> ValidationContext:
        """Validate user data"""
        return ValidationService._validate_with_schema(data, UserValidationSchema)
    
    @staticmethod
    def _validate_with_schema(data: Dict[str, Any], schema_class: type) -> ValidationContext:
        """Validate data with the given schema"""
        context = ValidationContext()
        
        try:
            validated_model = schema_class(**data)
            context.results.extend(validated_model.validation_context.results)
        except ValueError as e:
            context.add_error(
                field="data",
                message=str(e),
                code="validation_error"
            )
        except Exception as e:
            context.add_error(
                field="data",
                message=f"Validation failed: {str(e)}",
                code="validation_system_error"
            )
        
        return context

def create_validation_middleware(app, config: Optional[Dict[str, Any]] = None):
    """Factory function to create validation middleware"""
    return ValidationMiddleware(app, config)