"""Error Handling Middleware for Beverly Knits AI Supply Chain Planner API"""

import logging
import traceback
import json
from typing import Callable
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR
from pydantic import ValidationError
import time

logger = logging.getLogger(__name__)

class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Middleware for centralized error handling"""
    
    def __init__(self, app, debug: bool = False):
        super().__init__(app)
        self.debug = debug
    
    async def dispatch(self, request: Request, call_next: Callable) -> JSONResponse:
        try:
            response = await call_next(request)
            return response
            
        except HTTPException as e:
            return await self._handle_http_exception(request, e)
            
        except ValidationError as e:
            return await self._handle_validation_error(request, e)
            
        except Exception as e:
            return await self._handle_generic_exception(request, e)
    
    async def _handle_http_exception(self, request: Request, exc: HTTPException) -> JSONResponse:
        """Handle HTTP exceptions"""
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        error_response = {
            "success": False,
            "error_code": f"HTTP_{exc.status_code}",
            "message": exc.detail,
            "timestamp": time.time(),
            "request_id": request_id,
            "path": request.url.path
        }
        
        # Log the error
        logger.warning(
            f"HTTP Exception: {exc.status_code} - {exc.detail} - "
            f"Path: {request.url.path} - Request ID: {request_id}"
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response
        )
    
    async def _handle_validation_error(self, request: Request, exc: ValidationError) -> JSONResponse:
        """Handle Pydantic validation errors"""
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        # Format validation errors
        validation_errors = []
        for error in exc.errors():
            validation_errors.append({
                "field": ".".join(str(x) for x in error["loc"]),
                "message": error["msg"],
                "type": error["type"],
                "input": error.get("input")
            })
        
        error_response = {
            "success": False,
            "error_code": "VALIDATION_ERROR",
            "message": "Validation failed",
            "validation_errors": validation_errors,
            "timestamp": time.time(),
            "request_id": request_id,
            "path": request.url.path
        }
        
        # Log the validation error
        logger.warning(
            f"Validation Error: {len(validation_errors)} errors - "
            f"Path: {request.url.path} - Request ID: {request_id} - "
            f"Errors: {json.dumps(validation_errors)}"
        )
        
        return JSONResponse(
            status_code=422,
            content=error_response
        )
    
    async def _handle_generic_exception(self, request: Request, exc: Exception) -> JSONResponse:
        """Handle generic exceptions"""
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        # Get exception details
        exc_name = type(exc).__name__
        exc_message = str(exc)
        
        error_response = {
            "success": False,
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": "An internal server error occurred",
            "timestamp": time.time(),
            "request_id": request_id,
            "path": request.url.path
        }
        
        # Add debug information if enabled
        if self.debug:
            error_response["debug"] = {
                "exception_type": exc_name,
                "exception_message": exc_message,
                "traceback": traceback.format_exc()
            }
        
        # Log the error
        logger.error(
            f"Unhandled Exception: {exc_name} - {exc_message} - "
            f"Path: {request.url.path} - Request ID: {request_id}\n"
            f"Traceback: {traceback.format_exc()}"
        )
        
        return JSONResponse(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response
        )

class DetailedErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Enhanced error handling middleware with detailed error tracking"""
    
    def __init__(self, app, debug: bool = False, track_errors: bool = True):
        super().__init__(app)
        self.debug = debug
        self.track_errors = track_errors
        self.error_counts = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> JSONResponse:
        try:
            response = await call_next(request)
            return response
            
        except HTTPException as e:
            await self._track_error("HTTP_EXCEPTION", str(e.status_code))
            return await self._create_error_response(request, e, "HTTP_EXCEPTION")
            
        except ValidationError as e:
            await self._track_error("VALIDATION_ERROR", "validation_failed")
            return await self._create_validation_error_response(request, e)
            
        except ConnectionError as e:
            await self._track_error("CONNECTION_ERROR", "connection_failed")
            return await self._create_error_response(request, e, "CONNECTION_ERROR")
            
        except TimeoutError as e:
            await self._track_error("TIMEOUT_ERROR", "request_timeout")
            return await self._create_error_response(request, e, "TIMEOUT_ERROR")
            
        except FileNotFoundError as e:
            await self._track_error("FILE_NOT_FOUND", "file_missing")
            return await self._create_error_response(request, e, "FILE_NOT_FOUND")
            
        except PermissionError as e:
            await self._track_error("PERMISSION_ERROR", "access_denied")
            return await self._create_error_response(request, e, "PERMISSION_ERROR")
            
        except Exception as e:
            await self._track_error("UNHANDLED_EXCEPTION", type(e).__name__)
            return await self._create_error_response(request, e, "INTERNAL_SERVER_ERROR")
    
    async def _track_error(self, error_type: str, error_subtype: str):
        """Track error occurrences"""
        if not self.track_errors:
            return
        
        key = f"{error_type}:{error_subtype}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1
        
        # Log error statistics periodically
        if self.error_counts[key] % 10 == 0:
            logger.warning(f"Error pattern detected: {key} occurred {self.error_counts[key]} times")
    
    async def _create_error_response(self, request: Request, exc: Exception, error_type: str) -> JSONResponse:
        """Create standardized error response"""
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        # Determine status code
        if isinstance(exc, HTTPException):
            status_code = exc.status_code
            message = exc.detail
        elif error_type == "CONNECTION_ERROR":
            status_code = 503
            message = "Service temporarily unavailable"
        elif error_type == "TIMEOUT_ERROR":
            status_code = 504
            message = "Request timeout"
        elif error_type == "FILE_NOT_FOUND":
            status_code = 404
            message = "Resource not found"
        elif error_type == "PERMISSION_ERROR":
            status_code = 403
            message = "Access denied"
        else:
            status_code = 500
            message = "Internal server error"
        
        error_response = {
            "success": False,
            "error_code": error_type,
            "message": message,
            "timestamp": time.time(),
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method
        }
        
        # Add debug information if enabled
        if self.debug:
            error_response["debug"] = {
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc()
            }
        
        # Add error context
        error_response["context"] = {
            "user_agent": request.headers.get("user-agent", "Unknown"),
            "client_ip": self._get_client_ip(request),
            "query_params": dict(request.query_params)
        }
        
        # Log the error
        log_message = (
            f"Error: {error_type} - {message} - "
            f"Path: {request.url.path} - Method: {request.method} - "
            f"Request ID: {request_id} - Client: {self._get_client_ip(request)}"
        )
        
        if status_code >= 500:
            logger.error(f"{log_message}\nTraceback: {traceback.format_exc()}")
        else:
            logger.warning(log_message)
        
        return JSONResponse(
            status_code=status_code,
            content=error_response
        )
    
    async def _create_validation_error_response(self, request: Request, exc: ValidationError) -> JSONResponse:
        """Create validation error response"""
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        # Format validation errors
        validation_errors = []
        for error in exc.errors():
            validation_errors.append({
                "field": ".".join(str(x) for x in error["loc"]),
                "message": error["msg"],
                "type": error["type"],
                "input": error.get("input")
            })
        
        error_response = {
            "success": False,
            "error_code": "VALIDATION_ERROR",
            "message": "Request validation failed",
            "validation_errors": validation_errors,
            "timestamp": time.time(),
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method
        }
        
        # Log validation error
        logger.warning(
            f"Validation Error: {len(validation_errors)} errors - "
            f"Path: {request.url.path} - Request ID: {request_id}"
        )
        
        return JSONResponse(
            status_code=422,
            content=error_response
        )
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request"""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def get_error_statistics(self) -> dict:
        """Get error statistics"""
        return dict(self.error_counts)