"""Logging Middleware for Beverly Knits AI Supply Chain Planner API"""

import time
import logging
import json
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import StreamingResponse
import uuid

logger = logging.getLogger(__name__)

class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests and responses"""
    
    def __init__(self, app, log_requests: bool = True, log_responses: bool = True):
        super().__init__(app)
        self.log_requests = log_requests
        self.log_responses = log_responses
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        # Log request
        if self.log_requests:
            await self._log_request(request, request_id)
        
        # Process request
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(process_time)
        
        # Log response
        if self.log_responses:
            await self._log_response(response, request_id, process_time)
        
        return response
    
    async def _log_request(self, request: Request, request_id: str):
        """Log HTTP request details"""
        try:
            # Get client IP
            client_ip = self._get_client_ip(request)
            
            # Get request headers (excluding sensitive ones)
            headers = dict(request.headers)
            sensitive_headers = {"authorization", "cookie", "x-api-key"}
            filtered_headers = {
                k: "[REDACTED]" if k.lower() in sensitive_headers else v
                for k, v in headers.items()
            }
            
            # Log request info
            log_data = {
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "client_ip": client_ip,
                "user_agent": headers.get("user-agent", "Unknown"),
                "headers": filtered_headers,
                "content_type": headers.get("content-type"),
                "content_length": headers.get("content-length")
            }
            
            logger.info(f"Request: {json.dumps(log_data)}")
            
        except Exception as e:
            logger.error(f"Error logging request: {e}")
    
    async def _log_response(self, response: Response, request_id: str, process_time: float):
        """Log HTTP response details"""
        try:
            # Get response size
            response_size = None
            if hasattr(response, 'headers') and 'content-length' in response.headers:
                response_size = response.headers['content-length']
            
            # Log response info
            log_data = {
                "request_id": request_id,
                "status_code": response.status_code,
                "content_type": response.headers.get("content-type"),
                "content_length": response_size,
                "process_time_seconds": round(process_time, 4)
            }
            
            # Log at appropriate level based on status code
            if response.status_code >= 500:
                logger.error(f"Response: {json.dumps(log_data)}")
            elif response.status_code >= 400:
                logger.warning(f"Response: {json.dumps(log_data)}")
            else:
                logger.info(f"Response: {json.dumps(log_data)}")
                
        except Exception as e:
            logger.error(f"Error logging response: {e}")
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request"""
        # Check for forwarded headers (reverse proxy)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fallback to direct connection
        if request.client:
            return request.client.host
        
        return "unknown"

class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """Enhanced middleware for structured logging with metrics"""
    
    def __init__(self, app, include_request_body: bool = False, include_response_body: bool = False):
        super().__init__(app)
        self.include_request_body = include_request_body
        self.include_response_body = include_response_body
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request context
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Add to request state
        request.state.request_id = request_id
        request.state.start_time = start_time
        
        # Capture request body if enabled
        request_body = None
        if self.include_request_body and request.method in ["POST", "PUT", "PATCH"]:
            request_body = await self._capture_request_body(request)
        
        # Process request
        response = await call_next(request)
        end_time = time.time()
        
        # Capture response body if enabled
        response_body = None
        if self.include_response_body and response.status_code < 400:
            response_body = await self._capture_response_body(response)
        
        # Log structured data
        await self._log_structured_data(
            request, response, request_id, start_time, end_time,
            request_body, response_body
        )
        
        # Add headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(round(end_time - start_time, 4))
        
        return response
    
    async def _capture_request_body(self, request: Request) -> str:
        """Capture request body for logging"""
        try:
            body = await request.body()
            if body:
                return body.decode('utf-8')[:1000]  # Limit to 1000 chars
        except Exception as e:
            logger.debug(f"Could not capture request body: {e}")
        return None
    
    async def _capture_response_body(self, response: Response) -> str:
        """Capture response body for logging"""
        try:
            if isinstance(response, StreamingResponse):
                return "[STREAMING_RESPONSE]"
            
            if hasattr(response, 'body') and response.body:
                body = response.body
                if isinstance(body, bytes):
                    return body.decode('utf-8')[:1000]  # Limit to 1000 chars
                return str(body)[:1000]
        except Exception as e:
            logger.debug(f"Could not capture response body: {e}")
        return None
    
    async def _log_structured_data(
        self, request: Request, response: Response, request_id: str,
        start_time: float, end_time: float, request_body: str, response_body: str
    ):
        """Log structured request/response data"""
        try:
            # Build structured log entry
            log_entry = {
                "timestamp": time.time(),
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "status_code": response.status_code,
                "process_time_seconds": round(end_time - start_time, 4),
                "client_ip": self._get_client_ip(request),
                "user_agent": request.headers.get("user-agent", "Unknown"),
                "content_type": request.headers.get("content-type"),
                "response_content_type": response.headers.get("content-type"),
                "success": response.status_code < 400
            }
            
            # Add request body if captured
            if request_body:
                log_entry["request_body"] = request_body
            
            # Add response body if captured
            if response_body:
                log_entry["response_body"] = response_body
            
            # Log with appropriate level
            if response.status_code >= 500:
                logger.error(f"HTTP_REQUEST: {json.dumps(log_entry)}")
            elif response.status_code >= 400:
                logger.warning(f"HTTP_REQUEST: {json.dumps(log_entry)}")
            else:
                logger.info(f"HTTP_REQUEST: {json.dumps(log_entry)}")
                
        except Exception as e:
            logger.error(f"Error in structured logging: {e}")
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request"""
        # Check for forwarded headers (reverse proxy)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fallback to direct connection
        if request.client:
            return request.client.host
        
        return "unknown"