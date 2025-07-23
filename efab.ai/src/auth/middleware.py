"""Authentication and Security Middleware for Beverly Knits AI Supply Chain Planner"""

import time
from typing import Optional
from fastapi import Request, Response, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import logging

from .auth_service import auth_service

logger = logging.getLogger(__name__)

class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Middleware for handling authentication and security"""
    
    def __init__(self, app, exclude_paths: Optional[list] = None):
        super().__init__(app)
        self.exclude_paths = exclude_paths or [
            "/",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/api/v1/auth/login",
            "/api/v1/auth/register",
            "/health",
            "/ping"
        ]
        self.security = HTTPBearer(auto_error=False)
    
    async def dispatch(self, request: Request, call_next):
        """Process request through authentication middleware"""
        start_time = time.time()
        
        # Skip authentication for excluded paths
        if request.url.path in self.exclude_paths:
            response = await call_next(request)
            return response
        
        # Skip authentication for preflight requests
        if request.method == "OPTIONS":
            response = await call_next(request)
            return response
        
        # Extract token from Authorization header
        authorization = request.headers.get("Authorization")
        if not authorization:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Authorization header required"}
            )
        
        try:
            # Validate token format
            if not authorization.startswith("Bearer "):
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Invalid authorization header format"}
                )
            
            token = authorization.split(" ")[1]
            
            # Verify token and get user
            user = auth_service.get_current_user(token)
            
            # Add user to request state
            request.state.user = user
            
            # Continue with request
            response = await call_next(request)
            
            # Add security headers
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            
            # Log request
            process_time = time.time() - start_time
            logger.info(
                f"Request: {request.method} {request.url.path} - "
                f"User: {user.username} - "
                f"Time: {process_time:.4f}s - "
                f"Status: {response.status_code}"
            )
            
            return response
            
        except HTTPException as e:
            logger.warning(f"Authentication failed for {request.url.path}: {e.detail}")
            return JSONResponse(
                status_code=e.status_code,
                content={"detail": e.detail}
            )
        except Exception as e:
            logger.error(f"Authentication error for {request.url.path}: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Internal authentication error"}
            )

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting"""
    
    def __init__(self, app, requests_per_minute: int = 100):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.request_counts = {}
        self.request_times = {}
    
    async def dispatch(self, request: Request, call_next):
        """Process request through rate limiting"""
        client_ip = request.client.host
        current_time = time.time()
        
        # Clean old entries
        cutoff_time = current_time - 60  # 1 minute ago
        self.request_counts = {
            ip: count for ip, count in self.request_counts.items()
            if self.request_times.get(ip, 0) > cutoff_time
        }
        
        # Check current request count
        if client_ip in self.request_counts:
            if self.request_counts[client_ip] >= self.requests_per_minute:
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={"detail": "Too many requests"}
                )
            self.request_counts[client_ip] += 1
        else:
            self.request_counts[client_ip] = 1
        
        self.request_times[client_ip] = current_time
        
        response = await call_next(request)
        return response

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for adding security headers"""
    
    async def dispatch(self, request: Request, call_next):
        """Add security headers to response"""
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        
        return response

class CORSMiddleware(BaseHTTPMiddleware):
    """Middleware for handling CORS"""
    
    def __init__(self, app, allow_origins: list = None, allow_methods: list = None):
        super().__init__(app)
        self.allow_origins = allow_origins or ["*"]
        self.allow_methods = allow_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    
    async def dispatch(self, request: Request, call_next):
        """Handle CORS headers"""
        if request.method == "OPTIONS":
            response = Response()
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allow_methods)
            response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
            response.headers["Access-Control-Max-Age"] = "86400"
            return response
        
        response = await call_next(request)
        
        # Add CORS headers
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allow_methods)
        response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
        
        return response

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests"""
    
    async def dispatch(self, request: Request, call_next):
        """Log request details"""
        start_time = time.time()
        
        # Log request
        logger.info(
            f"Request started: {request.method} {request.url.path} - "
            f"Client: {request.client.host}"
        )
        
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        logger.info(
            f"Request completed: {request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Time: {process_time:.4f}s"
        )
        
        return response

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for handling errors"""
    
    async def dispatch(self, request: Request, call_next):
        """Handle errors gracefully"""
        try:
            response = await call_next(request)
            return response
        except HTTPException as e:
            logger.warning(f"HTTP error: {e.status_code} - {e.detail}")
            return JSONResponse(
                status_code=e.status_code,
                content={"detail": e.detail}
            )
        except Exception as e:
            logger.error(f"Unhandled error: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Internal server error"}
            )