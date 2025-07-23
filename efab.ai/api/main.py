"""
FastAPI Application - Beverly Knits AI Supply Chain Planner API
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import uvicorn
from pathlib import Path
import sys
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.routers import auth, materials, suppliers, planning, forecasting, analytics
from api.websocket_endpoints import websocket_router
from api.middleware.logging_middleware import LoggingMiddleware
from api.middleware.error_middleware import ErrorHandlerMiddleware
from src.config.config_manager import ConfigManager
from src.auth.middleware import (
    SecurityHeadersMiddleware,
    RateLimitMiddleware,
    RequestLoggingMiddleware,
    ErrorHandlingMiddleware
)
from src.validation.middleware import ValidationMiddleware

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize configuration
config = ConfigManager()

# Initialize database
from src.database.connection import init_db
init_db()

# Create FastAPI application
app = FastAPI(
    title="Beverly Knits AI Supply Chain Planner API",
    description="Intelligent supply chain optimization API for textile manufacturing",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
app.add_middleware(ErrorHandlerMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RateLimitMiddleware, requests_per_minute=100)
app.add_middleware(RequestLoggingMiddleware)

# Add validation middleware
validation_config = {
    "enabled": True,
    "strict_mode": False
}
app.add_middleware(ValidationMiddleware, validation_config=validation_config)

# Include routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(materials.router, prefix="/api/v1/materials", tags=["Materials"])
app.include_router(suppliers.router, prefix="/api/v1/suppliers", tags=["Suppliers"])
app.include_router(planning.router, prefix="/api/v1/planning", tags=["Planning"])
app.include_router(forecasting.router, prefix="/api/v1/forecasting", tags=["Forecasting"])
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["Analytics"])
app.include_router(websocket_router, prefix="/api/v1", tags=["WebSocket"])

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Beverly Knits AI Supply Chain Planner API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    from src.database.connection import health_check as db_health_check
    
    db_status = db_health_check()
    
    return {
        "status": "healthy",
        "timestamp": "2025-01-17T18:00:00Z",
        "version": "1.0.0",
        "services": {
            "api": "operational",
            "database": db_status["status"],
            "ml_models": "operational"
        }
    }

@app.get("/api/v1/info")
async def api_info():
    """API information endpoint"""
    return {
        "api_name": "Beverly Knits AI Supply Chain Planner",
        "version": "1.0.0",
        "description": "Intelligent supply chain optimization for textile manufacturing",
        "endpoints": {
            "authentication": "/api/v1/auth",
            "materials": "/api/v1/materials",
            "suppliers": "/api/v1/suppliers",
            "planning": "/api/v1/planning",
            "forecasting": "/api/v1/forecasting",
            "analytics": "/api/v1/analytics"
        }
    }

# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Beverly Knits AI Supply Chain Planner API",
        version="1.0.0",
        description="Intelligent supply chain optimization API for textile manufacturing",
        routes=app.routes,
    )
    
    # Add custom schema information
    openapi_schema["info"]["x-logo"] = {
        "url": "https://example.com/logo.png"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )