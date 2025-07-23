"""Database Connection Management for Beverly Knits AI Supply Chain Planner"""

from sqlalchemy import create_engine, MetaData, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
from typing import Generator
import logging
from pathlib import Path

from ..config.settings import settings

logger = logging.getLogger(__name__)

# SQLAlchemy setup
Base = declarative_base()
metadata = MetaData()

# Global database components
engine = None
SessionLocal = None

def get_database_url() -> str:
    """Get database URL from configuration"""
    return settings.get_database_url()

def init_db() -> None:
    """Initialize database connection and create tables"""
    global engine, SessionLocal
    
    database_url = get_database_url()
    logger.info(f"Initializing database: {database_url.split('://')[0]}://...")
    
    # Create engine with appropriate settings
    if database_url.startswith("sqlite"):
        engine = create_engine(
            database_url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=settings.database.echo_queries
        )
    else:
        engine = create_engine(
            database_url,
            pool_size=settings.database.pool_size,
            max_overflow=settings.database.max_overflow,
            pool_recycle=settings.database.pool_recycle,
            pool_timeout=settings.database.pool_timeout,
            echo=settings.database.echo_queries
        )
    
    # Create session factory
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Import models to ensure they're registered
    from .models import (
        MaterialModel, SupplierModel, SupplierMaterialModel, 
        InventoryModel, BOMModel, ForecastModel, 
        ProcurementRecommendationModel, UserModel
    )
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")
    
    # Add event listeners for SQLite
    if database_url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

def get_db() -> Generator[Session, None, None]:
    """Get database session"""
    if SessionLocal is None:
        init_db()
    
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_db_context():
    """Context manager for database sessions"""
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

def get_session() -> Session:
    """Get database session - wrapper for compatibility"""
    if SessionLocal is None:
        init_db()
    return SessionLocal()

def reset_db() -> None:
    """Reset database (drop and recreate all tables)"""
    global engine
    
    if engine is None:
        init_db()
    
    logger.warning("Resetting database - all data will be lost")
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    logger.info("Database reset completed")

def get_db_stats() -> dict:
    """Get database statistics"""
    if engine is None:
        return {"status": "not_initialized"}
    
    try:
        with get_db_context() as db:
            # Import models for table inspection
            from .models import (
                MaterialModel, SupplierModel, SupplierMaterialModel,
                InventoryModel, BOMModel, ForecastModel,
                ProcurementRecommendationModel, UserModel
            )
            
            stats = {
                "status": "connected",
                "database_url": str(engine.url).replace(engine.url.password or '', '***'),
                "pool_size": engine.pool.size() if hasattr(engine.pool, 'size') else None,
                "checked_out": engine.pool.checkedout() if hasattr(engine.pool, 'checkedout') else None,
                "table_counts": {
                    "materials": db.query(MaterialModel).count(),
                    "suppliers": db.query(SupplierModel).count(),
                    "supplier_materials": db.query(SupplierMaterialModel).count(),
                    "inventory": db.query(InventoryModel).count(),
                    "boms": db.query(BOMModel).count(),
                    "forecasts": db.query(ForecastModel).count(),
                    "recommendations": db.query(ProcurementRecommendationModel).count(),
                    "users": db.query(UserModel).count(),
                }
            }
            
            return stats
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

def health_check() -> dict:
    """Database health check"""
    try:
        with get_db_context() as db:
            # Simple query to test connection
            db.execute("SELECT 1")
            return {
                "status": "healthy",
                "database_type": str(engine.url).split("://")[0] if engine else "unknown"
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }