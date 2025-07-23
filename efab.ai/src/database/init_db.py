"""Database Initialization for Beverly Knits AI Supply Chain Planner"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .connection import get_database_url, engine, SessionLocal
from .models.base import BaseModel
from .models import *  # Import all models to ensure they're registered

def create_tables():
    """Create all database tables"""
    try:
        # Create all tables
        BaseModel.metadata.create_all(bind=engine)
        print("✅ Database tables created successfully")
        return True
    except Exception as e:
        print(f"❌ Error creating database tables: {e}")
        return False

def drop_tables():
    """Drop all database tables"""
    try:
        BaseModel.metadata.drop_all(bind=engine)
        print("✅ Database tables dropped successfully")
        return True
    except Exception as e:
        print(f"❌ Error dropping database tables: {e}")
        return False

def recreate_tables():
    """Drop and recreate all database tables"""
    print("🔄 Recreating database tables...")
    drop_tables()
    return create_tables()

def check_database_connection():
    """Check if database connection is working"""
    try:
        with SessionLocal() as session:
            session.execute("SELECT 1")
        print("✅ Database connection successful")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def init_database():
    """Initialize the database with tables"""
    print("🚀 Initializing Beverly Knits AI Supply Chain Planner Database...")
    
    # Check connection
    if not check_database_connection():
        return False
    
    # Create tables
    if not create_tables():
        return False
    
    print("✅ Database initialization completed successfully")
    return True

def reset_database():
    """Reset the database (drop and recreate tables)"""
    print("🔄 Resetting Beverly Knits AI Supply Chain Planner Database...")
    
    # Check connection
    if not check_database_connection():
        return False
    
    # Recreate tables
    if not recreate_tables():
        return False
    
    print("✅ Database reset completed successfully")
    return True

def get_database_info():
    """Get database information"""
    try:
        db_url = get_database_url()
        with SessionLocal() as session:
            # Get table names
            result = session.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in result.fetchall()]
            
            # Get database size (for SQLite)
            if db_url.startswith("sqlite"):
                db_path = db_url.replace("sqlite:///", "")
                db_size = os.path.getsize(db_path) if os.path.exists(db_path) else 0
                db_size_mb = db_size / (1024 * 1024)
            else:
                db_size_mb = "N/A"
            
            return {
                "database_url": db_url,
                "tables": tables,
                "table_count": len(tables),
                "database_size_mb": db_size_mb
            }
    except Exception as e:
        print(f"❌ Error getting database info: {e}")
        return None

def print_database_info():
    """Print database information"""
    info = get_database_info()
    if info:
        print("\n📊 Database Information:")
        print(f"   URL: {info['database_url']}")
        print(f"   Tables: {info['table_count']}")
        print(f"   Table Names: {', '.join(info['tables'])}")
        print(f"   Database Size: {info['database_size_mb']:.2f} MB")
    else:
        print("❌ Could not retrieve database information")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "init":
            init_database()
        elif command == "reset":
            reset_database()
        elif command == "info":
            print_database_info()
        elif command == "check":
            check_database_connection()
        elif command == "create":
            create_tables()
        elif command == "drop":
            drop_tables()
        else:
            print("Usage: python init_db.py [init|reset|info|check|create|drop]")
    else:
        # Default action
        init_database()
        print_database_info()