# 🗄️ Database Integration Layer Plan

**Version:** 1.0.0  
**Last Updated:** January 2025  
**Status:** Planning Phase  
**Document Type:** Technical Implementation Plan

---

## 📋 Overview

This document outlines the plan for implementing a database integration layer for the Beverly Knits AI Supply Chain Optimization Planner, transitioning from file-based CSV processing to a robust database-driven architecture.

## 🎯 Current State Analysis

### ✅ **Current File-Based Implementation**
- **Data Storage**: CSV files in `data/live/` directory
- **Processing**: Pandas-based data manipulation
- **Integration**: `BeverlyKnitsLiveDataIntegrator` class
- **Advantages**: Simple, no database setup required
- **Limitations**: No concurrent access, limited scalability, no data relationships

### 🔄 **Target Database Architecture**
- **Primary Database**: PostgreSQL for transactional data
- **Analytics Database**: TimescaleDB for time-series data
- **Caching Layer**: Redis for session data and frequently accessed queries
- **ORM**: SQLAlchemy for Python database abstraction
- **Migration**: Alembic for database schema versioning

---

## 🏗️ Database Schema Design

### Core Tables

#### 1. Materials Table
```sql
CREATE TABLE materials (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    material_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    type VARCHAR(50) NOT NULL,
    description TEXT,
    specifications JSONB,
    is_critical BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

#### 2. Suppliers Table
```sql
CREATE TABLE suppliers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    supplier_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    contact_info TEXT,
    default_lead_time_days INTEGER,
    reliability_score DECIMAL(3,2) CHECK (reliability_score >= 0 AND reliability_score <= 1),
    risk_level VARCHAR(20) CHECK (risk_level IN ('low', 'medium', 'high')),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

#### 3. Supplier Materials Table
```sql
CREATE TABLE supplier_materials (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    supplier_id UUID REFERENCES suppliers(id),
    material_id UUID REFERENCES materials(id),
    cost_per_unit DECIMAL(10,2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    moq_amount DECIMAL(10,2) NOT NULL,
    moq_unit VARCHAR(20) NOT NULL,
    lead_time_days INTEGER NOT NULL,
    contract_qty_limit DECIMAL(10,2),
    reliability_score DECIMAL(3,2),
    ordering_cost DECIMAL(10,2) DEFAULT 100.00,
    holding_cost_rate DECIMAL(3,2) DEFAULT 0.25,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(supplier_id, material_id)
);
```

#### 4. Inventory Table
```sql
CREATE TABLE inventory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    material_id UUID REFERENCES materials(id),
    on_hand_qty DECIMAL(10,2) NOT NULL,
    on_hand_unit VARCHAR(20) NOT NULL,
    open_po_qty DECIMAL(10,2) DEFAULT 0,
    open_po_unit VARCHAR(20),
    po_expected_date DATE,
    safety_stock_qty DECIMAL(10,2) DEFAULT 0,
    safety_stock_unit VARCHAR(20),
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(material_id)
);
```

#### 5. BOMs Table
```sql
CREATE TABLE boms (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sku_id VARCHAR(50) NOT NULL,
    material_id UUID REFERENCES materials(id),
    qty_per_unit DECIMAL(10,4) NOT NULL,
    unit VARCHAR(20) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(sku_id, material_id)
);
```

#### 6. Forecasts Table
```sql
CREATE TABLE forecasts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sku_id VARCHAR(50) NOT NULL,
    forecast_qty DECIMAL(10,2) NOT NULL,
    forecast_unit VARCHAR(20) NOT NULL,
    forecast_date DATE NOT NULL,
    source VARCHAR(50) NOT NULL,
    confidence_score DECIMAL(3,2) DEFAULT 0.8,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

#### 7. Procurement Recommendations Table
```sql
CREATE TABLE procurement_recommendations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    material_id UUID REFERENCES materials(id),
    supplier_id UUID REFERENCES suppliers(id),
    recommended_order_qty DECIMAL(10,2) NOT NULL,
    recommended_order_unit VARCHAR(20) NOT NULL,
    unit_cost DECIMAL(10,2) NOT NULL,
    total_cost DECIMAL(10,2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    expected_lead_time_days INTEGER,
    risk_flag VARCHAR(20),
    reasoning TEXT,
    urgency_score DECIMAL(3,2),
    planning_cycle_id UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### Time-Series Tables (TimescaleDB)

#### 8. Sales Data Table
```sql
CREATE TABLE sales_data (
    time TIMESTAMP WITH TIME ZONE NOT NULL,
    sku_id VARCHAR(50) NOT NULL,
    quantity DECIMAL(10,2) NOT NULL,
    unit VARCHAR(20) NOT NULL,
    unit_price DECIMAL(10,2),
    total_value DECIMAL(10,2),
    region VARCHAR(50),
    customer_segment VARCHAR(50)
);

SELECT create_hypertable('sales_data', 'time');
```

#### 9. Demand History Table
```sql
CREATE TABLE demand_history (
    time TIMESTAMP WITH TIME ZONE NOT NULL,
    material_id UUID REFERENCES materials(id),
    demand_qty DECIMAL(10,2) NOT NULL,
    demand_unit VARCHAR(20) NOT NULL,
    source VARCHAR(50),
    actual_vs_forecast DECIMAL(10,2)
);

SELECT create_hypertable('demand_history', 'time');
```

---

## 🔧 Repository Pattern Implementation

### Base Repository
```python
# src/infrastructure/database/base_repository.py
from abc import ABC, abstractmethod
from typing import List, Optional, TypeVar, Generic
from sqlalchemy.orm import Session

T = TypeVar('T')

class BaseRepository(Generic[T], ABC):
    def __init__(self, session: Session, model_class: type):
        self.session = session
        self.model_class = model_class
    
    def get_by_id(self, id: str) -> Optional[T]:
        return self.session.query(self.model_class).filter(
            self.model_class.id == id
        ).first()
    
    def get_all(self) -> List[T]:
        return self.session.query(self.model_class).all()
    
    def create(self, entity: T) -> T:
        self.session.add(entity)
        self.session.commit()
        self.session.refresh(entity)
        return entity
    
    def update(self, entity: T) -> T:
        self.session.commit()
        self.session.refresh(entity)
        return entity
    
    def delete(self, entity: T) -> None:
        self.session.delete(entity)
        self.session.commit()
```

### Material Repository
```python
# src/infrastructure/database/material_repository.py
from typing import List, Optional
from sqlalchemy.orm import Session
from .base_repository import BaseRepository
from .models import MaterialModel
from ...core.domain.entities import Material

class MaterialRepository(BaseRepository[MaterialModel]):
    def __init__(self, session: Session):
        super().__init__(session, MaterialModel)
    
    def get_by_material_id(self, material_id: str) -> Optional[MaterialModel]:
        return self.session.query(MaterialModel).filter(
            MaterialModel.material_id == material_id
        ).first()
    
    def get_by_type(self, material_type: str) -> List[MaterialModel]:
        return self.session.query(MaterialModel).filter(
            MaterialModel.type == material_type
        ).all()
    
    def get_critical_materials(self) -> List[MaterialModel]:
        return self.session.query(MaterialModel).filter(
            MaterialModel.is_critical == True
        ).all()
```

### Supplier Repository
```python
# src/infrastructure/database/supplier_repository.py
from typing import List, Optional
from sqlalchemy.orm import Session
from .base_repository import BaseRepository
from .models import SupplierModel

class SupplierRepository(BaseRepository[SupplierModel]):
    def __init__(self, session: Session):
        super().__init__(session, SupplierModel)
    
    def get_by_supplier_id(self, supplier_id: str) -> Optional[SupplierModel]:
        return self.session.query(SupplierModel).filter(
            SupplierModel.supplier_id == supplier_id
        ).first()
    
    def get_active_suppliers(self) -> List[SupplierModel]:
        return self.session.query(SupplierModel).filter(
            SupplierModel.is_active == True
        ).all()
    
    def get_by_risk_level(self, risk_level: str) -> List[SupplierModel]:
        return self.session.query(SupplierModel).filter(
            SupplierModel.risk_level == risk_level
        ).all()
```

---

## 📊 Data Migration Strategy

### Phase 1: Database Setup
1. **PostgreSQL Installation** - Docker container setup
2. **Schema Creation** - Run DDL scripts
3. **Index Creation** - Performance optimization
4. **Connection Setup** - SQLAlchemy configuration

### Phase 2: Data Migration
1. **CSV to Database** - Migrate existing CSV data
2. **Data Validation** - Ensure data integrity
3. **Relationship Mapping** - Establish foreign key relationships
4. **Historical Data** - Load historical sales/demand data

### Phase 3: Application Updates
1. **Repository Implementation** - Replace CSV processing
2. **Service Layer Updates** - Update use cases
3. **API Integration** - Database-backed APIs
4. **Testing** - Comprehensive database testing

---

## 🔄 Implementation Phases

### Phase 1: Core Database Setup (2-3 weeks)
- PostgreSQL setup with Docker
- Base schema creation
- SQLAlchemy models
- Repository pattern implementation
- Basic CRUD operations

### Phase 2: Data Migration (1-2 weeks)
- CSV to database migration scripts
- Data validation and cleanup
- Relationship establishment
- Historical data loading

### Phase 3: Application Integration (2-3 weeks)
- Update data integrator
- Modify planning engine
- Update Streamlit UI
- Comprehensive testing

### Phase 4: Advanced Features (1-2 weeks)
- Query optimization
- Caching implementation
- Performance monitoring
- Documentation updates

---

## 📈 Performance Considerations

### Database Optimization
- **Indexing Strategy**: Primary keys, foreign keys, and query-specific indexes
- **Query Optimization**: Optimized queries with proper joins
- **Connection Pooling**: SQLAlchemy connection pool configuration
- **Caching**: Redis for frequently accessed data

### Scalability
- **Horizontal Scaling**: Read replicas for analytics queries
- **Partitioning**: Time-based partitioning for historical data
- **Archiving**: Automated archiving of old data
- **Monitoring**: Database performance monitoring

---

## 🧪 Testing Strategy

### Database Testing
```python
# tests/infrastructure/test_repositories.py
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.infrastructure.database.models import Base
from src.infrastructure.database.material_repository import MaterialRepository

@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()

def test_material_repository_crud(db_session):
    repo = MaterialRepository(db_session)
    # Test create, read, update, delete operations
    pass
```

### Integration Testing
- **Data Migration Tests**: Verify CSV to database migration
- **Repository Tests**: CRUD operations testing
- **Service Layer Tests**: Business logic with database
- **Performance Tests**: Query performance validation

---

## 📋 Configuration Updates

### Database Configuration
```python
# src/config/database_config.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    database: str = "beverly_knits"
    username: str = "postgres"
    password: str = ""
    pool_size: int = 5
    max_overflow: int = 10
    echo: bool = False
    
    @property
    def url(self) -> str:
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
```

### Application Configuration Updates
```json
{
  "database": {
    "host": "localhost",
    "port": 5432,
    "database": "beverly_knits",
    "username": "postgres",
    "password": "",
    "pool_size": 5,
    "max_overflow": 10,
    "echo": false
  },
  "redis": {
    "host": "localhost",
    "port": 6379,
    "db": 0,
    "password": ""
  }
}
```

---

## 📚 Next Steps

1. **Docker Setup** - PostgreSQL and Redis containers
2. **Schema Implementation** - Create database schema
3. **Model Development** - SQLAlchemy models
4. **Repository Implementation** - Data access layer
5. **Migration Scripts** - CSV to database migration
6. **Testing Framework** - Database testing setup
7. **Documentation** - Database usage guides

---

*This plan will be updated as database integration progresses and requirements evolve.*