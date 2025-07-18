# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The Beverly Knits AI Supply Chain Planner is a sophisticated supply chain optimization system for textile manufacturing. It uses a **6-phase planning engine** with ML-powered forecasting and risk assessment to generate intelligent procurement recommendations.

**Key Architecture:** Domain-driven design with clean architecture patterns, featuring rich domain entities, ML integration, and a Streamlit web interface.

## Essential Commands

### Development Setup
```bash
# Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running the Application
```bash
# Launch main Streamlit application
streamlit run main.py

# Alternative: Run simple app version
streamlit run simple_app.py

# Launch FastAPI REST API server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Alternative: Run API with Python
python -m uvicorn api.main:app --reload
```

### Testing
```bash
# Run full test suite with coverage
python tests/run_tests.py

# Run specific test files
pytest tests/test_planning_engine.py -v
pytest tests/test_eoq_optimizer.py -v
pytest tests/test_data_integration.py -v
pytest tests/test_domain_entities.py -v
pytest tests/test_multi_supplier_optimizer.py -v

# Run single test function
pytest tests/test_planning_engine.py::test_execute_planning_cycle -v

# Test FastAPI endpoints
python test_api.py
```

### Data Processing
```bash
# Generate sample data for testing
python -m src.utils.sample_data_generator

# Run live data integration demo
python scripts/demo_live_data.py

# Run planning engine demo
python scripts/demo.py
```

### ML Model Training
```bash
# Train basic ML models
python train_basic_ml.py

# Train enhanced ML models
python train_enhanced_ml.py

# Train ML models directly
python train_ml_models_direct.py

# Test ML integration
python test_ml_integration.py
```

## Architecture Overview

### Core Planning Engine (6-Phase Process)
The heart of the system is `src/engine/planning_engine.py`, which executes:
1. **Forecast Unification** - Weighted consolidation of demand signals
2. **BOM Explosion** - SKU-to-material requirement conversion  
3. **Inventory Netting** - Current stock and open PO accounting
4. **Procurement Optimization** - EOQ and safety stock calculations
5. **Supplier Selection** - Multi-criteria optimization
6. **Output Generation** - Actionable recommendations with audit trails

### Domain-Driven Design Structure
- **Domain Entities** (`src/core/domain/entities.py`): Rich business objects (Material, Supplier, BOM, Forecast, ProcurementRecommendation)
- **Value Objects** (`src/core/domain/value_objects.py`): Immutable types (Money, Quantity, MaterialId, SupplierId)
- **Use Cases** (`src/core/use_cases/`): Business logic services (SupplyChainPlanningService, DataQualityService, ReportingService)

### ML Integration Layer
- **ML Model Manager** (`src/engine/ml_model_manager.py`): Orchestrates ARIMA, Prophet, LSTM, XGBoost models
- **ML Risk Assessor** (`src/engine/ml_risk_assessor.py`): Supplier risk scoring and anomaly detection
- **Forecasting Models** (`src/engine/forecasting/`): Individual ML model implementations
- **Production ML Loader** (`src/engine/production_ml_loader.py`): Production-ready model serving

### Data Integration Pipeline
- **Beverly Knits Data Integrator** (`src/data/beverly_knits_live_data_integrator.py`): Real data processing with quality fixes
- **Data Quality Fixer** (`src/utils/data_quality_fixer.py`): Automated data cleaning and normalization
- **Data Validation Pipeline** (`src/utils/data_validation_pipeline.py`): Schema validation and business rules

## Key Implementation Patterns

### Working with Domain Objects
```python
# Always import from domain module
from src.core.domain import Material, Supplier, BOM, Forecast, ProcurementRecommendation
from src.core.domain import Money, Quantity, MaterialId, SupplierId

# Create rich domain objects with validation
material = Material(
    id=MaterialId(value="YARN001"),
    name="Cotton Yarn",
    type=MaterialType.YARN,
    is_critical=True
)

# Use value objects for type safety
cost = Money(amount=Decimal("15.50"), currency="USD")
quantity = Quantity(amount=Decimal("100"), unit="pounds")
```

### Planning Engine Usage
```python
from src.engine.planning_engine import PlanningEngine
from src.data.beverly_knits_live_data_integrator import BeverlyKnitsLiveDataIntegrator

# Standard workflow
integrator = BeverlyKnitsLiveDataIntegrator(data_path="data/live/")
engine = PlanningEngine()

# Load and process data
domain_objects = integrator.integrate_live_data()

# Execute planning cycle
recommendations = engine.execute_planning_cycle(
    forecasts=domain_objects['forecasts'],
    boms=domain_objects['boms'],
    inventory=domain_objects['inventory'],
    suppliers=domain_objects['supplier_materials']
)
```

### Data Processing Pipeline
Data flows through automatic quality fixes in this order:
1. **Raw CSV Loading** - Multiple encoding support
2. **Data Quality Fixes** - Negative inventory correction, BOM normalization
3. **Domain Object Creation** - Rich objects with validation
4. **Planning Engine Processing** - 6-phase optimization
5. **Recommendation Generation** - Actionable output with reasoning

## Configuration Management

### Main Configuration
- **App Config** (`config/app_config.json`): Core application settings
- **Planning Parameters**: Source weights, safety stock, horizon, optimization toggles
- **Data Integration Settings**: File paths, validation rules, auto-fix flags
- **ML Configuration**: Model toggles, forecasting parameters

### Environment Configuration
- **Development** (`config/app_config.json`): Debug mode, local paths
- **Production** (not yet implemented): Performance optimizations, cloud settings

## Data File Structure

### Expected CSV Files (in `data/live/`)
- `Yarn_ID_1.csv` - Master yarn data
- `Yarn_ID_Current_Inventory.csv` - Current inventory levels
- `Supplier_ID.csv` - Supplier information
- `Style_BOM.csv` - Bill of materials
- `Sales Activity Report.csv` - Sales history
- `cfab_Yarn_Demand_By_Style.csv` - Demand by style
- `Yarn_Demand_2025-06-27_0442.csv` - Demand forecasts

### Automatic Data Quality Fixes
The system automatically corrects:
- Negative inventory balances → 0
- BOM percentages > 99% → Normalized to 100%
- Missing cost data → Industry-standard assignments
- Invalid data types → Converted and standardized

## Testing Strategy

### Test Categories
- **Unit Tests** (`tests/test_*.py`): Domain logic, algorithms, utilities
- **Integration Tests** (`tests/test_data_integration.py`): Data pipeline validation
- **End-to-End Tests** (`scripts/demo*.py`): Complete workflow validation

### Test Data
- **Sample Data Generator** (`src/utils/sample_data_generator.py`): Creates realistic test data
- **Live Data** (`data/live/`): Real Beverly Knits data files
- **Fixtures** (`tests/`): Test-specific data and mocks

## Development Workflow

### Adding New Features
1. **Domain First**: Update entities/value objects if needed
2. **Use Cases**: Implement business logic in use case services
3. **Engine Integration**: Add to planning engine phases
4. **UI Integration**: Update Streamlit interface
5. **Tests**: Add comprehensive test coverage

### Working with ML Models
- **Model Training**: Use `train_*.py` scripts for training
- **Model Serving**: Use `MLModelManager` for production inference
- **Model Monitoring**: Built-in performance tracking and retraining triggers

### Data Integration
- **New Data Sources**: Extend `BeverlyKnitsLiveDataIntegrator`
- **Quality Fixes**: Add fixes to `DataQualityFixer`
- **Validation**: Update `DataValidationPipeline`

## Known Limitations

### Current Implementation
- **File-based storage only** - No database persistence (DATABASE LAYER IN PROGRESS)
- **REST API implemented** - FastAPI with 15+ endpoints and OpenAPI docs
- **Basic authentication** - Token-based auth with role management
- **Manual deployment** - No CI/CD pipeline

### FastAPI REST API Layer

**Base URL:** `http://localhost:8000`  
**Documentation:** `http://localhost:8000/docs` (Swagger UI)  
**Alternative Docs:** `http://localhost:8000/redoc` (ReDoc)

**Available Endpoints:**
- `POST /api/v1/auth/login` - User authentication
- `GET /api/v1/auth/me` - Get current user info
- `GET /api/v1/materials/` - List materials with pagination
- `POST /api/v1/materials/` - Create new material
- `GET /api/v1/suppliers/` - List suppliers with filtering
- `POST /api/v1/planning/execute` - Execute planning cycle
- `POST /api/v1/forecasting/ml/generate` - Generate ML forecasts
- `GET /api/v1/analytics/dashboard` - Get analytics dashboard

**Authentication:**
```bash
# Login to get access token
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'

# Use token in subsequent requests
curl -X GET "http://localhost:8000/api/v1/materials/" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

### Future Enhancements (See AI_IMPLEMENTATION_ROADMAP.md)
- Database integration with PostgreSQL
- JWT-based authentication with bcrypt
- Performance optimization and caching
- Real-time processing capabilities

## Important Notes

### ML Model Dependencies
The system includes sophisticated ML capabilities with TensorFlow, PyTorch, Prophet, and XGBoost. Models are trained on historical data and used for demand forecasting and risk assessment.

### Data Quality Focus
The system emphasizes automatic data quality fixes and validation. Always expect data inconsistencies and let the quality pipeline handle corrections.

### Domain-Driven Design
Maintain the rich domain model. Business logic belongs in domain entities and use cases, not in UI or data access layers.

### Performance Considerations
- Large datasets processed in batches
- ML models cached for repeated inference
- Streamlit performance optimized with session state
- Database queries not yet optimized (future enhancement)