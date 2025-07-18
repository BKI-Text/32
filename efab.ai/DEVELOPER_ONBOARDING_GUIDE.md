# Beverly Knits AI Supply Chain Planner - Developer Onboarding Guide

Welcome to the Beverly Knits AI Supply Chain Planner development team! This guide will help you get up to speed with our sophisticated supply chain optimization system.

## 🎯 System Overview

The Beverly Knits AI Supply Chain Planner is a production-ready, AI-powered supply chain optimization system for textile manufacturing. It features:

- **6-Phase Planning Engine**: Intelligent supply chain planning
- **Multi-Model ML Pipeline**: ARIMA, Prophet, LSTM, XGBoost
- **FastAPI REST API**: 15+ endpoints with comprehensive authentication
- **Domain-Driven Design**: Clean architecture with rich domain models
- **Real-time Analytics**: Interactive dashboards and reporting

## 🚀 Quick Start (30 Minutes)

### 1. Environment Setup
```bash
# Clone the repository
git clone [repository-url]
cd efab.ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Database Setup
```bash
# Initialize database
cd src/database
python -m alembic upgrade head

# Seed initial data
python seed_database.py
```

### 3. Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env  # Set database URL, secret keys, etc.
```

### 4. Run the Application
```bash
# Terminal 1: Start FastAPI server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Start Streamlit interface
streamlit run main.py

# Terminal 3: Run tests
pytest tests/ -v
```

### 5. Verify Installation
- **API Documentation**: http://localhost:8000/docs
- **Streamlit Interface**: http://localhost:8501
- **API Health Check**: http://localhost:8000/health

## 🏗️ Architecture Understanding

### Core Components

#### 1. Domain Layer (`src/core/domain/`)
**Rich business entities with complex logic**
```python
# Example: Material entity
class Material:
    def __init__(self, id: MaterialId, name: str, type: MaterialType):
        self.id = id
        self.name = name
        self.type = type
        self.is_critical = False
        
    def calculate_safety_stock(self, demand: Quantity, lead_time: int) -> Quantity:
        # Complex business logic for safety stock calculation
        return Quantity(amount=demand.amount * lead_time * 0.5, unit=demand.unit)
```

#### 2. Use Cases (`src/core/use_cases/`)
**Business logic orchestration**
```python
# Example: Planning service
class SupplyChainPlanningService:
    def execute_planning_cycle(self, forecasts: List[Forecast]) -> List[ProcurementRecommendation]:
        # 6-phase planning process
        unified_forecasts = self.unify_forecasts(forecasts)
        material_requirements = self.explode_bom(unified_forecasts)
        net_requirements = self.net_inventory(material_requirements)
        optimized_orders = self.optimize_procurement(net_requirements)
        supplier_selections = self.select_suppliers(optimized_orders)
        return self.generate_recommendations(supplier_selections)
```

#### 3. Infrastructure Layer (`src/database/`, `src/engine/`)
**Data access and ML processing**
```python
# Example: Repository pattern
class MaterialRepository:
    def find_by_id(self, material_id: MaterialId) -> Optional[Material]:
        # Database query logic
        return self.db.query(MaterialModel).filter_by(id=material_id.value).first()
```

### API Architecture

#### Authentication Flow
```python
# JWT-based authentication
@app.post("/api/v1/auth/login")
async def login(credentials: LoginRequest):
    user = authenticate_user(credentials.username, credentials.password)
    token = create_access_token(user.id)
    return {"access_token": token, "token_type": "bearer"}

# Protected endpoint
@app.get("/api/v1/materials/")
async def get_materials(current_user: User = Depends(get_current_user)):
    return material_service.get_all_materials()
```

#### Validation Pipeline
```python
# Pydantic models for request validation
class MaterialCreateRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=255)
    type: MaterialType = Field(...)
    specifications: Dict[str, str] = Field(default_factory=dict)
    
    @validator('name')
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError('Name cannot be empty')
        return v.strip()
```

### ML Pipeline Architecture

#### Model Management
```python
# Multi-model ensemble system
class MLModelManager:
    def __init__(self):
        self.models = {
            'arima': ARIMAModel(),
            'prophet': ProphetModel(),
            'lstm': LSTMModel(),
            'xgboost': XGBoostModel()
        }
    
    def generate_forecast(self, data: pd.DataFrame) -> Dict[str, Any]:
        forecasts = {}
        for name, model in self.models.items():
            forecasts[name] = model.predict(data)
        return self.ensemble_predictions(forecasts)
```

## 🛠️ Development Workflow

### Git Workflow
```bash
# Feature development
git checkout -b feature/new-feature
# Make changes
git add .
git commit -m "Add new feature: detailed description"
git push origin feature/new-feature
# Create pull request
```

### Testing Strategy
```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# End-to-end tests
pytest tests/e2e/ -v

# Coverage report
pytest --cov=src tests/ --cov-report=html
```

### Code Quality
```bash
# Linting
flake8 src/
black src/  # Code formatting
mypy src/  # Type checking

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

## 📊 Key Concepts

### 1. Domain-Driven Design
- **Entities**: Objects with identity (Material, Supplier, BOM)
- **Value Objects**: Immutable objects (Money, Quantity, MaterialId)
- **Aggregates**: Consistency boundaries around entities
- **Services**: Stateless operations that don't belong to entities

### 2. Planning Engine (6 Phases)
1. **Forecast Unification**: Combine multiple demand sources
2. **BOM Explosion**: Convert SKU demand to material requirements
3. **Inventory Netting**: Account for current stock and open orders
4. **Procurement Optimization**: Calculate optimal order quantities
5. **Supplier Selection**: Choose best suppliers based on criteria
6. **Output Generation**: Create actionable procurement recommendations

### 3. ML Model Types
- **ARIMA**: Traditional time series forecasting
- **Prophet**: Handles seasonality and trends
- **LSTM**: Deep learning for complex patterns
- **XGBoost**: Gradient boosting for feature-based predictions

### 4. Data Flow
```
Raw Data → Data Quality → Domain Objects → ML Processing → Business Logic → Recommendations
```

## 🔧 Common Development Tasks

### Adding a New API Endpoint
```python
# 1. Define request/response models
class NewFeatureRequest(BaseModel):
    parameter: str = Field(..., description="Parameter description")

class NewFeatureResponse(BaseModel):
    result: str
    success: bool

# 2. Add to router
@router.post("/new-feature/", response_model=NewFeatureResponse)
async def new_feature(
    request: NewFeatureRequest,
    current_user: User = Depends(get_current_user)
):
    result = new_feature_service.process(request.parameter)
    return NewFeatureResponse(result=result, success=True)

# 3. Update API documentation
# 4. Add tests
def test_new_feature():
    response = client.post("/api/v1/new-feature/", json={"parameter": "test"})
    assert response.status_code == 200
    assert response.json()["success"] is True
```

### Adding a New ML Model
```python
# 1. Create model class
class NewMLModel:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
    
    def train(self, data: pd.DataFrame):
        # Training logic
        pass
    
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        # Prediction logic
        pass

# 2. Register in model manager
model_manager.register_model("new_model", NewMLModel)

# 3. Add to configuration
ML_MODELS = {
    "new_model": {
        "enabled": True,
        "config": {...}
    }
}
```

### Adding a New Domain Entity
```python
# 1. Create entity class
class NewEntity:
    def __init__(self, id: NewEntityId, name: str):
        self.id = id
        self.name = name
        self.created_at = datetime.utcnow()
    
    def business_method(self) -> str:
        # Business logic
        return f"Processing {self.name}"

# 2. Create value object for ID
class NewEntityId:
    def __init__(self, value: str):
        if not value:
            raise ValueError("ID cannot be empty")
        self.value = value

# 3. Add database model
class NewEntityModel(BaseModel):
    __tablename__ = "new_entities"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

# 4. Create repository
class NewEntityRepository:
    def save(self, entity: NewEntity) -> None:
        # Database save logic
        pass
```

## 📚 Important Files to Know

### Configuration Files
- `requirements.txt` - Python dependencies
- `.env` - Environment variables
- `config/app_config.json` - Application configuration
- `src/config/settings.py` - Settings management

### Core Application Files
- `main.py` - Streamlit web interface
- `api/main.py` - FastAPI REST API server
- `src/engine/planning_engine.py` - Core planning logic
- `src/core/domain/entities.py` - Domain entities
- `src/core/use_cases/` - Business logic services

### Infrastructure Files
- `src/database/models/` - SQLAlchemy models
- `src/database/repositories/` - Data access layer
- `src/auth/auth_service.py` - Authentication service
- `src/validation/` - Input validation system

### Documentation Files
- `README.md` - Project overview
- `CLAUDE.md` - Developer guidance
- `AI_IMPLEMENTATION_ROADMAP.md` - Implementation roadmap
- `VALIDATION_SYSTEM.md` - Validation system documentation

## 🐛 Debugging and Troubleshooting

### Common Issues

#### Import Errors
```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Add project root to path
export PYTHONPATH="${PYTHONPATH}:${PWD}"
```

#### Database Issues
```bash
# Reset database
rm -f development.db
python -m alembic upgrade head

# Check database connections
python -c "from src.database.connection import get_db; print('DB OK')"
```

#### ML Model Issues
```bash
# Check model files
ls -la models/trained/

# Test model loading
python -c "from src.engine.production_ml_loader import production_ml_loader; print('ML OK')"
```

### Debugging Tools
- **Logging**: Use structured logging for debugging
- **Debugger**: Python debugger (pdb) for step-through debugging
- **Testing**: Write tests to reproduce issues
- **Profiling**: Use cProfile for performance debugging

### Performance Optimization
```python
# Database query optimization
def optimized_query():
    return db.query(MaterialModel).options(
        joinedload(MaterialModel.supplier_materials)
    ).filter(MaterialModel.is_active == True)

# Caching expensive operations
@lru_cache(maxsize=128)
def expensive_calculation(param: str) -> Dict[str, Any]:
    # Expensive operation
    pass
```

## 📋 Development Checklist

### Before Starting Work
- [ ] Pull latest changes from main branch
- [ ] Create feature branch
- [ ] Verify all tests pass
- [ ] Check environment configuration

### During Development
- [ ] Follow coding standards
- [ ] Add tests for new functionality
- [ ] Update documentation
- [ ] Use proper error handling

### Before Committing
- [ ] Run full test suite
- [ ] Check code quality (linting, formatting)
- [ ] Update relevant documentation
- [ ] Test API endpoints manually

### Code Review Checklist
- [ ] Code follows architectural patterns
- [ ] Proper error handling
- [ ] Tests cover new functionality
- [ ] Documentation is updated
- [ ] Security considerations addressed

## 🎯 Next Steps

### Week 1: Get Familiar
1. Set up development environment
2. Run the application locally
3. Explore the codebase structure
4. Read architecture documentation
5. Run tests and understand test patterns

### Week 2: First Contributions
1. Fix a small bug or add a minor feature
2. Write tests for your changes
3. Submit your first pull request
4. Understand code review process

### Week 3: Domain Understanding
1. Study the planning engine in detail
2. Understand ML model pipeline
3. Learn about domain entities and business logic
4. Explore API endpoints and their usage

### Week 4: Advanced Features
1. Contribute to a major feature
2. Add new ML model or optimization
3. Improve system performance
4. Enhance documentation

## 📞 Getting Help

### Resources
- **Code Documentation**: Inline comments and docstrings
- **API Documentation**: http://localhost:8000/docs
- **Architecture Docs**: `SYSTEM_ARCHITECTURE_OVERVIEW.md`
- **Git History**: `git log --oneline` for recent changes

### Team Communication
- **Daily Standups**: Progress updates and blockers
- **Code Reviews**: Collaborative code improvement
- **Technical Discussions**: Architecture and design decisions
- **Documentation**: Keep documentation updated

### Learning Resources
- **Domain-Driven Design**: Eric Evans' book
- **FastAPI**: Official documentation
- **SQLAlchemy**: ORM patterns and best practices
- **Machine Learning**: scikit-learn and TensorFlow docs

Welcome to the team! This system represents a sophisticated implementation of AI-driven supply chain optimization. Take your time to understand the architecture, and don't hesitate to ask questions. The codebase is well-structured and follows industry best practices, making it a great learning environment.

Happy coding! 🚀