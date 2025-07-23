#!/usr/bin/env python3
"""
ML Model Versioning System
Beverly Knits AI Supply Chain Planner
"""

import os
import json
import pickle
import logging
import hashlib
import shutil
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import joblib
import sqlite3
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class ModelStatus(Enum):
    """Model status enumeration"""
    TRAINING = "training"
    TRAINED = "trained"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"

class ModelType(Enum):
    """Model type enumeration"""
    ARIMA = "arima"
    PROPHET = "prophet"
    LSTM = "lstm"
    XGBOOST = "xgboost"
    ENSEMBLE = "ensemble"
    STACKING = "stacking"
    BLENDING = "blending"
    VOTING = "voting"
    CUSTOM = "custom"

@dataclass
class ModelMetadata:
    """Model metadata"""
    model_id: str
    model_name: str
    model_type: ModelType
    version: str
    description: str
    created_at: datetime
    updated_at: datetime
    status: ModelStatus
    
    # Model characteristics
    input_features: List[str]
    output_targets: List[str]
    model_params: Dict[str, Any]
    
    # Performance metrics
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    
    # Technical details
    framework: str
    python_version: str
    dependencies: Dict[str, str]
    
    # File information
    model_file: str
    model_size: int
    model_hash: str
    
    # Deployment information
    deployment_date: Optional[datetime] = None
    deployment_environment: Optional[str] = None
    deployment_config: Dict[str, Any] = field(default_factory=dict)
    
    # Tags and labels
    tags: List[str] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        # Convert datetime objects to ISO format
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat() if value else None
            elif isinstance(value, Enum):
                data[key] = value.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary"""
        # Convert datetime strings back to datetime objects
        for key in ['created_at', 'updated_at', 'deployment_date']:
            if key in data and data[key]:
                data[key] = datetime.fromisoformat(data[key])
        
        # Convert enum strings back to enums
        if 'model_type' in data:
            data['model_type'] = ModelType(data['model_type'])
        if 'status' in data:
            data['status'] = ModelStatus(data['status'])
            
        return cls(**data)

class ModelVersioningSystem:
    """ML Model Versioning System"""
    
    def __init__(self, storage_path: str = "data/models", db_path: str = "data/models/versions.db"):
        self.storage_path = Path(storage_path)
        self.db_path = Path(db_path)
        
        # Create directories
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        logger.info(f"Model versioning system initialized at {self.storage_path}")
    
    def _init_database(self):
        """Initialize SQLite database for metadata"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS model_versions (
                    model_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    version TEXT NOT NULL,
                    description TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    input_features TEXT,
                    output_targets TEXT,
                    model_params TEXT,
                    training_metrics TEXT,
                    validation_metrics TEXT,
                    test_metrics TEXT,
                    framework TEXT,
                    python_version TEXT,
                    dependencies TEXT,
                    model_file TEXT NOT NULL,
                    model_size INTEGER,
                    model_hash TEXT,
                    deployment_date TEXT,
                    deployment_environment TEXT,
                    deployment_config TEXT,
                    tags TEXT,
                    labels TEXT
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_model_name ON model_versions(model_name)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_model_type ON model_versions(model_type)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_status ON model_versions(status)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_version ON model_versions(version)
            ''')
    
    @contextmanager
    def _get_db_connection(self):
        """Get database connection with context manager"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _calculate_model_hash(self, model_file: Path) -> str:
        """Calculate SHA256 hash of model file"""
        hash_sha256 = hashlib.sha256()
        with open(model_file, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _generate_model_id(self, model_name: str, version: str) -> str:
        """Generate unique model ID"""
        return f"{model_name}_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _generate_version(self, model_name: str) -> str:
        """Generate next version number"""
        with self._get_db_connection() as conn:
            cursor = conn.execute(
                "SELECT version FROM model_versions WHERE model_name = ? ORDER BY created_at DESC LIMIT 1",
                (model_name,)
            )
            row = cursor.fetchone()
            
            if row:
                # Parse version (e.g., "1.2.3")
                version_parts = row['version'].split('.')
                major, minor, patch = int(version_parts[0]), int(version_parts[1]), int(version_parts[2])
                return f"{major}.{minor}.{patch + 1}"
            else:
                return "1.0.0"
    
    def register_model(self, model: BaseEstimator, model_name: str, model_type: ModelType,
                      description: str = "", input_features: List[str] = None,
                      output_targets: List[str] = None, model_params: Dict[str, Any] = None,
                      training_metrics: Dict[str, float] = None,
                      validation_metrics: Dict[str, float] = None,
                      test_metrics: Dict[str, float] = None,
                      framework: str = "sklearn", tags: List[str] = None,
                      labels: Dict[str, str] = None) -> str:
        """
        Register a new model version
        
        Args:
            model: The trained model object
            model_name: Name of the model
            model_type: Type of the model
            description: Model description
            input_features: List of input feature names
            output_targets: List of output target names
            model_params: Model parameters
            training_metrics: Training performance metrics
            validation_metrics: Validation performance metrics
            test_metrics: Test performance metrics
            framework: ML framework used
            tags: List of tags
            labels: Dictionary of labels
            
        Returns:
            Model ID
        """
        logger.info(f"Registering model: {model_name}")
        
        # Generate version and model ID
        version = self._generate_version(model_name)
        model_id = self._generate_model_id(model_name, version)
        
        # Create model directory
        model_dir = self.storage_path / model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_file = model_dir / f"{model_id}.pkl"
        try:
            joblib.dump(model, model_file)
            logger.info(f"Model saved to {model_file}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
        
        # Calculate model hash and size
        model_hash = self._calculate_model_hash(model_file)
        model_size = model_file.stat().st_size
        
        # Get Python version and dependencies
        import sys
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            model_name=model_name,
            model_type=model_type,
            version=version,
            description=description,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            status=ModelStatus.TRAINED,
            input_features=input_features or [],
            output_targets=output_targets or [],
            model_params=model_params or {},
            training_metrics=training_metrics or {},
            validation_metrics=validation_metrics or {},
            test_metrics=test_metrics or {},
            framework=framework,
            python_version=python_version,
            dependencies=self._get_dependencies(),
            model_file=str(model_file.relative_to(self.storage_path)),
            model_size=model_size,
            model_hash=model_hash,
            tags=tags or [],
            labels=labels or {}
        )
        
        # Save metadata to database
        self._save_metadata(metadata)
        
        # Save metadata to JSON file
        metadata_file = model_dir / f"{model_id}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        logger.info(f"Model registered successfully with ID: {model_id}")
        return model_id
    
    def _get_dependencies(self) -> Dict[str, str]:
        """Get current package dependencies"""
        try:
            import pkg_resources
            dependencies = {}
            for package in ['scikit-learn', 'numpy', 'pandas', 'tensorflow', 'xgboost', 'prophet']:
                try:
                    version = pkg_resources.get_distribution(package).version
                    dependencies[package] = version
                except pkg_resources.DistributionNotFound:
                    pass
            return dependencies
        except ImportError:
            return {}
    
    def _save_metadata(self, metadata: ModelMetadata):
        """Save metadata to database"""
        with self._get_db_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO model_versions (
                    model_id, model_name, model_type, version, description,
                    created_at, updated_at, status, input_features, output_targets,
                    model_params, training_metrics, validation_metrics, test_metrics,
                    framework, python_version, dependencies, model_file, model_size,
                    model_hash, deployment_date, deployment_environment, deployment_config,
                    tags, labels
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metadata.model_id,
                metadata.model_name,
                metadata.model_type.value,
                metadata.version,
                metadata.description,
                metadata.created_at.isoformat(),
                metadata.updated_at.isoformat(),
                metadata.status.value,
                json.dumps(metadata.input_features),
                json.dumps(metadata.output_targets),
                json.dumps(metadata.model_params),
                json.dumps(metadata.training_metrics),
                json.dumps(metadata.validation_metrics),
                json.dumps(metadata.test_metrics),
                metadata.framework,
                metadata.python_version,
                json.dumps(metadata.dependencies),
                metadata.model_file,
                metadata.model_size,
                metadata.model_hash,
                metadata.deployment_date.isoformat() if metadata.deployment_date else None,
                metadata.deployment_environment,
                json.dumps(metadata.deployment_config),
                json.dumps(metadata.tags),
                json.dumps(metadata.labels)
            ))
            conn.commit()
    
    def load_model(self, model_id: str) -> Tuple[BaseEstimator, ModelMetadata]:
        """
        Load a model and its metadata
        
        Args:
            model_id: Model ID
            
        Returns:
            Tuple of (model, metadata)
        """
        logger.info(f"Loading model: {model_id}")
        
        # Get metadata
        metadata = self.get_model_metadata(model_id)
        if not metadata:
            raise ValueError(f"Model not found: {model_id}")
        
        # Load model
        model_file = self.storage_path / metadata.model_file
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        try:
            model = joblib.load(model_file)
            logger.info(f"Model loaded successfully: {model_id}")
            return model, metadata
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            raise
    
    def get_model_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """
        Get model metadata by ID
        
        Args:
            model_id: Model ID
            
        Returns:
            Model metadata or None
        """
        with self._get_db_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM model_versions WHERE model_id = ?",
                (model_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return self._row_to_metadata(row)
            return None
    
    def _row_to_metadata(self, row: sqlite3.Row) -> ModelMetadata:
        """Convert database row to ModelMetadata"""
        return ModelMetadata(
            model_id=row['model_id'],
            model_name=row['model_name'],
            model_type=ModelType(row['model_type']),
            version=row['version'],
            description=row['description'],
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at']),
            status=ModelStatus(row['status']),
            input_features=json.loads(row['input_features']),
            output_targets=json.loads(row['output_targets']),
            model_params=json.loads(row['model_params']),
            training_metrics=json.loads(row['training_metrics']),
            validation_metrics=json.loads(row['validation_metrics']),
            test_metrics=json.loads(row['test_metrics']),
            framework=row['framework'],
            python_version=row['python_version'],
            dependencies=json.loads(row['dependencies']),
            model_file=row['model_file'],
            model_size=row['model_size'],
            model_hash=row['model_hash'],
            deployment_date=datetime.fromisoformat(row['deployment_date']) if row['deployment_date'] else None,
            deployment_environment=row['deployment_environment'],
            deployment_config=json.loads(row['deployment_config']),
            tags=json.loads(row['tags']),
            labels=json.loads(row['labels'])
        )
    
    def list_models(self, model_name: Optional[str] = None, model_type: Optional[ModelType] = None,
                   status: Optional[ModelStatus] = None, limit: int = 100) -> List[ModelMetadata]:
        """
        List models with optional filtering
        
        Args:
            model_name: Filter by model name
            model_type: Filter by model type
            status: Filter by status
            limit: Maximum number of results
            
        Returns:
            List of model metadata
        """
        query = "SELECT * FROM model_versions WHERE 1=1"
        params = []
        
        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)
        
        if model_type:
            query += " AND model_type = ?"
            params.append(model_type.value)
        
        if status:
            query += " AND status = ?"
            params.append(status.value)
        
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        with self._get_db_connection() as conn:
            cursor = conn.execute(query, params)
            return [self._row_to_metadata(row) for row in cursor.fetchall()]
    
    def get_latest_model(self, model_name: str, status: Optional[ModelStatus] = None) -> Optional[ModelMetadata]:
        """
        Get the latest model version
        
        Args:
            model_name: Model name
            status: Filter by status
            
        Returns:
            Latest model metadata or None
        """
        query = "SELECT * FROM model_versions WHERE model_name = ?"
        params = [model_name]
        
        if status:
            query += " AND status = ?"
            params.append(status.value)
        
        query += " ORDER BY created_at DESC LIMIT 1"
        
        with self._get_db_connection() as conn:
            cursor = conn.execute(query, params)
            row = cursor.fetchone()
            
            if row:
                return self._row_to_metadata(row)
            return None
    
    def update_model_status(self, model_id: str, status: ModelStatus,
                           deployment_environment: Optional[str] = None,
                           deployment_config: Optional[Dict[str, Any]] = None):
        """
        Update model status
        
        Args:
            model_id: Model ID
            status: New status
            deployment_environment: Deployment environment
            deployment_config: Deployment configuration
        """
        logger.info(f"Updating model {model_id} status to {status.value}")
        
        deployment_date = datetime.now() if status == ModelStatus.DEPLOYED else None
        
        with self._get_db_connection() as conn:
            conn.execute('''
                UPDATE model_versions 
                SET status = ?, updated_at = ?, deployment_date = ?, 
                    deployment_environment = ?, deployment_config = ?
                WHERE model_id = ?
            ''', (
                status.value,
                datetime.now().isoformat(),
                deployment_date.isoformat() if deployment_date else None,
                deployment_environment,
                json.dumps(deployment_config or {}),
                model_id
            ))
            conn.commit()
    
    def add_model_tags(self, model_id: str, tags: List[str]):
        """Add tags to a model"""
        metadata = self.get_model_metadata(model_id)
        if not metadata:
            raise ValueError(f"Model not found: {model_id}")
        
        updated_tags = list(set(metadata.tags + tags))
        
        with self._get_db_connection() as conn:
            conn.execute(
                "UPDATE model_versions SET tags = ?, updated_at = ? WHERE model_id = ?",
                (json.dumps(updated_tags), datetime.now().isoformat(), model_id)
            )
            conn.commit()
    
    def remove_model_tags(self, model_id: str, tags: List[str]):
        """Remove tags from a model"""
        metadata = self.get_model_metadata(model_id)
        if not metadata:
            raise ValueError(f"Model not found: {model_id}")
        
        updated_tags = [tag for tag in metadata.tags if tag not in tags]
        
        with self._get_db_connection() as conn:
            conn.execute(
                "UPDATE model_versions SET tags = ?, updated_at = ? WHERE model_id = ?",
                (json.dumps(updated_tags), datetime.now().isoformat(), model_id)
            )
            conn.commit()
    
    def compare_models(self, model_id1: str, model_id2: str) -> Dict[str, Any]:
        """
        Compare two models
        
        Args:
            model_id1: First model ID
            model_id2: Second model ID
            
        Returns:
            Comparison results
        """
        metadata1 = self.get_model_metadata(model_id1)
        metadata2 = self.get_model_metadata(model_id2)
        
        if not metadata1 or not metadata2:
            raise ValueError("One or both models not found")
        
        comparison = {
            'model1': {
                'id': model_id1,
                'name': metadata1.model_name,
                'version': metadata1.version,
                'type': metadata1.model_type.value,
                'created_at': metadata1.created_at.isoformat(),
                'metrics': {
                    'training': metadata1.training_metrics,
                    'validation': metadata1.validation_metrics,
                    'test': metadata1.test_metrics
                }
            },
            'model2': {
                'id': model_id2,
                'name': metadata2.model_name,
                'version': metadata2.version,
                'type': metadata2.model_type.value,
                'created_at': metadata2.created_at.isoformat(),
                'metrics': {
                    'training': metadata2.training_metrics,
                    'validation': metadata2.validation_metrics,
                    'test': metadata2.test_metrics
                }
            },
            'differences': {
                'model_type': metadata1.model_type != metadata2.model_type,
                'framework': metadata1.framework != metadata2.framework,
                'features': metadata1.input_features != metadata2.input_features,
                'targets': metadata1.output_targets != metadata2.output_targets,
                'size': metadata1.model_size != metadata2.model_size
            }
        }
        
        # Compare metrics
        for metric_type in ['training', 'validation', 'test']:
            metrics1 = getattr(metadata1, f'{metric_type}_metrics')
            metrics2 = getattr(metadata2, f'{metric_type}_metrics')
            
            comparison['differences'][f'{metric_type}_metrics'] = {}
            for metric_name in set(metrics1.keys()) | set(metrics2.keys()):
                val1 = metrics1.get(metric_name)
                val2 = metrics2.get(metric_name)
                
                if val1 is not None and val2 is not None:
                    comparison['differences'][f'{metric_type}_metrics'][metric_name] = {
                        'model1': val1,
                        'model2': val2,
                        'difference': val2 - val1,
                        'improvement': ((val2 - val1) / val1) * 100 if val1 != 0 else 0
                    }
        
        return comparison
    
    def archive_model(self, model_id: str):
        """Archive a model"""
        logger.info(f"Archiving model: {model_id}")
        
        metadata = self.get_model_metadata(model_id)
        if not metadata:
            raise ValueError(f"Model not found: {model_id}")
        
        # Create archive directory
        archive_dir = self.storage_path / "archive" / metadata.model_name
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Move model files
        current_model_file = self.storage_path / metadata.model_file
        archive_model_file = archive_dir / current_model_file.name
        
        if current_model_file.exists():
            shutil.move(str(current_model_file), str(archive_model_file))
        
        # Update metadata
        metadata.status = ModelStatus.ARCHIVED
        metadata.updated_at = datetime.now()
        metadata.model_file = str(archive_model_file.relative_to(self.storage_path))
        
        self._save_metadata(metadata)
        
        logger.info(f"Model archived successfully: {model_id}")
    
    def delete_model(self, model_id: str, force: bool = False):
        """
        Delete a model
        
        Args:
            model_id: Model ID
            force: Force deletion even if deployed
        """
        logger.warning(f"Deleting model: {model_id}")
        
        metadata = self.get_model_metadata(model_id)
        if not metadata:
            raise ValueError(f"Model not found: {model_id}")
        
        if metadata.status == ModelStatus.DEPLOYED and not force:
            raise ValueError("Cannot delete deployed model without force=True")
        
        # Delete model file
        model_file = self.storage_path / metadata.model_file
        if model_file.exists():
            model_file.unlink()
        
        # Delete metadata file
        metadata_file = model_file.parent / f"{model_id}_metadata.json"
        if metadata_file.exists():
            metadata_file.unlink()
        
        # Delete from database
        with self._get_db_connection() as conn:
            conn.execute("DELETE FROM model_versions WHERE model_id = ?", (model_id,))
            conn.commit()
        
        logger.info(f"Model deleted successfully: {model_id}")
    
    def get_model_lineage(self, model_id: str) -> Dict[str, Any]:
        """
        Get model lineage information
        
        Args:
            model_id: Model ID
            
        Returns:
            Model lineage information
        """
        metadata = self.get_model_metadata(model_id)
        if not metadata:
            raise ValueError(f"Model not found: {model_id}")
        
        # Get all versions of the same model
        all_versions = self.list_models(model_name=metadata.model_name, limit=1000)
        
        lineage = {
            'model_name': metadata.model_name,
            'current_version': metadata.version,
            'total_versions': len(all_versions),
            'versions': []
        }
        
        for version_metadata in all_versions:
            lineage['versions'].append({
                'id': version_metadata.model_id,
                'version': version_metadata.version,
                'created_at': version_metadata.created_at.isoformat(),
                'status': version_metadata.status.value,
                'description': version_metadata.description,
                'metrics': {
                    'training': version_metadata.training_metrics,
                    'validation': version_metadata.validation_metrics,
                    'test': version_metadata.test_metrics
                }
            })
        
        return lineage
    
    def export_model_registry(self, output_file: str):
        """Export model registry to JSON file"""
        logger.info(f"Exporting model registry to {output_file}")
        
        all_models = self.list_models(limit=10000)
        
        registry_data = {
            'exported_at': datetime.now().isoformat(),
            'total_models': len(all_models),
            'models': [model.to_dict() for model in all_models]
        }
        
        with open(output_file, 'w') as f:
            json.dump(registry_data, f, indent=2)
        
        logger.info(f"Model registry exported successfully: {output_file}")
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get model registry statistics"""
        with self._get_db_connection() as conn:
            # Total models
            total_models = conn.execute("SELECT COUNT(*) as count FROM model_versions").fetchone()['count']
            
            # Models by type
            type_stats = {}
            cursor = conn.execute("SELECT model_type, COUNT(*) as count FROM model_versions GROUP BY model_type")
            for row in cursor:
                type_stats[row['model_type']] = row['count']
            
            # Models by status
            status_stats = {}
            cursor = conn.execute("SELECT status, COUNT(*) as count FROM model_versions GROUP BY status")
            for row in cursor:
                status_stats[row['status']] = row['count']
            
            # Storage stats
            storage_stats = conn.execute(
                "SELECT SUM(model_size) as total_size, AVG(model_size) as avg_size FROM model_versions"
            ).fetchone()
            
            return {
                'total_models': total_models,
                'models_by_type': type_stats,
                'models_by_status': status_stats,
                'storage': {
                    'total_size_bytes': storage_stats['total_size'] or 0,
                    'average_size_bytes': storage_stats['avg_size'] or 0,
                    'total_size_mb': (storage_stats['total_size'] or 0) / (1024 * 1024)
                }
            }

# Global model versioning instance
model_versioning_system = ModelVersioningSystem()

def get_model_versioning_system() -> ModelVersioningSystem:
    """Get the global model versioning system"""
    return model_versioning_system

def register_model(*args, **kwargs) -> str:
    """Register a model using the global versioning system"""
    return model_versioning_system.register_model(*args, **kwargs)

def load_model(*args, **kwargs) -> Tuple[BaseEstimator, ModelMetadata]:
    """Load a model using the global versioning system"""
    return model_versioning_system.load_model(*args, **kwargs)

def get_latest_model(*args, **kwargs) -> Optional[ModelMetadata]:
    """Get the latest model using the global versioning system"""
    return model_versioning_system.get_latest_model(*args, **kwargs)