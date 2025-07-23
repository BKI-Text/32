#!/usr/bin/env python3
"""
Automated Model Retraining Scheduler
Beverly Knits AI Supply Chain Planner
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from pathlib import Path
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import pickle

logger = logging.getLogger(__name__)

class ModelStatus(Enum):
    """Model status enumeration"""
    ACTIVE = "active"
    RETRAINING = "retraining"
    FAILED = "failed"
    DEPRECATED = "deprecated"

class RetrainingTrigger(Enum):
    """Retraining trigger types"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_DRIFT = "data_drift"
    TIME_BASED = "time_based"
    MANUAL = "manual"

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    model_id: str
    model_type: str
    accuracy: float
    mae: float
    mse: float
    rmse: float
    r2_score: float
    last_updated: datetime
    data_points: int
    confidence_interval: float = 0.95
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'model_id': self.model_id,
            'model_type': self.model_type,
            'accuracy': self.accuracy,
            'mae': self.mae,
            'mse': self.mse,
            'rmse': self.rmse,
            'r2_score': self.r2_score,
            'last_updated': self.last_updated.isoformat(),
            'data_points': self.data_points,
            'confidence_interval': self.confidence_interval
        }

@dataclass
class RetrainingConfig:
    """Retraining configuration"""
    model_id: str
    model_type: str
    performance_threshold: float = 0.8  # Minimum acceptable performance
    data_drift_threshold: float = 0.3   # Maximum acceptable data drift
    max_age_days: int = 30              # Maximum model age in days
    min_data_points: int = 100          # Minimum data points for retraining
    retraining_interval_hours: int = 24  # Check interval in hours
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'model_id': self.model_id,
            'model_type': self.model_type,
            'performance_threshold': self.performance_threshold,
            'data_drift_threshold': self.data_drift_threshold,
            'max_age_days': self.max_age_days,
            'min_data_points': self.min_data_points,
            'retraining_interval_hours': self.retraining_interval_hours,
            'enabled': self.enabled
        }

@dataclass
class RetrainingJob:
    """Retraining job"""
    job_id: str
    model_id: str
    model_type: str
    trigger: RetrainingTrigger
    scheduled_time: datetime
    started_time: Optional[datetime] = None
    completed_time: Optional[datetime] = None
    status: str = "pending"
    error_message: Optional[str] = None
    old_metrics: Optional[ModelMetrics] = None
    new_metrics: Optional[ModelMetrics] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'job_id': self.job_id,
            'model_id': self.model_id,
            'model_type': self.model_type,
            'trigger': self.trigger.value,
            'scheduled_time': self.scheduled_time.isoformat(),
            'started_time': self.started_time.isoformat() if self.started_time else None,
            'completed_time': self.completed_time.isoformat() if self.completed_time else None,
            'status': self.status,
            'error_message': self.error_message,
            'old_metrics': self.old_metrics.to_dict() if self.old_metrics else None,
            'new_metrics': self.new_metrics.to_dict() if self.new_metrics else None
        }

class ModelPerformanceMonitor:
    """Monitor model performance and detect degradation"""
    
    def __init__(self, config: RetrainingConfig):
        self.config = config
        self.metrics_history: List[ModelMetrics] = []
        self.baseline_metrics: Optional[ModelMetrics] = None
        
    def update_metrics(self, metrics: ModelMetrics):
        """Update model metrics"""
        self.metrics_history.append(metrics)
        
        # Keep only recent metrics (last 100 entries)
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
            
        # Set baseline if not exists
        if self.baseline_metrics is None:
            self.baseline_metrics = metrics
            
    def check_performance_degradation(self) -> bool:
        """Check if model performance has degraded"""
        if not self.metrics_history or not self.baseline_metrics:
            return False
            
        latest_metrics = self.metrics_history[-1]
        
        # Check if performance dropped below threshold
        if latest_metrics.accuracy < self.config.performance_threshold:
            logger.warning(f"Model {self.config.model_id} performance degraded: {latest_metrics.accuracy:.3f} < {self.config.performance_threshold}")
            return True
            
        # Check if performance dropped significantly from baseline
        performance_drop = (self.baseline_metrics.accuracy - latest_metrics.accuracy) / self.baseline_metrics.accuracy
        if performance_drop > 0.2:  # 20% drop
            logger.warning(f"Model {self.config.model_id} performance dropped {performance_drop:.1%} from baseline")
            return True
            
        return False
        
    def check_data_drift(self) -> bool:
        """Check for data drift (simplified implementation)"""
        if len(self.metrics_history) < 10:
            return False
            
        # Calculate recent performance variance
        recent_accuracies = [m.accuracy for m in self.metrics_history[-10:]]
        variance = np.var(recent_accuracies)
        
        # If variance is too high, it might indicate data drift
        if variance > self.config.data_drift_threshold:
            logger.warning(f"Model {self.config.model_id} shows high variance: {variance:.3f}")
            return True
            
        return False
        
    def check_model_age(self) -> bool:
        """Check if model is too old"""
        if not self.baseline_metrics:
            return False
            
        age_days = (datetime.now() - self.baseline_metrics.last_updated).days
        if age_days > self.config.max_age_days:
            logger.info(f"Model {self.config.model_id} is {age_days} days old (max: {self.config.max_age_days})")
            return True
            
        return False
        
    def needs_retraining(self) -> Optional[RetrainingTrigger]:
        """Check if model needs retraining"""
        if not self.config.enabled:
            return None
            
        if self.check_performance_degradation():
            return RetrainingTrigger.PERFORMANCE_DEGRADATION
            
        if self.check_data_drift():
            return RetrainingTrigger.DATA_DRIFT
            
        if self.check_model_age():
            return RetrainingTrigger.TIME_BASED
            
        return None

class ModelRetrainingScheduler:
    """Automated model retraining scheduler"""
    
    def __init__(self, data_dir: str = "data/models"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.monitors: Dict[str, ModelPerformanceMonitor] = {}
        self.configs: Dict[str, RetrainingConfig] = {}
        self.jobs: Dict[str, RetrainingJob] = {}
        self.retraining_functions: Dict[str, Callable] = {}
        
        self.running = False
        self.scheduler_thread = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        
    def register_model(self, config: RetrainingConfig, retraining_function: Callable):
        """Register a model for automated retraining"""
        self.configs[config.model_id] = config
        self.monitors[config.model_id] = ModelPerformanceMonitor(config)
        self.retraining_functions[config.model_id] = retraining_function
        
        logger.info(f"Registered model {config.model_id} for automated retraining")
        
    def update_model_metrics(self, model_id: str, metrics: ModelMetrics):
        """Update model performance metrics"""
        if model_id in self.monitors:
            self.monitors[model_id].update_metrics(metrics)
            
    def schedule_retraining(self, model_id: str, trigger: RetrainingTrigger, 
                          scheduled_time: Optional[datetime] = None) -> str:
        """Schedule a retraining job"""
        if scheduled_time is None:
            scheduled_time = datetime.now()
            
        job_id = f"{model_id}_{int(time.time())}"
        config = self.configs[model_id]
        
        job = RetrainingJob(
            job_id=job_id,
            model_id=model_id,
            model_type=config.model_type,
            trigger=trigger,
            scheduled_time=scheduled_time
        )
        
        self.jobs[job_id] = job
        logger.info(f"Scheduled retraining job {job_id} for model {model_id} (trigger: {trigger.value})")
        
        return job_id
        
    async def execute_retraining_job(self, job_id: str):
        """Execute a retraining job"""
        job = self.jobs.get(job_id)
        if not job:
            logger.error(f"Job {job_id} not found")
            return
            
        try:
            job.started_time = datetime.now()
            job.status = "running"
            logger.info(f"Starting retraining job {job_id}")
            
            # Get current metrics
            monitor = self.monitors.get(job.model_id)
            if monitor and monitor.metrics_history:
                job.old_metrics = monitor.metrics_history[-1]
                
            # Execute retraining function
            retraining_function = self.retraining_functions.get(job.model_id)
            if retraining_function:
                # Run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                new_metrics = await loop.run_in_executor(
                    self.executor, 
                    retraining_function
                )
                
                if isinstance(new_metrics, ModelMetrics):
                    job.new_metrics = new_metrics
                    # Update monitor with new metrics
                    if monitor:
                        monitor.update_metrics(new_metrics)
                        
                job.status = "completed"
                job.completed_time = datetime.now()
                logger.info(f"Completed retraining job {job_id}")
                
            else:
                job.status = "failed"
                job.error_message = f"No retraining function found for model {job.model_id}"
                logger.error(job.error_message)
                
        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            job.completed_time = datetime.now()
            logger.error(f"Retraining job {job_id} failed: {e}")
            
    def check_retraining_triggers(self):
        """Check all models for retraining triggers"""
        for model_id, monitor in self.monitors.items():
            trigger = monitor.needs_retraining()
            if trigger:
                # Check if there's already a pending job
                pending_jobs = [j for j in self.jobs.values() 
                              if j.model_id == model_id and j.status in ["pending", "running"]]
                
                if not pending_jobs:
                    self.schedule_retraining(model_id, trigger)
                    
    async def process_pending_jobs(self):
        """Process pending retraining jobs"""
        pending_jobs = [j for j in self.jobs.values() if j.status == "pending"]
        
        for job in pending_jobs:
            if job.scheduled_time <= datetime.now():
                await self.execute_retraining_job(job.job_id)
                
    async def scheduler_loop(self):
        """Main scheduler loop"""
        logger.info("Started model retraining scheduler")
        
        while self.running:
            try:
                # Check for retraining triggers
                self.check_retraining_triggers()
                
                # Process pending jobs
                await self.process_pending_jobs()
                
                # Save state
                self.save_state()
                
                # Wait before next check
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
                
    def start(self):
        """Start the scheduler"""
        if self.running:
            logger.warning("Scheduler is already running")
            return
            
        self.running = True
        self.load_state()
        
        # Start scheduler in background thread
        def run_scheduler():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.scheduler_loop())
            
        self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("Model retraining scheduler started")
        
    def stop(self):
        """Stop the scheduler"""
        if not self.running:
            return
            
        self.running = False
        
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
            
        self.save_state()
        logger.info("Model retraining scheduler stopped")
        
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status"""
        return {
            'running': self.running,
            'registered_models': len(self.configs),
            'active_jobs': len([j for j in self.jobs.values() if j.status in ["pending", "running"]]),
            'completed_jobs': len([j for j in self.jobs.values() if j.status == "completed"]),
            'failed_jobs': len([j for j in self.jobs.values() if j.status == "failed"]),
            'models': {
                model_id: {
                    'config': config.to_dict(),
                    'last_metrics': monitor.metrics_history[-1].to_dict() if monitor.metrics_history else None,
                    'needs_retraining': monitor.needs_retraining().value if monitor.needs_retraining() else None
                }
                for model_id, (config, monitor) in 
                zip(self.configs.keys(), zip(self.configs.values(), self.monitors.values()))
            }
        }
        
    def get_job_history(self, model_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get job history"""
        jobs = list(self.jobs.values())
        
        if model_id:
            jobs = [j for j in jobs if j.model_id == model_id]
            
        return [job.to_dict() for job in sorted(jobs, key=lambda x: x.scheduled_time, reverse=True)]
        
    def save_state(self):
        """Save scheduler state to disk"""
        try:
            state = {
                'configs': {k: v.to_dict() for k, v in self.configs.items()},
                'jobs': {k: v.to_dict() for k, v in self.jobs.items()},
                'metrics_history': {
                    model_id: [m.to_dict() for m in monitor.metrics_history[-10:]]  # Save last 10 metrics
                    for model_id, monitor in self.monitors.items()
                }
            }
            
            state_file = self.data_dir / "scheduler_state.json"
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save scheduler state: {e}")
            
    def load_state(self):
        """Load scheduler state from disk"""
        try:
            state_file = self.data_dir / "scheduler_state.json"
            if not state_file.exists():
                return
                
            with open(state_file, 'r') as f:
                state = json.load(f)
                
            # Load configs
            for model_id, config_data in state.get('configs', {}).items():
                config = RetrainingConfig(
                    model_id=config_data['model_id'],
                    model_type=config_data['model_type'],
                    performance_threshold=config_data.get('performance_threshold', 0.8),
                    data_drift_threshold=config_data.get('data_drift_threshold', 0.3),
                    max_age_days=config_data.get('max_age_days', 30),
                    min_data_points=config_data.get('min_data_points', 100),
                    retraining_interval_hours=config_data.get('retraining_interval_hours', 24),
                    enabled=config_data.get('enabled', True)
                )
                self.configs[model_id] = config
                self.monitors[model_id] = ModelPerformanceMonitor(config)
                
            # Load jobs
            for job_id, job_data in state.get('jobs', {}).items():
                job = RetrainingJob(
                    job_id=job_data['job_id'],
                    model_id=job_data['model_id'],
                    model_type=job_data['model_type'],
                    trigger=RetrainingTrigger(job_data['trigger']),
                    scheduled_time=datetime.fromisoformat(job_data['scheduled_time']),
                    started_time=datetime.fromisoformat(job_data['started_time']) if job_data.get('started_time') else None,
                    completed_time=datetime.fromisoformat(job_data['completed_time']) if job_data.get('completed_time') else None,
                    status=job_data.get('status', 'pending'),
                    error_message=job_data.get('error_message')
                )
                self.jobs[job_id] = job
                
            # Load metrics history
            for model_id, metrics_data in state.get('metrics_history', {}).items():
                if model_id in self.monitors:
                    monitor = self.monitors[model_id]
                    for metric_data in metrics_data:
                        metrics = ModelMetrics(
                            model_id=metric_data['model_id'],
                            model_type=metric_data['model_type'],
                            accuracy=metric_data['accuracy'],
                            mae=metric_data['mae'],
                            mse=metric_data['mse'],
                            rmse=metric_data['rmse'],
                            r2_score=metric_data['r2_score'],
                            last_updated=datetime.fromisoformat(metric_data['last_updated']),
                            data_points=metric_data['data_points'],
                            confidence_interval=metric_data.get('confidence_interval', 0.95)
                        )
                        monitor.update_metrics(metrics)
                        
            logger.info("Loaded scheduler state from disk")
            
        except Exception as e:
            logger.error(f"Failed to load scheduler state: {e}")

# Global scheduler instance
model_scheduler = ModelRetrainingScheduler()

def get_model_scheduler() -> ModelRetrainingScheduler:
    """Get the global model scheduler instance"""
    return model_scheduler

def start_model_scheduler():
    """Start the global model scheduler"""
    model_scheduler.start()

def stop_model_scheduler():
    """Stop the global model scheduler"""
    model_scheduler.stop()