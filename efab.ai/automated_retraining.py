#!/usr/bin/env python3
"""
Automated Model Retraining Pipeline for Beverly Knits AI
Automatically retrains models based on performance degradation or new data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
import pickle
import shutil
from typing import Dict, Any, List, Optional

# Import custom modules
from model_monitoring import ModelMonitor
from train_basic_ml import BasicMLTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('automated_retraining.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutomatedRetrainingPipeline:
    """Automated pipeline for model retraining"""
    
    def __init__(self, data_path: str = "data/live/", models_path: str = "models/trained/"):
        self.data_path = Path(data_path)
        self.models_path = Path(models_path)
        self.backup_path = Path("models/backup/")
        self.backup_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.monitor = ModelMonitor()
        self.trainer = BasicMLTrainer()
        
        # Retraining triggers
        self.retrain_triggers = {
            'performance_degradation': {
                'mape_threshold': 50.0,  # Retrain if MAPE > 50%
                'performance_change_threshold': 25.0  # Retrain if performance degrades by 25%
            },
            'data_freshness': {
                'days_since_training': 30,  # Retrain every 30 days
                'min_new_records': 50  # Retrain if 50+ new records
            },
            'manual_trigger': False  # Manual override
        }
        
        # Load retraining history
        self.retraining_history = self.load_retraining_history()
    
    def load_retraining_history(self) -> List[Dict[str, Any]]:
        """Load retraining history"""
        try:
            history_path = self.models_path / "retraining_history.json"
            if history_path.exists():
                with open(history_path, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"Error loading retraining history: {e}")
            return []
    
    def save_retraining_history(self, entry: Dict[str, Any]):
        """Save retraining history entry"""
        try:
            self.retraining_history.append(entry)
            history_path = self.models_path / "retraining_history.json"
            with open(history_path, 'w') as f:
                json.dump(self.retraining_history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving retraining history: {e}")
    
    def check_retraining_triggers(self) -> Dict[str, Any]:
        """Check if retraining should be triggered"""
        triggers = {
            'should_retrain': False,
            'triggered_by': [],
            'reasons': []
        }
        
        # Check performance degradation
        monitoring_report = self.monitor.generate_monitoring_report()
        
        for model_name, performance in monitoring_report['performance_results'].items():
            if performance.get('status') == 'success':
                # Check MAPE threshold
                if 'mape' in performance:
                    if performance['mape'] > self.retrain_triggers['performance_degradation']['mape_threshold']:
                        triggers['should_retrain'] = True
                        triggers['triggered_by'].append('performance_degradation')
                        triggers['reasons'].append(f"{model_name} MAPE ({performance['mape']:.2f}%) exceeds threshold")
                
                # Check performance change
                if 'performance_change' in performance:
                    if performance['performance_change'] > self.retrain_triggers['performance_degradation']['performance_change_threshold']:
                        triggers['should_retrain'] = True
                        triggers['triggered_by'].append('performance_degradation')
                        triggers['reasons'].append(f"{model_name} performance degraded by {performance['performance_change']:.1f}%")
        
        # Check data freshness
        last_training_date = self.get_last_training_date()
        if last_training_date:
            days_since_training = (datetime.now() - last_training_date).days
            if days_since_training >= self.retrain_triggers['data_freshness']['days_since_training']:
                triggers['should_retrain'] = True
                triggers['triggered_by'].append('data_freshness')
                triggers['reasons'].append(f"Last training was {days_since_training} days ago")
        
        # Check for new data
        new_records_count = self.count_new_records(last_training_date)
        if new_records_count >= self.retrain_triggers['data_freshness']['min_new_records']:
            triggers['should_retrain'] = True
            triggers['triggered_by'].append('new_data')
            triggers['reasons'].append(f"{new_records_count} new records available")
        
        return triggers
    
    def get_last_training_date(self) -> Optional[datetime]:
        """Get the date of last training"""
        try:
            if self.retraining_history:
                last_entry = self.retraining_history[-1]
                return datetime.fromisoformat(last_entry['timestamp'])
            
            # Check training results
            results_path = self.models_path / "training_results_basic.json"
            if results_path.exists():
                with open(results_path, 'r') as f:
                    results = json.load(f)
                    if 'training_completed' in results:
                        return datetime.fromisoformat(results['training_completed'])
            
            return None
        except Exception as e:
            logger.error(f"Error getting last training date: {e}")
            return None
    
    def count_new_records(self, since_date: Optional[datetime]) -> int:
        """Count new records since last training"""
        try:
            if not since_date:
                return 0
            
            sales_file = self.data_path / "Sales Activity Report.csv"
            if not sales_file.exists():
                return 0
            
            sales_data = pd.read_csv(sales_file, encoding='utf-8-sig')
            sales_data['date'] = pd.to_datetime(sales_data['Invoice Date'], errors='coerce')
            
            # Count records after last training
            new_records = sales_data[sales_data['date'] > since_date]
            return len(new_records)
            
        except Exception as e:
            logger.error(f"Error counting new records: {e}")
            return 0
    
    def backup_current_models(self) -> bool:
        """Backup current models before retraining"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir = self.backup_path / f"backup_{timestamp}"
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy model files
            model_files = list(self.models_path.glob("*.pkl"))
            model_files.extend(list(self.models_path.glob("*.json")))
            
            for model_file in model_files:
                shutil.copy2(model_file, backup_dir / model_file.name)
            
            logger.info(f"✅ Models backed up to {backup_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error backing up models: {e}")
            return False
    
    def retrain_models(self) -> Dict[str, Any]:
        """Retrain all models"""
        try:
            logger.info("🔄 Starting model retraining...")
            
            # Backup current models
            if not self.backup_current_models():
                return {'status': 'error', 'error': 'Failed to backup current models'}
            
            # Load fresh data
            daily_data = self.trainer.load_sales_data()
            if daily_data.empty:
                return {'status': 'error', 'error': 'No data available for retraining'}
            
            # Retrain demand forecasting
            demand_result = self.trainer.train_demand_forecasting(daily_data)
            
            # Retrain price prediction
            price_result = self.trainer.train_price_prediction(daily_data)
            
            # Retrain anomaly detection
            anomaly_result = self.trainer.train_anomaly_detection()
            
            # Create retraining results
            retraining_results = {
                'timestamp': datetime.now().isoformat(),
                'data_records': len(daily_data),
                'demand_forecasting': demand_result,
                'price_prediction': price_result,
                'anomaly_detection': anomaly_result,
                'status': 'completed'
            }
            
            # Save results
            results_path = self.models_path / "retraining_results.json"
            with open(results_path, 'w') as f:
                json.dump(retraining_results, f, indent=2)
            
            logger.info("✅ Model retraining completed successfully")
            return retraining_results
            
        except Exception as e:
            logger.error(f"Error during retraining: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def validate_retrained_models(self) -> Dict[str, Any]:
        """Validate retrained models"""
        try:
            logger.info("🔍 Validating retrained models...")
            
            # Generate monitoring report for new models
            validation_report = self.monitor.generate_monitoring_report()
            
            # Compare with backup performance
            validation_results = {
                'timestamp': datetime.now().isoformat(),
                'validation_status': 'passed',
                'performance_comparison': {},
                'issues': []
            }
            
            # Check each model
            for model_name, performance in validation_report['performance_results'].items():
                if performance.get('status') == 'success':
                    # Check if performance is reasonable
                    if 'mape' in performance:
                        if performance['mape'] > 200:  # Very high error
                            validation_results['validation_status'] = 'failed'
                            validation_results['issues'].append(f"{model_name} MAPE too high: {performance['mape']:.2f}%")
                        
                        validation_results['performance_comparison'][model_name] = {
                            'mape': performance['mape'],
                            'mae': performance.get('mae', 0),
                            'status': 'ok' if performance['mape'] < 100 else 'warning'
                        }
                else:
                    validation_results['validation_status'] = 'failed'
                    validation_results['issues'].append(f"{model_name} failed validation: {performance.get('error', 'Unknown error')}")
            
            logger.info(f"📊 Validation status: {validation_results['validation_status']}")
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating models: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def rollback_models(self) -> bool:
        """Rollback to backup models if validation fails"""
        try:
            logger.info("🔄 Rolling back to backup models...")
            
            # Find most recent backup
            backup_dirs = sorted([d for d in self.backup_path.iterdir() if d.is_dir()], reverse=True)
            if not backup_dirs:
                logger.error("No backup available for rollback")
                return False
            
            latest_backup = backup_dirs[0]
            
            # Copy backup files back
            backup_files = list(latest_backup.glob("*.pkl"))
            backup_files.extend(list(latest_backup.glob("*.json")))
            
            for backup_file in backup_files:
                shutil.copy2(backup_file, self.models_path / backup_file.name)
            
            logger.info(f"✅ Models rolled back from {latest_backup}")
            return True
            
        except Exception as e:
            logger.error(f"Error rolling back models: {e}")
            return False
    
    def run_automated_retraining(self) -> Dict[str, Any]:
        """Run the complete automated retraining pipeline"""
        pipeline_start = datetime.now()
        logger.info("🚀 Starting Automated Retraining Pipeline")
        
        try:
            # Check if retraining is needed
            triggers = self.check_retraining_triggers()
            
            if not triggers['should_retrain']:
                logger.info("✅ No retraining needed - models are performing well")
                return {
                    'status': 'skipped',
                    'reason': 'No retraining triggers met',
                    'timestamp': pipeline_start.isoformat()
                }
            
            logger.info(f"🔔 Retraining triggered by: {', '.join(triggers['triggered_by'])}")
            for reason in triggers['reasons']:
                logger.info(f"   - {reason}")
            
            # Retrain models
            retraining_results = self.retrain_models()
            
            if retraining_results.get('status') == 'error':
                logger.error(f"❌ Retraining failed: {retraining_results.get('error')}")
                return retraining_results
            
            # Validate retrained models
            validation_results = self.validate_retrained_models()
            
            if validation_results.get('validation_status') == 'failed':
                logger.warning("⚠️ Validation failed - rolling back models")
                rollback_success = self.rollback_models()
                
                return {
                    'status': 'rolled_back',
                    'retraining_results': retraining_results,
                    'validation_results': validation_results,
                    'rollback_success': rollback_success,
                    'timestamp': pipeline_start.isoformat()
                }
            
            # Save successful retraining history
            pipeline_results = {
                'timestamp': pipeline_start.isoformat(),
                'triggers': triggers,
                'retraining_results': retraining_results,
                'validation_results': validation_results,
                'status': 'success',
                'duration_minutes': (datetime.now() - pipeline_start).total_seconds() / 60
            }
            
            self.save_retraining_history(pipeline_results)
            
            logger.info("🎉 Automated retraining pipeline completed successfully!")
            return pipeline_results
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': pipeline_start.isoformat()
            }
    
    def schedule_retraining(self, schedule_type: str = 'daily') -> str:
        """Generate cron schedule for automated retraining"""
        schedules = {
            'daily': '0 2 * * *',  # 2 AM daily
            'weekly': '0 2 * * 0',  # 2 AM Sunday
            'monthly': '0 2 1 * *'  # 2 AM first day of month
        }
        
        cron_schedule = schedules.get(schedule_type, schedules['daily'])
        
        # Create cron command
        script_path = Path(__file__).absolute()
        python_path = sys.executable
        
        cron_command = f"{cron_schedule} {python_path} {script_path} >> /var/log/beverly_knits_retraining.log 2>&1"
        
        logger.info(f"📅 Suggested cron schedule ({schedule_type}): {cron_command}")
        return cron_command

if __name__ == "__main__":
    pipeline = AutomatedRetrainingPipeline()
    
    # Run automated retraining
    results = pipeline.run_automated_retraining()
    
    # Print results
    print("\n" + "="*60)
    print("🤖 AUTOMATED RETRAINING PIPELINE RESULTS")
    print("="*60)
    print(f"Status: {results['status']}")
    print(f"Timestamp: {results['timestamp']}")
    
    if results['status'] == 'success':
        print("✅ Retraining completed successfully!")
        duration = results.get('duration_minutes', 0)
        print(f"Duration: {duration:.1f} minutes")
        
        # Show performance summary
        if 'validation_results' in results:
            validation = results['validation_results']
            print(f"\n📊 Model Performance:")
            for model_name, perf in validation.get('performance_comparison', {}).items():
                print(f"  {model_name}: MAPE {perf['mape']:.2f}% ({perf['status']})")
    
    elif results['status'] == 'skipped':
        print("✅ No retraining needed - models are performing well")
    
    elif results['status'] == 'rolled_back':
        print("⚠️ Retraining validation failed - models rolled back")
        
    else:
        print(f"❌ Pipeline failed: {results.get('error', 'Unknown error')}")
    
    print("\n💡 To schedule automatic retraining:")
    print("   Add to crontab: " + pipeline.schedule_retraining('weekly'))