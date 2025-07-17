#!/usr/bin/env python3
"""
ML Model Training Script for Beverly Knits AI Supply Chain Planner

This script trains all ML models using the current data sources:
- Sales Activity Report (historical sales data)
- Yarn Demand (time series demand data)
- Supplier and Inventory data (risk assessment)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import our ML components
from src.engine.planning_engine import PlanningEngine
from src.engine.ml_model_manager import MLModelManager
from src.engine.ml_risk_assessor import MLRiskAssessor

class MLTrainingPipeline:
    """Complete ML training pipeline for Beverly Knits data"""
    
    def __init__(self, data_path: str = "data/live/"):
        self.data_path = Path(data_path)
        self.models_path = Path("models/trained/")
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ML components
        self.planning_engine = PlanningEngine()
        self.ml_model_manager = MLModelManager()
        self.ml_risk_assessor = MLRiskAssessor()
        
        # Training results
        self.training_results = {}
        
    def load_data_sources(self) -> dict:
        """Load all available data sources"""
        logger.info("Loading data sources...")
        
        data_sources = {}
        
        try:
            # Sales Activity Report - Historical sales data
            sales_path = self.data_path / "Sales Activity Report.csv"
            if sales_path.exists():
                sales_data = pd.read_csv(sales_path, encoding='utf-8-sig')
                data_sources['sales'] = sales_data
                logger.info(f"Loaded sales data: {len(sales_data)} records")
            
            # Yarn Demand - Time series demand data
            demand_path = self.data_path / "Yarn_Demand_2025-06-27_0442.csv"
            if demand_path.exists():
                demand_data = pd.read_csv(demand_path, encoding='utf-8-sig')
                data_sources['demand'] = demand_data
                logger.info(f"Loaded demand data: {len(demand_data)} records")
            
            # Inventory data
            inventory_path = self.data_path / "Yarn_ID_Current_Inventory.csv"
            if inventory_path.exists():
                inventory_data = pd.read_csv(inventory_path, encoding='utf-8-sig')
                data_sources['inventory'] = inventory_data
                logger.info(f"Loaded inventory data: {len(inventory_data)} records")
            
            # Style BOM data
            bom_path = self.data_path / "Style_BOM.csv"
            if bom_path.exists():
                bom_data = pd.read_csv(bom_path, encoding='utf-8-sig')
                data_sources['bom'] = bom_data
                logger.info(f"Loaded BOM data: {len(bom_data)} records")
            
            # Supplier data
            supplier_path = self.data_path / "Supplier_ID.csv"
            if supplier_path.exists():
                supplier_data = pd.read_csv(supplier_path, encoding='utf-8-sig')
                data_sources['suppliers'] = supplier_data
                logger.info(f"Loaded supplier data: {len(supplier_data)} records")
                
        except Exception as e:
            logger.error(f"Error loading data sources: {e}")
            
        return data_sources
    
    def prepare_time_series_data(self, data_sources: dict) -> pd.DataFrame:
        """Prepare time series data for forecasting models"""
        logger.info("Preparing time series data...")
        
        try:
            # Method 1: Use Sales Activity Report for historical demand
            if 'sales' in data_sources:
                sales_data = data_sources['sales'].copy()
                
                # Convert Invoice Date to datetime
                sales_data['Invoice Date'] = pd.to_datetime(sales_data['Invoice Date'])
                
                # Clean and convert numeric columns
                sales_data['Yds_ordered'] = pd.to_numeric(sales_data['Yds_ordered'], errors='coerce')
                sales_data['Line Price'] = sales_data['Line Price'].str.replace('$', '').str.replace(',', '')
                sales_data['Line Price'] = pd.to_numeric(sales_data['Line Price'], errors='coerce')
                
                # Aggregate daily demand by date
                daily_demand = sales_data.groupby('Invoice Date').agg({
                    'Yds_ordered': 'sum',
                    'Line Price': 'sum'
                }).reset_index()
                
                # Create complete date range
                date_range = pd.date_range(
                    start=daily_demand['Invoice Date'].min(),
                    end=daily_demand['Invoice Date'].max(),
                    freq='D'
                )
                
                # Reindex to fill missing dates
                daily_demand = daily_demand.set_index('Invoice Date')
                daily_demand = daily_demand.reindex(date_range, fill_value=0)
                daily_demand.index.name = 'date'
                
                # Rename columns for ML models
                daily_demand = daily_demand.rename(columns={
                    'Yds_ordered': 'demand',
                    'Line Price': 'revenue'
                })
                
                logger.info(f"Prepared time series data: {len(daily_demand)} daily records")
                logger.info(f"Date range: {daily_demand.index.min()} to {daily_demand.index.max()}")
                logger.info(f"Average daily demand: {daily_demand['demand'].mean():.2f} yards")
                
                return daily_demand
                
            # Method 2: Use Yarn Demand data (weekly format)
            elif 'demand' in data_sources:
                demand_data = data_sources['demand'].copy()
                
                # Extract weekly demand columns
                demand_cols = [col for col in demand_data.columns if 'Demand Week' in col]
                
                if demand_cols:
                    # Create time series from weekly data
                    weekly_demand = []
                    base_date = datetime(2025, 6, 27)  # Based on filename
                    
                    for i, col in enumerate(demand_cols):
                        week_num = int(col.split('Week ')[1])
                        week_date = base_date + timedelta(weeks=week_num-27)
                        
                        total_demand = demand_data[col].sum()
                        weekly_demand.append({
                            'date': week_date,
                            'demand': total_demand
                        })
                    
                    weekly_df = pd.DataFrame(weekly_demand)
                    weekly_df = weekly_df.set_index('date')
                    
                    # Convert to daily by interpolation
                    daily_range = pd.date_range(
                        start=weekly_df.index.min(),
                        end=weekly_df.index.max(),
                        freq='D'
                    )
                    
                    daily_demand = weekly_df.reindex(daily_range).interpolate()
                    daily_demand.index.name = 'date'
                    
                    logger.info(f"Prepared time series from weekly data: {len(daily_demand)} daily records")
                    return daily_demand
                    
        except Exception as e:
            logger.error(f"Error preparing time series data: {e}")
            
        # Fallback: Generate synthetic data for training
        logger.warning("Using synthetic data for training")
        return self.generate_synthetic_time_series()
    
    def generate_synthetic_time_series(self) -> pd.DataFrame:
        """Generate synthetic time series data for training"""
        logger.info("Generating synthetic time series data...")
        
        # Create 2 years of daily data
        date_range = pd.date_range(
            start='2023-01-01',
            end='2024-12-31',
            freq='D'
        )
        
        # Generate realistic demand pattern
        n_days = len(date_range)
        
        # Base demand with seasonal pattern
        base_demand = 1000
        seasonal_pattern = 200 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)
        weekly_pattern = 50 * np.sin(2 * np.pi * np.arange(n_days) / 7)
        
        # Add trend and noise
        trend = np.linspace(0, 100, n_days)
        noise = np.random.normal(0, 50, n_days)
        
        demand = base_demand + seasonal_pattern + weekly_pattern + trend + noise
        demand = np.maximum(demand, 0)  # Ensure non-negative
        
        synthetic_data = pd.DataFrame({
            'demand': demand,
            'revenue': demand * np.random.uniform(8, 12, n_days)  # Price per yard
        }, index=date_range)
        
        synthetic_data.index.name = 'date'
        
        logger.info(f"Generated synthetic data: {len(synthetic_data)} records")
        return synthetic_data
    
    def prepare_supplier_risk_data(self, data_sources: dict) -> pd.DataFrame:
        """Prepare supplier data for risk assessment training"""
        logger.info("Preparing supplier risk data...")
        
        try:
            # Combine inventory and supplier data
            supplier_data = []
            
            if 'inventory' in data_sources and 'suppliers' in data_sources:
                inventory_df = data_sources['inventory'].copy()
                supplier_df = data_sources['suppliers'].copy()
                
                # Merge inventory with supplier data
                for _, row in inventory_df.iterrows():
                    supplier_info = {
                        'supplier_id': row.get('Supplier', 'Unknown'),
                        'material_id': row.get('Yarn_ID', 'Unknown'),
                        'cost_per_unit': self._parse_cost(row.get('Cost_Pound', '$0')),
                        'current_inventory': self._parse_numeric(row.get('Inventory', '0')),
                        'on_order': self._parse_numeric(row.get('On_Order', '0')),
                        'total_cost': self._parse_numeric(row.get('Total_Cost', '0')),
                        'lead_time_days': np.random.randint(7, 45),  # Estimated
                        'moq_amount': max(100, abs(self._parse_numeric(row.get('On_Order', '100')))),
                        'reliability_score': max(0.5, min(1.0, np.random.uniform(0.7, 0.95))),
                        'quality_score': max(0.6, min(1.0, np.random.uniform(0.8, 1.0))),
                        'created_at': datetime.now() - timedelta(days=np.random.randint(30, 365))
                    }
                    supplier_data.append(supplier_info)
                    
            else:
                # Generate synthetic supplier data
                suppliers = ['FERR', 'DECA GLOBAL', 'MIKE BECKER INC', 'PROMPTEX YARNS', 'DUNAWAY YARNS', 'VERTEX', 'CAP YARNS LLC']
                materials = [f'MAT{i:03d}' for i in range(1, 51)]
                
                for i in range(200):
                    supplier_info = {
                        'supplier_id': np.random.choice(suppliers),
                        'material_id': np.random.choice(materials),
                        'cost_per_unit': np.random.uniform(1.0, 15.0),
                        'current_inventory': np.random.uniform(-1000, 5000),
                        'on_order': np.random.uniform(0, 10000),
                        'total_cost': np.random.uniform(100, 50000),
                        'lead_time_days': np.random.randint(7, 60),
                        'moq_amount': np.random.randint(100, 2000),
                        'reliability_score': np.random.uniform(0.6, 1.0),
                        'quality_score': np.random.uniform(0.7, 1.0),
                        'created_at': datetime.now() - timedelta(days=np.random.randint(1, 365))
                    }
                    supplier_data.append(supplier_info)
            
            supplier_df = pd.DataFrame(supplier_data)
            logger.info(f"Prepared supplier risk data: {len(supplier_df)} records")
            
            return supplier_df
            
        except Exception as e:
            logger.error(f"Error preparing supplier risk data: {e}")
            return pd.DataFrame()
    
    def _parse_cost(self, cost_str: str) -> float:
        """Parse cost string to float"""
        try:
            if isinstance(cost_str, str):
                cost_str = cost_str.replace('$', '').replace(',', '')
            return float(cost_str)
        except:
            return 0.0
    
    def _parse_numeric(self, value: str) -> float:
        """Parse numeric string to float"""
        try:
            if isinstance(value, str):
                value = value.replace(',', '').replace('$', '').replace('(', '-').replace(')', '')
            return float(value)
        except:
            return 0.0
    
    def train_forecasting_models(self, time_series_data: pd.DataFrame) -> dict:
        """Train all forecasting models"""
        logger.info("Training forecasting models...")
        
        training_results = {}
        
        # Split data for training and validation
        split_date = time_series_data.index[-30]  # Last 30 days for validation
        train_data = time_series_data[time_series_data.index < split_date]
        val_data = time_series_data[time_series_data.index >= split_date]
        
        logger.info(f"Training data: {len(train_data)} records")
        logger.info(f"Validation data: {len(val_data)} records")
        
        # Try to train each model
        models_to_train = [
            ('ARIMA', 'arima'),
            ('Prophet', 'prophet'),
            ('LSTM', 'lstm'),
            ('XGBoost', 'xgboost')
        ]
        
        for model_name, model_key in models_to_train:
            try:
                logger.info(f"Training {model_name} model...")
                
                # Train model using the ML model manager
                success = self.ml_model_manager.train_model(
                    model_name=model_key,
                    data=train_data,
                    target_column='demand'
                )
                
                if success:
                    logger.info(f"✅ {model_name} model trained successfully")
                    
                    # Generate predictions for validation
                    predictions = self.ml_model_manager.predict(model_key, val_data)
                    
                    if predictions is not None:
                        # Calculate metrics
                        actual = val_data['demand'].values
                        pred_values = predictions[:len(actual)]
                        
                        mape = np.mean(np.abs((actual - pred_values) / actual)) * 100
                        mae = np.mean(np.abs(actual - pred_values))
                        rmse = np.sqrt(np.mean((actual - pred_values) ** 2))
                        
                        training_results[model_key] = {
                            'status': 'success',
                            'mape': mape,
                            'mae': mae,
                            'rmse': rmse,
                            'training_samples': len(train_data)
                        }
                        
                        logger.info(f"   MAPE: {mape:.2f}%")
                        logger.info(f"   MAE: {mae:.2f}")
                        logger.info(f"   RMSE: {rmse:.2f}")
                    else:
                        logger.warning(f"   Could not generate predictions for {model_name}")
                        training_results[model_key] = {
                            'status': 'trained_no_predictions',
                            'training_samples': len(train_data)
                        }
                else:
                    logger.error(f"❌ {model_name} model training failed")
                    training_results[model_key] = {
                        'status': 'failed',
                        'error': 'Training failed'
                    }
                    
            except Exception as e:
                logger.error(f"❌ Error training {model_name}: {e}")
                training_results[model_key] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return training_results
    
    def train_risk_assessment_models(self, supplier_data: pd.DataFrame) -> dict:
        """Train risk assessment models"""
        logger.info("Training risk assessment models...")
        
        training_results = {}
        
        try:
            # Train risk model
            logger.info("Training supplier risk model...")
            self.ml_risk_assessor.train_risk_model(supplier_data)
            
            # Train anomaly detector
            logger.info("Training anomaly detection model...")
            self.ml_risk_assessor.train_anomaly_detector(supplier_data)
            
            # Test risk assessment
            risk_scores = self.ml_risk_assessor.predict_supplier_risk(supplier_data.head(10))
            anomalies = self.ml_risk_assessor.detect_anomalies(supplier_data.head(10))
            
            training_results['risk_assessment'] = {
                'status': 'success',
                'risk_model_trained': self.ml_risk_assessor.is_risk_model_trained,
                'anomaly_detector_trained': self.ml_risk_assessor.is_anomaly_detector_trained,
                'test_risk_scores': len(risk_scores),
                'test_anomalies': len(anomalies),
                'training_samples': len(supplier_data)
            }
            
            logger.info(f"✅ Risk assessment models trained successfully")
            logger.info(f"   Risk scores generated: {len(risk_scores)}")
            logger.info(f"   Anomalies detected: {sum(1 for a in anomalies if a.is_anomaly)}")
            
        except Exception as e:
            logger.error(f"❌ Error training risk assessment models: {e}")
            training_results['risk_assessment'] = {
                'status': 'error',
                'error': str(e)
            }
        
        return training_results
    
    def save_training_results(self, training_results: dict):
        """Save training results to file"""
        logger.info("Saving training results...")
        
        try:
            # Save results as JSON
            results_path = self.models_path / "training_results.json"
            
            # Convert datetime objects to strings for JSON serialization
            serializable_results = {}
            for key, value in training_results.items():
                if isinstance(value, dict):
                    serializable_results[key] = {}
                    for k, v in value.items():
                        if isinstance(v, datetime):
                            serializable_results[key][k] = v.isoformat()
                        else:
                            serializable_results[key][k] = v
                else:
                    serializable_results[key] = value
            
            with open(results_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Training results saved to {results_path}")
            
            # Save summary report
            report_path = self.models_path / "training_report.txt"
            with open(report_path, 'w') as f:
                f.write("Beverly Knits ML Model Training Report\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("Forecasting Models:\n")
                f.write("-" * 20 + "\n")
                for model in ['arima', 'prophet', 'lstm', 'xgboost']:
                    if model in training_results:
                        result = training_results[model]
                        f.write(f"{model.upper()}: {result['status']}\n")
                        if result['status'] == 'success':
                            f.write(f"  MAPE: {result['mape']:.2f}%\n")
                            f.write(f"  MAE: {result['mae']:.2f}\n")
                            f.write(f"  RMSE: {result['rmse']:.2f}\n")
                        f.write("\n")
                
                f.write("Risk Assessment Models:\n")
                f.write("-" * 20 + "\n")
                if 'risk_assessment' in training_results:
                    result = training_results['risk_assessment']
                    f.write(f"Risk Assessment: {result['status']}\n")
                    if result['status'] == 'success':
                        f.write(f"  Risk Model Trained: {result['risk_model_trained']}\n")
                        f.write(f"  Anomaly Detector Trained: {result['anomaly_detector_trained']}\n")
                        f.write(f"  Training Samples: {result['training_samples']}\n")
                
            logger.info(f"Training report saved to {report_path}")
            
        except Exception as e:
            logger.error(f"Error saving training results: {e}")
    
    def run_complete_training(self):
        """Run the complete training pipeline"""
        logger.info("🚀 Starting ML Model Training Pipeline for Beverly Knits")
        logger.info("=" * 60)
        
        try:
            # Step 1: Load data sources
            data_sources = self.load_data_sources()
            
            if not data_sources:
                logger.error("No data sources found! Please check data directory.")
                return
            
            # Step 2: Prepare time series data
            time_series_data = self.prepare_time_series_data(data_sources)
            
            # Step 3: Train forecasting models
            forecasting_results = self.train_forecasting_models(time_series_data)
            
            # Step 4: Prepare supplier risk data
            supplier_data = self.prepare_supplier_risk_data(data_sources)
            
            # Step 5: Train risk assessment models
            if not supplier_data.empty:
                risk_results = self.train_risk_assessment_models(supplier_data)
            else:
                risk_results = {'risk_assessment': {'status': 'no_data'}}
            
            # Step 6: Combine results
            all_results = {
                **forecasting_results,
                **risk_results,
                'training_completed': datetime.now().isoformat(),
                'data_sources_used': list(data_sources.keys()),
                'time_series_records': len(time_series_data),
                'supplier_records': len(supplier_data) if not supplier_data.empty else 0
            }
            
            # Step 7: Save results
            self.save_training_results(all_results)
            
            # Step 8: Print summary
            self.print_training_summary(all_results)
            
            logger.info("✅ ML Model Training Pipeline Completed Successfully!")
            
        except Exception as e:
            logger.error(f"❌ Training pipeline failed: {e}")
            raise
    
    def print_training_summary(self, results: dict):
        """Print training summary"""
        print("\n" + "=" * 60)
        print("🎉 TRAINING SUMMARY")
        print("=" * 60)
        
        print(f"📊 Data Sources Used: {', '.join(results.get('data_sources_used', []))}")
        print(f"📈 Time Series Records: {results.get('time_series_records', 0)}")
        print(f"🏭 Supplier Records: {results.get('supplier_records', 0)}")
        
        print("\n🤖 FORECASTING MODELS:")
        print("-" * 30)
        for model in ['arima', 'prophet', 'lstm', 'xgboost']:
            if model in results:
                result = results[model]
                status = result['status']
                if status == 'success':
                    print(f"  ✅ {model.upper()}: MAPE {result['mape']:.2f}%")
                else:
                    print(f"  ❌ {model.upper()}: {status}")
        
        print("\n⚠️  RISK ASSESSMENT:")
        print("-" * 30)
        if 'risk_assessment' in results:
            result = results['risk_assessment']
            if result['status'] == 'success':
                print(f"  ✅ Risk Model: Trained")
                print(f"  ✅ Anomaly Detector: Trained")
                print(f"  📊 Training Samples: {result['training_samples']}")
            else:
                print(f"  ❌ Risk Assessment: {result['status']}")
        
        print("\n📁 FILES CREATED:")
        print("-" * 30)
        print(f"  📄 Training Results: models/trained/training_results.json")
        print(f"  📄 Training Report: models/trained/training_report.txt")
        print(f"  📄 Training Log: ml_training.log")
        
        print("\n🚀 NEXT STEPS:")
        print("-" * 30)
        print("  1. Run the Streamlit app: streamlit run main.py")
        print("  2. Navigate to ML Forecasting page")
        print("  3. Test trained models with real data")
        print("  4. Monitor model performance over time")
        print("=" * 60)

if __name__ == "__main__":
    # Create and run training pipeline
    trainer = MLTrainingPipeline()
    trainer.run_complete_training()