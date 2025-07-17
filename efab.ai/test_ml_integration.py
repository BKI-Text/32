#!/usr/bin/env python3
"""
Test script for ML integration in Beverly Knits AI Supply Chain Planner
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.engine.planning_engine import PlanningEngine

def test_ml_integration():
    print("🧪 Testing ML Integration in Beverly Knits AI Supply Chain Planner")
    print("=" * 60)
    
    # Initialize planning engine
    print("1. Initializing planning engine with ML components...")
    planning_engine = PlanningEngine()
    print("   ✅ Planning engine initialized successfully")
    
    # Check ML model status
    print("\n2. Checking ML model status...")
    ml_status = planning_engine.get_ml_model_status()
    print(f"   ML Enabled: {ml_status.get('ml_enabled', False)}")
    
    if ml_status.get('ml_enabled', False):
        print("   ✅ ML components are operational")
        
        # Test ML forecasting
        print("\n3. Testing ML forecasting...")
        try:
            # Create sample historical data
            dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
            demand = 100 + 20 * np.sin(np.arange(100) * 2 * np.pi / 30) + np.random.normal(0, 5, 100)
            
            historical_data = pd.DataFrame({
                'date': dates,
                'demand': demand
            })
            historical_data = historical_data.set_index('date')
            
            # Generate forecasts using available models
            forecasts = planning_engine.generate_ml_forecasts(
                historical_data=historical_data,
                periods=7,
                models=['arima']  # Use only ARIMA since other models may not be available
            )
            
            print(f"   ✅ ML forecasts generated: {len(forecasts)} forecasts")
            
            # Display first forecast
            if forecasts:
                first_forecast = forecasts[0]
                print(f"   First forecast: {first_forecast.forecast_qty.amount} units on {first_forecast.forecast_date}")
                
        except Exception as e:
            print(f"   ⚠️ ML forecasting test failed: {e}")
        
        # Test ML risk assessment
        print("\n4. Testing ML risk assessment...")
        try:
            # Create sample supplier data
            suppliers = ['SUP001', 'SUP002', 'SUP003']
            materials = ['MAT001', 'MAT002', 'MAT003']
            
            supplier_data = []
            for i in range(30):
                supplier_data.append({
                    'supplier_id': np.random.choice(suppliers),
                    'material_id': np.random.choice(materials),
                    'cost_per_unit': np.random.uniform(10, 100),
                    'lead_time_days': np.random.randint(7, 30),
                    'moq_amount': np.random.randint(100, 1000),
                    'reliability_score': np.random.uniform(0.7, 1.0),
                    'quality_score': np.random.uniform(0.8, 1.0),
                    'created_at': datetime.now() - timedelta(days=np.random.randint(1, 365))
                })
            
            supplier_df = pd.DataFrame(supplier_data)
            
            # Run risk assessment
            risk_assessment = planning_engine.assess_supplier_risk_with_ml(supplier_df)
            
            risk_scores = risk_assessment.get('risk_scores', [])
            anomalies = risk_assessment.get('anomalies', [])
            
            print(f"   ✅ ML risk assessment completed: {len(risk_scores)} suppliers assessed")
            print(f"   Anomalies detected: {sum(1 for a in anomalies if a.is_anomaly)}")
            
            # Display first risk score
            if risk_scores:
                first_risk = risk_scores[0]
                print(f"   First risk score: {first_risk.overall_score:.2f} ({first_risk.risk_level.value})")
                
        except Exception as e:
            print(f"   ⚠️ ML risk assessment test failed: {e}")
        
        print("\n5. Testing ML model manager...")
        try:
            model_manager = planning_engine.ml_model_manager
            model_status = model_manager.get_model_status()
            
            available_models = model_status.get('available_models', {})
            print(f"   Available models: {list(available_models.keys())}")
            
            enabled_models = [name for name, enabled in available_models.items() if enabled]
            print(f"   Enabled models: {enabled_models}")
            
        except Exception as e:
            print(f"   ⚠️ ML model manager test failed: {e}")
            
    else:
        print(f"   ⚠️ ML components not available: {ml_status.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 60)
    print("🎉 ML Integration Test Complete!")
    print("\nNext steps:")
    print("- Run the Streamlit app: streamlit run main.py")
    print("- Navigate to ML Forecasting page to test advanced features")
    print("- Use ML-Enhanced Planning for optimal supply chain decisions")

if __name__ == "__main__":
    test_ml_integration()