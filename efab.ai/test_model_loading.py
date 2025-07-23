#!/usr/bin/env python3
"""
Test ML model loading to diagnose the issue
"""

import joblib
import numpy as np
from pathlib import Path

def test_model_loading():
    """Test loading and using individual models"""
    model_dir = Path("models/trained")
    
    if not model_dir.exists():
        print("❌ No models directory found")
        return
    
    model_files = list(model_dir.glob("*.pkl"))
    print(f"Found {len(model_files)} model files")
    
    for model_file in model_files:
        print(f"\n🔍 Testing model: {model_file.name}")
        
        try:
            # Load the model
            model = joblib.load(model_file)
            print(f"   ✅ Loaded successfully")
            print(f"   📊 Model type: {type(model)}")
            
            # Check if it has predict method
            if hasattr(model, 'predict'):
                print(f"   ✅ Has predict method")
                
                # Try to predict with dummy data
                if 'demand' in model_file.name.lower():
                    # Demand forecasting models expect 10 features
                    test_features = np.array([[6, 1000, 1.0, 2, 1, 0, 0, 0, 0, 0]])
                    try:
                        prediction = model.predict(test_features)
                        print(f"   ✅ Prediction works: {prediction}")
                    except Exception as e:
                        print(f"   ❌ Prediction failed: {e}")
                        # Try with different feature count
                        for n_features in [5, 7, 8, 9, 11, 12]:
                            try:
                                test_features = np.random.rand(1, n_features)
                                prediction = model.predict(test_features)
                                print(f"   ✅ Works with {n_features} features: {prediction}")
                                break
                            except:
                                continue
                
                elif 'inventory' in model_file.name.lower() or 'optimizer' in model_file.name.lower():
                    # Inventory models expect 5 features
                    test_features = np.array([[500, 21, 0.2, 50, 0.85]])
                    try:
                        prediction = model.predict(test_features)
                        print(f"   ✅ Prediction works: {prediction}")
                    except Exception as e:
                        print(f"   ❌ Prediction failed: {e}")
                
                elif 'supplier' in model_file.name.lower():
                    # Supplier models expect 10 features
                    test_features = np.array([[21, 4.0, 0.15, 5, 30, 1, 0, 0, 0, 0]])
                    try:
                        prediction = model.predict(test_features)
                        print(f"   ✅ Prediction works: {prediction}")
                    except Exception as e:
                        print(f"   ❌ Prediction failed: {e}")
                
                else:
                    print(f"   ⚠️ Unknown model type, skipping prediction test")
                    
            else:
                print(f"   ❌ No predict method")
                print(f"   📝 Available methods: {dir(model)}")
                
        except Exception as e:
            print(f"   ❌ Failed to load: {e}")

if __name__ == "__main__":
    test_model_loading()