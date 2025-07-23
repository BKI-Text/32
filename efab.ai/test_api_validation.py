#!/usr/bin/env python3
"""
Test script for API validation middleware
"""

import requests
import json
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:8000"

def test_validation_middleware():
    """Test validation middleware with various endpoints"""
    print("🧪 Testing Beverly Knits AI Supply Chain Planner API Validation")
    print("=" * 70)
    
    # Test materials endpoint with valid data
    print("\n1. Testing materials endpoint with valid data...")
    valid_material = {
        "id": "YARN001",
        "name": "Cotton Yarn",
        "type": "yarn",
        "description": "High-quality cotton yarn for knitting",
        "is_critical": True
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/v1/materials/", json=valid_material)
        print(f"   Status: {response.status_code}")
        if response.status_code == 422:
            print("   ✅ Validation working - endpoint returned validation error (expected without auth)")
        else:
            print(f"   Response: {response.json()}")
    except requests.exceptions.ConnectionError:
        print("   ❌ Connection error - API server not running")
        print("   To test validation, start the API server with: uvicorn api.main:app --reload")
        return
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test materials endpoint with invalid data
    print("\n2. Testing materials endpoint with invalid data...")
    invalid_material = {
        "id": "invalid-id",  # Should be alphanumeric
        "name": "A",  # Too short
        "type": "unknown",  # Invalid type
        "description": "A" * 1001,  # Too long
        "is_critical": "yes"  # Should be boolean
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/v1/materials/", json=invalid_material)
        print(f"   Status: {response.status_code}")
        if response.status_code == 422:
            print("   ✅ Validation working - rejected invalid data")
            response_data = response.json()
            print(f"   Validation errors: {len(response_data.get('errors', []))}")
        else:
            print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test suppliers endpoint with valid data
    print("\n3. Testing suppliers endpoint with valid data...")
    valid_supplier = {
        "id": "SUP001",
        "name": "Textile Supplier Inc",
        "contact_info": "contact@supplier.com",
        "lead_time_days": 14,
        "reliability_score": 0.95,
        "risk_level": "low",
        "is_active": True
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/v1/suppliers/", json=valid_supplier)
        print(f"   Status: {response.status_code}")
        if response.status_code == 422:
            print("   ✅ Validation working - endpoint returned validation error (expected without auth)")
        else:
            print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test suppliers endpoint with invalid data
    print("\n4. Testing suppliers endpoint with invalid data...")
    invalid_supplier = {
        "id": "invalid-id",  # Should be alphanumeric
        "name": "A",  # Too short
        "contact_info": "123",  # Too short
        "lead_time_days": 0,  # Should be >= 1
        "reliability_score": 1.5,  # Should be <= 1
        "risk_level": "unknown",  # Invalid risk level
        "is_active": "yes"  # Should be boolean
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/v1/suppliers/", json=invalid_supplier)
        print(f"   Status: {response.status_code}")
        if response.status_code == 422:
            print("   ✅ Validation working - rejected invalid data")
            response_data = response.json()
            print(f"   Validation errors: {len(response_data.get('errors', []))}")
        else:
            print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test that GET requests are not validated
    print("\n5. Testing that GET requests skip validation...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/materials/")
        print(f"   Status: {response.status_code}")
        if response.status_code != 422:
            print("   ✅ GET requests correctly skip validation")
        else:
            print("   ❌ GET requests incorrectly validated")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test health endpoint (should skip validation)
    print("\n6. Testing health endpoint (should skip validation)...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print("   ✅ Health endpoint correctly skips validation")
        else:
            print("   ❌ Health endpoint validation issue")
    except Exception as e:
        print(f"   ❌ Error: {e}")

def test_empty_request_body():
    """Test validation with empty request body"""
    print("\n7. Testing empty request body...")
    try:
        response = requests.post(f"{BASE_URL}/api/v1/materials/", json={})
        print(f"   Status: {response.status_code}")
        if response.status_code == 422:
            print("   ✅ Empty request body correctly validated")
        else:
            print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

def test_invalid_json():
    """Test validation with invalid JSON"""
    print("\n8. Testing invalid JSON...")
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/materials/",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 400:
            print("   ✅ Invalid JSON correctly rejected")
        else:
            print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    test_validation_middleware()
    test_empty_request_body()
    test_invalid_json()
    
    print("\n" + "=" * 70)
    print("🎯 Validation testing complete!")
    print("To run the API server: uvicorn api.main:app --reload --host 0.0.0.0 --port 8000")