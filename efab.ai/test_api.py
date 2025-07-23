#!/usr/bin/env python3
"""
Test script for Beverly Knits AI Supply Chain Planner API
"""

import requests
import json
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:8000"

def test_api_endpoints():
    """Test basic API endpoints"""
    print("🧪 Testing Beverly Knits AI Supply Chain Planner API")
    print("=" * 60)
    
    # Test health check
    print("\n1. Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Status: {response.json()['status']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    # Test API info
    print("\n2. Testing API info...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/info")
        if response.status_code == 200:
            print("✅ API info retrieved")
            info = response.json()
            print(f"   API: {info['api_name']}")
            print(f"   Version: {info['version']}")
        else:
            print(f"❌ API info failed: {response.status_code}")
    except Exception as e:
        print(f"❌ API info error: {e}")
    
    # Test authentication
    print("\n3. Testing authentication...")
    try:
        # Login with demo credentials
        login_data = {
            "username": "admin",
            "password": "admin123"
        }
        response = requests.post(f"{BASE_URL}/api/v1/auth/login", json=login_data)
        if response.status_code == 200:
            print("✅ Authentication successful")
            auth_data = response.json()
            access_token = auth_data['access_token']
            print(f"   Token type: {auth_data['token_type']}")
            print(f"   Expires in: {auth_data['expires_in']} seconds")
            
            # Test authenticated endpoint
            headers = {"Authorization": f"Bearer {access_token}"}
            response = requests.get(f"{BASE_URL}/api/v1/auth/me", headers=headers)
            if response.status_code == 200:
                print("✅ Token validation successful")
                user = response.json()
                print(f"   User: {user['username']} ({user['role']})")
            else:
                print(f"❌ Token validation failed: {response.status_code}")
                
        else:
            print(f"❌ Authentication failed: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"❌ Authentication error: {e}")
    
    # Test materials endpoint
    print("\n4. Testing materials endpoint...")
    try:
        # Get access token first
        login_data = {"username": "admin", "password": "admin123"}
        auth_response = requests.post(f"{BASE_URL}/api/v1/auth/login", json=login_data)
        if auth_response.status_code == 200:
            token = auth_response.json()['access_token']
            headers = {"Authorization": f"Bearer {token}"}
            
            response = requests.get(f"{BASE_URL}/api/v1/materials/", headers=headers)
            if response.status_code == 200:
                print("✅ Materials endpoint successful")
                materials = response.json()
                print(f"   Total materials: {materials['total']}")
                print(f"   Materials returned: {len(materials['materials'])}") 
            else:
                print(f"❌ Materials endpoint failed: {response.status_code}")
        else:
            print("❌ Could not get auth token for materials test")
    except Exception as e:
        print(f"❌ Materials endpoint error: {e}")
    
    # Test suppliers endpoint
    print("\n5. Testing suppliers endpoint...")
    try:
        # Get access token first
        login_data = {"username": "admin", "password": "admin123"}
        auth_response = requests.post(f"{BASE_URL}/api/v1/auth/login", json=login_data)
        if auth_response.status_code == 200:
            token = auth_response.json()['access_token']
            headers = {"Authorization": f"Bearer {token}"}
            
            response = requests.get(f"{BASE_URL}/api/v1/suppliers/", headers=headers)
            if response.status_code == 200:
                print("✅ Suppliers endpoint successful")
                suppliers = response.json()
                print(f"   Total suppliers: {suppliers['total']}")
                print(f"   Suppliers returned: {len(suppliers['suppliers'])}")
            else:
                print(f"❌ Suppliers endpoint failed: {response.status_code}")
        else:
            print("❌ Could not get auth token for suppliers test")
    except Exception as e:
        print(f"❌ Suppliers endpoint error: {e}")
    
    # Test analytics endpoint
    print("\n6. Testing analytics endpoint...")
    try:
        # Get access token first
        login_data = {"username": "admin", "password": "admin123"}
        auth_response = requests.post(f"{BASE_URL}/api/v1/auth/login", json=login_data)
        if auth_response.status_code == 200:
            token = auth_response.json()['access_token']
            headers = {"Authorization": f"Bearer {token}"}
            
            response = requests.get(f"{BASE_URL}/api/v1/analytics/dashboard", headers=headers)
            if response.status_code == 200:
                print("✅ Analytics endpoint successful")
                analytics = response.json()
                print(f"   Metrics returned: {len(analytics['metrics'])}")
                print(f"   Charts returned: {len(analytics['charts'])}")
            else:
                print(f"❌ Analytics endpoint failed: {response.status_code}")
        else:
            print("❌ Could not get auth token for analytics test")
    except Exception as e:
        print(f"❌ Analytics endpoint error: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 API testing completed!")
    print("\n📚 API Documentation available at: http://localhost:8000/docs")
    print("📋 Alternative docs at: http://localhost:8000/redoc")

if __name__ == "__main__":
    test_api_endpoints()