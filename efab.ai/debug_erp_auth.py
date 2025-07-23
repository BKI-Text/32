#!/usr/bin/env python3
"""
Debug ERP Authentication
Beverly Knits AI Supply Chain Planner
"""

import sys
import os
import logging
import requests
from datetime import datetime
import json
import re

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_erp_auth():
    """Debug ERP authentication process"""
    logger.info("🔍 Debugging ERP Authentication")
    
    # Create session
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    })
    
    base_url = 'https://efab.bkiapps.com'
    username = 'psytz'
    password = 'big$cat'
    
    try:
        # Step 1: Get main page
        logger.info("🔗 Step 1: Getting main page")
        response = session.get(base_url)
        logger.info(f"Status: {response.status_code}")
        logger.info(f"URL: {response.url}")
        logger.info(f"Content length: {len(response.content)}")
        
        # Save HTML for analysis
        with open('debug_main_page.html', 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        # Look for forms
        forms = re.findall(r'<form[^>]*>.*?</form>', response.text, re.DOTALL | re.IGNORECASE)
        logger.info(f"Found {len(forms)} forms")
        
        for i, form in enumerate(forms):
            logger.info(f"Form {i+1}: {form[:200]}...")
            
            # Extract form action
            action_match = re.search(r'action=["\']([^"\']*)["\']', form, re.IGNORECASE)
            if action_match:
                logger.info(f"  Action: {action_match.group(1)}")
            
            # Extract input fields
            inputs = re.findall(r'<input[^>]*>', form, re.IGNORECASE)
            logger.info(f"  Found {len(inputs)} inputs")
            
            for input_tag in inputs:
                name_match = re.search(r'name=["\']([^"\']+)["\']', input_tag)
                type_match = re.search(r'type=["\']([^"\']+)["\']', input_tag)
                value_match = re.search(r'value=["\']([^"\']*)["\']', input_tag)
                
                if name_match:
                    name = name_match.group(1)
                    input_type = type_match.group(1) if type_match else 'text'
                    value = value_match.group(1) if value_match else ''
                    logger.info(f"    {name} ({input_type}): {value}")
        
        # Step 2: Try different login approaches
        logger.info("\n🔐 Step 2: Testing login approaches")
        
        # Approach 1: Look for login form specifically
        login_form_pattern = r'<form[^>]*(?:login|auth)[^>]*>.*?</form>'
        login_forms = re.findall(login_form_pattern, response.text, re.DOTALL | re.IGNORECASE)
        
        if login_forms:
            logger.info(f"Found {len(login_forms)} login forms")
            for i, form in enumerate(login_forms):
                logger.info(f"Login form {i+1}: {form[:300]}...")
        
        # Approach 2: Try POST to main page with different field names
        field_combinations = [
            {'username': username, 'password': password},
            {'email': username, 'password': password},
            {'login': username, 'pass': password},
            {'user': username, 'pwd': password},
            {'userid': username, 'passwd': password}
        ]
        
        for i, fields in enumerate(field_combinations):
            logger.info(f"\n🧪 Testing field combination {i+1}: {list(fields.keys())}")
            
            try:
                login_response = session.post(base_url, data=fields, allow_redirects=True)
                logger.info(f"  Status: {login_response.status_code}")
                logger.info(f"  Final URL: {login_response.url}")
                logger.info(f"  Content length: {len(login_response.content)}")
                
                # Check if we got redirected or content changed
                if login_response.url != base_url:
                    logger.info(f"  ✅ Redirected to: {login_response.url}")
                    
                    # Try accessing protected endpoints
                    test_endpoints = [
                        '/yarn',
                        '/report/expected_yarn', 
                        '/report/yarn_demand',
                        '/fabric/so/list',
                        '/yarn/po/list'
                    ]
                    
                    for endpoint in test_endpoints:
                        try:
                            test_url = f"{base_url}{endpoint}"
                            test_response = session.get(test_url)
                            logger.info(f"    {endpoint}: {test_response.status_code}")
                            
                            if test_response.status_code == 200:
                                # Check if we got actual data or another login page
                                if 'login' not in test_response.text.lower() and 'password' not in test_response.text.lower():
                                    logger.info(f"    ✅ {endpoint} - Got data! (Length: {len(test_response.content)})")
                                    
                                    # Save sample data
                                    endpoint_name = endpoint.replace('/', '_').strip('_')
                                    with open(f'debug_{endpoint_name}.html', 'w', encoding='utf-8') as f:
                                        f.write(test_response.text)
                                else:
                                    logger.info(f"    ❌ {endpoint} - Still showing login page")
                                    
                        except Exception as e:
                            logger.info(f"    ❌ {endpoint} - Error: {e}")
                
                # Save response for analysis
                with open(f'debug_login_attempt_{i+1}.html', 'w', encoding='utf-8') as f:
                    f.write(login_response.text)
                    
            except Exception as e:
                logger.error(f"  ❌ Login attempt {i+1} failed: {e}")
        
        # Step 3: Try /login endpoint specifically
        logger.info("\n🔐 Step 3: Testing /login endpoint")
        
        try:
            login_url = f"{base_url}/login"
            login_page = session.get(login_url)
            logger.info(f"Login page status: {login_page.status_code}")
            
            if login_page.status_code == 200:
                with open('debug_login_page.html', 'w', encoding='utf-8') as f:
                    f.write(login_page.text)
                
                # Try POST to /login
                for i, fields in enumerate(field_combinations[:3]):  # Test first 3 combinations
                    try:
                        login_response = session.post(login_url, data=fields, allow_redirects=True)
                        logger.info(f"  Login POST {i+1}: {login_response.status_code} -> {login_response.url}")
                        
                        if login_response.status_code == 200 and login_response.url != login_url:
                            logger.info(f"  ✅ Possible success - redirected to: {login_response.url}")
                            
                            # Test protected endpoints
                            yarn_response = session.get(f"{base_url}/yarn")
                            logger.info(f"  Yarn test: {yarn_response.status_code}")
                            
                            if yarn_response.status_code == 200:
                                logger.info("  🎉 SUCCESS! Can access /yarn endpoint")
                                with open('debug_yarn_success.html', 'w', encoding='utf-8') as f:
                                    f.write(yarn_response.text)
                                return True
                                
                    except Exception as e:
                        logger.error(f"  Login POST {i+1} failed: {e}")
                        
        except Exception as e:
            logger.error(f"Login endpoint test failed: {e}")
        
        logger.info("\n📊 Debug Summary:")
        logger.info("- Main page HTML saved to: debug_main_page.html")
        logger.info("- Login attempts saved to: debug_login_attempt_*.html")
        logger.info("- Check these files to understand the authentication flow")
        
        return False
        
    except Exception as e:
        logger.error(f"Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main debug function"""
    logger.info("🚀 ERP Authentication Debug Tool")
    logger.info("This will help us understand how to authenticate with your ERP system")
    
    success = debug_erp_auth()
    
    if success:
        logger.info("\n✅ Authentication method discovered!")
    else:
        logger.info("\n❌ Authentication method not found")
        logger.info("Please check the debug HTML files for more information")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)