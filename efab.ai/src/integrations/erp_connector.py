#!/usr/bin/env python3
"""
ERP Connector Base Classes
Beverly Knits AI Supply Chain Planner
"""

import requests
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
from abc import ABC, abstractmethod
import hashlib
import base64
from urllib.parse import urljoin, urlparse
import time

logger = logging.getLogger(__name__)

@dataclass
class ERPCredentials:
    """ERP authentication credentials"""
    username: str
    password: str
    base_url: str
    additional_params: Dict[str, Any] = None

class ERPAuthentication:
    """Handles ERP authentication and session management"""
    
    def __init__(self, credentials: ERPCredentials):
        self.credentials = credentials
        self.session = requests.Session()
        self.is_authenticated = False
        self.auth_token = None
        self.session_cookies = None
        self.last_auth_time = None
        
        # Configure session defaults
        self.session.headers.update({
            'User-Agent': 'Beverly-Knits-AI-Supply-Chain-Planner/1.0',
            'Accept': 'application/json, text/html, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Content-Type': 'application/x-www-form-urlencoded'
        })
        
        # Set reasonable timeouts
        self.session.timeout = 30
    
    def authenticate(self) -> bool:
        """
        Authenticate with the ERP system
        
        Returns:
            bool: True if authentication successful
        """
        try:
            logger.info(f"Attempting to authenticate with ERP at {self.credentials.base_url}")
            
            # Try multiple authentication approaches
            
            # Approach 1: Direct login to main page
            if self._try_direct_login():
                return True
            
            # Approach 2: Login via /login endpoint
            if self._try_login_endpoint():
                return True
            
            # Approach 3: Form-based login
            if self._try_form_login():
                return True
                
            logger.error("❌ All authentication methods failed")
            return False
                
        except Exception as e:
            logger.error(f"❌ Error during ERP authentication: {e}")
            return False
    
    def _try_direct_login(self) -> bool:
        """Try authenticating directly to the main page"""
        try:
            # First get the main page to see the login form
            response = self.session.get(self.credentials.base_url)
            
            if response.status_code == 200:
                # Extract form data from the page
                form_data = self._extract_login_form_data(response.text)
                
                if form_data:
                    # Add credentials to form data
                    form_data.update({
                        'username': self.credentials.username,
                        'password': self.credentials.password
                    })
                    
                    # Add additional parameters
                    if self.credentials.additional_params:
                        form_data.update(self.credentials.additional_params)
                    
                    # Submit login form to the correct endpoint
                    login_response = self.session.post(
                        urljoin(self.credentials.base_url, '/login'), 
                        data=form_data, 
                        allow_redirects=True
                    )
                    
                    if self._is_login_successful(login_response):
                        self.is_authenticated = True
                        self.session_cookies = self.session.cookies
                        self.last_auth_time = datetime.now()
                        logger.info("✅ Direct login successful")
                        return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Direct login failed: {e}")
            return False
    
    def _try_login_endpoint(self) -> bool:
        """Try authenticating via /login endpoint"""
        try:
            login_url = urljoin(self.credentials.base_url, '/login')
            
            # Get login page
            response = self.session.get(login_url)
            
            if response.status_code == 200:
                # Extract login form data and tokens
                form_data = self._extract_login_form_data(response.text)
                csrf_token = self._extract_csrf_token(response.text)
                
                # Prepare login data
                login_data = {
                    'username': self.credentials.username,
                    'password': self.credentials.password
                }
                
                # Add form data (including return_url)
                if form_data:
                    login_data.update(form_data)
                
                # Add CSRF token if found
                if csrf_token:
                    login_data['csrf_token'] = csrf_token
                    logger.info("Added CSRF token to login request")
                
                # Add additional parameters
                if self.credentials.additional_params:
                    login_data.update(self.credentials.additional_params)
                
                # Perform login
                login_response = self.session.post(login_url, data=login_data, allow_redirects=True)
                
                if self._is_login_successful(login_response):
                    self.is_authenticated = True
                    self.session_cookies = self.session.cookies
                    self.last_auth_time = datetime.now()
                    logger.info("✅ Login endpoint authentication successful")
                    return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Login endpoint failed: {e}")
            return False
    
    def _try_form_login(self) -> bool:
        """Try form-based authentication with common field names"""
        try:
            # Common login field combinations
            field_combinations = [
                {'user': 'username', 'pass': 'password'},
                {'user': 'email', 'pass': 'password'},
                {'user': 'login', 'pass': 'pass'},
                {'user': 'user_name', 'pass': 'user_password'},
                {'user': 'userid', 'pass': 'passwd'}
            ]
            
            for fields in field_combinations:
                try:
                    login_data = {
                        fields['user']: self.credentials.username,
                        fields['pass']: self.credentials.password
                    }
                    
                    # Add additional parameters
                    if self.credentials.additional_params:
                        login_data.update(self.credentials.additional_params)
                    
                    # Try POST to login endpoint
                    login_response = self.session.post(
                        urljoin(self.credentials.base_url, '/login'), 
                        data=login_data, 
                        allow_redirects=True
                    )
                    
                    if self._is_login_successful(login_response):
                        self.is_authenticated = True
                        self.session_cookies = self.session.cookies
                        self.last_auth_time = datetime.now()
                        logger.info(f"✅ Form login successful with fields: {fields}")
                        return True
                        
                except Exception as e:
                    logger.debug(f"Form login with {fields} failed: {e}")
                    continue
            
            return False
            
        except Exception as e:
            logger.debug(f"Form login failed: {e}")
            return False
    
    def _extract_login_form_data(self, html_content: str) -> Dict[str, str]:
        """Extract login form data from HTML"""
        import re
        
        form_data = {}
        
        # Look for hidden input fields
        hidden_inputs = re.findall(
            r'<input[^>]*type=["\']hidden["\'][^>]*>', 
            html_content, 
            re.IGNORECASE
        )
        
        for input_tag in hidden_inputs:
            # Extract name and value
            name_match = re.search(r'name=["\']([^"\']+)["\']', input_tag)
            value_match = re.search(r'value=["\']([^"\']*)["\']', input_tag)
            
            if name_match:
                name = name_match.group(1)
                value = value_match.group(1) if value_match else ''
                form_data[name] = value
        
        return form_data
    
    def _extract_csrf_token(self, html_content: str) -> Optional[str]:
        """Extract CSRF token from HTML content"""
        import re
        
        # Common CSRF token patterns
        patterns = [
            r'name=["\']csrf_token["\'] value=["\']([^"\']+)["\']',
            r'name=["\']_token["\'] value=["\']([^"\']+)["\']',
            r'<input[^>]+name=["\']csrf["\'][^>]+value=["\']([^"\']+)["\']',
            r'csrf["\']:\s*["\']([^"\']+)["\']'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, html_content, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _is_login_successful(self, response: requests.Response) -> bool:
        """Check if login was successful based on response"""
        # Check status code
        if response.status_code not in [200, 302, 301]:
            return False
        
        # Check if we're still on login page - this indicates failure
        if '/login' in response.url:
            logger.debug("Still on login page - authentication failed")
            return False
        
        # If redirected away from login page, likely successful
        if response.history and '/login' not in response.url:
            logger.debug(f"Redirected to {response.url} - likely successful")
            return True
        
        # Check for common success indicators in content
        success_indicators = [
            'dashboard', 'welcome', 'home', 'main', 'menu',
            'yarn', 'report', 'fabric', 'logout', 'user'
        ]
        
        # Check for common failure indicators in content
        failure_indicators = [
            'invalid', 'incorrect', 'failed', 'error', 
            'username', 'password', 'login', 'sign in'
        ]
        
        response_text = response.text.lower()
        
        # Check response content
        has_success = any(indicator in response_text for indicator in success_indicators)
        has_failure = any(indicator in response_text for indicator in failure_indicators)
        
        logger.debug(f"Success indicators found: {has_success}")
        logger.debug(f"Failure indicators found: {has_failure}")
        
        # Success if we have success indicators and no failure indicators
        return has_success and not has_failure
    
    def is_session_valid(self) -> bool:
        """Check if current session is still valid"""
        if not self.is_authenticated or not self.last_auth_time:
            return False
        
        # Check if session has expired (24 hours)
        if datetime.now() - self.last_auth_time > timedelta(hours=24):
            logger.warning("ERP session expired")
            return False
        
        try:
            # Test session with a simple request
            test_url = urljoin(self.credentials.base_url, '/api/test')
            response = self.session.get(test_url, timeout=10)
            return response.status_code in [200, 404]  # 404 is ok, means we're authenticated but endpoint doesn't exist
            
        except Exception:
            return False
    
    def refresh_session(self) -> bool:
        """Refresh the authentication session"""
        logger.info("Refreshing ERP session")
        self.is_authenticated = False
        self.auth_token = None
        self.session_cookies = None
        return self.authenticate()

class ERPDataSync:
    """Handles data synchronization between ERP and local system"""
    
    def __init__(self, auth: ERPAuthentication):
        self.auth = auth
        self.last_sync_time = None
        self.sync_status = {}
    
    def sync_materials(self) -> Dict[str, Any]:
        """Sync materials data from ERP"""
        if not self.auth.is_session_valid():
            if not self.auth.refresh_session():
                return {'success': False, 'error': 'Authentication failed'}
        
        try:
            # Try common ERP endpoints for materials
            endpoints = [
                '/api/materials',
                '/api/inventory/materials',
                '/materials/list',
                '/api/products',
                '/inventory'
            ]
            
            for endpoint in endpoints:
                try:
                    url = urljoin(self.auth.credentials.base_url, endpoint)
                    response = self.auth.session.get(url)
                    
                    if response.status_code == 200:
                        data = self._parse_response(response)
                        if data:
                            logger.info(f"✅ Materials data synced from {endpoint}")
                            return {
                                'success': True,
                                'data': data,
                                'endpoint': endpoint,
                                'count': len(data) if isinstance(data, list) else 1
                            }
                except Exception as e:
                    logger.debug(f"Endpoint {endpoint} failed: {e}")
                    continue
            
            return {'success': False, 'error': 'No valid materials endpoint found'}
            
        except Exception as e:
            logger.error(f"Error syncing materials: {e}")
            return {'success': False, 'error': str(e)}
    
    def sync_suppliers(self) -> Dict[str, Any]:
        """Sync suppliers data from ERP"""
        if not self.auth.is_session_valid():
            if not self.auth.refresh_session():
                return {'success': False, 'error': 'Authentication failed'}
        
        try:
            endpoints = [
                '/api/suppliers',
                '/api/vendors',
                '/suppliers/list',
                '/purchasing/suppliers'
            ]
            
            for endpoint in endpoints:
                try:
                    url = urljoin(self.auth.credentials.base_url, endpoint)
                    response = self.auth.session.get(url)
                    
                    if response.status_code == 200:
                        data = self._parse_response(response)
                        if data:
                            logger.info(f"✅ Suppliers data synced from {endpoint}")
                            return {
                                'success': True,
                                'data': data,
                                'endpoint': endpoint,
                                'count': len(data) if isinstance(data, list) else 1
                            }
                except Exception as e:
                    logger.debug(f"Endpoint {endpoint} failed: {e}")
                    continue
            
            return {'success': False, 'error': 'No valid suppliers endpoint found'}
            
        except Exception as e:
            logger.error(f"Error syncing suppliers: {e}")
            return {'success': False, 'error': str(e)}
    
    def sync_inventory(self) -> Dict[str, Any]:
        """Sync inventory data from ERP"""
        if not self.auth.is_session_valid():
            if not self.auth.refresh_session():
                return {'success': False, 'error': 'Authentication failed'}
        
        try:
            endpoints = [
                '/api/inventory',
                '/api/stock',
                '/inventory/current',
                '/api/inventory/levels'
            ]
            
            for endpoint in endpoints:
                try:
                    url = urljoin(self.auth.credentials.base_url, endpoint)
                    response = self.auth.session.get(url)
                    
                    if response.status_code == 200:
                        data = self._parse_response(response)
                        if data:
                            logger.info(f"✅ Inventory data synced from {endpoint}")
                            return {
                                'success': True,
                                'data': data,
                                'endpoint': endpoint,
                                'count': len(data) if isinstance(data, list) else 1
                            }
                except Exception as e:
                    logger.debug(f"Endpoint {endpoint} failed: {e}")
                    continue
            
            return {'success': False, 'error': 'No valid inventory endpoint found'}
            
        except Exception as e:
            logger.error(f"Error syncing inventory: {e}")
            return {'success': False, 'error': str(e)}
    
    def sync_orders(self, date_from: datetime = None) -> Dict[str, Any]:
        """Sync orders/sales data from ERP"""
        if not self.auth.is_session_valid():
            if not self.auth.refresh_session():
                return {'success': False, 'error': 'Authentication failed'}
        
        try:
            # Default to last 30 days if no date specified
            if not date_from:
                date_from = datetime.now() - timedelta(days=30)
            
            endpoints = [
                '/api/orders',
                '/api/sales',
                '/orders/list',
                '/sales/orders'
            ]
            
            for endpoint in endpoints:
                try:
                    url = urljoin(self.auth.credentials.base_url, endpoint)
                    
                    # Add date parameters
                    params = {
                        'date_from': date_from.strftime('%Y-%m-%d'),
                        'date_to': datetime.now().strftime('%Y-%m-%d')
                    }
                    
                    response = self.auth.session.get(url, params=params)
                    
                    if response.status_code == 200:
                        data = self._parse_response(response)
                        if data:
                            logger.info(f"✅ Orders data synced from {endpoint}")
                            return {
                                'success': True,
                                'data': data,
                                'endpoint': endpoint,
                                'count': len(data) if isinstance(data, list) else 1
                            }
                except Exception as e:
                    logger.debug(f"Endpoint {endpoint} failed: {e}")
                    continue
            
            return {'success': False, 'error': 'No valid orders endpoint found'}
            
        except Exception as e:
            logger.error(f"Error syncing orders: {e}")
            return {'success': False, 'error': str(e)}
    
    def _parse_response(self, response: requests.Response) -> Optional[Union[Dict, List]]:
        """Parse HTTP response to extract data"""
        try:
            content_type = response.headers.get('content-type', '').lower()
            
            if 'application/json' in content_type:
                return response.json()
            elif 'text/html' in content_type:
                # Try to extract data from HTML tables or embedded JSON
                return self._extract_data_from_html(response.text)
            else:
                logger.warning(f"Unknown content type: {content_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return None
    
    def _extract_data_from_html(self, html_content: str) -> Optional[Union[Dict, List]]:
        """Extract structured data from HTML content"""
        try:
            import re
            
            # Look for embedded JSON data
            json_patterns = [
                r'var\s+data\s*=\s*(\{.*?\});',
                r'window\.data\s*=\s*(\{.*?\});',
                r'data:\s*(\[.*?\])',
                r'<script[^>]*>.*?(\{.*?\}).*?</script>'
            ]
            
            for pattern in json_patterns:
                matches = re.findall(pattern, html_content, re.DOTALL)
                for match in matches:
                    try:
                        return json.loads(match)
                    except:
                        continue
            
            # If no JSON found, try to parse HTML tables
            return self._parse_html_tables(html_content)
            
        except Exception as e:
            logger.error(f"Error extracting data from HTML: {e}")
            return None
    
    def _parse_html_tables(self, html_content: str) -> Optional[List[Dict]]:
        """Parse HTML tables to extract data"""
        try:
            import pandas as pd
            from io import StringIO
            
            # Use pandas to parse HTML tables
            tables = pd.read_html(StringIO(html_content))
            
            if tables:
                # Convert first table to dict
                df = tables[0]
                return df.to_dict('records')
            
            return None
            
        except Exception:
            return None

class ERPConnector(ABC):
    """Abstract base class for ERP connectors"""
    
    def __init__(self, credentials: ERPCredentials):
        self.credentials = credentials
        self.auth = ERPAuthentication(credentials)
        self.data_sync = ERPDataSync(self.auth)
        self.connection_status = 'disconnected'
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to the ERP system"""
        pass
    
    @abstractmethod
    def test_connection(self) -> Dict[str, Any]:
        """Test the ERP connection"""
        pass
    
    def disconnect(self):
        """Disconnect from ERP system"""
        self.auth.is_authenticated = False
        self.connection_status = 'disconnected'
        logger.info("Disconnected from ERP system")
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information"""
        return {
            'base_url': self.credentials.base_url,
            'username': self.credentials.username,
            'status': self.connection_status,
            'authenticated': self.auth.is_authenticated,
            'last_auth': self.auth.last_auth_time.isoformat() if self.auth.last_auth_time else None
        }