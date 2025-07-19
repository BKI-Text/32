#!/usr/bin/env python3
"""
ERP Web Crawler for Beverly Knits AI Supply Chain Planner
Comprehensive deep crawl of Beverly Knits ERP website to discover additional data sources
"""

import requests
import json
import logging
import time
import re
import sys
import os
from typing import Dict, List, Set, Optional, Any, Tuple
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse, parse_qs
from dataclasses import dataclass

# Add the parent directory to the path to import the efab_integration
sys.path.append('/mnt/c/Users/psytz/32/efab.ai/src/integrations')
from efab_integration import EfabERPIntegration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataSourceAnalysis:
    """Analysis of discovered data source"""
    url: str
    endpoint: str
    status_code: int
    response_size: int
    data_type: str
    fields: List[str]
    sample_data: Dict[str, Any]
    ml_potential: Dict[str, Any]
    integration_complexity: str
    requires_params: bool
    param_options: Dict[str, Any]
    temporal_data: bool
    relationship_data: bool
    volume_estimate: str
    priority: str
    notes: str

class ERPWebCrawler:
    """Comprehensive web crawler for Beverly Knits ERP system"""
    
    def __init__(self, username: str, password: str):
        """Initialize the crawler with ERP credentials"""
        self.erp_integration = EfabERPIntegration(username, password)
        self.discovered_urls = set()
        self.tested_endpoints = {}
        self.data_sources = []
        self.crawl_report = {
            'timestamp': datetime.now().isoformat(),
            'total_urls_discovered': 0,
            'data_sources_found': 0,
            'ml_training_candidates': 0,
            'integration_ready': 0,
            'phase_1_recommendations': [],
            'phase_2_recommendations': [],
            'phase_3_recommendations': [],
            'detailed_findings': []
        }
        
        # Known endpoints to start from
        self.seed_urls = [
            '/yarn',
            '/fabric/so/list',
            '/report/yarn_demand',
            '/report/expected_yarn',
            '/yarn/po/list',
            '/dashboard',
            '/home',
            '/main',
            '/reports',
            '/inventory',
            '/api',
            '/admin',
            '/settings',
            '/production',
            '/planning',
            '/suppliers',
            '/orders',
            '/procurement',
            '/quality',
            '/analytics',
            '/export'
        ]
        
        # Patterns to look for in URLs and content
        self.data_patterns = {
            'reports': ['/report/', '/reports/', '/analytics/', '/dashboard/'],
            'api_endpoints': ['/api/', '/data/', '/json/', '/xml/'],
            'inventory': ['/inventory/', '/stock/', '/warehouse/', '/materials/'],
            'orders': ['/orders/', '/sales/', '/purchase/', '/procurement/'],
            'production': ['/production/', '/manufacturing/', '/planning/'],
            'suppliers': ['/suppliers/', '/vendors/', '/partners/'],
            'quality': ['/quality/', '/testing/', '/inspection/'],
            'historical': ['/history/', '/archive/', '/historical/', '/trend/'],
            'realtime': ['/live/', '/current/', '/real-time/', '/now/'],
            'export': ['/export/', '/download/', '/csv/', '/excel/']
        }
    
    def crawl_erp_system(self) -> Dict[str, Any]:
        """Perform comprehensive crawl of the ERP system"""
        try:
            logger.info("🕷️ Starting comprehensive ERP web crawl...")
            
            # Connect to ERP system
            if not self.erp_integration.connect():
                logger.error("❌ Failed to connect to ERP system")
                return self.crawl_report
            
            # Phase 1: Discover all accessible URLs
            logger.info("📍 Phase 1: URL Discovery")
            self._discover_urls()
            
            # Phase 2: Analyze discovered endpoints
            logger.info("🔍 Phase 2: Endpoint Analysis")
            self._analyze_endpoints()
            
            # Phase 3: Test data extraction
            logger.info("🧪 Phase 3: Data Extraction Testing")
            self._test_data_extraction()
            
            # Phase 4: ML potential assessment
            logger.info("🤖 Phase 4: ML Potential Assessment")
            self._assess_ml_potential()
            
            # Phase 5: Integration planning
            logger.info("📋 Phase 5: Integration Planning")
            self._plan_integration()
            
            # Generate final report
            self._generate_report()
            
            logger.info("✅ ERP web crawl completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Error during ERP crawl: {e}")
            self.crawl_report['error'] = str(e)
        
        return self.crawl_report
    
    def _discover_urls(self):
        """Discover all accessible URLs in the ERP system"""
        urls_to_process = set(self.seed_urls)
        processed_urls = set()
        
        while urls_to_process:
            current_url = urls_to_process.pop()
            
            if current_url in processed_urls:
                continue
            
            processed_urls.add(current_url)
            
            try:
                # Get the page
                full_url = urljoin(self.erp_integration.credentials.base_url, current_url)
                response = self.erp_integration.auth.session.get(full_url, timeout=10)
                
                if response.status_code == 200:
                    logger.info(f"✅ Discovered: {current_url}")
                    self.discovered_urls.add(current_url)
                    
                    # Extract additional URLs from the page
                    new_urls = self._extract_urls_from_content(response.text, current_url)
                    
                    # Add new URLs to processing queue
                    for url in new_urls:
                        if url not in processed_urls:
                            urls_to_process.add(url)
                
                # Add small delay to avoid overwhelming the server
                time.sleep(0.5)
                
            except Exception as e:
                logger.debug(f"Error accessing {current_url}: {e}")
                continue
        
        logger.info(f"📊 URL Discovery complete: {len(self.discovered_urls)} URLs found")
        self.crawl_report['total_urls_discovered'] = len(self.discovered_urls)
    
    def _extract_urls_from_content(self, html_content: str, base_url: str) -> Set[str]:
        """Extract URLs from HTML content using regex"""
        urls = set()
        
        try:
            # Extract from links using regex
            link_patterns = [
                r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>',
                r'<a[^>]+href=([^\\s>]+)[^>]*>',
                r'href=["\']([^"\']+)["\']'
            ]
            
            for pattern in link_patterns:
                matches = re.findall(pattern, html_content, re.IGNORECASE)
                for href in matches:
                    if href.startswith('/'):
                        urls.add(href)
                    elif href.startswith(self.erp_integration.credentials.base_url):
                        path = urlparse(href).path
                        if path:
                            urls.add(path)
            
            # Extract from form actions
            form_patterns = [
                r'<form[^>]+action=["\']([^"\']+)["\'][^>]*>',
                r'<form[^>]+action=([^\\s>]+)[^>]*>',
                r'action=["\']([^"\']+)["\']'
            ]
            
            for pattern in form_patterns:
                matches = re.findall(pattern, html_content, re.IGNORECASE)
                for action in matches:
                    if action.startswith('/'):
                        urls.add(action)
            
            # Extract from script tags (look for AJAX endpoints)
            script_content = re.findall(r'<script[^>]*>(.*?)</script>', html_content, re.DOTALL | re.IGNORECASE)
            for script in script_content:
                # Look for URL patterns in JavaScript
                js_urls = re.findall(r'["\'](/[^"\']+)["\']', script)
                for url in js_urls:
                    if any(pattern in url for pattern_list in self.data_patterns.values() for pattern in pattern_list):
                        urls.add(url)
            
            # Look for API documentation or endpoint lists
            api_patterns = [
                r'api/v\d+/[^"\'\\s]+',
                r'/api/[^"\'\\s]+',
                r'/data/[^"\'\\s]+',
                r'/json/[^"\'\\s]+',
                r'/report/[^"\'\\s]+',
                r'/export/[^"\'\\s]+'
            ]
            
            for pattern in api_patterns:
                matches = re.findall(pattern, html_content, re.IGNORECASE)
                for match in matches:
                    if match.startswith('/'):
                        urls.add(match)
                    else:
                        urls.add('/' + match)
            
        except Exception as e:
            logger.debug(f"Error extracting URLs from {base_url}: {e}")
        
        return urls
    
    def _analyze_endpoints(self):
        """Analyze discovered endpoints for data potential"""
        logger.info(f"🔍 Analyzing {len(self.discovered_urls)} discovered endpoints...")
        
        for url in self.discovered_urls:
            try:
                # Test endpoint accessibility
                full_url = urljoin(self.erp_integration.credentials.base_url, url)
                response = self.erp_integration.auth.session.get(full_url, timeout=10)
                
                endpoint_analysis = {
                    'url': url,
                    'status_code': response.status_code,
                    'response_size': len(response.content),
                    'content_type': response.headers.get('content-type', ''),
                    'accessible': response.status_code == 200,
                    'data_indicators': self._detect_data_indicators(url, response),
                    'requires_params': self._check_parameter_requirements(response),
                    'forms_present': self._detect_forms(response.text if response.status_code == 200 else ''),
                    'table_data': self._detect_table_data(response.text if response.status_code == 200 else ''),
                    'json_data': self._detect_json_data(response),
                    'category': self._categorize_endpoint(url)
                }
                
                self.tested_endpoints[url] = endpoint_analysis
                
                # Add delay to avoid overwhelming server
                time.sleep(0.3)
                
            except Exception as e:
                logger.debug(f"Error analyzing endpoint {url}: {e}")
                self.tested_endpoints[url] = {
                    'url': url,
                    'status_code': 0,
                    'accessible': False,
                    'error': str(e)
                }
        
        logger.info(f"📊 Endpoint analysis complete: {len(self.tested_endpoints)} endpoints analyzed")
    
    def _detect_data_indicators(self, url: str, response: requests.Response) -> List[str]:
        """Detect indicators that an endpoint contains valuable data"""
        indicators = []
        
        # URL pattern indicators
        for category, patterns in self.data_patterns.items():
            if any(pattern in url.lower() for pattern in patterns):
                indicators.append(f"url_pattern_{category}")
        
        if response.status_code == 200:
            content = response.text.lower()
            
            # Content indicators
            data_keywords = [
                'data', 'records', 'results', 'items', 'list', 'table',
                'json', 'xml', 'csv', 'export', 'download', 'report',
                'inventory', 'stock', 'orders', 'sales', 'purchase',
                'material', 'supplier', 'vendor', 'customer', 'product',
                'quantity', 'price', 'cost', 'amount', 'date', 'time',
                'forecast', 'demand', 'production', 'planning', 'schedule'
            ]
            
            for keyword in data_keywords:
                if keyword in content:
                    indicators.append(f"content_{keyword}")
            
            # HTML structure indicators
            if '<table' in content:
                indicators.append('html_table')
            if '<form' in content:
                indicators.append('html_form')
            if 'thead' in content and 'tbody' in content:
                indicators.append('structured_table')
            
            # JavaScript data indicators
            if 'var data' in content or 'window.data' in content:
                indicators.append('js_data_object')
            if 'ajax' in content or 'fetch(' in content:
                indicators.append('ajax_endpoint')
        
        return indicators
    
    def _check_parameter_requirements(self, response: requests.Response) -> Dict[str, Any]:
        """Check if endpoint requires parameters using regex"""
        param_info = {
            'requires_params': False,
            'detected_params': [],
            'forms': [],
            'url_params': []
        }
        
        if response.status_code == 200:
            try:
                html_content = response.text
                
                # Check for forms using regex
                form_matches = re.findall(r'<form[^>]*>(.*?)</form>', html_content, re.DOTALL | re.IGNORECASE)
                for form_content in form_matches:
                    # Extract form action and method
                    action_match = re.search(r'<form[^>]*action=["\']([^"\']+)["\']', form_content, re.IGNORECASE)
                    method_match = re.search(r'<form[^>]*method=["\']([^"\']+)["\']', form_content, re.IGNORECASE)
                    
                    form_info = {
                        'action': action_match.group(1) if action_match else '',
                        'method': method_match.group(1) if method_match else 'get',
                        'inputs': []
                    }
                    
                    # Extract input fields
                    input_patterns = [
                        r'<input[^>]*name=["\']([^"\']+)["\'][^>]*>',
                        r'<select[^>]*name=["\']([^"\']+)["\'][^>]*>',
                        r'<textarea[^>]*name=["\']([^"\']+)["\'][^>]*>'
                    ]
                    
                    for pattern in input_patterns:
                        input_matches = re.findall(pattern, form_content, re.IGNORECASE)
                        for input_name in input_matches:
                            input_info = {
                                'name': input_name,
                                'type': 'input',
                                'required': 'required' in form_content.lower()
                            }
                            form_info['inputs'].append(input_info)
                    
                    param_info['forms'].append(form_info)
                    if form_info['inputs']:
                        param_info['requires_params'] = True
                
                # Check for URL parameters in links
                link_matches = re.findall(r'href=["\']([^"\']+\?[^"\']+)["\']', html_content, re.IGNORECASE)
                for href in link_matches:
                    if '?' in href:
                        query_params = parse_qs(urlparse(href).query)
                        for param in query_params.keys():
                            if param not in param_info['url_params']:
                                param_info['url_params'].append(param)
                                param_info['requires_params'] = True
                
            except Exception as e:
                logger.debug(f"Error checking parameter requirements: {e}")
        
        return param_info
    
    def _detect_forms(self, html_content: str) -> List[Dict[str, Any]]:
        """Detect forms in HTML content using regex"""
        forms = []
        
        try:
            # Find all form tags
            form_matches = re.findall(r'<form[^>]*>(.*?)</form>', html_content, re.DOTALL | re.IGNORECASE)
            
            for form_content in form_matches:
                # Extract form attributes
                action_match = re.search(r'<form[^>]*action=["\']([^"\']+)["\']', form_content, re.IGNORECASE)
                method_match = re.search(r'<form[^>]*method=["\']([^"\']+)["\']', form_content, re.IGNORECASE)
                
                form_data = {
                    'action': action_match.group(1) if action_match else '',
                    'method': method_match.group(1) if method_match else 'get',
                    'fields': []
                }
                
                # Extract input fields
                input_patterns = [
                    r'<input[^>]*name=["\']([^"\']+)["\'][^>]*>',
                    r'<select[^>]*name=["\']([^"\']+)["\'][^>]*>',
                    r'<textarea[^>]*name=["\']([^"\']+)["\'][^>]*>'
                ]
                
                for pattern in input_patterns:
                    field_matches = re.findall(pattern, form_content, re.IGNORECASE)
                    for field_name in field_matches:
                        field_data = {
                            'name': field_name,
                            'type': 'input',
                            'required': 'required' in form_content.lower(),
                            'placeholder': '',
                            'value': ''
                        }
                        form_data['fields'].append(field_data)
                
                forms.append(form_data)
                
        except Exception as e:
            logger.debug(f"Error detecting forms: {e}")
        
        return forms
    
    def _detect_table_data(self, html_content: str) -> Dict[str, Any]:
        """Detect and analyze table data in HTML using regex"""
        table_info = {
            'has_tables': False,
            'table_count': 0,
            'structured_data': False,
            'headers': [],
            'sample_rows': []
        }
        
        try:
            # Find all table tags
            table_matches = re.findall(r'<table[^>]*>(.*?)</table>', html_content, re.DOTALL | re.IGNORECASE)
            
            if table_matches:
                table_info['has_tables'] = True
                table_info['table_count'] = len(table_matches)
                
                # Analyze first table
                first_table = table_matches[0]
                
                # Get headers from thead or first tr
                headers = []
                thead_match = re.search(r'<thead[^>]*>(.*?)</thead>', first_table, re.DOTALL | re.IGNORECASE)
                if thead_match:
                    # Extract headers from thead
                    th_matches = re.findall(r'<th[^>]*>(.*?)</th>', thead_match.group(1), re.DOTALL | re.IGNORECASE)
                    headers = [re.sub(r'<[^>]+>', '', th).strip() for th in th_matches]
                else:
                    # Try first row as headers
                    first_tr_match = re.search(r'<tr[^>]*>(.*?)</tr>', first_table, re.DOTALL | re.IGNORECASE)
                    if first_tr_match:
                        cell_matches = re.findall(r'<t[hd][^>]*>(.*?)</t[hd]>', first_tr_match.group(1), re.DOTALL | re.IGNORECASE)
                        headers = [re.sub(r'<[^>]+>', '', cell).strip() for cell in cell_matches]
                
                table_info['headers'] = headers
                
                # Get sample rows
                tr_matches = re.findall(r'<tr[^>]*>(.*?)</tr>', first_table, re.DOTALL | re.IGNORECASE)
                for i, row_content in enumerate(tr_matches[1:4]):  # Skip first row (headers), get up to 3 sample rows
                    row_data = []
                    cell_matches = re.findall(r'<td[^>]*>(.*?)</td>', row_content, re.DOTALL | re.IGNORECASE)
                    for cell in cell_matches:
                        cleaned_cell = re.sub(r'<[^>]+>', '', cell).strip()
                        row_data.append(cleaned_cell)
                    if row_data:
                        table_info['sample_rows'].append(row_data)
                
                if headers and table_info['sample_rows']:
                    table_info['structured_data'] = True
        
        except Exception as e:
            logger.debug(f"Error detecting table data: {e}")
        
        return table_info
    
    def _detect_json_data(self, response: requests.Response) -> Dict[str, Any]:
        """Detect JSON data in response"""
        json_info = {
            'is_json': False,
            'has_embedded_json': False,
            'data_structure': None,
            'sample_keys': []
        }
        
        try:
            content_type = response.headers.get('content-type', '')
            
            if 'application/json' in content_type:
                json_info['is_json'] = True
                try:
                    data = response.json()
                    json_info['data_structure'] = type(data).__name__
                    
                    if isinstance(data, dict):
                        json_info['sample_keys'] = list(data.keys())[:10]
                    elif isinstance(data, list) and data and isinstance(data[0], dict):
                        json_info['sample_keys'] = list(data[0].keys())[:10]
                        
                except:
                    pass
            
            # Check for embedded JSON in HTML
            if response.status_code == 200 and 'text/html' in content_type:
                html_content = response.text
                json_patterns = [
                    r'var\s+data\s*=\s*(\{.*?\});',
                    r'window\.data\s*=\s*(\{.*?\});',
                    r'data:\s*(\[.*?\])',
                    r'<script[^>]*>.*?(\{.*?\}).*?</script>'
                ]
                
                for pattern in json_patterns:
                    matches = re.findall(pattern, html_content, re.DOTALL)
                    if matches:
                        json_info['has_embedded_json'] = True
                        break
        
        except Exception as e:
            logger.debug(f"Error detecting JSON data: {e}")
        
        return json_info
    
    def _categorize_endpoint(self, url: str) -> str:
        """Categorize endpoint based on URL patterns"""
        url_lower = url.lower()
        
        # Category mapping
        categories = {
            'reports': ['report', 'analytics', 'dashboard'],
            'api': ['api', 'data', 'json', 'xml'],
            'inventory': ['inventory', 'stock', 'warehouse', 'material'],
            'orders': ['order', 'sales', 'purchase', 'procurement'],
            'production': ['production', 'manufacturing', 'planning'],
            'suppliers': ['supplier', 'vendor', 'partner'],
            'quality': ['quality', 'testing', 'inspection'],
            'historical': ['history', 'archive', 'historical', 'trend'],
            'realtime': ['live', 'current', 'real-time', 'now'],
            'export': ['export', 'download', 'csv', 'excel'],
            'admin': ['admin', 'settings', 'config'],
            'yarn': ['yarn'],
            'fabric': ['fabric']
        }
        
        for category, keywords in categories.items():
            if any(keyword in url_lower for keyword in keywords):
                return category
        
        return 'other'
    
    def _test_data_extraction(self):
        """Test data extraction from promising endpoints"""
        logger.info("🧪 Testing data extraction from promising endpoints...")
        
        # Filter endpoints that look promising for data
        promising_endpoints = []
        
        for url, analysis in self.tested_endpoints.items():
            if (analysis.get('accessible', False) and
                (analysis.get('data_indicators', []) or
                 analysis.get('table_data', {}).get('structured_data', False) or
                 analysis.get('json_data', {}).get('is_json', False))):
                promising_endpoints.append(url)
        
        logger.info(f"🎯 Found {len(promising_endpoints)} promising endpoints to test")
        
        for url in promising_endpoints:
            try:
                self._extract_and_analyze_data(url)
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                logger.debug(f"Error testing data extraction from {url}: {e}")
        
        logger.info(f"📊 Data extraction testing complete")
    
    def _extract_and_analyze_data(self, url: str):
        """Extract and analyze data from a specific endpoint"""
        try:
            full_url = urljoin(self.erp_integration.credentials.base_url, url)
            response = self.erp_integration.auth.session.get(full_url, timeout=10)
            
            if response.status_code != 200:
                return
            
            # Try to extract structured data
            extracted_data = self._extract_structured_data(response)
            
            if extracted_data:
                # Analyze the extracted data
                analysis = self._analyze_extracted_data(url, extracted_data)
                
                if analysis:
                    data_source = DataSourceAnalysis(
                        url=url,
                        endpoint=url,
                        status_code=response.status_code,
                        response_size=len(response.content),
                        data_type=analysis['data_type'],
                        fields=analysis['fields'],
                        sample_data=analysis['sample_data'],
                        ml_potential=analysis['ml_potential'],
                        integration_complexity=analysis['integration_complexity'],
                        requires_params=analysis['requires_params'],
                        param_options=analysis['param_options'],
                        temporal_data=analysis['temporal_data'],
                        relationship_data=analysis['relationship_data'],
                        volume_estimate=analysis['volume_estimate'],
                        priority=analysis['priority'],
                        notes=analysis['notes']
                    )
                    
                    self.data_sources.append(data_source)
                    logger.info(f"✅ Data source analyzed: {url}")
        
        except Exception as e:
            logger.debug(f"Error extracting data from {url}: {e}")
    
    def _extract_structured_data(self, response: requests.Response) -> Optional[Any]:
        """Extract structured data from response"""
        try:
            content_type = response.headers.get('content-type', '')
            
            # JSON response
            if 'application/json' in content_type:
                return response.json()
            
            # HTML response
            elif 'text/html' in content_type:
                # Try to extract from tables
                table_data = self._extract_table_data(response.text)
                if table_data:
                    return table_data
                
                # Try to extract embedded JSON
                json_data = self._extract_embedded_json(response.text)
                if json_data:
                    return json_data
            
            return None
            
        except Exception as e:
            logger.debug(f"Error extracting structured data: {e}")
            return None
    
    def _extract_table_data(self, html_content: str) -> Optional[List[Dict]]:
        """Extract data from HTML tables using regex"""
        try:
            # Find all table tags
            table_matches = re.findall(r'<table[^>]*>(.*?)</table>', html_content, re.DOTALL | re.IGNORECASE)
            
            if not table_matches:
                return None
            
            # Process first table
            table_content = table_matches[0]
            data = []
            
            # Get headers
            headers = []
            thead_match = re.search(r'<thead[^>]*>(.*?)</thead>', table_content, re.DOTALL | re.IGNORECASE)
            if thead_match:
                # Extract headers from thead
                th_matches = re.findall(r'<th[^>]*>(.*?)</th>', thead_match.group(1), re.DOTALL | re.IGNORECASE)
                headers = [re.sub(r'<[^>]+>', '', th).strip() for th in th_matches]
            else:
                # Try first row as headers
                tr_matches = re.findall(r'<tr[^>]*>(.*?)</tr>', table_content, re.DOTALL | re.IGNORECASE)
                if tr_matches:
                    first_row = tr_matches[0]
                    cell_matches = re.findall(r'<t[hd][^>]*>(.*?)</t[hd]>', first_row, re.DOTALL | re.IGNORECASE)
                    headers = [re.sub(r'<[^>]+>', '', cell).strip() for cell in cell_matches]
            
            # Get data rows
            tr_matches = re.findall(r'<tr[^>]*>(.*?)</tr>', table_content, re.DOTALL | re.IGNORECASE)
            start_index = 1 if headers else 0
            
            for row_content in tr_matches[start_index:]:
                row_data = {}
                cell_matches = re.findall(r'<td[^>]*>(.*?)</td>', row_content, re.DOTALL | re.IGNORECASE)
                
                for i, cell in enumerate(cell_matches):
                    cleaned_cell = re.sub(r'<[^>]+>', '', cell).strip()
                    if i < len(headers):
                        row_data[headers[i]] = cleaned_cell
                    else:
                        row_data[f'column_{i}'] = cleaned_cell
                
                if row_data:
                    data.append(row_data)
            
            return data if data else None
            
        except Exception as e:
            logger.debug(f"Error extracting table data: {e}")
            return None
    
    def _extract_embedded_json(self, html_content: str) -> Optional[Any]:
        """Extract embedded JSON from HTML"""
        try:
            json_patterns = [
                r'var\s+data\s*=\s*(\{.*?\});',
                r'window\.data\s*=\s*(\{.*?\});',
                r'data:\s*(\[.*?\])',
                r'"data":\s*(\{.*?\})',
                r'"data":\s*(\[.*?\])'
            ]
            
            for pattern in json_patterns:
                matches = re.findall(pattern, html_content, re.DOTALL)
                for match in matches:
                    try:
                        return json.loads(match)
                    except:
                        continue
            
            return None
            
        except Exception as e:
            logger.debug(f"Error extracting embedded JSON: {e}")
            return None
    
    def _analyze_extracted_data(self, url: str, data: Any) -> Optional[Dict[str, Any]]:
        """Analyze extracted data for ML potential and integration complexity"""
        try:
            analysis = {
                'data_type': type(data).__name__,
                'fields': [],
                'sample_data': {},
                'ml_potential': {},
                'integration_complexity': 'medium',
                'requires_params': False,
                'param_options': {},
                'temporal_data': False,
                'relationship_data': False,
                'volume_estimate': 'unknown',
                'priority': 'medium',
                'notes': ''
            }
            
            # Analyze data structure
            if isinstance(data, list) and data:
                analysis['volume_estimate'] = self._estimate_volume(len(data))
                
                # Analyze first record
                if isinstance(data[0], dict):
                    analysis['fields'] = list(data[0].keys())
                    analysis['sample_data'] = data[0]
                    
                    # Check for temporal data
                    temporal_fields = self._detect_temporal_fields(data[0])
                    if temporal_fields:
                        analysis['temporal_data'] = True
                        analysis['ml_potential']['temporal_fields'] = temporal_fields
                    
                    # Check for relationship data
                    relationship_fields = self._detect_relationship_fields(data[0])
                    if relationship_fields:
                        analysis['relationship_data'] = True
                        analysis['ml_potential']['relationship_fields'] = relationship_fields
            
            elif isinstance(data, dict):
                analysis['fields'] = list(data.keys())
                analysis['sample_data'] = data
                
                # Check for temporal data
                temporal_fields = self._detect_temporal_fields(data)
                if temporal_fields:
                    analysis['temporal_data'] = True
                    analysis['ml_potential']['temporal_fields'] = temporal_fields
                
                # Check for relationship data
                relationship_fields = self._detect_relationship_fields(data)
                if relationship_fields:
                    analysis['relationship_data'] = True
                    analysis['ml_potential']['relationship_fields'] = relationship_fields
            
            # Assess ML potential
            analysis['ml_potential'] = self._assess_ml_potential(analysis)
            
            # Determine integration complexity
            analysis['integration_complexity'] = self._assess_integration_complexity(url, analysis)
            
            # Determine priority
            analysis['priority'] = self._determine_priority(analysis)
            
            # Add notes
            analysis['notes'] = self._generate_notes(url, analysis)
            
            return analysis
            
        except Exception as e:
            logger.debug(f"Error analyzing extracted data: {e}")
            return None
    
    def _detect_temporal_fields(self, data: Dict) -> List[str]:
        """Detect temporal/time-series fields in data"""
        temporal_fields = []
        
        temporal_keywords = [
            'date', 'time', 'timestamp', 'created', 'updated', 'modified',
            'start', 'end', 'deadline', 'due', 'period', 'month', 'year',
            'week', 'day', 'hour', 'minute', 'second', 'schedule'
        ]
        
        for field in data.keys():
            field_lower = field.lower()
            
            # Check for temporal keywords
            if any(keyword in field_lower for keyword in temporal_keywords):
                temporal_fields.append(field)
            
            # Check for date-like values
            try:
                value = str(data[field])
                if self._is_date_like(value):
                    temporal_fields.append(field)
            except:
                pass
        
        return temporal_fields
    
    def _detect_relationship_fields(self, data: Dict) -> List[str]:
        """Detect relationship/foreign key fields in data"""
        relationship_fields = []
        
        relationship_keywords = [
            'id', 'ref', 'key', 'link', 'parent', 'child', 'supplier',
            'customer', 'vendor', 'material', 'product', 'order', 'user',
            'account', 'code', 'sku', 'part', 'item', 'category', 'type'
        ]
        
        for field in data.keys():
            field_lower = field.lower()
            
            # Check for relationship keywords
            if any(keyword in field_lower for keyword in relationship_keywords):
                relationship_fields.append(field)
            
            # Check for ID-like patterns
            if field_lower.endswith('_id') or field_lower.startswith('id_'):
                relationship_fields.append(field)
        
        return relationship_fields
    
    def _is_date_like(self, value: str) -> bool:
        """Check if a value looks like a date"""
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',   # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}',   # MM-DD-YYYY
            r'\d{4}/\d{2}/\d{2}',   # YYYY/MM/DD
            r'\d{1,2}-\d{1,2}-\d{4}',  # M-D-YYYY
            r'\d{1,2}/\d{1,2}/\d{4}'   # M/D/YYYY
        ]
        
        for pattern in date_patterns:
            if re.match(pattern, value):
                return True
        
        return False
    
    def _estimate_volume(self, count: int) -> str:
        """Estimate data volume category"""
        if count < 10:
            return 'very_low'
        elif count < 100:
            return 'low'
        elif count < 1000:
            return 'medium'
        elif count < 10000:
            return 'high'
        else:
            return 'very_high'
    
    def _assess_ml_potential(self, analysis: Dict) -> Dict[str, Any]:
        """Assess ML training potential of the data"""
        ml_potential = {
            'arima_suitable': False,
            'prophet_suitable': False,
            'lstm_suitable': False,
            'xgboost_suitable': False,
            'overall_score': 0,
            'reasons': []
        }
        
        # Check for ARIMA suitability (time series data)
        if analysis['temporal_data'] and analysis['volume_estimate'] in ['medium', 'high', 'very_high']:
            ml_potential['arima_suitable'] = True
            ml_potential['overall_score'] += 2
            ml_potential['reasons'].append('Has temporal data suitable for ARIMA time series forecasting')
        
        # Check for Prophet suitability (time series with seasonality)
        if analysis['temporal_data'] and analysis['volume_estimate'] in ['medium', 'high', 'very_high']:
            ml_potential['prophet_suitable'] = True
            ml_potential['overall_score'] += 2
            ml_potential['reasons'].append('Has temporal data suitable for Prophet forecasting')
        
        # Check for LSTM suitability (sequential data)
        if analysis['temporal_data'] and analysis['volume_estimate'] in ['high', 'very_high']:
            ml_potential['lstm_suitable'] = True
            ml_potential['overall_score'] += 3
            ml_potential['reasons'].append('Has sequential data suitable for LSTM neural networks')
        
        # Check for XGBoost suitability (tabular data with features)
        if (len(analysis['fields']) > 3 and 
            analysis['relationship_data'] and 
            analysis['volume_estimate'] in ['medium', 'high', 'very_high']):
            ml_potential['xgboost_suitable'] = True
            ml_potential['overall_score'] += 2
            ml_potential['reasons'].append('Has structured tabular data suitable for XGBoost')
        
        return ml_potential
    
    def _assess_integration_complexity(self, url: str, analysis: Dict) -> str:
        """Assess integration complexity"""
        complexity_score = 0
        
        # URL complexity
        if '/api/' in url:
            complexity_score -= 1  # API endpoints are easier
        if url.count('/') > 3:
            complexity_score += 1  # Deep URLs might be more complex
        
        # Data structure complexity
        if analysis['data_type'] == 'list':
            complexity_score -= 1  # Lists are easier to handle
        if len(analysis['fields']) > 10:
            complexity_score += 1  # Many fields = more complex
        
        # Parameter requirements
        if analysis['requires_params']:
            complexity_score += 2  # Parameters make it more complex
        
        # Temporal data handling
        if analysis['temporal_data']:
            complexity_score += 1  # Time series needs special handling
        
        # Determine complexity level
        if complexity_score <= 0:
            return 'low'
        elif complexity_score <= 2:
            return 'medium'
        else:
            return 'high'
    
    def _determine_priority(self, analysis: Dict) -> str:
        """Determine integration priority"""
        priority_score = 0
        
        # ML potential
        priority_score += analysis['ml_potential']['overall_score']
        
        # Data volume
        volume_scores = {
            'very_low': 0,
            'low': 1,
            'medium': 2,
            'high': 3,
            'very_high': 4
        }
        priority_score += volume_scores.get(analysis['volume_estimate'], 0)
        
        # Temporal data bonus
        if analysis['temporal_data']:
            priority_score += 2
        
        # Relationship data bonus
        if analysis['relationship_data']:
            priority_score += 1
        
        # Complexity penalty
        complexity_penalties = {
            'low': 0,
            'medium': -1,
            'high': -2
        }
        priority_score += complexity_penalties.get(analysis['integration_complexity'], 0)
        
        # Determine priority
        if priority_score >= 6:
            return 'high'
        elif priority_score >= 3:
            return 'medium'
        else:
            return 'low'
    
    def _generate_notes(self, url: str, analysis: Dict) -> str:
        """Generate notes about the data source"""
        notes = []
        
        # Data type notes
        if analysis['data_type'] == 'list':
            notes.append(f"Contains {analysis['volume_estimate']} volume list data")
        elif analysis['data_type'] == 'dict':
            notes.append("Contains structured dictionary data")
        
        # ML potential notes
        if analysis['ml_potential']['overall_score'] > 0:
            notes.append(f"ML potential score: {analysis['ml_potential']['overall_score']}/10")
        
        # Integration notes
        if analysis['integration_complexity'] == 'high':
            notes.append("High integration complexity - may require custom parsing")
        elif analysis['integration_complexity'] == 'low':
            notes.append("Low integration complexity - straightforward to integrate")
        
        # Special characteristics
        if analysis['temporal_data']:
            notes.append("Contains time-series data suitable for forecasting")
        if analysis['relationship_data']:
            notes.append("Contains relational data for building connections")
        
        return "; ".join(notes)
    
    def _assess_ml_potential(self):
        """Assess ML potential for all discovered data sources"""
        logger.info("🤖 Assessing ML potential for discovered data sources...")
        
        ml_candidates = []
        
        for data_source in self.data_sources:
            if data_source.ml_potential['overall_score'] > 0:
                ml_candidates.append(data_source)
        
        self.crawl_report['ml_training_candidates'] = len(ml_candidates)
        logger.info(f"📊 Found {len(ml_candidates)} ML training candidates")
    
    def _plan_integration(self):
        """Plan integration phases based on priority and complexity"""
        logger.info("📋 Planning integration phases...")
        
        # Sort data sources by priority and complexity
        sorted_sources = sorted(
            self.data_sources,
            key=lambda x: (
                {'high': 3, 'medium': 2, 'low': 1}[x.priority],
                -{'high': 3, 'medium': 2, 'low': 1}[x.integration_complexity]
            ),
            reverse=True
        )
        
        # Phase 1: High priority, low complexity
        phase_1 = [ds for ds in sorted_sources if ds.priority == 'high' and ds.integration_complexity == 'low']
        
        # Phase 2: High priority, medium complexity OR medium priority, low complexity
        phase_2 = [ds for ds in sorted_sources if 
                   (ds.priority == 'high' and ds.integration_complexity == 'medium') or
                   (ds.priority == 'medium' and ds.integration_complexity == 'low')]
        
        # Phase 3: Everything else
        phase_3 = [ds for ds in sorted_sources if ds not in phase_1 and ds not in phase_2]
        
        self.crawl_report['phase_1_recommendations'] = [
            {
                'url': ds.url,
                'priority': ds.priority,
                'complexity': ds.integration_complexity,
                'ml_score': ds.ml_potential['overall_score'],
                'notes': ds.notes
            } for ds in phase_1
        ]
        
        self.crawl_report['phase_2_recommendations'] = [
            {
                'url': ds.url,
                'priority': ds.priority,
                'complexity': ds.integration_complexity,
                'ml_score': ds.ml_potential['overall_score'],
                'notes': ds.notes
            } for ds in phase_2
        ]
        
        self.crawl_report['phase_3_recommendations'] = [
            {
                'url': ds.url,
                'priority': ds.priority,
                'complexity': ds.integration_complexity,
                'ml_score': ds.ml_potential['overall_score'],
                'notes': ds.notes
            } for ds in phase_3
        ]
        
        self.crawl_report['integration_ready'] = len(phase_1)
        
        logger.info(f"📊 Integration planning complete:")
        logger.info(f"  Phase 1 (immediate): {len(phase_1)} sources")
        logger.info(f"  Phase 2 (short-term): {len(phase_2)} sources")
        logger.info(f"  Phase 3 (long-term): {len(phase_3)} sources")
    
    def _generate_report(self):
        """Generate comprehensive crawl report"""
        logger.info("📊 Generating comprehensive crawl report...")
        
        # Update summary statistics
        self.crawl_report['data_sources_found'] = len(self.data_sources)
        
        # Add detailed findings
        for data_source in self.data_sources:
            finding = {
                'url': data_source.url,
                'endpoint': data_source.endpoint,
                'status_code': data_source.status_code,
                'data_type': data_source.data_type,
                'fields': data_source.fields,
                'sample_data': data_source.sample_data,
                'ml_potential': data_source.ml_potential,
                'integration_complexity': data_source.integration_complexity,
                'priority': data_source.priority,
                'temporal_data': data_source.temporal_data,
                'relationship_data': data_source.relationship_data,
                'volume_estimate': data_source.volume_estimate,
                'notes': data_source.notes
            }
            self.crawl_report['detailed_findings'].append(finding)
        
        # Add summary insights
        self.crawl_report['summary_insights'] = {
            'total_endpoints_tested': len(self.tested_endpoints),
            'accessible_endpoints': len([ep for ep in self.tested_endpoints.values() if ep.get('accessible', False)]),
            'api_endpoints_found': len([ds for ds in self.data_sources if '/api/' in ds.url]),
            'report_endpoints_found': len([ds for ds in self.data_sources if '/report/' in ds.url]),
            'high_priority_sources': len([ds for ds in self.data_sources if ds.priority == 'high']),
            'temporal_data_sources': len([ds for ds in self.data_sources if ds.temporal_data]),
            'relationship_data_sources': len([ds for ds in self.data_sources if ds.relationship_data]),
            'arima_suitable_sources': len([ds for ds in self.data_sources if ds.ml_potential['arima_suitable']]),
            'prophet_suitable_sources': len([ds for ds in self.data_sources if ds.ml_potential['prophet_suitable']]),
            'lstm_suitable_sources': len([ds for ds in self.data_sources if ds.ml_potential['lstm_suitable']]),
            'xgboost_suitable_sources': len([ds for ds in self.data_sources if ds.ml_potential['xgboost_suitable']])
        }
        
        logger.info("✅ Comprehensive crawl report generated")
    
    def save_report(self, filename: str = None):
        """Save the crawl report to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"/mnt/c/Users/psytz/32/efab.ai/efab.ai/beverly_knits_erp_crawl_report_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.crawl_report, f, indent=2, default=str)
            
            logger.info(f"📄 Crawl report saved to: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"❌ Error saving crawl report: {e}")
            return None

def main():
    """Main execution function"""
    logger.info("🚀 Starting Beverly Knits ERP Comprehensive Web Crawl")
    
    # Initialize crawler with known credentials
    crawler = ERPWebCrawler(username='psytz', password='big$cat')
    
    # Perform comprehensive crawl
    report = crawler.crawl_erp_system()
    
    # Save report
    report_file = crawler.save_report()
    
    # Print summary
    print("\n" + "="*80)
    print("📊 BEVERLY KNITS ERP CRAWL SUMMARY")
    print("="*80)
    print(f"🔍 Total URLs Discovered: {report['total_urls_discovered']}")
    print(f"📈 Data Sources Found: {report['data_sources_found']}")
    print(f"🤖 ML Training Candidates: {report['ml_training_candidates']}")
    print(f"✅ Integration Ready: {report['integration_ready']}")
    print(f"📋 Phase 1 Recommendations: {len(report['phase_1_recommendations'])}")
    print(f"📋 Phase 2 Recommendations: {len(report['phase_2_recommendations'])}")
    print(f"📋 Phase 3 Recommendations: {len(report['phase_3_recommendations'])}")
    
    if report_file:
        print(f"\n📄 Detailed report saved to: {report_file}")
    
    print("\n🎯 TOP PHASE 1 RECOMMENDATIONS:")
    for i, rec in enumerate(report['phase_1_recommendations'][:5], 1):
        print(f"  {i}. {rec['url']} (ML Score: {rec['ml_score']}, Priority: {rec['priority']})")
    
    print("\n" + "="*80)
    
    return report

if __name__ == "__main__":
    main()