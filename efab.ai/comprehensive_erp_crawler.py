#!/usr/bin/env python3
"""
Comprehensive ERP Crawler and Analysis
Beverly Knits AI Supply Chain Planner
"""

import sys
import os
import logging
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
from bs4 import BeautifulSoup
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.integrations.efab_integration import EfabERPIntegration

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ERPCrawlerAnalyzer:
    """Comprehensive ERP website crawler and data analyzer"""
    
    def __init__(self):
        self.erp = EfabERPIntegration(username='psytz', password='big$cat')
        self.analysis_results = {}
        self.discovered_endpoints = []
        self.data_structures = {}
        
    def connect_to_erp(self) -> bool:
        """Connect to ERP system"""
        try:
            if self.erp.connect():
                logger.info("✅ Connected to ERP successfully")
                return True
            else:
                logger.error("❌ Failed to connect to ERP")
                return False
        except Exception as e:
            logger.error(f"❌ ERP connection error: {e}")
            return False
    
    def discover_endpoints(self):
        """Discover available ERP endpoints"""
        logger.info("🔍 Discovering ERP endpoints...")
        
        # Known endpoints from previous analysis
        known_endpoints = [
            '/yarn',
            '/yarn/po/list',
            '/report/yarn_demand', 
            '/report/expected_yarn',
            '/fabric/so/list',
            '/inventory',
            '/suppliers',
            '/materials',
            '/orders',
            '/reports',
            '/admin',
            '/settings',
            '/users',
            '/dashboard'
        ]
        
        # Common ERP endpoint patterns
        common_patterns = [
            '/api/materials',
            '/api/suppliers',
            '/api/inventory',
            '/api/orders',
            '/data/yarn',
            '/data/fabric',
            '/export/yarn',
            '/export/inventory',
            '/reports/inventory',
            '/reports/suppliers',
            '/reports/materials',
            '/admin/materials',
            '/admin/suppliers'
        ]
        
        all_endpoints = known_endpoints + common_patterns
        accessible_endpoints = []
        
        for endpoint in all_endpoints:
            try:
                url = f"{self.erp.credentials.base_url}{endpoint}"
                response = self.erp.auth.session.get(url, timeout=10)
                
                endpoint_info = {
                    'endpoint': endpoint,
                    'status_code': response.status_code,
                    'accessible': response.status_code in [200, 401, 403],
                    'content_length': len(response.content),
                    'content_type': response.headers.get('content-type', 'unknown'),
                    'has_data': response.status_code == 200
                }
                
                if response.status_code == 200:
                    # Analyze content for data indicators
                    content = response.text.lower()
                    endpoint_info.update({
                        'has_tables': '<table' in content,
                        'has_forms': '<form' in content,
                        'has_devextreme': 'dx-data-grid' in content or 'devexpress' in content,
                        'has_json': 'application/json' in response.headers.get('content-type', ''),
                        'table_count': content.count('<table'),
                        'form_count': content.count('<form')
                    })
                    
                    accessible_endpoints.append(endpoint_info)
                    logger.info(f"✅ {endpoint}: {response.status_code} ({len(response.content)} bytes)")
                else:
                    logger.debug(f"❌ {endpoint}: {response.status_code}")
                
            except Exception as e:
                logger.debug(f"❌ {endpoint}: {e}")
        
        self.discovered_endpoints = accessible_endpoints
        logger.info(f"🎯 Found {len(accessible_endpoints)} accessible endpoints")
        
        return accessible_endpoints
    
    def analyze_data_structure(self, endpoint: str, save_sample: bool = True) -> Dict[str, Any]:
        """Analyze data structure of an endpoint"""
        logger.info(f"🔍 Analyzing data structure: {endpoint}")
        
        try:
            url = f"{self.erp.credentials.base_url}{endpoint}"
            response = self.erp.auth.session.get(url)
            
            if response.status_code != 200:
                return {'error': f"HTTP {response.status_code}"}
            
            analysis = {
                'endpoint': endpoint,
                'timestamp': datetime.now().isoformat(),
                'content_length': len(response.content),
                'content_type': response.headers.get('content-type', 'unknown')
            }
            
            # Parse HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract tables
            tables = soup.find_all('table')
            analysis['tables'] = []
            
            for i, table in enumerate(tables):
                table_analysis = self._analyze_table(table, f"table_{i}")
                analysis['tables'].append(table_analysis)
            
            # Extract forms
            forms = soup.find_all('form')
            analysis['forms'] = []
            
            for i, form in enumerate(forms):
                form_analysis = self._analyze_form(form, f"form_{i}")
                analysis['forms'].append(form_analysis)
            
            # Extract JavaScript data
            scripts = soup.find_all('script')
            analysis['javascript_data'] = []
            
            for script in scripts:
                if script.string:
                    js_data = self._extract_javascript_data(script.string)
                    if js_data:
                        analysis['javascript_data'].extend(js_data)
            
            # DevExtreme data grids
            analysis['devextreme_grids'] = self._analyze_devextreme_grids(response.text)
            
            # Save sample if requested
            if save_sample:
                sample_file = f"erp_sample_{endpoint.replace('/', '_').strip('_')}.html"
                with open(sample_file, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                analysis['sample_file'] = sample_file
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {endpoint}: {e}")
            return {'error': str(e)}
    
    def _analyze_table(self, table_soup, table_id: str) -> Dict[str, Any]:
        """Analyze HTML table structure"""
        analysis = {
            'id': table_id,
            'row_count': 0,
            'column_count': 0,
            'headers': [],
            'sample_rows': [],
            'data_types': {}
        }
        
        try:
            # Extract headers
            header_row = table_soup.find('tr')
            if header_row:
                headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
                analysis['headers'] = headers
                analysis['column_count'] = len(headers)
            
            # Extract sample rows
            rows = table_soup.find_all('tr')
            analysis['row_count'] = len(rows)
            
            for i, row in enumerate(rows[1:6]):  # First 5 data rows
                cells = [td.get_text(strip=True) for td in row.find_all(['td', 'th'])]
                analysis['sample_rows'].append(cells)
            
            # Analyze data types
            if analysis['sample_rows'] and analysis['headers']:
                for col_idx, header in enumerate(analysis['headers']):
                    values = [row[col_idx] for row in analysis['sample_rows'] if col_idx < len(row)]
                    analysis['data_types'][header] = self._detect_data_type(values)
            
        except Exception as e:
            analysis['error'] = str(e)
        
        return analysis
    
    def _analyze_form(self, form_soup, form_id: str) -> Dict[str, Any]:
        """Analyze HTML form structure"""
        analysis = {
            'id': form_id,
            'action': form_soup.get('action', ''),
            'method': form_soup.get('method', 'GET').upper(),
            'inputs': []
        }
        
        try:
            inputs = form_soup.find_all(['input', 'select', 'textarea'])
            
            for input_elem in inputs:
                input_info = {
                    'name': input_elem.get('name', ''),
                    'type': input_elem.get('type', input_elem.name),
                    'value': input_elem.get('value', ''),
                    'required': input_elem.has_attr('required'),
                    'placeholder': input_elem.get('placeholder', '')
                }
                
                if input_elem.name == 'select':
                    options = [opt.get_text(strip=True) for opt in input_elem.find_all('option')]
                    input_info['options'] = options
                
                analysis['inputs'].append(input_info)
            
        except Exception as e:
            analysis['error'] = str(e)
        
        return analysis
    
    def _extract_javascript_data(self, js_content: str) -> List[Dict]:
        """Extract structured data from JavaScript"""
        data_objects = []
        
        try:
            # Look for common data patterns
            patterns = [
                r'var\s+(\w+)\s*=\s*(\{.*?\});',
                r'window\.(\w+)\s*=\s*(\{.*?\});',
                r'data:\s*(\[.*?\])',
                r'gridData:\s*(\[.*?\])',
                r'tableData:\s*(\[.*?\])'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, js_content, re.DOTALL)
                for match in matches:
                    try:
                        if len(match) == 2:
                            var_name, data_str = match
                            # Try to parse as JSON
                            data = json.loads(data_str)
                            data_objects.append({
                                'variable': var_name,
                                'type': type(data).__name__,
                                'data': data
                            })
                        else:
                            # Single match (array data)
                            data = json.loads(match)
                            data_objects.append({
                                'variable': 'anonymous',
                                'type': type(data).__name__,
                                'data': data
                            })
                    except:
                        continue
        
        except Exception as e:
            logger.debug(f"JavaScript parsing error: {e}")
        
        return data_objects
    
    def _analyze_devextreme_grids(self, html_content: str) -> List[Dict]:
        """Analyze DevExtreme data grid configurations"""
        grids = []
        
        try:
            # Look for DevExtreme grid configurations
            grid_patterns = [
                r'("#\w+")\.dxDataGrid\(({.*?})\)',
                r'\.dxDataGrid\(\s*({.*?})\s*\)',
                r'dataSource:\s*({.*?})',
                r'columns:\s*(\[.*?\])'
            ]
            
            for pattern in grid_patterns:
                matches = re.findall(pattern, html_content, re.DOTALL)
                for match in matches:
                    try:
                        config_str = match if isinstance(match, str) else match[0]
                        # This would need more sophisticated parsing
                        grids.append({
                            'config_snippet': config_str[:200] + '...' if len(config_str) > 200 else config_str,
                            'estimated_columns': config_str.count('dataField'),
                            'has_editing': 'editing' in config_str,
                            'has_filtering': 'filtering' in config_str
                        })
                    except:
                        continue
        
        except Exception as e:
            logger.debug(f"DevExtreme parsing error: {e}")
        
        return grids
    
    def _detect_data_type(self, values: List[str]) -> str:
        """Detect data type from sample values"""
        if not values:
            return 'unknown'
        
        # Remove empty values
        non_empty = [v for v in values if v.strip()]
        if not non_empty:
            return 'empty'
        
        # Check for numbers
        numeric_count = 0
        date_count = 0
        
        for value in non_empty:
            # Check numeric
            try:
                float(value.replace(',', '').replace('$', ''))
                numeric_count += 1
            except:
                pass
            
            # Check date patterns
            if re.match(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', value) or \
               re.match(r'\d{4}-\d{2}-\d{2}', value):
                date_count += 1
        
        total = len(non_empty)
        
        if numeric_count / total > 0.8:
            return 'numeric'
        elif date_count / total > 0.6:
            return 'date'
        elif any(len(v) > 50 for v in non_empty):
            return 'text_long'
        else:
            return 'text'
    
    def map_to_domain_entities(self, analysis_results: Dict) -> Dict[str, Any]:
        """Map discovered data to Beverly Knits domain entities"""
        logger.info("🎯 Mapping ERP data to domain entities...")
        
        entity_mapping = {
            'materials': {
                'likely_endpoints': [],
                'data_fields': [],
                'domain_mapping': {}
            },
            'suppliers': {
                'likely_endpoints': [],
                'data_fields': [],
                'domain_mapping': {}
            },
            'inventory': {
                'likely_endpoints': [],
                'data_fields': [],
                'domain_mapping': {}
            },
            'orders': {
                'likely_endpoints': [],
                'data_fields': [],
                'domain_mapping': {}
            },
            'forecasts': {
                'likely_endpoints': [],
                'data_fields': [],
                'domain_mapping': {}
            }
        }
        
        # Analyze each endpoint for entity patterns
        for endpoint_data in self.discovered_endpoints:
            endpoint = endpoint_data['endpoint']
            
            # Classify endpoint by entity type
            if 'yarn' in endpoint.lower() or 'material' in endpoint.lower():
                entity_type = 'materials'
            elif 'supplier' in endpoint.lower() or 'vendor' in endpoint.lower():
                entity_type = 'suppliers'
            elif 'inventory' in endpoint.lower() or 'stock' in endpoint.lower():
                entity_type = 'inventory'
            elif 'order' in endpoint.lower() or 'so' in endpoint.lower():
                entity_type = 'orders'
            elif 'demand' in endpoint.lower() or 'forecast' in endpoint.lower():
                entity_type = 'forecasts'
            else:
                continue  # Skip unknown endpoints
            
            entity_mapping[entity_type]['likely_endpoints'].append(endpoint)
            
            # Extract data fields from analysis
            if endpoint in analysis_results:
                endpoint_analysis = analysis_results[endpoint]
                
                # From tables
                for table in endpoint_analysis.get('tables', []):
                    entity_mapping[entity_type]['data_fields'].extend(table.get('headers', []))
                
                # From forms
                for form in endpoint_analysis.get('forms', []):
                    field_names = [inp['name'] for inp in form.get('inputs', []) if inp['name']]
                    entity_mapping[entity_type]['data_fields'].extend(field_names)
        
        # Create domain mappings
        for entity_type, mapping in entity_mapping.items():
            unique_fields = list(set(mapping['data_fields']))
            mapping['data_fields'] = unique_fields
            mapping['domain_mapping'] = self._create_domain_mapping(entity_type, unique_fields)
        
        return entity_mapping
    
    def _create_domain_mapping(self, entity_type: str, data_fields: List[str]) -> Dict[str, str]:
        """Create mapping from ERP fields to domain entity fields"""
        mappings = {
            'materials': {
                'yarn_id': 'material_id',
                'item_id': 'material_id',
                'description': 'name',
                'yarn_description': 'name',
                'type': 'material_type',
                'blend': 'specifications.blend',
                'color': 'specifications.color',
                'cost': 'cost.amount',
                'unit_cost': 'cost.amount',
                'price': 'cost.amount'
            },
            'suppliers': {
                'supplier_id': 'supplier_id',
                'supplier': 'name',
                'vendor': 'name',
                'lead_time': 'lead_time_days',
                'moq': 'min_order_quantity',
                'reliability': 'reliability_score'
            },
            'inventory': {
                'inventory': 'on_hand_quantity',
                'on_hand': 'on_hand_quantity',
                'available': 'available_quantity',
                'on_order': 'open_po_quantity',
                'allocated': 'allocated_quantity',
                'safety_stock': 'safety_stock_quantity'
            },
            'orders': {
                'order_id': 'order_id',
                'so_number': 'order_id',
                'customer': 'customer_name',
                'quantity': 'quantity.amount',
                'due_date': 'required_date',
                'status': 'order_status'
            },
            'forecasts': {
                'demand': 'demand_quantity',
                'forecast': 'forecast_quantity',
                'period': 'forecast_period',
                'confidence': 'confidence_score'
            }
        }
        
        entity_mappings = mappings.get(entity_type, {})
        result = {}
        
        for field in data_fields:
            field_lower = field.lower()
            for erp_field, domain_field in entity_mappings.items():
                if erp_field in field_lower:
                    result[field] = domain_field
                    break
        
        return result
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        logger.info("📊 Generating comprehensive analysis report...")
        
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'erp_system': 'Efab ERP',
            'base_url': self.erp.credentials.base_url,
            'summary': {
                'total_endpoints_tested': len(self.discovered_endpoints) if hasattr(self, 'discovered_endpoints') else 0,
                'accessible_endpoints': len([ep for ep in getattr(self, 'discovered_endpoints', []) if ep.get('has_data', False)]),
                'endpoints_with_tables': len([ep for ep in getattr(self, 'discovered_endpoints', []) if ep.get('has_tables', False)]),
                'endpoints_with_grids': len([ep for ep in getattr(self, 'discovered_endpoints', []) if ep.get('has_devextreme', False)])
            },
            'discovered_endpoints': getattr(self, 'discovered_endpoints', []),
            'data_structures': getattr(self, 'data_structures', {}),
            'entity_mapping': {},
            'integration_recommendations': []
        }
        
        # Add entity mapping if available
        if hasattr(self, 'data_structures') and self.data_structures:
            report['entity_mapping'] = self.map_to_domain_entities(self.data_structures)
        
        # Generate integration recommendations
        report['integration_recommendations'] = self._generate_integration_recommendations(report)
        
        return report
    
    def _generate_integration_recommendations(self, report: Dict) -> List[Dict]:
        """Generate integration recommendations based on analysis"""
        recommendations = []
        
        # High-priority endpoints
        high_priority = ['/yarn', '/fabric/so/list', '/report/yarn_demand', '/report/expected_yarn']
        accessible_high_priority = [ep for ep in report['discovered_endpoints'] 
                                   if ep['endpoint'] in high_priority and ep.get('has_data', False)]
        
        if accessible_high_priority:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Data Integration',
                'title': 'Implement primary data endpoints',
                'description': f'Focus on integrating {len(accessible_high_priority)} key endpoints that contain business-critical data',
                'endpoints': [ep['endpoint'] for ep in accessible_high_priority],
                'estimated_effort': '1-2 weeks'
            })
        
        # DevExtreme grid handling
        grid_endpoints = [ep for ep in report['discovered_endpoints'] if ep.get('has_devextreme', False)]
        if grid_endpoints:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Technical Implementation',
                'title': 'Implement DevExtreme grid parsing',
                'description': f'Develop specialized parsing for {len(grid_endpoints)} endpoints using DevExtreme data grids',
                'technical_notes': 'Requires JavaScript execution or API endpoint discovery for data extraction',
                'estimated_effort': '1 week'
            })
        
        # Real-time sync capability
        recommendations.append({
            'priority': 'HIGH',
            'category': 'Architecture',
            'title': 'Implement real-time data synchronization',
            'description': 'Set up automated sync pipeline for critical inventory and order data',
            'requirements': ['Authentication session management', 'Error handling and retry logic', 'Data validation pipeline'],
            'estimated_effort': '1-2 weeks'
        })
        
        return recommendations

def main():
    """Main crawler execution"""
    logger.info("🚀 Beverly Knits AI Supply Chain Planner - ERP Comprehensive Analysis")
    logger.info("="*80)
    
    crawler = ERPCrawlerAnalyzer()
    
    try:
        # Step 1: Connect to ERP
        logger.info("🔗 Step 1: Connecting to ERP system...")
        if not crawler.connect_to_erp():
            logger.error("❌ Failed to connect to ERP. Exiting.")
            return False
        
        # Step 2: Discover endpoints
        logger.info("🔍 Step 2: Discovering ERP endpoints...")
        endpoints = crawler.discover_endpoints()
        
        # Step 3: Analyze key endpoints
        logger.info("📊 Step 3: Analyzing data structures...")
        key_endpoints = [ep['endpoint'] for ep in endpoints if ep.get('has_data', False)][:8]  # Top 8 endpoints
        
        for endpoint in key_endpoints:
            logger.info(f"Analyzing {endpoint}...")
            analysis = crawler.analyze_data_structure(endpoint)
            crawler.data_structures[endpoint] = analysis
        
        # Step 4: Generate comprehensive report
        logger.info("📋 Step 4: Generating comprehensive report...")
        report = crawler.generate_comprehensive_report()
        
        # Save report
        report_file = f"comprehensive_erp_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"✅ Comprehensive report saved to: {report_file}")
        
        # Display summary
        logger.info("\n" + "="*80)
        logger.info("📊 ANALYSIS SUMMARY")
        logger.info("="*80)
        
        summary = report['summary']
        logger.info(f"Endpoints tested: {summary['total_endpoints_tested']}")
        logger.info(f"Accessible endpoints: {summary['accessible_endpoints']}")
        logger.info(f"Endpoints with tables: {summary['endpoints_with_tables']}")
        logger.info(f"Endpoints with data grids: {summary['endpoints_with_grids']}")
        
        logger.info("\n🎯 KEY FINDINGS:")
        for rec in report['integration_recommendations']:
            logger.info(f"• {rec['title']} ({rec['priority']} priority)")
        
        logger.info("\n🎉 Comprehensive ERP analysis completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)