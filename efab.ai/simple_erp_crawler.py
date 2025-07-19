#!/usr/bin/env python3
"""
Simple ERP Crawler and Analysis
Beverly Knits AI Supply Chain Planner
"""

import sys
import os
import logging
import json
import re
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.integrations.efab_integration import EfabERPIntegration

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleERPCrawler:
    """Simple ERP website crawler and analyzer"""
    
    def __init__(self):
        self.erp = EfabERPIntegration(username='psytz', password='big$cat')
        self.discovered_data = {}
        
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
    
    def analyze_key_endpoints(self):
        """Analyze key ERP endpoints"""
        logger.info("🔍 Analyzing key ERP endpoints...")
        
        # Key endpoints we know work
        endpoints = {
            'yarn_inventory': '/yarn',
            'yarn_demand': '/report/yarn_demand',
            'expected_yarn': '/report/expected_yarn',
            'fabric_orders': '/fabric/so/list',
            'yarn_po': '/yarn/po/list'
        }
        
        analysis_results = {}
        
        for name, endpoint in endpoints.items():
            try:
                logger.info(f"🔍 Analyzing {name}: {endpoint}")
                
                url = f"{self.erp.credentials.base_url}{endpoint}"
                response = self.erp.auth.session.get(url)
                
                if response.status_code == 200:
                    analysis = self._analyze_html_content(response.text, endpoint)
                    analysis_results[name] = analysis
                    
                    # Save sample data
                    sample_file = f"sample_{name}.html"
                    with open(sample_file, 'w', encoding='utf-8') as f:
                        f.write(response.text)
                    
                    logger.info(f"✅ {name}: {analysis['summary']}")
                else:
                    logger.warning(f"❌ {name}: HTTP {response.status_code}")
                    
            except Exception as e:
                logger.error(f"❌ Error analyzing {name}: {e}")
        
        return analysis_results
    
    def _analyze_html_content(self, html: str, endpoint: str) -> Dict[str, Any]:
        """Analyze HTML content for data structures"""
        analysis = {
            'endpoint': endpoint,
            'content_length': len(html),
            'summary': '',
            'tables': [],
            'forms': [],
            'data_indicators': {},
            'field_names': []
        }
        
        # Count HTML elements
        table_count = html.count('<table')
        form_count = html.count('<form')
        input_count = html.count('<input')
        
        # Check for data grid frameworks
        has_devextreme = 'dx-data-grid' in html or 'DevExpress' in html
        has_datatables = 'DataTables' in html or 'datatables' in html
        
        # Extract table headers (simple regex approach)
        table_headers = []
        header_patterns = [
            r'<th[^>]*>(.*?)</th>',
            r'<td[^>]*class="[^"]*header[^"]*"[^>]*>(.*?)</td>',
            r'dataField:\s*["\']([^"\']+)["\']'
        ]
        
        for pattern in header_patterns:
            matches = re.findall(pattern, html, re.IGNORECASE | re.DOTALL)
            for match in matches:
                clean_header = re.sub(r'<[^>]+>', '', match).strip()
                if clean_header and len(clean_header) < 50:
                    table_headers.append(clean_header)
        
        # Extract form field names
        form_fields = []
        field_patterns = [
            r'name=["\']([^"\']+)["\']',
            r'id=["\']([^"\']+)["\']'
        ]
        
        for pattern in field_patterns:
            matches = re.findall(pattern, html, re.IGNORECASE)
            for match in matches:
                if match and len(match) < 30 and not match.startswith('_'):
                    form_fields.append(match)
        
        # Look for JavaScript data
        js_data_patterns = [
            r'var\s+\w+\s*=\s*\[.*?\]',
            r'data:\s*\[.*?\]',
            r'dataSource:\s*\[.*?\]'
        ]
        
        has_js_data = any(re.search(pattern, html, re.DOTALL) for pattern in js_data_patterns)
        
        # Update analysis
        analysis.update({
            'tables': {'count': table_count, 'headers': list(set(table_headers))},
            'forms': {'count': form_count, 'fields': list(set(form_fields))},
            'data_indicators': {
                'has_devextreme': has_devextreme,
                'has_datatables': has_datatables,
                'has_js_data': has_js_data,
                'input_count': input_count
            },
            'field_names': list(set(table_headers + form_fields))
        })
        
        # Generate summary
        summary_parts = []
        if table_count > 0:
            summary_parts.append(f"{table_count} tables")
        if form_count > 0:
            summary_parts.append(f"{form_count} forms")
        if has_devextreme:
            summary_parts.append("DevExtreme grids")
        if has_js_data:
            summary_parts.append("JavaScript data")
        
        analysis['summary'] = ', '.join(summary_parts) if summary_parts else 'Basic HTML content'
        
        return analysis
    
    def map_to_domain_entities(self, analysis_results: Dict) -> Dict[str, Any]:
        """Map discovered fields to Beverly Knits domain entities"""
        logger.info("🎯 Mapping ERP data to domain entities...")
        
        # Domain entity field mappings
        entity_mappings = {
            'materials': {
                'potential_fields': ['yarn_id', 'item_id', 'description', 'type', 'blend', 'color', 'cost', 'price'],
                'domain_fields': ['material_id', 'name', 'material_type', 'specifications', 'cost'],
                'found_fields': []
            },
            'suppliers': {
                'potential_fields': ['supplier_id', 'supplier', 'vendor', 'lead_time', 'moq', 'reliability'],
                'domain_fields': ['supplier_id', 'name', 'lead_time_days', 'min_order_quantity', 'reliability_score'],
                'found_fields': []
            },
            'inventory': {
                'potential_fields': ['inventory', 'on_hand', 'available', 'on_order', 'allocated', 'safety_stock'],
                'domain_fields': ['on_hand_quantity', 'available_quantity', 'open_po_quantity', 'allocated_quantity'],
                'found_fields': []
            },
            'orders': {
                'potential_fields': ['order_id', 'so_number', 'customer', 'quantity', 'due_date', 'status'],
                'domain_fields': ['order_id', 'customer_name', 'quantity', 'required_date', 'status'],
                'found_fields': []
            },
            'forecasts': {
                'potential_fields': ['demand', 'forecast', 'period', 'confidence', 'expected'],
                'domain_fields': ['demand_quantity', 'forecast_period', 'confidence_score'],
                'found_fields': []
            }
        }
        
        # Analyze each endpoint's fields
        for endpoint_name, endpoint_analysis in analysis_results.items():
            all_fields = endpoint_analysis.get('field_names', [])
            
            # Match fields to entities
            for entity_type, mapping in entity_mappings.items():
                for field in all_fields:
                    field_lower = field.lower()
                    for potential_field in mapping['potential_fields']:
                        if potential_field in field_lower:
                            mapping['found_fields'].append({
                                'erp_field': field,
                                'endpoint': endpoint_name,
                                'matched_pattern': potential_field
                            })
        
        return entity_mappings
    
    def generate_integration_plan(self, analysis_results: Dict, entity_mappings: Dict) -> Dict[str, Any]:
        """Generate integration plan based on analysis"""
        logger.info("📋 Generating integration plan...")
        
        plan = {
            'priority_endpoints': [],
            'data_extraction_strategy': {},
            'entity_integration_plan': {},
            'technical_requirements': [],
            'estimated_timeline': {}
        }
        
        # Priority endpoints (based on data richness)
        endpoint_scores = {}
        for name, analysis in analysis_results.items():
            score = 0
            score += len(analysis.get('field_names', [])) * 2  # Field count
            score += analysis.get('tables', {}).get('count', 0) * 10  # Table count
            score += analysis.get('forms', {}).get('count', 0) * 5  # Form count
            if analysis.get('data_indicators', {}).get('has_devextreme', False):
                score += 20  # DevExtreme bonus
            endpoint_scores[name] = score
        
        # Sort by score
        sorted_endpoints = sorted(endpoint_scores.items(), key=lambda x: x[1], reverse=True)
        plan['priority_endpoints'] = [
            {'endpoint': name, 'score': score, 'priority': 'HIGH' if score > 30 else 'MEDIUM' if score > 15 else 'LOW'}
            for name, score in sorted_endpoints
        ]
        
        # Data extraction strategy
        for name, analysis in analysis_results.items():
            strategy = []
            
            if analysis.get('data_indicators', {}).get('has_devextreme', False):
                strategy.append('DevExtreme grid API extraction')
            if analysis.get('tables', {}).get('count', 0) > 0:
                strategy.append('HTML table parsing')
            if analysis.get('data_indicators', {}).get('has_js_data', False):
                strategy.append('JavaScript data extraction')
            
            plan['data_extraction_strategy'][name] = strategy
        
        # Entity integration plan
        for entity_type, mapping in entity_mappings.items():
            if mapping['found_fields']:
                plan['entity_integration_plan'][entity_type] = {
                    'field_mappings': mapping['found_fields'],
                    'confidence': 'HIGH' if len(mapping['found_fields']) > 3 else 'MEDIUM' if len(mapping['found_fields']) > 1 else 'LOW',
                    'implementation_notes': f"Found {len(mapping['found_fields'])} relevant fields"
                }
        
        # Technical requirements
        plan['technical_requirements'] = [
            'Session-based authentication with form login',
            'HTML parsing for table data extraction',
            'DevExtreme data grid API integration',
            'JavaScript execution for dynamic data',
            'Data validation and quality checks',
            'Error handling and retry mechanisms'
        ]
        
        # Timeline estimation
        plan['estimated_timeline'] = {
            'Phase 1 - Authentication Setup': '3-5 days',
            'Phase 2 - Data Extraction Development': '1-2 weeks',
            'Phase 3 - Domain Entity Mapping': '3-5 days',
            'Phase 4 - Integration Testing': '3-5 days',
            'Phase 5 - Production Deployment': '2-3 days',
            'Total Estimated Time': '3-4 weeks'
        }
        
        return plan
    
    def generate_comprehensive_report(self, analysis_results: Dict, entity_mappings: Dict, integration_plan: Dict) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        logger.info("📊 Generating comprehensive report...")
        
        report = {
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'erp_system': 'Efab ERP',
                'base_url': self.erp.credentials.base_url,
                'analysis_version': '1.0'
            },
            'executive_summary': {
                'endpoints_analyzed': len(analysis_results),
                'total_fields_discovered': sum(len(a.get('field_names', [])) for a in analysis_results.values()),
                'entities_mappable': len([e for e in entity_mappings.values() if e['found_fields']]),
                'integration_feasibility': 'HIGH'
            },
            'detailed_analysis': analysis_results,
            'entity_mappings': entity_mappings,
            'integration_plan': integration_plan,
            'recommendations': []
        }
        
        # Generate recommendations
        recommendations = []
        
        # High-priority recommendations
        high_priority_endpoints = [ep for ep in integration_plan['priority_endpoints'] if ep['priority'] == 'HIGH']
        if high_priority_endpoints:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Data Integration',
                'title': 'Focus on high-value endpoints first',
                'description': f"Start with {len(high_priority_endpoints)} endpoints that have the richest data structures",
                'action_items': [ep['endpoint'] for ep in high_priority_endpoints]
            })
        
        # DevExtreme handling
        devextreme_endpoints = [name for name, analysis in analysis_results.items() 
                              if analysis.get('data_indicators', {}).get('has_devextreme', False)]
        if devextreme_endpoints:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Technical Implementation',
                'title': 'Implement DevExtreme grid data extraction',
                'description': f"Develop specialized handling for {len(devextreme_endpoints)} endpoints using DevExtreme",
                'technical_notes': 'Consider using browser automation or finding API endpoints for structured data access'
            })
        
        # Entity completeness
        complete_entities = [entity for entity, mapping in entity_mappings.items() 
                           if len(mapping['found_fields']) >= 3]
        if complete_entities:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Data Modeling',
                'title': 'Prioritize complete entity mappings',
                'description': f"Focus on {len(complete_entities)} entities with sufficient field mappings",
                'entities': complete_entities
            })
        
        report['recommendations'] = recommendations
        
        return report

def main():
    """Main crawler execution"""
    logger.info("🚀 Beverly Knits AI Supply Chain Planner - Simple ERP Analysis")
    logger.info("="*80)
    
    crawler = SimpleERPCrawler()
    
    try:
        # Step 1: Connect to ERP
        logger.info("🔗 Step 1: Connecting to ERP system...")
        if not crawler.connect_to_erp():
            logger.error("❌ Failed to connect to ERP. Exiting.")
            return False
        
        # Step 2: Analyze key endpoints
        logger.info("🔍 Step 2: Analyzing key endpoints...")
        analysis_results = crawler.analyze_key_endpoints()
        
        # Step 3: Map to domain entities
        logger.info("🎯 Step 3: Mapping to domain entities...")
        entity_mappings = crawler.map_to_domain_entities(analysis_results)
        
        # Step 4: Generate integration plan
        logger.info("📋 Step 4: Generating integration plan...")
        integration_plan = crawler.generate_integration_plan(analysis_results, entity_mappings)
        
        # Step 5: Generate comprehensive report
        logger.info("📊 Step 5: Generating comprehensive report...")
        report = crawler.generate_comprehensive_report(analysis_results, entity_mappings, integration_plan)
        
        # Save report
        report_file = f"erp_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"✅ Analysis report saved to: {report_file}")
        
        # Display key findings
        logger.info("\n" + "="*80)
        logger.info("📊 KEY FINDINGS")
        logger.info("="*80)
        
        exec_summary = report['executive_summary']
        logger.info(f"📈 Endpoints analyzed: {exec_summary['endpoints_analyzed']}")
        logger.info(f"📈 Fields discovered: {exec_summary['total_fields_discovered']}")
        logger.info(f"📈 Entities mappable: {exec_summary['entities_mappable']}")
        logger.info(f"📈 Integration feasibility: {exec_summary['integration_feasibility']}")
        
        logger.info("\n🎯 PRIORITY ENDPOINTS:")
        for ep in integration_plan['priority_endpoints'][:3]:
            logger.info(f"• {ep['endpoint']} ({ep['priority']} priority, score: {ep['score']})")
        
        logger.info("\n🔧 ENTITY MAPPINGS:")
        for entity, mapping in entity_mappings.items():
            if mapping['found_fields']:
                logger.info(f"• {entity}: {len(mapping['found_fields'])} fields mapped")
        
        logger.info("\n📋 RECOMMENDATIONS:")
        for rec in report['recommendations']:
            logger.info(f"• {rec['title']} ({rec['priority']} priority)")
        
        logger.info("\n🎉 ERP analysis completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)