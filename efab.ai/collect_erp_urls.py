#!/usr/bin/env python3
"""
ERP URL Collection and Validation Tool
Beverly Knits AI Supply Chain Planner
"""

import sys
import os
import logging
import json
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.integrations.efab_integration import EfabERPIntegration

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ERPURLCollector:
    """Collect and validate additional ERP URLs"""
    
    def __init__(self):
        self.erp = EfabERPIntegration(username='psytz', password='big$cat')
        self.validated_urls = {}
    
    def connect(self) -> bool:
        """Connect to ERP"""
        return self.erp.connect()
    
    def validate_url(self, url_name: str, url_path: str) -> Dict[str, Any]:
        """Validate a single URL"""
        logger.info(f"🔍 Validating {url_name}: {url_path}")
        
        try:
            # Handle both full URLs and path-only URLs
            if url_path.startswith('http'):
                full_url = url_path
            else:
                if not url_path.startswith('/'):
                    url_path = '/' + url_path
                full_url = f"{self.erp.credentials.base_url}{url_path}"
            
            response = self.erp.auth.session.get(full_url, timeout=15)
            
            validation_result = {
                'url_name': url_name,
                'url_path': url_path,
                'full_url': full_url,
                'status_code': response.status_code,
                'accessible': response.status_code == 200,
                'content_length': len(response.content),
                'content_type': response.headers.get('content-type', 'unknown'),
                'validation_timestamp': datetime.now().isoformat()
            }
            
            if response.status_code == 200:
                # Analyze content
                content = response.text.lower()
                validation_result.update({
                    'has_tables': '<table' in content,
                    'has_forms': '<form' in content,
                    'has_devextreme': 'dx-data-grid' in content or 'devextreme' in content,
                    'table_count': content.count('<table'),
                    'estimated_data_richness': self._estimate_data_richness(content)
                })
                
                # Save sample if data-rich
                if validation_result['estimated_data_richness'] > 5:
                    sample_file = f"sample_{url_name.replace(' ', '_').lower()}.html"
                    with open(sample_file, 'w', encoding='utf-8') as f:
                        f.write(response.text)
                    validation_result['sample_file'] = sample_file
                
                logger.info(f"✅ {url_name}: {response.status_code} ({len(response.content)} bytes, richness: {validation_result['estimated_data_richness']})")
            else:
                logger.warning(f"❌ {url_name}: HTTP {response.status_code}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"❌ Error validating {url_name}: {e}")
            return {
                'url_name': url_name,
                'url_path': url_path,
                'error': str(e),
                'accessible': False
            }
    
    def _estimate_data_richness(self, html_content: str) -> int:
        """Estimate how data-rich a page is"""
        score = 0
        
        # Count data indicators
        score += html_content.count('<table') * 5
        score += html_content.count('<tr') * 1
        score += html_content.count('dx-data-grid') * 10
        score += html_content.count('devextreme') * 8
        score += html_content.count('datafield') * 2
        score += html_content.count('json') * 3
        
        # Look for form inputs
        score += html_content.count('<input') * 1
        score += html_content.count('<select') * 2
        
        return min(score, 100)  # Cap at 100
    
    def collect_urls_interactively(self):
        """Collect URLs interactively from user"""
        logger.info("📝 ERP URL Collection - Enter URLs for validation")
        logger.info("You mentioned: inventory reports, sales order report, yarn demand, etc.")
        logger.info("Enter URLs one by one (press Enter with empty input to finish):")
        
        urls_to_validate = {}
        
        while True:
            url_name = input("\nReport name (e.g., 'Inventory Report'): ").strip()
            if not url_name:
                break
                
            url_path = input(f"URL path for {url_name}: ").strip()
            if not url_path:
                continue
                
            urls_to_validate[url_name] = url_path
            logger.info(f"Added: {url_name} -> {url_path}")
        
        return urls_to_validate
    
    def validate_all_urls(self, urls: Dict[str, str]) -> Dict[str, Any]:
        """Validate all provided URLs"""
        logger.info(f"🔍 Validating {len(urls)} URLs...")
        
        validation_results = {}
        
        for url_name, url_path in urls.items():
            validation_results[url_name] = self.validate_url(url_name, url_path)
        
        return validation_results
    
    def generate_integration_report(self, validation_results: Dict) -> Dict[str, Any]:
        """Generate integration report based on validated URLs"""
        logger.info("📊 Generating integration report...")
        
        accessible_urls = {name: result for name, result in validation_results.items() 
                          if result.get('accessible', False)}
        
        report = {
            'validation_timestamp': datetime.now().isoformat(),
            'total_urls_tested': len(validation_results),
            'accessible_urls_count': len(accessible_urls),
            'success_rate': len(accessible_urls) / len(validation_results) * 100 if validation_results else 0,
            'url_validation_results': validation_results,
            'integration_recommendations': [],
            'priority_urls': []
        }
        
        # Rank URLs by data richness
        ranked_urls = sorted(
            [(name, result) for name, result in accessible_urls.items()],
            key=lambda x: x[1].get('estimated_data_richness', 0),
            reverse=True
        )
        
        report['priority_urls'] = [
            {
                'name': name,
                'url': result['url_path'],
                'data_richness': result.get('estimated_data_richness', 0),
                'priority': 'HIGH' if result.get('estimated_data_richness', 0) > 20 else 'MEDIUM' if result.get('estimated_data_richness', 0) > 10 else 'LOW'
            }
            for name, result in ranked_urls[:10]  # Top 10
        ]
        
        # Generate recommendations
        high_priority_count = len([url for url in report['priority_urls'] if url['priority'] == 'HIGH'])
        
        recommendations = []
        
        if high_priority_count > 0:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Data Integration',
                'title': f'Integrate {high_priority_count} high-priority data sources',
                'description': 'These URLs contain rich data structures that will significantly enhance the AI planning system',
                'urls': [url['name'] for url in report['priority_urls'] if url['priority'] == 'HIGH']
            })
        
        if len(accessible_urls) > 5:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Comprehensive Integration',
                'title': 'Develop comprehensive data sync pipeline',
                'description': f'With {len(accessible_urls)} accessible endpoints, create a unified sync pipeline for all data sources',
                'estimated_effort': '2-3 weeks'
            })
        
        recommendations.append({
            'priority': 'HIGH',
            'category': 'Production Deployment',
            'title': 'Deploy enhanced Beverly Knits AI system',
            'description': 'With validated ERP URLs, the system is ready for production deployment with daily planning capabilities',
            'next_steps': ['Configure daily sync schedule', 'Set up monitoring and alerts', 'Train users on new features']
        })
        
        report['integration_recommendations'] = recommendations
        
        return report

def main():
    """Main URL collection and validation"""
    logger.info("🎯 Beverly Knits ERP - URL Collection and Validation Tool")
    logger.info("="*60)
    
    collector = ERPURLCollector()
    
    try:
        # Connect to ERP
        logger.info("🔗 Connecting to ERP...")
        if not collector.connect():
            logger.error("❌ Failed to connect to ERP")
            return False
        
        logger.info("✅ Connected to ERP successfully")
        
        # Option 1: Collect URLs interactively
        print("\n" + "="*60)
        print("OPTION 1: Interactive URL Collection")
        print("="*60)
        
        collect_interactively = input("\nWould you like to enter URLs interactively? (y/n): ").strip().lower()
        
        if collect_interactively == 'y':
            urls = collector.collect_urls_interactively()
        else:
            # Option 2: Use sample URLs for demonstration
            logger.info("Using sample URLs for demonstration...")
            urls = {
                'Current Inventory': '/yarn',
                'Sales Orders': '/fabric/so/list',
                'Yarn Demand Report': '/report/yarn_demand',
                'Expected Yarn Report': '/report/expected_yarn',
                'Purchase Orders': '/yarn/po/list',
                'Supplier Performance': '/suppliers',
                'Cost Analysis': '/reports/cost',
                'Material Master': '/materials'
            }
        
        if not urls:
            logger.warning("No URLs provided. Exiting.")
            return False
        
        # Validate all URLs
        logger.info(f"\n🔍 Validating {len(urls)} URLs...")
        validation_results = collector.validate_all_urls(urls)
        
        # Generate integration report
        report = collector.generate_integration_report(validation_results)
        
        # Save report
        report_file = f"erp_url_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"\n📄 Validation report saved to: {report_file}")
        
        # Display summary
        logger.info("\n" + "="*60)
        logger.info("📊 URL VALIDATION SUMMARY")
        logger.info("="*60)
        
        logger.info(f"URLs tested: {report['total_urls_tested']}")
        logger.info(f"Accessible URLs: {report['accessible_urls_count']}")
        logger.info(f"Success rate: {report['success_rate']:.1f}%")
        
        logger.info("\n🎯 PRIORITY URLS:")
        for url in report['priority_urls'][:5]:
            logger.info(f"• {url['name']} ({url['priority']} priority, richness: {url['data_richness']})")
        
        logger.info("\n📋 INTEGRATION RECOMMENDATIONS:")
        for rec in report['integration_recommendations']:
            logger.info(f"• {rec['title']} ({rec['priority']} priority)")
        
        if report['success_rate'] > 70:
            logger.info("\n🎉 URL validation completed successfully!")
            logger.info("✅ Ready to integrate validated URLs into Beverly Knits AI system")
        else:
            logger.warning("\n⚠️ Some URLs failed validation - please check the report")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ URL collection failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)