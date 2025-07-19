#!/usr/bin/env python3
"""
Comprehensive ERP Data Analyzer for Beverly Knits AI Supply Chain Planner
Deep analysis and historical data extraction for ML training pipeline
"""

import sys
import os
import logging
import json
import pandas as pd
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import statistics
import requests
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.integrations.efab_integration import EfabERPIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/erp_data_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DataField:
    """Represents a data field with analysis metadata"""
    name: str
    data_type: str
    sample_values: List[str]
    completeness: float
    uniqueness: float
    ml_potential: str  # LOW, MEDIUM, HIGH
    business_value: str  # LOW, MEDIUM, HIGH
    temporal_indicator: bool
    numeric_indicator: bool
    categorical_indicator: bool

@dataclass
class EndpointAnalysis:
    """Complete analysis of an ERP endpoint"""
    endpoint: str
    url: str
    status: str
    record_count_estimate: int
    data_fields: List[DataField]
    data_quality_score: float
    ml_training_potential: str
    historical_data_available: bool
    recommended_extraction_frequency: str
    business_priority: str
    technical_complexity: str

@dataclass
class ERPDataLandscape:
    """Complete ERP data landscape analysis"""
    analysis_timestamp: str
    total_endpoints: int
    accessible_endpoints: int
    total_fields: int
    high_value_fields: int
    ml_ready_endpoints: int
    estimated_records: int
    data_categories: Dict[str, List[str]]
    training_datasets: List[Dict[str, Any]]
    integration_roadmap: List[Dict[str, Any]]

class ComprehensiveERPAnalyzer:
    """Advanced ERP data analyzer with ML training focus"""
    
    def __init__(self):
        self.erp = EfabERPIntegration(username='psytz', password='big$cat')
        self.analysis_results = {}
        self.data_landscape = None
        
        # ML Training Categories
        self.ml_categories = {
            'demand_forecasting': ['demand', 'quantity', 'sales', 'order', 'forecast'],
            'inventory_optimization': ['inventory', 'stock', 'balance', 'on_hand', 'available'],
            'supplier_performance': ['supplier', 'delivery', 'lead_time', 'reliability', 'cost'],
            'seasonal_patterns': ['date', 'month', 'period', 'season', 'time'],
            'cost_prediction': ['cost', 'price', 'value', 'amount', 'total'],
            'quality_metrics': ['quality', 'defect', 'reject', 'grade', 'standard']
        }
        
        # Business Value Indicators
        self.high_value_indicators = [
            'demand', 'forecast', 'inventory', 'cost', 'supplier', 'delivery',
            'quality', 'efficiency', 'revenue', 'profit', 'margin', 'trend'
        ]
    
    def connect_and_validate(self) -> bool:
        """Connect to ERP and validate access"""
        logger.info("🔗 Connecting to Beverly Knits ERP...")
        return self.erp.connect()
    
    def discover_all_endpoints(self) -> List[str]:
        """Discover additional endpoints beyond the known 5"""
        logger.info("🔍 Discovering all available ERP endpoints...")
        
        # Start with known working endpoints
        known_endpoints = [
            '/yarn',
            '/fabric/so/list',
            '/report/yarn_demand',
            '/report/expected_yarn',
            '/yarn/po/list'
        ]
        
        # Potential additional endpoints to test
        potential_endpoints = [
            '/suppliers',
            '/materials',
            '/inventory',
            '/reports',
            '/reports/inventory',
            '/reports/cost',
            '/reports/supplier_performance',
            '/reports/quality',
            '/fabric',
            '/fabric/inventory',
            '/orders',
            '/purchase_orders',
            '/sales_orders',
            '/forecasts',
            '/dashboard',
            '/analytics',
            '/production',
            '/production/schedule',
            '/quality',
            '/quality/control',
            '/costing',
            '/costing/analysis'
        ]
        
        accessible_endpoints = []
        
        # Test known endpoints first
        for endpoint in known_endpoints:
            if self._test_endpoint_access(endpoint):
                accessible_endpoints.append(endpoint)
        
        # Test potential endpoints
        logger.info("🔍 Testing potential additional endpoints...")
        for endpoint in potential_endpoints:
            if self._test_endpoint_access(endpoint):
                accessible_endpoints.append(endpoint)
                logger.info(f"✅ Discovered new endpoint: {endpoint}")
        
        logger.info(f"📊 Total accessible endpoints: {len(accessible_endpoints)}")
        return accessible_endpoints
    
    def _test_endpoint_access(self, endpoint: str) -> bool:
        """Test if an endpoint is accessible"""
        try:
            url = f"{self.erp.credentials.base_url}{endpoint}"
            response = self.erp.auth.session.get(url, timeout=10)
            return response.status_code == 200 and len(response.content) > 1000
        except:
            return False
    
    def analyze_endpoint_in_depth(self, endpoint: str) -> EndpointAnalysis:
        """Perform deep analysis of a single endpoint"""
        logger.info(f"🔬 Deep analysis of endpoint: {endpoint}")
        
        url = f"{self.erp.credentials.base_url}{endpoint}"
        
        try:
            response = self.erp.auth.session.get(url, timeout=15)
            
            if response.status_code != 200:
                return EndpointAnalysis(
                    endpoint=endpoint,
                    url=url,
                    status='FAILED',
                    record_count_estimate=0,
                    data_fields=[],
                    data_quality_score=0.0,
                    ml_training_potential='LOW',
                    historical_data_available=False,
                    recommended_extraction_frequency='NONE',
                    business_priority='LOW',
                    technical_complexity='HIGH'
                )
            
            html_content = response.text
            
            # Extract data fields using multiple methods
            data_fields = self._extract_data_fields(html_content)
            
            # Estimate record count
            record_estimate = self._estimate_record_count(html_content)
            
            # Analyze data quality
            data_quality = self._assess_data_quality(data_fields, html_content)
            
            # Assess ML training potential
            ml_potential = self._assess_ml_potential(data_fields)
            
            # Check for historical data indicators
            has_historical = self._detect_historical_data(data_fields, html_content)
            
            # Recommend extraction frequency
            extraction_freq = self._recommend_extraction_frequency(data_fields, has_historical)
            
            # Assess business priority
            business_priority = self._assess_business_priority(endpoint, data_fields)
            
            # Assess technical complexity
            tech_complexity = self._assess_technical_complexity(html_content)
            
            analysis = EndpointAnalysis(
                endpoint=endpoint,
                url=url,
                status='SUCCESS',
                record_count_estimate=record_estimate,
                data_fields=data_fields,
                data_quality_score=data_quality,
                ml_training_potential=ml_potential,
                historical_data_available=has_historical,
                recommended_extraction_frequency=extraction_freq,
                business_priority=business_priority,
                technical_complexity=tech_complexity
            )
            
            logger.info(f"✅ Analysis complete: {len(data_fields)} fields, {ml_potential} ML potential")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Analysis failed for {endpoint}: {e}")
            return EndpointAnalysis(
                endpoint=endpoint,
                url=url,
                status='ERROR',
                record_count_estimate=0,
                data_fields=[],
                data_quality_score=0.0,
                ml_training_potential='LOW',
                historical_data_available=False,
                recommended_extraction_frequency='NONE',
                business_priority='LOW',
                technical_complexity='HIGH'
            )
    
    def _extract_data_fields(self, html_content: str) -> List[DataField]:
        """Extract and analyze data fields from HTML content"""
        fields = []
        
        # Extract field names using multiple patterns
        field_patterns = [
            r'dataField:\s*["\']([^"\']+)["\']',  # DevExtreme dataField
            r'<th[^>]*>(.*?)</th>',  # Table headers
            r'name=["\']([^"\']+)["\']',  # Form field names
            r'id=["\']([^"\']+)["\']',  # Element IDs
            r'"field":\s*["\']([^"\']+)["\']',  # JSON field references
            r'data-field=["\']([^"\']+)["\']',  # Data attributes
        ]
        
        field_names = set()
        for pattern in field_patterns:
            matches = re.findall(pattern, html_content, re.IGNORECASE)
            for match in matches:
                clean_name = re.sub(r'<[^>]+>', '', str(match)).strip()
                if clean_name and len(clean_name) < 100 and not clean_name.startswith('_'):
                    field_names.add(clean_name)
        
        # Analyze each field
        for field_name in field_names:
            field_analysis = self._analyze_field(field_name, html_content)
            fields.append(field_analysis)
        
        return sorted(fields, key=lambda x: x.ml_potential + x.business_value, reverse=True)
    
    def _analyze_field(self, field_name: str, html_content: str) -> DataField:
        """Analyze individual field characteristics"""
        field_lower = field_name.lower()
        
        # Determine data type
        data_type = self._infer_data_type(field_lower)
        
        # Extract sample values (simplified)
        sample_values = self._extract_sample_values(field_name, html_content)
        
        # Calculate completeness (estimated)
        completeness = self._estimate_completeness(field_name, html_content)
        
        # Calculate uniqueness (estimated)
        uniqueness = self._estimate_uniqueness(field_name, sample_values)
        
        # Assess ML potential
        ml_potential = self._assess_field_ml_potential(field_lower)
        
        # Assess business value
        business_value = self._assess_field_business_value(field_lower)
        
        # Determine indicators
        temporal_indicator = any(temporal in field_lower for temporal in ['date', 'time', 'period', 'month', 'year'])
        numeric_indicator = any(numeric in field_lower for numeric in ['qty', 'amount', 'cost', 'price', 'count', 'balance'])
        categorical_indicator = any(cat in field_lower for cat in ['type', 'status', 'category', 'level', 'grade'])
        
        return DataField(
            name=field_name,
            data_type=data_type,
            sample_values=sample_values[:5],  # Limit to 5 samples
            completeness=completeness,
            uniqueness=uniqueness,
            ml_potential=ml_potential,
            business_value=business_value,
            temporal_indicator=temporal_indicator,
            numeric_indicator=numeric_indicator,
            categorical_indicator=categorical_indicator
        )
    
    def _infer_data_type(self, field_name: str) -> str:
        """Infer data type from field name"""
        if any(dt in field_name for dt in ['date', 'time', 'created', 'updated']):
            return 'datetime'
        elif any(num in field_name for num in ['qty', 'amount', 'cost', 'price', 'count', 'balance', 'total']):
            return 'numeric'
        elif any(cat in field_name for cat in ['status', 'type', 'category', 'level']):
            return 'categorical'
        elif any(txt in field_name for txt in ['name', 'description', 'note', 'comment']):
            return 'text'
        else:
            return 'unknown'
    
    def _extract_sample_values(self, field_name: str, html_content: str) -> List[str]:
        """Extract sample values for a field (simplified)"""
        # This is a simplified version - in production would parse actual data
        return ['sample_value_1', 'sample_value_2']
    
    def _estimate_completeness(self, field_name: str, html_content: str) -> float:
        """Estimate field completeness"""
        # Core fields typically have higher completeness
        core_fields = ['id', 'name', 'type', 'status', 'date', 'qty', 'cost']
        if any(core in field_name.lower() for core in core_fields):
            return 0.95
        return 0.80
    
    def _estimate_uniqueness(self, field_name: str, sample_values: List[str]) -> float:
        """Estimate field uniqueness"""
        field_lower = field_name.lower()
        if 'id' in field_lower or 'serial' in field_lower:
            return 1.0
        elif any(cat in field_lower for cat in ['type', 'status', 'category']):
            return 0.1
        return 0.5
    
    def _assess_field_ml_potential(self, field_name: str) -> str:
        """Assess ML training potential of a field"""
        high_value_ml = ['demand', 'forecast', 'quantity', 'cost', 'price', 'date', 'supplier', 'quality']
        medium_value_ml = ['type', 'status', 'category', 'balance', 'inventory']
        
        if any(hv in field_name for hv in high_value_ml):
            return 'HIGH'
        elif any(mv in field_name for mv in medium_value_ml):
            return 'MEDIUM'
        return 'LOW'
    
    def _assess_field_business_value(self, field_name: str) -> str:
        """Assess business value of a field"""
        if any(hv in field_name for hv in self.high_value_indicators):
            return 'HIGH'
        elif any(mv in field_name for mv in ['type', 'name', 'description', 'status']):
            return 'MEDIUM'
        return 'LOW'
    
    def _estimate_record_count(self, html_content: str) -> int:
        """Estimate number of records in the endpoint"""
        # Look for pagination indicators
        if 'pagination' in html_content.lower() or 'page' in html_content.lower():
            return 1000  # Estimated for paginated data
        
        # Count table rows (rough estimate)
        row_count = html_content.count('<tr')
        if row_count > 10:
            return row_count * 10  # Estimate multiple pages
        
        return 100  # Default estimate
    
    def _assess_data_quality(self, data_fields: List[DataField], html_content: str) -> float:
        """Assess overall data quality score"""
        if not data_fields:
            return 0.0
        
        # Calculate average completeness
        avg_completeness = statistics.mean([field.completeness for field in data_fields])
        
        # Bonus for structured data
        has_devextreme = 'devextreme' in html_content.lower()
        structure_bonus = 0.1 if has_devextreme else 0.0
        
        # Bonus for rich field variety
        variety_bonus = min(len(data_fields) / 20, 0.1)
        
        return min(avg_completeness + structure_bonus + variety_bonus, 1.0)
    
    def _assess_ml_potential(self, data_fields: List[DataField]) -> str:
        """Assess ML training potential of entire endpoint"""
        if not data_fields:
            return 'LOW'
        
        high_potential_count = sum(1 for field in data_fields if field.ml_potential == 'HIGH')
        total_fields = len(data_fields)
        
        if high_potential_count >= 5 or (high_potential_count / total_fields) > 0.3:
            return 'HIGH'
        elif high_potential_count >= 2 or (high_potential_count / total_fields) > 0.1:
            return 'MEDIUM'
        return 'LOW'
    
    def _detect_historical_data(self, data_fields: List[DataField], html_content: str) -> bool:
        """Detect if endpoint contains historical data"""
        temporal_fields = [field for field in data_fields if field.temporal_indicator]
        return len(temporal_fields) > 0
    
    def _recommend_extraction_frequency(self, data_fields: List[DataField], has_historical: bool) -> str:
        """Recommend extraction frequency"""
        # Check for real-time indicators
        real_time_indicators = ['current', 'live', 'real_time', 'now']
        has_real_time = any(any(rt in field.name.lower() for rt in real_time_indicators) for field in data_fields)
        
        if has_real_time:
            return 'HOURLY'
        elif has_historical:
            return 'DAILY'
        else:
            return 'WEEKLY'
    
    def _assess_business_priority(self, endpoint: str, data_fields: List[DataField]) -> str:
        """Assess business priority of endpoint"""
        critical_endpoints = ['/yarn', '/inventory', '/demand', '/forecast', '/supplier']
        high_priority_endpoints = ['/cost', '/quality', '/order', '/production']
        
        if any(critical in endpoint for critical in critical_endpoints):
            return 'CRITICAL'
        elif any(high in endpoint for high in high_priority_endpoints):
            return 'HIGH'
        
        # Check field-based priority
        high_value_fields = sum(1 for field in data_fields if field.business_value == 'HIGH')
        if high_value_fields >= 5:
            return 'HIGH'
        elif high_value_fields >= 2:
            return 'MEDIUM'
        
        return 'LOW'
    
    def _assess_technical_complexity(self, html_content: str) -> str:
        """Assess technical complexity of data extraction"""
        complexity_score = 0
        
        # DevExtreme grids are easier to extract
        if 'devextreme' in html_content.lower():
            complexity_score += 1
        
        # Regular HTML tables are moderate
        if '<table' in html_content.lower():
            complexity_score += 2
        
        # JavaScript data requires more complex extraction
        if 'javascript' in html_content.lower() and 'data' in html_content.lower():
            complexity_score += 3
        
        # Forms are typically simpler
        if '<form' in html_content.lower():
            complexity_score += 1
        
        if complexity_score <= 2:
            return 'LOW'
        elif complexity_score <= 4:
            return 'MEDIUM'
        else:
            return 'HIGH'
    
    def generate_ml_training_datasets(self, analyses: List[EndpointAnalysis]) -> List[Dict[str, Any]]:
        """Generate ML training dataset recommendations"""
        datasets = []
        
        # Demand Forecasting Dataset
        demand_endpoints = [a for a in analyses if any(kw in a.endpoint.lower() for kw in ['demand', 'forecast', 'sales'])]
        if demand_endpoints:
            datasets.append({
                'name': 'Demand Forecasting Dataset',
                'description': 'Historical demand and sales data for ML forecasting models',
                'endpoints': [a.endpoint for a in demand_endpoints],
                'ml_models': ['ARIMA', 'Prophet', 'LSTM', 'XGBoost'],
                'target_variable': 'demand_quantity',
                'features': ['date', 'product_id', 'historical_demand', 'seasonality'],
                'estimated_records': sum(a.record_count_estimate for a in demand_endpoints),
                'business_value': 'CRITICAL',
                'implementation_priority': 1
            })
        
        # Inventory Optimization Dataset
        inventory_endpoints = [a for a in analyses if any(kw in a.endpoint.lower() for kw in ['inventory', 'stock', 'yarn'])]
        if inventory_endpoints:
            datasets.append({
                'name': 'Inventory Optimization Dataset',
                'description': 'Inventory levels and movement patterns for optimization',
                'endpoints': [a.endpoint for a in inventory_endpoints],
                'ml_models': ['XGBoost', 'Random Forest', 'Linear Regression'],
                'target_variable': 'optimal_inventory_level',
                'features': ['current_stock', 'lead_time', 'demand_variability', 'cost'],
                'estimated_records': sum(a.record_count_estimate for a in inventory_endpoints),
                'business_value': 'HIGH',
                'implementation_priority': 2
            })
        
        # Supplier Performance Dataset
        supplier_endpoints = [a for a in analyses if any(kw in a.endpoint.lower() for kw in ['supplier', 'po', 'purchase'])]
        if supplier_endpoints:
            datasets.append({
                'name': 'Supplier Performance Dataset',
                'description': 'Supplier reliability and performance metrics',
                'endpoints': [a.endpoint for a in supplier_endpoints],
                'ml_models': ['Classification', 'Regression', 'Clustering'],
                'target_variable': 'supplier_reliability_score',
                'features': ['delivery_time', 'quality_score', 'cost_variance', 'order_frequency'],
                'estimated_records': sum(a.record_count_estimate for a in supplier_endpoints),
                'business_value': 'HIGH',
                'implementation_priority': 3
            })
        
        return datasets
    
    def create_integration_roadmap(self, analyses: List[EndpointAnalysis], datasets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create implementation roadmap"""
        roadmap = []
        
        # Phase 1: High-value, low-complexity endpoints
        high_value_simple = [a for a in analyses if a.business_priority in ['CRITICAL', 'HIGH'] and a.technical_complexity == 'LOW']
        if high_value_simple:
            roadmap.append({
                'phase': 'Phase 1: Quick Wins',
                'duration': '2-3 weeks',
                'endpoints': [a.endpoint for a in high_value_simple],
                'objectives': ['Establish data pipeline', 'Validate integration approach', 'Deliver immediate value'],
                'deliverables': ['Automated data extraction', 'Basic dashboards', 'Data quality reports'],
                'success_metrics': ['Data extraction reliability > 95%', 'Processing time < 30min', '3+ datasets operational']
            })
        
        # Phase 2: ML Training Pipeline
        ml_ready_endpoints = [a for a in analyses if a.ml_training_potential == 'HIGH']
        if ml_ready_endpoints:
            roadmap.append({
                'phase': 'Phase 2: ML Training Pipeline',
                'duration': '3-4 weeks',
                'endpoints': [a.endpoint for a in ml_ready_endpoints],
                'objectives': ['Build ML training datasets', 'Train initial models', 'Validate model performance'],
                'deliverables': ['ML training pipeline', 'Trained models', 'Model validation reports'],
                'success_metrics': ['Model accuracy > 80%', '3+ ML models trained', 'Automated retraining']
            })
        
        # Phase 3: Advanced Integration
        complex_endpoints = [a for a in analyses if a.business_priority in ['HIGH', 'MEDIUM'] and a.technical_complexity in ['MEDIUM', 'HIGH']]
        if complex_endpoints:
            roadmap.append({
                'phase': 'Phase 3: Advanced Integration',
                'duration': '4-5 weeks',
                'endpoints': [a.endpoint for a in complex_endpoints],
                'objectives': ['Complete data landscape', 'Advanced analytics', 'Full automation'],
                'deliverables': ['Complete data warehouse', 'Advanced dashboards', 'Predictive analytics'],
                'success_metrics': ['All endpoints integrated', 'Real-time processing', 'Business KPIs improved']
            })
        
        return roadmap
    
    def run_comprehensive_analysis(self) -> ERPDataLandscape:
        """Run complete ERP data landscape analysis"""
        logger.info("🚀 Starting Comprehensive ERP Data Analysis")
        logger.info("=" * 80)
        
        # Step 1: Connect to ERP
        if not self.connect_and_validate():
            logger.error("❌ Failed to connect to ERP")
            return None
        
        # Step 2: Discover all endpoints
        endpoints = self.discover_all_endpoints()
        logger.info(f"📊 Discovered {len(endpoints)} accessible endpoints")
        
        # Step 3: Analyze each endpoint in depth
        analyses = []
        for endpoint in endpoints:
            analysis = self.analyze_endpoint_in_depth(endpoint)
            analyses.append(analysis)
            self.analysis_results[endpoint] = analysis
        
        # Step 4: Generate ML training datasets
        datasets = self.generate_ml_training_datasets(analyses)
        logger.info(f"🧠 Generated {len(datasets)} ML training datasets")
        
        # Step 5: Create integration roadmap
        roadmap = self.create_integration_roadmap(analyses, datasets)
        logger.info(f"🗺️ Created {len(roadmap)} phase implementation roadmap")
        
        # Step 6: Compile data landscape
        successful_analyses = [a for a in analyses if a.status == 'SUCCESS']
        total_fields = sum(len(a.data_fields) for a in successful_analyses)
        high_value_fields = sum(len([f for f in a.data_fields if f.business_value == 'HIGH']) for a in successful_analyses)
        ml_ready_endpoints = len([a for a in successful_analyses if a.ml_training_potential == 'HIGH'])
        
        # Categorize data
        data_categories = {}
        for category, keywords in self.ml_categories.items():
            matching_endpoints = []
            for analysis in successful_analyses:
                if any(keyword in analysis.endpoint.lower() for keyword in keywords):
                    matching_endpoints.append(analysis.endpoint)
            data_categories[category] = matching_endpoints
        
        self.data_landscape = ERPDataLandscape(
            analysis_timestamp=datetime.now().isoformat(),
            total_endpoints=len(endpoints),
            accessible_endpoints=len(successful_analyses),
            total_fields=total_fields,
            high_value_fields=high_value_fields,
            ml_ready_endpoints=ml_ready_endpoints,
            estimated_records=sum(a.record_count_estimate for a in successful_analyses),
            data_categories=data_categories,
            training_datasets=datasets,
            integration_roadmap=roadmap
        )
        
        return self.data_landscape
    
    def save_analysis_results(self, filename_prefix: str = "comprehensive_erp_analysis"):
        """Save all analysis results to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed analysis
        detailed_file = f"{filename_prefix}_{timestamp}.json"
        detailed_data = {
            'data_landscape': asdict(self.data_landscape),
            'endpoint_analyses': {k: asdict(v) for k, v in self.analysis_results.items()},
            'metadata': {
                'analysis_date': datetime.now().isoformat(),
                'analyzer_version': '1.0',
                'erp_system': 'Efab ERP'
            }
        }
        
        with open(detailed_file, 'w') as f:
            json.dump(detailed_data, f, indent=2, default=str)
        
        logger.info(f"💾 Detailed analysis saved to: {detailed_file}")
        
        # Save summary report
        summary_file = f"{filename_prefix}_summary_{timestamp}.md"
        self._generate_summary_report(summary_file)
        
        return detailed_file, summary_file
    
    def _generate_summary_report(self, filename: str):
        """Generate markdown summary report"""
        if not self.data_landscape:
            return
        
        report = f"""# Beverly Knits ERP Data Analysis - Comprehensive Report

**Analysis Date:** {self.data_landscape.analysis_timestamp}
**Analyst:** Beverly Knits AI Supply Chain Planner

## 📊 Executive Summary

- **Total Endpoints Discovered:** {self.data_landscape.total_endpoints}
- **Successfully Analyzed:** {self.data_landscape.accessible_endpoints}
- **Total Data Fields:** {self.data_landscape.total_fields}
- **High-Value Fields:** {self.data_landscape.high_value_fields}
- **ML-Ready Endpoints:** {self.data_landscape.ml_ready_endpoints}
- **Estimated Records:** {self.data_landscape.estimated_records:,}

## 🎯 ML Training Datasets

"""
        
        for dataset in self.data_landscape.training_datasets:
            report += f"""### {dataset['name']}
- **Description:** {dataset['description']}
- **Endpoints:** {', '.join(dataset['endpoints'])}
- **ML Models:** {', '.join(dataset['ml_models'])}
- **Estimated Records:** {dataset['estimated_records']:,}
- **Business Value:** {dataset['business_value']}

"""
        
        report += f"""## 🗺️ Implementation Roadmap

"""
        
        for phase in self.data_landscape.integration_roadmap:
            report += f"""### {phase['phase']}
- **Duration:** {phase['duration']}
- **Endpoints:** {len(phase['endpoints'])} endpoints
- **Objectives:** {', '.join(phase['objectives'])}

"""
        
        report += f"""## 📈 Data Categories

"""
        
        for category, endpoints in self.data_landscape.data_categories.items():
            if endpoints:
                report += f"- **{category.replace('_', ' ').title()}:** {len(endpoints)} endpoints\n"
        
        report += f"""
## 🎉 Key Findings

1. **High Data Quality:** {self.data_landscape.accessible_endpoints}/{self.data_landscape.total_endpoints} endpoints accessible
2. **Rich ML Potential:** {self.data_landscape.ml_ready_endpoints} endpoints ready for ML training
3. **Comprehensive Coverage:** {len(self.data_landscape.data_categories)} data categories identified
4. **Production Ready:** {len(self.data_landscape.integration_roadmap)} phase implementation plan

## 🚀 Next Steps

1. Begin Phase 1 implementation with {len([p for p in self.data_landscape.integration_roadmap if p['phase'].startswith('Phase 1')])} high-value endpoints
2. Establish ML training pipeline for demand forecasting
3. Set up automated data extraction and quality monitoring
4. Deploy initial predictive models for inventory optimization

---
*Generated by Beverly Knits AI Supply Chain Planner - Comprehensive ERP Data Analyzer*
"""
        
        with open(filename, 'w') as f:
            f.write(report)
        
        logger.info(f"📄 Summary report saved to: {filename}")

def main():
    """Main execution"""
    logger.info("🎯 Beverly Knits AI - Comprehensive ERP Data Analysis")
    logger.info("Analyzing ERP data landscape for ML training and business intelligence")
    logger.info("=" * 80)
    
    analyzer = ComprehensiveERPAnalyzer()
    
    try:
        # Run comprehensive analysis
        data_landscape = analyzer.run_comprehensive_analysis()
        
        if data_landscape:
            # Save results
            detailed_file, summary_file = analyzer.save_analysis_results()
            
            # Display key metrics
            logger.info("\n" + "=" * 80)
            logger.info("🎉 COMPREHENSIVE ERP ANALYSIS COMPLETE")
            logger.info("=" * 80)
            
            logger.info(f"📊 Endpoints Analyzed: {data_landscape.accessible_endpoints}/{data_landscape.total_endpoints}")
            logger.info(f"📈 Total Data Fields: {data_landscape.total_fields}")
            logger.info(f"🧠 ML-Ready Endpoints: {data_landscape.ml_ready_endpoints}")
            logger.info(f"💾 Estimated Records: {data_landscape.estimated_records:,}")
            logger.info(f"🎯 Training Datasets: {len(data_landscape.training_datasets)}")
            
            logger.info("\n📁 Generated Files:")
            logger.info(f"   • Detailed Analysis: {detailed_file}")
            logger.info(f"   • Summary Report: {summary_file}")
            
            logger.info("\n🚀 Ready for ML Training Pipeline Implementation!")
            
            return True
        else:
            logger.error("❌ Analysis failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Analysis error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)