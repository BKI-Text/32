#!/usr/bin/env python3
"""
Efab ERP Integration
Beverly Knits AI Supply Chain Planner
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
from .erp_connector import ERPConnector, ERPCredentials, ERPAuthentication, ERPDataSync
from ..core.domain.entities import Material, Supplier, Forecast, BOM
from ..core.domain.value_objects import MaterialId, SupplierId, Money, Quantity
from decimal import Decimal
import json

logger = logging.getLogger(__name__)

class EfabERPIntegration(ERPConnector):
    """Efab ERP Integration for Beverly Knits AI Supply Chain Planner"""
    
    def __init__(self, username: str, password: str):
        """
        Initialize Efab ERP integration
        
        Args:
            username: ERP username
            password: ERP password
        """
        credentials = ERPCredentials(
            username=username,
            password=password,
            base_url='https://efab.bkiapps.com',
            additional_params={
                'remember_me': 'on',
                'login_type': 'user'
            }
        )
        
        super().__init__(credentials)
        self.erp_name = 'Efab ERP'
        self.api_endpoints = {
            'materials': ['/yarn', '/inventory/materials', '/api/materials'],
            'suppliers': ['/suppliers', '/vendors', '/api/suppliers'],
            'inventory': ['/yarn', '/inventory', '/stock'],
            'orders': ['/fabric/so/list', '/orders', '/sales'],
            'bom': ['/bill-of-materials', '/bom', '/api/bom'],
            'forecasts': ['/report/yarn_demand', '/report/expected_yarn', '/forecasts'],
            'yarn_data': ['/yarn'],
            'yarn_reports': ['/report/expected_yarn', '/report/yarn_demand'],
            'fabric_orders': ['/fabric/so/list']
        }
    
    def connect(self) -> bool:
        """
        Connect to Efab ERP system
        
        Returns:
            bool: True if connection successful
        """
        try:
            logger.info(f"🔗 Connecting to {self.erp_name}")
            
            # Authenticate with ERP
            if self.auth.authenticate():
                self.connection_status = 'connected'
                logger.info(f"✅ Successfully connected to {self.erp_name}")
                return True
            else:
                self.connection_status = 'failed'
                logger.error(f"❌ Failed to connect to {self.erp_name}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error connecting to {self.erp_name}: {e}")
            self.connection_status = 'error'
            return False
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test the ERP connection
        
        Returns:
            Dict with connection test results
        """
        test_results = {
            'connection_status': self.connection_status,
            'authenticated': self.auth.is_authenticated,
            'base_url': self.credentials.base_url,
            'username': self.credentials.username,
            'test_timestamp': datetime.now().isoformat(),
            'endpoints_tested': {},
            'success': False
        }
        
        try:
            # Test authentication
            if not self.auth.is_session_valid():
                if not self.connect():
                    test_results['error'] = 'Authentication failed'
                    return test_results
            
            # Test various endpoints
            for endpoint_type, endpoints in self.api_endpoints.items():
                test_results['endpoints_tested'][endpoint_type] = {}
                
                for endpoint in endpoints:
                    try:
                        url = f"{self.credentials.base_url}{endpoint}"
                        response = self.auth.session.get(url, timeout=10)
                        
                        test_results['endpoints_tested'][endpoint_type][endpoint] = {
                            'status_code': response.status_code,
                            'accessible': response.status_code in [200, 401, 403],  # 401/403 means endpoint exists but may need auth
                            'response_size': len(response.content)
                        }
                        
                        if response.status_code == 200:
                            logger.info(f"✅ Endpoint {endpoint} is accessible")
                            test_results['success'] = True
                        else:
                            logger.warning(f"⚠️ Endpoint {endpoint} returned {response.status_code}")
                            
                    except Exception as e:
                        test_results['endpoints_tested'][endpoint_type][endpoint] = {
                            'error': str(e),
                            'accessible': False
                        }
                        logger.debug(f"Endpoint {endpoint} test failed: {e}")
            
            if test_results['success']:
                logger.info(f"✅ {self.erp_name} connection test passed")
            else:
                logger.warning(f"⚠️ {self.erp_name} connection test completed with warnings")
                
        except Exception as e:
            logger.error(f"❌ {self.erp_name} connection test failed: {e}")
            test_results['error'] = str(e)
        
        return test_results
    
    def sync_all_data(self) -> Dict[str, Any]:
        """
        Sync all data from ERP
        
        Returns:
            Dict with sync results
        """
        sync_results = {
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'materials': {},
            'suppliers': {},
            'inventory': {},
            'orders': {},
            'bom': {},
            'forecasts': {},
            'summary': {}
        }
        
        try:
            logger.info(f"🔄 Starting full data sync with {self.erp_name}")
            
            # Ensure connection is valid
            if not self.auth.is_session_valid():
                if not self.connect():
                    sync_results['error'] = 'Connection failed'
                    return sync_results
            
            # Sync materials
            sync_results['materials'] = self.data_sync.sync_materials()
            
            # Sync suppliers
            sync_results['suppliers'] = self.data_sync.sync_suppliers()
            
            # Sync inventory
            sync_results['inventory'] = self.data_sync.sync_inventory()
            
            # Sync orders
            sync_results['orders'] = self.data_sync.sync_orders()
            
            # Sync BOM data
            sync_results['bom'] = self.sync_bom_data()
            
            # Sync forecasts
            sync_results['forecasts'] = self.sync_forecasts_data()
            
            # Sync yarn-specific data
            sync_results['yarn_data'] = self.sync_yarn_data()
            
            # Sync yarn reports
            sync_results['yarn_reports'] = self.sync_yarn_reports()
            
            # Sync fabric orders
            sync_results['fabric_orders'] = self.sync_fabric_orders()
            
            # Calculate summary
            entity_keys = ['materials', 'suppliers', 'inventory', 'orders', 'bom', 'forecasts', 'yarn_data', 'yarn_reports', 'fabric_orders']
            successful_syncs = sum(1 for key in entity_keys 
                                 if sync_results.get(key, {}).get('success', False))
            
            sync_results['summary'] = {
                'total_entities': len(entity_keys),
                'successful_syncs': successful_syncs,
                'success_rate': successful_syncs / len(entity_keys) * 100,
                'materials_count': sync_results['materials'].get('count', 0),
                'suppliers_count': sync_results['suppliers'].get('count', 0),
                'inventory_count': sync_results['inventory'].get('count', 0),
                'orders_count': sync_results['orders'].get('count', 0),
                'yarn_data_count': sync_results['yarn_data'].get('count', 0),
                'yarn_reports_count': sync_results['yarn_reports'].get('count', 0),
                'fabric_orders_count': sync_results['fabric_orders'].get('count', 0)
            }
            
            sync_results['success'] = successful_syncs > 0
            
            if sync_results['success']:
                logger.info(f"✅ Data sync completed: {successful_syncs}/6 entities synced")
            else:
                logger.warning(f"⚠️ Data sync completed with issues: {successful_syncs}/6 entities synced")
                
        except Exception as e:
            logger.error(f"❌ Data sync failed: {e}")
            sync_results['error'] = str(e)
        
        return sync_results
    
    def sync_bom_data(self) -> Dict[str, Any]:
        """Sync Bill of Materials data"""
        if not self.auth.is_session_valid():
            return {'success': False, 'error': 'Authentication failed'}
        
        try:
            for endpoint in self.api_endpoints['bom']:
                try:
                    url = f"{self.credentials.base_url}{endpoint}"
                    response = self.auth.session.get(url)
                    
                    if response.status_code == 200:
                        data = self._parse_response(response)
                        if data:
                            logger.info(f"✅ BOM data synced from {endpoint}")
                            return {
                                'success': True,
                                'data': data,
                                'endpoint': endpoint,
                                'count': len(data) if isinstance(data, list) else 1
                            }
                except Exception as e:
                    logger.debug(f"BOM endpoint {endpoint} failed: {e}")
                    continue
            
            return {'success': False, 'error': 'No valid BOM endpoint found'}
            
        except Exception as e:
            logger.error(f"Error syncing BOM data: {e}")
            return {'success': False, 'error': str(e)}
    
    def sync_forecasts_data(self) -> Dict[str, Any]:
        """Sync forecasts/demand data"""
        if not self.auth.is_session_valid():
            return {'success': False, 'error': 'Authentication failed'}
        
        try:
            for endpoint in self.api_endpoints['forecasts']:
                try:
                    url = f"{self.credentials.base_url}{endpoint}"
                    response = self.auth.session.get(url)
                    
                    if response.status_code == 200:
                        data = self._parse_response(response)
                        if data:
                            logger.info(f"✅ Forecasts data synced from {endpoint}")
                            return {
                                'success': True,
                                'data': data,
                                'endpoint': endpoint,
                                'count': len(data) if isinstance(data, list) else 1
                            }
                except Exception as e:
                    logger.debug(f"Forecasts endpoint {endpoint} failed: {e}")
                    continue
            
            return {'success': False, 'error': 'No valid forecasts endpoint found'}
            
        except Exception as e:
            logger.error(f"Error syncing forecasts data: {e}")
            return {'success': False, 'error': str(e)}
    
    def sync_yarn_data(self) -> Dict[str, Any]:
        """Sync yarn inventory and materials data"""
        if not self.auth.is_session_valid():
            return {'success': False, 'error': 'Authentication failed'}
        
        try:
            url = f"{self.credentials.base_url}/yarn"
            response = self.auth.session.get(url)
            
            if response.status_code == 200:
                data = self._parse_response(response)
                if data:
                    logger.info("✅ Yarn data synced from /yarn endpoint")
                    return {
                        'success': True,
                        'data': data,
                        'endpoint': '/yarn',
                        'count': len(data) if isinstance(data, list) else 1
                    }
            
            return {'success': False, 'error': 'Yarn endpoint not accessible'}
            
        except Exception as e:
            logger.error(f"Error syncing yarn data: {e}")
            return {'success': False, 'error': str(e)}
    
    def sync_yarn_reports(self) -> Dict[str, Any]:
        """Sync yarn demand and expected yarn reports"""
        if not self.auth.is_session_valid():
            return {'success': False, 'error': 'Authentication failed'}
        
        try:
            reports_data = {}
            
            # Get expected yarn report
            try:
                url = f"{self.credentials.base_url}/report/expected_yarn"
                response = self.auth.session.get(url)
                
                if response.status_code == 200:
                    data = self._parse_response(response)
                    if data:
                        reports_data['expected_yarn'] = data
                        logger.info("✅ Expected yarn report synced")
            except Exception as e:
                logger.debug(f"Expected yarn report failed: {e}")
            
            # Get yarn demand report
            try:
                url = f"{self.credentials.base_url}/report/yarn_demand"
                response = self.auth.session.get(url)
                
                if response.status_code == 200:
                    data = self._parse_response(response)
                    if data:
                        reports_data['yarn_demand'] = data
                        logger.info("✅ Yarn demand report synced")
            except Exception as e:
                logger.debug(f"Yarn demand report failed: {e}")
            
            if reports_data:
                return {
                    'success': True,
                    'data': reports_data,
                    'endpoint': '/report/*',
                    'count': len(reports_data)
                }
            else:
                return {'success': False, 'error': 'No yarn reports accessible'}
            
        except Exception as e:
            logger.error(f"Error syncing yarn reports: {e}")
            return {'success': False, 'error': str(e)}
    
    def sync_fabric_orders(self) -> Dict[str, Any]:
        """Sync fabric sales orders"""
        if not self.auth.is_session_valid():
            return {'success': False, 'error': 'Authentication failed'}
        
        try:
            url = f"{self.credentials.base_url}/fabric/so/list"
            response = self.auth.session.get(url)
            
            if response.status_code == 200:
                data = self._parse_response(response)
                if data:
                    logger.info("✅ Fabric orders synced from /fabric/so/list")
                    return {
                        'success': True,
                        'data': data,
                        'endpoint': '/fabric/so/list',
                        'count': len(data) if isinstance(data, list) else 1
                    }
            
            return {'success': False, 'error': 'Fabric orders endpoint not accessible'}
            
        except Exception as e:
            logger.error(f"Error syncing fabric orders: {e}")
            return {'success': False, 'error': str(e)}
    
    def _parse_response(self, response) -> Optional[Any]:
        """Parse HTTP response"""
        return self.data_sync._parse_response(response)
    
    def convert_to_domain_objects(self, sync_results: Dict[str, Any]) -> Dict[str, List]:
        """
        Convert ERP data to domain objects
        
        Args:
            sync_results: Results from sync_all_data()
            
        Returns:
            Dict with domain objects
        """
        domain_objects = {
            'materials': [],
            'suppliers': [],
            'forecasts': [],
            'boms': [],
            'inventory': []
        }
        
        try:
            # Convert materials
            if sync_results['materials'].get('success') and sync_results['materials'].get('data'):
                materials_data = sync_results['materials']['data']
                if isinstance(materials_data, list):
                    for item in materials_data:
                        domain_objects['materials'].append(self._convert_to_material(item))
            
            # Convert suppliers
            if sync_results['suppliers'].get('success') and sync_results['suppliers'].get('data'):
                suppliers_data = sync_results['suppliers']['data']
                if isinstance(suppliers_data, list):
                    for item in suppliers_data:
                        domain_objects['suppliers'].append(self._convert_to_supplier(item))
            
            # Convert forecasts
            if sync_results['forecasts'].get('success') and sync_results['forecasts'].get('data'):
                forecasts_data = sync_results['forecasts']['data']
                if isinstance(forecasts_data, list):
                    for item in forecasts_data:
                        domain_objects['forecasts'].append(self._convert_to_forecast(item))
            
            # Convert BOMs
            if sync_results['bom'].get('success') and sync_results['bom'].get('data'):
                bom_data = sync_results['bom']['data']
                if isinstance(bom_data, list):
                    for item in bom_data:
                        domain_objects['boms'].append(self._convert_to_bom(item))
            
            logger.info(f"✅ Converted to domain objects: {len(domain_objects['materials'])} materials, "
                       f"{len(domain_objects['suppliers'])} suppliers, {len(domain_objects['forecasts'])} forecasts")
            
        except Exception as e:
            logger.error(f"Error converting to domain objects: {e}")
        
        return domain_objects
    
    def _convert_to_material(self, data: Dict) -> Optional[Material]:
        """Convert ERP data to Material domain object"""
        try:
            # Extract material information from ERP data
            material_id = data.get('id', data.get('material_id', data.get('sku', 'UNKNOWN')))
            name = data.get('name', data.get('description', data.get('material_name', 'Unknown Material')))
            cost = data.get('cost', data.get('unit_cost', data.get('price', 0)))
            
            return Material(
                id=MaterialId(value=str(material_id)),
                name=name,
                type=data.get('type', 'RAW_MATERIAL'),
                cost=Money(amount=Decimal(str(cost)), currency='USD'),
                is_critical=data.get('is_critical', False)
            )
        except Exception as e:
            logger.error(f"Error converting material: {e}")
            return None
    
    def _convert_to_supplier(self, data: Dict) -> Optional[Supplier]:
        """Convert ERP data to Supplier domain object"""
        try:
            supplier_id = data.get('id', data.get('supplier_id', data.get('vendor_id', 'UNKNOWN')))
            name = data.get('name', data.get('supplier_name', data.get('vendor_name', 'Unknown Supplier')))
            
            return Supplier(
                id=SupplierId(value=str(supplier_id)),
                name=name,
                contact_info=data.get('contact_info', {}),
                reliability_score=data.get('reliability_score', 0.8),
                lead_time_days=data.get('lead_time_days', 30)
            )
        except Exception as e:
            logger.error(f"Error converting supplier: {e}")
            return None
    
    def _convert_to_forecast(self, data: Dict) -> Optional[Forecast]:
        """Convert ERP data to Forecast domain object"""
        try:
            material_id = data.get('material_id', data.get('sku', 'UNKNOWN'))
            demand = data.get('demand', data.get('quantity', data.get('forecast_quantity', 0)))
            
            return Forecast(
                material_id=MaterialId(value=str(material_id)),
                period=data.get('period', datetime.now().strftime('%Y-%m')),
                demand=Quantity(amount=Decimal(str(demand)), unit='units'),
                confidence=data.get('confidence', 0.8),
                source='ERP_EFAB'
            )
        except Exception as e:
            logger.error(f"Error converting forecast: {e}")
            return None
    
    def _convert_to_bom(self, data: Dict) -> Optional[BOM]:
        """Convert ERP data to BOM domain object"""
        try:
            # BOM conversion depends on ERP structure
            # This is a placeholder implementation
            return BOM(
                finished_good_id=MaterialId(value=str(data.get('finished_good_id', 'UNKNOWN'))),
                components={
                    MaterialId(value=str(data.get('component_id', 'UNKNOWN'))): 
                    Quantity(amount=Decimal(str(data.get('quantity', 1))), unit='units')
                }
            )
        except Exception as e:
            logger.error(f"Error converting BOM: {e}")
            return None
    
    def push_recommendations(self, recommendations: List[Dict]) -> Dict[str, Any]:
        """
        Push procurement recommendations back to ERP
        
        Args:
            recommendations: List of procurement recommendations
            
        Returns:
            Dict with push results
        """
        if not self.auth.is_session_valid():
            return {'success': False, 'error': 'Authentication failed'}
        
        try:
            # Try to find procurement/purchase order endpoints
            po_endpoints = [
                '/api/purchase_orders',
                '/api/po',
                '/purchase_orders',
                '/procurement'
            ]
            
            results = {
                'success': False,
                'pushed_count': 0,
                'failed_count': 0,
                'details': []
            }
            
            for recommendation in recommendations:
                try:
                    # Convert recommendation to ERP format
                    erp_data = self._convert_recommendation_to_erp(recommendation)
                    
                    # Try to post to ERP
                    for endpoint in po_endpoints:
                        try:
                            url = f"{self.credentials.base_url}{endpoint}"
                            response = self.auth.session.post(url, json=erp_data)
                            
                            if response.status_code in [200, 201]:
                                results['pushed_count'] += 1
                                results['details'].append({
                                    'recommendation_id': recommendation.get('id', 'unknown'),
                                    'status': 'success',
                                    'endpoint': endpoint
                                })
                                break
                                
                        except Exception as e:
                            logger.debug(f"Failed to push to {endpoint}: {e}")
                            continue
                    else:
                        results['failed_count'] += 1
                        results['details'].append({
                            'recommendation_id': recommendation.get('id', 'unknown'),
                            'status': 'failed',
                            'error': 'No valid endpoint found'
                        })
                        
                except Exception as e:
                    results['failed_count'] += 1
                    results['details'].append({
                        'recommendation_id': recommendation.get('id', 'unknown'),
                        'status': 'failed',
                        'error': str(e)
                    })
            
            results['success'] = results['pushed_count'] > 0
            
            logger.info(f"✅ Push completed: {results['pushed_count']} successful, {results['failed_count']} failed")
            
            return results
            
        except Exception as e:
            logger.error(f"Error pushing recommendations: {e}")
            return {'success': False, 'error': str(e)}
    
    def _convert_recommendation_to_erp(self, recommendation: Dict) -> Dict:
        """Convert procurement recommendation to ERP format"""
        return {
            'material_id': recommendation.get('material_id'),
            'supplier_id': recommendation.get('supplier_id'),
            'quantity': recommendation.get('quantity'),
            'unit_cost': recommendation.get('unit_cost'),
            'total_cost': recommendation.get('total_cost'),
            'requested_date': recommendation.get('required_date'),
            'priority': recommendation.get('priority', 'normal'),
            'notes': f"Generated by Beverly Knits AI Supply Chain Planner - {recommendation.get('reasoning', '')}"
        }
    
    def get_sync_status(self) -> Dict[str, Any]:
        """Get current sync status"""
        return {
            'connection_status': self.connection_status,
            'authenticated': self.auth.is_authenticated,
            'last_sync': getattr(self.data_sync, 'last_sync_time', None),
            'sync_status': getattr(self.data_sync, 'sync_status', {}),
            'erp_name': self.erp_name,
            'base_url': self.credentials.base_url
        }