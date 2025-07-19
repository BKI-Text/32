#!/usr/bin/env python3
"""
ERP Manager
Beverly Knits AI Supply Chain Planner
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import os
from pathlib import Path

from .efab_integration import EfabERPIntegration

logger = logging.getLogger(__name__)

class ERPManager:
    """Manages ERP integrations for Beverly Knits AI Supply Chain Planner"""
    
    def __init__(self, config_path: str = "config/erp_config.json"):
        """
        Initialize ERP Manager
        
        Args:
            config_path: Path to ERP configuration file
        """
        self.config_path = config_path
        self.erp_integration = None
        self.config = {}
        self.last_sync_time = None
        self.sync_status = {}
        
        # Load configuration
        self._load_config()
    
    def _load_config(self):
        """Load ERP configuration"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
                logger.info(f"✅ ERP configuration loaded from {self.config_path}")
            else:
                # Create default config
                self.config = {
                    'erp_type': 'efab',
                    'base_url': 'https://efab.bkiapps.com',
                    'username': '',
                    'password': '',
                    'auto_sync': True,
                    'sync_interval_hours': 24,
                    'sync_on_startup': True
                }
                self._save_config()
                logger.info(f"✅ Default ERP configuration created at {self.config_path}")
        except Exception as e:
            logger.error(f"❌ Error loading ERP configuration: {e}")
            self.config = {}
    
    def _save_config(self):
        """Save ERP configuration"""
        try:
            Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"✅ ERP configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"❌ Error saving ERP configuration: {e}")
    
    def set_credentials(self, username: str, password: str):
        """
        Set ERP credentials
        
        Args:
            username: ERP username
            password: ERP password
        """
        self.config['username'] = username
        self.config['password'] = password
        self._save_config()
        logger.info("✅ ERP credentials updated")
    
    def connect_erp(self) -> bool:
        """
        Connect to ERP system
        
        Returns:
            bool: True if connection successful
        """
        try:
            if not self.config.get('username') or not self.config.get('password'):
                logger.error("❌ ERP credentials not configured")
                return False
            
            # Initialize ERP integration
            if self.config['erp_type'] == 'efab':
                self.erp_integration = EfabERPIntegration(
                    username=self.config['username'],
                    password=self.config['password']
                )
            else:
                logger.error(f"❌ Unsupported ERP type: {self.config['erp_type']}")
                return False
            
            # Connect to ERP
            if self.erp_integration.connect():
                logger.info("✅ ERP connection established")
                return True
            else:
                logger.error("❌ Failed to connect to ERP")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error connecting to ERP: {e}")
            return False
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test ERP connection
        
        Returns:
            Dict with test results
        """
        if not self.erp_integration:
            if not self.connect_erp():
                return {'success': False, 'error': 'Failed to connect to ERP'}
        
        try:
            return self.erp_integration.test_connection()
        except Exception as e:
            logger.error(f"❌ Error testing ERP connection: {e}")
            return {'success': False, 'error': str(e)}
    
    def sync_data(self) -> Dict[str, Any]:
        """
        Sync data from ERP
        
        Returns:
            Dict with sync results
        """
        if not self.erp_integration:
            if not self.connect_erp():
                return {'success': False, 'error': 'Failed to connect to ERP'}
        
        try:
            sync_results = self.erp_integration.sync_all_data()
            
            if sync_results['success']:
                self.last_sync_time = datetime.now()
                self.sync_status = sync_results['summary']
                logger.info("✅ ERP data sync completed successfully")
            else:
                logger.error("❌ ERP data sync failed")
            
            return sync_results
            
        except Exception as e:
            logger.error(f"❌ Error syncing ERP data: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_domain_objects(self) -> Dict[str, List]:
        """
        Get domain objects from last sync
        
        Returns:
            Dict with domain objects
        """
        if not self.erp_integration:
            return {'materials': [], 'suppliers': [], 'forecasts': [], 'boms': []}
        
        try:
            # Get latest sync results
            sync_results = self.sync_data()
            
            if sync_results['success']:
                return self.erp_integration.convert_to_domain_objects(sync_results)
            else:
                logger.error("❌ Failed to get domain objects: sync failed")
                return {'materials': [], 'suppliers': [], 'forecasts': [], 'boms': []}
                
        except Exception as e:
            logger.error(f"❌ Error getting domain objects: {e}")
            return {'materials': [], 'suppliers': [], 'forecasts': [], 'boms': []}
    
    def push_recommendations(self, recommendations: List[Dict]) -> Dict[str, Any]:
        """
        Push procurement recommendations to ERP
        
        Args:
            recommendations: List of recommendations
            
        Returns:
            Dict with push results
        """
        if not self.erp_integration:
            return {'success': False, 'error': 'ERP not connected'}
        
        try:
            return self.erp_integration.push_recommendations(recommendations)
        except Exception as e:
            logger.error(f"❌ Error pushing recommendations: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get ERP integration status
        
        Returns:
            Dict with status information
        """
        if not self.erp_integration:
            return {
                'connected': False,
                'erp_type': self.config.get('erp_type', 'unknown'),
                'base_url': self.config.get('base_url', 'unknown'),
                'username': self.config.get('username', 'not set'),
                'last_sync': None,
                'sync_status': {}
            }
        
        return {
            'connected': True,
            'erp_type': self.config.get('erp_type', 'efab'),
            'base_url': self.config.get('base_url', 'https://efab.bkiapps.com'),
            'username': self.config.get('username', 'not set'),
            'last_sync': self.last_sync_time.isoformat() if self.last_sync_time else None,
            'sync_status': self.sync_status,
            'integration_status': self.erp_integration.get_sync_status()
        }
    
    def is_connected(self) -> bool:
        """Check if ERP is connected"""
        return self.erp_integration is not None and self.erp_integration.connection_status == 'connected'
    
    def disconnect(self):
        """Disconnect from ERP"""
        if self.erp_integration:
            self.erp_integration.disconnect()
            self.erp_integration = None
        logger.info("✅ ERP disconnected")

# Global ERP manager instance
erp_manager = ERPManager()