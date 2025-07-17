import pytest
import pandas as pd
from decimal import Decimal
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from src.data.data_integration import DataIntegrator
from src.core.domain import (
    Material, Supplier, SupplierMaterial, Inventory, BOM, Forecast,
    MaterialType, ForecastSource, RiskLevel,
    Money, Quantity, MaterialId, SupplierId, SkuId, LeadTime
)

class TestDataIntegrator:
    def setup_method(self):
        self.integrator = DataIntegrator(data_path="test_data/")
        
        # Create sample data
        self.sample_yarn_master = pd.DataFrame({
            'Yarn_ID': ['YARN-001', 'YARN-002', 'YARN-003'],
            'Yarn_Description': ['Cotton Yarn', 'Polyester Yarn', 'Wool Yarn'],
            'Yarn_Cost': ['$5.00', '$4.50', '$6.00'],
            'Supplier_ID': ['SUP-001', 'SUP-002', 'SUP-001'],
            'Blend': ['Cotton', 'Polyester', 'Wool'],
            'Type': ['Natural', 'Synthetic', 'Natural'],
            'Color': ['White', 'Blue', 'Gray']
        })
        
        self.sample_yarn_inventory = pd.DataFrame({
            'Yarn_ID': ['YARN-001', 'YARN-002', 'YARN-003'],
            'Inventory': [500, -100, 300],  # Include negative inventory
            'On_Order': [200, 150, 0],
            'Allocated': [50, 25, 75],
            'Planning_Balance': [650, 25, 225],
            'Cost_Pound': ['$5.00', '$4.50', '$6.00'],
            'Total_Cost': ['$2500.00', '$-450.00', '$1800.00']
        })
        
        self.sample_suppliers = pd.DataFrame({
            'Supplier_ID': ['SUP-001', 'SUP-002', 'SUP-003'],
            'Name': ['ABC Textiles', 'XYZ Materials', 'Remove Supplier'],
            'Type': ['Import', 'Domestic', 'Import'],
            'Lead_time': [14, 21, 30],
            'MOQ': [100, 150, 200],
            'Remove': ['', '', 'Remove']  # SUP-003 marked for removal
        })
        
        self.sample_style_bom = pd.DataFrame({
            'Style_ID': ['STYLE-001', 'STYLE-001', 'STYLE-002'],
            'Yarn_ID': ['YARN-001', 'YARN-002', 'YARN-003'],
            'Percentage': [60.0, 40.0, 100.0]  # Some > 99%
        })
        
        self.sample_sales_orders = pd.DataFrame({
            'Style_ID': ['STYLE-001', 'STYLE-002'],
            'Quantity': [100, 50],
            'Ship_Date': ['2024-01-15', '2024-02-01'],
            'Customer': ['CUST-001', 'CUST-002']
        })
    
    @patch('src.data.data_integration.pd.read_csv')
    @patch('pathlib.Path.exists')
    def test_load_raw_data(self, mock_exists, mock_read_csv):
        # Setup mocks
        mock_exists.return_value = True
        mock_read_csv.side_effect = [
            self.sample_yarn_master,
            self.sample_yarn_inventory,
            self.sample_suppliers,
            self.sample_style_bom,
            self.sample_sales_orders,
            pd.DataFrame()  # Empty sales activity
        ]
        
        raw_data = self.integrator._load_raw_data()
        
        assert 'yarn_master' in raw_data
        assert 'yarn_inventory' in raw_data
        assert 'suppliers' in raw_data
        assert 'style_bom' in raw_data
        assert 'sales_orders' in raw_data
        assert len(raw_data['yarn_master']) == 3
        assert len(raw_data['yarn_inventory']) == 3
    
    def test_fix_inventory_data(self):
        # Test fixing negative inventory
        df = self.sample_yarn_inventory.copy()
        
        fixed_df = self.integrator._fix_inventory_data(df)
        
        # Negative inventory should be fixed to 0
        assert fixed_df.loc[fixed_df['Yarn_ID'] == 'YARN-002', 'Inventory'].iloc[0] == 0
        
        # Check that fix was recorded
        assert any('negative inventory' in fix.lower() for fix in self.integrator.fixes_applied)
        
        # Cost data should be cleaned
        assert fixed_df['Cost_Pound'].dtype in ['float64', 'int64']
    
    def test_fix_bom_data(self):
        # Test fixing BOM percentages
        df = self.sample_style_bom.copy()
        
        fixed_df = self.integrator._fix_bom_data(df)
        
        # Percentages should be converted to decimals
        assert fixed_df['Percentage'].max() <= 1.0
        assert fixed_df['Percentage'].min() >= 0.0
        
        # Check that fix was recorded if any percentage > 0.99
        if any(df['Percentage'] > 99):
            assert any('BOM percentage' in fix.lower() for fix in self.integrator.fixes_applied)
    
    def test_fix_material_data(self):
        # Test fixing material data
        df = self.sample_yarn_master.copy()
        
        fixed_df = self.integrator._fix_material_data(df)
        
        # Cost data should be cleaned
        assert fixed_df['Yarn_Cost'].dtype in ['float64', 'int64']
        
        # Check for zero cost warning
        if any(fixed_df['Yarn_Cost'] == 0):
            assert any('zero cost' in issue.lower() for issue in self.integrator.quality_issues)
    
    def test_fix_supplier_data(self):
        # Test fixing supplier data
        df = self.sample_suppliers.copy()
        
        fixed_df = self.integrator._fix_supplier_data(df)
        
        # Supplier marked for removal should be removed
        assert 'SUP-003' not in fixed_df['Supplier_ID'].values
        assert len(fixed_df) == 2
        
        # Check that fix was recorded
        assert any('removed' in fix.lower() for fix in self.integrator.fixes_applied)
    
    def test_create_materials(self):
        df = self.sample_yarn_master.copy()
        
        materials = self.integrator._create_materials(df)
        
        assert len(materials) == 3
        assert all(isinstance(material, Material) for material in materials)
        
        # Check first material
        material = materials[0]
        assert material.id.value == 'YARN-001'
        assert material.name == 'Cotton Yarn'
        assert material.type == MaterialType.YARN
        assert material.specifications['blend'] == 'Cotton'
    
    def test_create_suppliers(self):
        df = self.sample_suppliers.copy()
        
        suppliers = self.integrator._create_suppliers(df)
        
        assert len(suppliers) == 3
        assert all(isinstance(supplier, Supplier) for supplier in suppliers)
        
        # Check first supplier
        supplier = suppliers[0]
        assert supplier.id.value == 'SUP-001'
        assert supplier.name == 'ABC Textiles'
        assert supplier.lead_time.days == 14
        assert supplier.reliability_score == 0.85
    
    def test_create_supplier_materials(self):
        yarn_df = self.sample_yarn_master.copy()
        inventory_df = self.sample_yarn_inventory.copy()
        
        # Clean the data first
        yarn_df = self.integrator._fix_material_data(yarn_df)
        inventory_df = self.integrator._fix_inventory_data(inventory_df)
        
        supplier_materials = self.integrator._create_supplier_materials(yarn_df, inventory_df)
        
        assert len(supplier_materials) == 3
        assert all(isinstance(sm, SupplierMaterial) for sm in supplier_materials)
        
        # Check first supplier material
        sm = supplier_materials[0]
        assert sm.supplier_id.value == 'SUP-001'
        assert sm.material_id.value == 'YARN-001'
        assert sm.cost_per_unit.amount == Decimal('5.00')
        assert sm.moq.amount == Decimal('100')
    
    def test_create_inventory(self):
        df = self.sample_yarn_inventory.copy()
        df = self.integrator._fix_inventory_data(df)
        
        inventory = self.integrator._create_inventory(df)
        
        assert len(inventory) == 3
        assert all(isinstance(inv, Inventory) for inv in inventory)
        
        # Check first inventory item
        inv = inventory[0]
        assert inv.material_id.value == 'YARN-001'
        assert inv.on_hand_qty.amount == Decimal('500')
        assert inv.open_po_qty.amount == Decimal('200')
    
    def test_create_boms(self):
        df = self.sample_style_bom.copy()
        df = self.integrator._fix_bom_data(df)
        
        boms = self.integrator._create_boms(df)
        
        assert len(boms) == 3
        assert all(isinstance(bom, BOM) for bom in boms)
        
        # Check first BOM
        bom = boms[0]
        assert bom.sku_id.value == 'STYLE-001'
        assert bom.material_id.value == 'YARN-001'
        assert bom.qty_per_unit.amount == Decimal('0.6')  # 60% converted to decimal
    
    def test_create_forecasts(self):
        df = self.sample_sales_orders.copy()
        
        forecasts = self.integrator._create_forecasts(df)
        
        assert len(forecasts) == 2
        assert all(isinstance(forecast, Forecast) for forecast in forecasts)
        
        # Check first forecast
        forecast = forecasts[0]
        assert forecast.sku_id.value == 'STYLE-001'
        assert forecast.forecast_qty.amount == Decimal('100')
        assert forecast.source == ForecastSource.SALES_ORDER
        assert forecast.confidence_score == 0.9
    
    @patch('src.data.data_integration.pd.read_csv')
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_run_full_integration(self, mock_file, mock_exists, mock_read_csv):
        # Setup mocks
        mock_exists.return_value = True
        mock_read_csv.side_effect = [
            self.sample_yarn_master,
            self.sample_yarn_inventory,
            self.sample_suppliers,
            self.sample_style_bom,
            self.sample_sales_orders,
            pd.DataFrame()  # Empty sales activity
        ]
        
        # Run integration
        domain_objects = self.integrator.run_full_integration()
        
        # Check that all domain objects were created
        assert 'materials' in domain_objects
        assert 'suppliers' in domain_objects
        assert 'supplier_materials' in domain_objects
        assert 'inventory' in domain_objects
        assert 'boms' in domain_objects
        assert 'forecasts' in domain_objects
        
        # Check counts
        assert len(domain_objects['materials']) == 3
        assert len(domain_objects['forecasts']) == 2
    
    @patch('builtins.open', new_callable=mock_open)
    def test_generate_quality_report(self, mock_file):
        # Add some test data
        self.integrator.fixes_applied = ['Fixed 1 negative inventory balance']
        self.integrator.quality_issues = ['2 materials have zero cost']
        
        self.integrator._generate_quality_report()
        
        # Check that file was opened for writing
        mock_file.assert_called_once()
        handle = mock_file.return_value
        
        # Check that content was written
        written_content = ''.join(call.args[0] for call in handle.write.call_args_list)
        assert 'Beverly Knits Data Quality Report' in written_content
        assert 'Fixed 1 negative inventory balance' in written_content
        assert '2 materials have zero cost' in written_content
    
    @patch('pandas.DataFrame.to_csv')
    def test_save_integrated_data(self, mock_to_csv):
        cleaned_data = {
            'yarn_master': self.sample_yarn_master,
            'yarn_inventory': self.sample_yarn_inventory
        }
        
        self.integrator._save_integrated_data(cleaned_data)
        
        # Check that CSV files were saved
        assert mock_to_csv.call_count == 2
    
    def test_empty_data_handling(self):
        # Test with empty data
        empty_df = pd.DataFrame()
        
        # Should not crash with empty data
        materials = self.integrator._create_materials(empty_df)
        suppliers = self.integrator._create_suppliers(empty_df)
        
        assert materials == []
        assert suppliers == []
    
    def test_missing_columns_handling(self):
        # Test with missing columns
        incomplete_df = pd.DataFrame({
            'Yarn_ID': ['YARN-001'],
            # Missing other expected columns
        })
        
        # Should handle missing columns gracefully
        materials = self.integrator._create_materials(incomplete_df)
        assert len(materials) == 1
        assert materials[0].id.value == 'YARN-001'
    
    def test_data_type_conversion(self):
        # Test data type conversions
        df = pd.DataFrame({
            'Yarn_ID': ['YARN-001'],
            'Inventory': ['500.5'],  # String that should be converted to float
            'Cost_Pound': ['$5.00'],  # String with currency symbol
        })
        
        fixed_df = self.integrator._fix_inventory_data(df)
        
        # Should convert to numeric
        assert fixed_df['Inventory'].dtype in ['float64', 'int64']
        assert fixed_df['Cost_Pound'].dtype in ['float64', 'int64']
        assert fixed_df['Inventory'].iloc[0] == 500.5
        assert fixed_df['Cost_Pound'].iloc[0] == 5.0