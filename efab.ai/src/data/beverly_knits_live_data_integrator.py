"""
Beverly Knits Live Data Integration System

This module processes the actual Beverly Knits data files and integrates them 
with the AI Supply Chain Planner system, applying intelligent data quality fixes
and preparing data for the planning engine.
"""

import pandas as pd
import numpy as np
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, date
import logging
import json
from pathlib import Path
import re

from ..core.domain import (
    Material, Supplier, SupplierMaterial, Inventory, BOM, Forecast,
    MaterialType, ForecastSource, RiskLevel,
    Money, Quantity, MaterialId, SupplierId, SkuId, LeadTime
)

logger = logging.getLogger(__name__)

class BeverlyKnitsLiveDataIntegrator:
    """Live data integrator for Beverly Knits real data files"""
    
    def __init__(self, data_path: str = "data/live/"):
        self.data_path = Path(data_path)
        self.quality_issues = []
        self.fixes_applied = []
        self.integration_stats = {}
        
        # File mappings
        self.file_mappings = {
            'yarn_master': 'Yarn_ID_1.csv',
            'yarn_inventory': 'Yarn_ID_Current_Inventory.csv',
            'suppliers': 'Supplier_ID.csv',
            'style_bom': 'Style_BOM.csv',
            'sales_orders': 'eFab_SO_List.csv',
            'sales_activity': 'Sales Activity Report.csv',
            'yarn_demand_by_style': 'cfab_Yarn_Demand_By_Style.csv',
            'yarn_demand_forecast': 'Yarn_Demand_2025-06-27_0442.csv'
        }
    
    def integrate_live_data(self) -> Dict[str, Any]:
        """Main integration method for Beverly Knits live data"""
        logger.info("Starting Beverly Knits live data integration")
        
        # Load raw data
        raw_data = self._load_live_data()
        
        # Apply data quality fixes
        cleaned_data = self._apply_live_data_fixes(raw_data)
        
        # Create domain objects
        domain_objects = self._create_live_domain_objects(cleaned_data)
        
        # Generate integration report
        self._generate_integration_report()
        
        logger.info("Live data integration completed successfully")
        return domain_objects
    
    def _load_live_data(self) -> Dict[str, pd.DataFrame]:
        """Load Beverly Knits live data files"""
        raw_data = {}
        
        for key, filename in self.file_mappings.items():
            file_path = self.data_path / filename
            if file_path.exists():
                try:
                    # Handle different encodings
                    for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
                        try:
                            df = pd.read_csv(file_path, encoding=encoding)
                            raw_data[key] = df
                            logger.info(f"Loaded {filename}: {len(df)} rows")
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        logger.error(f"Could not decode {filename} with any encoding")
                        self.quality_issues.append(f"Encoding issue with {filename}")
                        
                except Exception as e:
                    logger.error(f"Failed to load {filename}: {e}")
                    self.quality_issues.append(f"Failed to load {filename}: {e}")
            else:
                logger.warning(f"File not found: {filename}")
                self.quality_issues.append(f"File not found: {filename}")
        
        return raw_data
    
    def _apply_live_data_fixes(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Apply data quality fixes specific to Beverly Knits data"""
        cleaned_data = {}
        
        for key, df in raw_data.items():
            if key == 'yarn_inventory':
                cleaned_data[key] = self._fix_yarn_inventory_data(df)
            elif key == 'suppliers':
                cleaned_data[key] = self._fix_supplier_data(df)
            elif key == 'style_bom':
                cleaned_data[key] = self._fix_bom_data(df)
            elif key == 'sales_orders':
                cleaned_data[key] = self._fix_sales_orders_data(df)
            elif key == 'sales_activity':
                cleaned_data[key] = self._fix_sales_activity_data(df)
            elif key == 'yarn_demand_by_style':
                cleaned_data[key] = self._fix_yarn_demand_data(df)
            else:
                cleaned_data[key] = df.copy()
        
        return cleaned_data
    
    def _fix_yarn_inventory_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix yarn inventory data issues"""
        df = df.copy()
        original_rows = len(df)
        
        # Clean numeric columns - remove commas, dollar signs, parentheses
        numeric_columns = ['Inventory', 'On_Order', 'Allocated', 'Planning_Ballance', 'Cost_Pound', 'Total_Cost']
        
        for col in numeric_columns:
            if col in df.columns:
                # Convert to string first
                df[col] = df[col].astype(str)
                
                # Remove formatting characters
                df[col] = df[col].str.replace(',', '', regex=False)
                df[col] = df[col].str.replace('$', '', regex=False)
                df[col] = df[col].str.replace(' ', '', regex=False)
                
                # Handle negative values in parentheses
                df[col] = df[col].str.replace(r'\((.*?)\)', r'-\1', regex=True)
                
                # Convert to numeric
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Fix negative inventory (set to 0 for on-hand, preserve planning balance)
        if 'Inventory' in df.columns:
            negative_inventory = (df['Inventory'] < 0).sum()
            if negative_inventory > 0:
                df.loc[df['Inventory'] < 0, 'Inventory'] = 0
                self.fixes_applied.append(f"Fixed {negative_inventory} negative inventory balances")
        
        # Handle missing supplier data
        if 'Supplier' in df.columns:
            missing_suppliers = df['Supplier'].isna().sum()
            if missing_suppliers > 0:
                df['Supplier'] = df['Supplier'].fillna('UNKNOWN_SUPPLIER')
                self.quality_issues.append(f"{missing_suppliers} yarn records missing supplier information")
        
        self.integration_stats['yarn_inventory'] = {
            'original_rows': original_rows,
            'processed_rows': len(df),
            'negative_inventory_fixed': negative_inventory if 'Inventory' in df.columns else 0
        }
        
        return df
    
    def _fix_supplier_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix supplier data issues"""
        df = df.copy()
        original_rows = len(df)
        
        # Remove suppliers marked for removal
        if 'Lead_time' in df.columns:
            remove_mask = df['Lead_time'] == 'Remove'
            removed_count = remove_mask.sum()
            df = df[~remove_mask]
            
            if removed_count > 0:
                self.fixes_applied.append(f"Removed {removed_count} suppliers marked for removal")
        
        # Clean lead time data
        if 'Lead_time' in df.columns:
            df['Lead_time'] = pd.to_numeric(df['Lead_time'], errors='coerce').fillna(14)
        
        # Clean MOQ data
        if 'MOQ' in df.columns:
            df['MOQ'] = pd.to_numeric(df['MOQ'], errors='coerce').fillna(100)
        
        # Standardize supplier types
        if 'Type' in df.columns:
            df['Type'] = df['Type'].fillna('Domestic')
        
        self.integration_stats['suppliers'] = {
            'original_rows': original_rows,
            'processed_rows': len(df),
            'removed_suppliers': removed_count if 'Lead_time' in df.columns else 0
        }
        
        return df
    
    def _fix_bom_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix BOM data issues"""
        df = df.copy()
        original_rows = len(df)
        
        # Ensure BOM_Percentage is numeric
        if 'BOM_Percentage' in df.columns:
            df['BOM_Percentage'] = pd.to_numeric(df['BOM_Percentage'], errors='coerce').fillna(0)
            
            # Check for BOMs that don't sum to 1.0 per style
            bom_sums = df.groupby('Style_ID')['BOM_Percentage'].sum()
            problematic_styles = bom_sums[abs(bom_sums - 1.0) > 0.01]
            
            if len(problematic_styles) > 0:
                self.quality_issues.append(f"{len(problematic_styles)} styles have BOM percentages that don't sum to 1.0")
                
                # Fix styles where percentages are very close to 1.0
                for style_id in problematic_styles.index:
                    style_mask = df['Style_ID'] == style_id
                    current_sum = df.loc[style_mask, 'BOM_Percentage'].sum()
                    
                    if 0.99 <= current_sum <= 1.01:
                        # Normalize to 1.0
                        df.loc[style_mask, 'BOM_Percentage'] = df.loc[style_mask, 'BOM_Percentage'] / current_sum
                        self.fixes_applied.append(f"Normalized BOM percentages for style {style_id}")
        
        self.integration_stats['bom'] = {
            'original_rows': original_rows,
            'processed_rows': len(df),
            'problematic_styles': len(problematic_styles) if 'BOM_Percentage' in df.columns else 0
        }
        
        return df
    
    def _fix_sales_orders_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix sales orders data"""
        df = df.copy()
        original_rows = len(df)
        
        # Clean unit price
        if 'Unit Price' in df.columns:
            df['Unit Price'] = df['Unit Price'].astype(str).str.replace('$', '').str.replace(' (yds)', '')
            df['Unit Price'] = pd.to_numeric(df['Unit Price'], errors='coerce').fillna(0)
        
        # Clean ordered quantity
        if 'Ordered' in df.columns:
            df['Ordered'] = pd.to_numeric(df['Ordered'], errors='coerce').fillna(0)
        
        # Parse ship date
        if 'Ship Date' in df.columns:
            df['Ship Date'] = pd.to_datetime(df['Ship Date'], errors='coerce')
        
        self.integration_stats['sales_orders'] = {
            'original_rows': original_rows,
            'processed_rows': len(df)
        }
        
        return df
    
    def _fix_sales_activity_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix sales activity data"""
        df = df.copy()
        original_rows = len(df)
        
        # Clean unit price
        if 'Unit Price' in df.columns:
            df['Unit Price'] = df['Unit Price'].astype(str).str.replace('$', '').str.replace(' ', '')
            df['Unit Price'] = pd.to_numeric(df['Unit Price'], errors='coerce').fillna(0)
        
        # Clean line price
        if 'Line Price' in df.columns:
            df['Line Price'] = df['Line Price'].astype(str).str.replace('$', '').str.replace(',', '').str.replace(' ', '')
            df['Line Price'] = pd.to_numeric(df['Line Price'], errors='coerce').fillna(0)
        
        # Clean yards ordered
        if 'Yds_ordered' in df.columns:
            df['Yds_ordered'] = pd.to_numeric(df['Yds_ordered'], errors='coerce').fillna(0)
        
        # Parse invoice date
        if 'Invoice Date' in df.columns:
            df['Invoice Date'] = pd.to_datetime(df['Invoice Date'], errors='coerce')
        
        self.integration_stats['sales_activity'] = {
            'original_rows': original_rows,
            'processed_rows': len(df)
        }
        
        return df
    
    def _fix_yarn_demand_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix yarn demand data"""
        df = df.copy()
        original_rows = len(df)
        
        # Clean numeric columns - remove commas
        numeric_columns = ['Percentage', 'This Week', 'Week 17', 'Week 18', 'Week 19', 'Week 20', 'Week 21', 'Week 22', 'Week 23', 'Week 24', 'Later', 'Total']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '')
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        self.integration_stats['yarn_demand'] = {
            'original_rows': original_rows,
            'processed_rows': len(df)
        }
        
        return df
    
    def _create_live_domain_objects(self, cleaned_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Create domain objects from cleaned Beverly Knits data"""
        domain_objects = {}
        
        # Create materials from yarn master
        if 'yarn_master' in cleaned_data:
            materials = self._create_materials_from_yarn_master(cleaned_data['yarn_master'])
            domain_objects['materials'] = materials
        
        # Create suppliers
        if 'suppliers' in cleaned_data:
            suppliers = self._create_suppliers_from_supplier_data(cleaned_data['suppliers'])
            domain_objects['suppliers'] = suppliers
        
        # Create supplier-material relationships
        if 'yarn_inventory' in cleaned_data:
            supplier_materials = self._create_supplier_materials_from_inventory(cleaned_data['yarn_inventory'])
            domain_objects['supplier_materials'] = supplier_materials
        
        # Create inventory
        if 'yarn_inventory' in cleaned_data:
            inventory = self._create_inventory_from_yarn_inventory(cleaned_data['yarn_inventory'])
            domain_objects['inventory'] = inventory
        
        # Create BOMs
        if 'style_bom' in cleaned_data:
            boms = self._create_boms_from_style_bom(cleaned_data['style_bom'])
            domain_objects['boms'] = boms
        
        # Create forecasts from sales data
        forecasts = []
        if 'sales_orders' in cleaned_data:
            sales_forecasts = self._create_forecasts_from_sales_orders(cleaned_data['sales_orders'])
            forecasts.extend(sales_forecasts)
        
        if 'yarn_demand_by_style' in cleaned_data:
            demand_forecasts = self._create_forecasts_from_yarn_demand(cleaned_data['yarn_demand_by_style'])
            forecasts.extend(demand_forecasts)
        
        domain_objects['forecasts'] = forecasts
        
        return domain_objects
    
    def _create_materials_from_yarn_master(self, df: pd.DataFrame) -> List[Material]:
        """Create Material objects from yarn master data"""
        materials = []
        
        for _, row in df.iterrows():
            material = Material(
                id=MaterialId(value=str(row['Yarn_ID'])),
                name=row.get('Description', f"Yarn {row['Yarn_ID']}"),
                type=MaterialType.YARN,
                description=f"{row.get('Blend', '')} {row.get('Type', '')} yarn",
                specifications={
                    'supplier': str(row.get('Supplier', '')) if pd.notna(row.get('Supplier')) else '',
                    'blend': str(row.get('Blend', '')) if pd.notna(row.get('Blend')) else '',
                    'type': str(row.get('Type', '')) if pd.notna(row.get('Type')) else '',
                    'color': str(row.get('Color', '')) if pd.notna(row.get('Color')) else '',
                    'desc_1': str(row.get('Desc_1', '')) if pd.notna(row.get('Desc_1')) else '',
                    'desc_2': str(row.get('Desc_2', '')) if pd.notna(row.get('Desc_2')) else '',
                    'desc_3': str(row.get('Desc_3', '')) if pd.notna(row.get('Desc_3')) else ''
                }
            )
            materials.append(material)
        
        return materials
    
    def _create_suppliers_from_supplier_data(self, df: pd.DataFrame) -> List[Supplier]:
        """Create Supplier objects from supplier data"""
        suppliers = []
        
        for _, row in df.iterrows():
            # Calculate reliability score based on supplier type
            reliability_score = 0.90 if row.get('Type') == 'Domestic' else 0.80
            
            supplier = Supplier(
                id=SupplierId(value=str(row['Supplier_ID'])),
                name=row.get('Supplier', f"Supplier {row['Supplier_ID']}"),
                lead_time=LeadTime(days=int(row.get('Lead_time', 14))),
                reliability_score=reliability_score,
                risk_level=RiskLevel.LOW if reliability_score > 0.85 else RiskLevel.MEDIUM
            )
            suppliers.append(supplier)
        
        return suppliers
    
    def _create_supplier_materials_from_inventory(self, df: pd.DataFrame) -> List[SupplierMaterial]:
        """Create SupplierMaterial relationships from inventory data"""
        supplier_materials = []
        
        for _, row in df.iterrows():
            # Skip if essential data is missing
            if pd.isna(row.get('Yarn_ID')) or pd.isna(row.get('Supplier')):
                continue
            
            supplier_material = SupplierMaterial(
                supplier_id=SupplierId(value=str(row['Supplier'])),
                material_id=MaterialId(value=str(row['Yarn_ID'])),
                cost_per_unit=Money(
                    amount=Decimal(str(row.get('Cost_Pound', 0))),
                    currency="USD"
                ),
                moq=Quantity(amount=Decimal("100"), unit="lb"),  # Default MOQ
                lead_time=LeadTime(days=14),  # Default lead time
                reliability_score=0.85  # Default reliability
            )
            supplier_materials.append(supplier_material)
        
        return supplier_materials
    
    def _create_inventory_from_yarn_inventory(self, df: pd.DataFrame) -> List[Inventory]:
        """Create Inventory objects from yarn inventory data"""
        inventory = []
        
        for _, row in df.iterrows():
            inv = Inventory(
                material_id=MaterialId(value=str(row['Yarn_ID'])),
                on_hand_qty=Quantity(
                    amount=Decimal(str(row.get('Inventory', 0))),
                    unit="lb"
                ),
                open_po_qty=Quantity(
                    amount=Decimal(str(row.get('On_Order', 0))),
                    unit="lb"
                ),
                safety_stock=Quantity(
                    amount=Decimal(str(row.get('Allocated', 0))),
                    unit="lb"
                )
            )
            inventory.append(inv)
        
        return inventory
    
    def _create_boms_from_style_bom(self, df: pd.DataFrame) -> List[BOM]:
        """Create BOM objects from style BOM data"""
        boms = []
        
        for _, row in df.iterrows():
            bom = BOM(
                sku_id=SkuId(value=str(row['Style_ID'])),
                material_id=MaterialId(value=str(row['Yarn_ID'])),
                qty_per_unit=Quantity(
                    amount=Decimal(str(row.get('BOM_Percentage', 0))),
                    unit=row.get('unit', 'lbs')
                ),
                unit=row.get('unit', 'lbs')
            )
            boms.append(bom)
        
        return boms
    
    def _create_forecasts_from_sales_orders(self, df: pd.DataFrame) -> List[Forecast]:
        """Create Forecast objects from sales orders"""
        forecasts = []
        
        for _, row in df.iterrows():
            # Extract style from fBase column
            style_id = row.get('fBase', 'UNKNOWN_STYLE')
            
            forecast = Forecast(
                sku_id=SkuId(value=str(style_id)),
                forecast_qty=Quantity(
                    amount=Decimal(str(row.get('Ordered', 0))),
                    unit="unit"
                ),
                forecast_date=date.today(),
                source=ForecastSource.SALES_ORDER,
                confidence_score=0.95
            )
            forecasts.append(forecast)
        
        return forecasts
    
    def _create_forecasts_from_yarn_demand(self, df: pd.DataFrame) -> List[Forecast]:
        """Create Forecast objects from yarn demand data"""
        forecasts = []
        
        for _, row in df.iterrows():
            # Create forecast for total demand
            forecast = Forecast(
                sku_id=SkuId(value=str(row['Style'])),
                forecast_qty=Quantity(
                    amount=Decimal(str(row.get('Total', 0))),
                    unit="lb"
                ),
                forecast_date=date.today(),
                source=ForecastSource.PROD_PLAN,
                confidence_score=0.85
            )
            forecasts.append(forecast)
        
        return forecasts
    
    def _generate_integration_report(self):
        """Generate integration report"""
        report_path = Path("data/output/live_data_integration_report.txt")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write("Beverly Knits Live Data Integration Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("INTEGRATION STATISTICS:\n")
            for dataset, stats in self.integration_stats.items():
                f.write(f"\n{dataset.upper()}:\n")
                for key, value in stats.items():
                    f.write(f"  - {key}: {value}\n")
            
            f.write("\nFIXES APPLIED:\n")
            for fix in self.fixes_applied:
                f.write(f"✓ {fix}\n")
            
            f.write("\nQUALITY ISSUES IDENTIFIED:\n")
            for issue in self.quality_issues:
                f.write(f"⚠ {issue}\n")
        
        logger.info(f"Integration report generated: {report_path}")

def integrate_beverly_knits_live_data(data_path: str = "data/live/") -> Dict[str, Any]:
    """Convenience function to integrate Beverly Knits live data"""
    integrator = BeverlyKnitsLiveDataIntegrator(data_path)
    return integrator.integrate_live_data()

if __name__ == "__main__":
    # Run live data integration
    domain_objects = integrate_beverly_knits_live_data()
    
    print("\n🧶 Beverly Knits Live Data Integration Summary")
    print("=" * 50)
    for key, objects in domain_objects.items():
        print(f"{key.title()}: {len(objects)} items")
    
    print("\n📊 Integration completed successfully!")
    print("📝 Check data/output/live_data_integration_report.txt for details")