import pandas as pd
import numpy as np
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, date
import logging
import json
from pathlib import Path

from ..core.domain import (
    Material, Supplier, SupplierMaterial, Inventory, BOM, Forecast,
    MaterialType, ForecastSource, RiskLevel,
    Money, Quantity, MaterialId, SupplierId, SkuId, LeadTime
)

logger = logging.getLogger(__name__)

class DataIntegrator:
    def __init__(self, data_path: str = "data/"):
        self.data_path = Path(data_path)
        self.quality_issues = []
        self.fixes_applied = []
        
    def run_full_integration(self) -> Dict[str, Any]:
        """Run complete data integration pipeline with automatic quality fixes"""
        logger.info("Starting enhanced data integration pipeline")
        
        # Load and clean raw data
        raw_data = self._load_raw_data()
        
        # Apply automatic fixes
        cleaned_data = self._apply_automatic_fixes(raw_data)
        
        # Create domain objects
        domain_objects = self._create_domain_objects(cleaned_data)
        
        # Generate quality report
        self._generate_quality_report()
        
        # Save integrated data
        self._save_integrated_data(cleaned_data)
        
        logger.info("Data integration pipeline completed successfully")
        return domain_objects
    
    def _load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """Load all raw CSV files"""
        raw_data = {}
        
        files_to_load = [
            ("yarn_master", "Yarn_ID_1.csv"),
            ("yarn_inventory", "Yarn_ID_Current_Inventory.csv"),
            ("suppliers", "Supplier_ID.csv"),
            ("style_bom", "Style_BOM.csv"),
            ("sales_orders", "eFab_SO_List.csv"),
            ("sales_activity", "Sales Activity Report.csv")
        ]
        
        for key, filename in files_to_load:
            file_path = self.data_path / filename
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path, encoding='utf-8')
                    raw_data[key] = df
                    logger.info(f"Loaded {filename}: {len(df)} rows")
                except Exception as e:
                    logger.error(f"Failed to load {filename}: {e}")
                    self.quality_issues.append(f"Failed to load {filename}: {e}")
            else:
                logger.warning(f"File not found: {filename}")
                self.quality_issues.append(f"File not found: {filename}")
        
        return raw_data
    
    def _apply_automatic_fixes(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Apply automatic data quality fixes"""
        cleaned_data = {}
        
        for key, df in raw_data.items():
            cleaned_df = df.copy()
            
            if key == "yarn_inventory":
                cleaned_df = self._fix_inventory_data(cleaned_df)
            elif key == "style_bom":
                cleaned_df = self._fix_bom_data(cleaned_df)
            elif key == "yarn_master":
                cleaned_df = self._fix_material_data(cleaned_df)
            elif key == "suppliers":
                cleaned_df = self._fix_supplier_data(cleaned_df)
            
            cleaned_data[key] = cleaned_df
        
        return cleaned_data
    
    def _fix_inventory_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix inventory data quality issues"""
        original_negatives = 0
        
        # Fix negative inventory balances
        if 'Inventory' in df.columns:
            negative_mask = pd.to_numeric(df['Inventory'], errors='coerce') < 0
            original_negatives = negative_mask.sum()
            
            if original_negatives > 0:
                df.loc[negative_mask, 'Inventory'] = 0
                self.fixes_applied.append(f"Fixed {original_negatives} negative inventory balances")
        
        # Clean cost data
        if 'Cost_Pound' in df.columns:
            df['Cost_Pound'] = df['Cost_Pound'].astype(str).str.replace('$', '').str.replace(',', '')
            df['Cost_Pound'] = pd.to_numeric(df['Cost_Pound'], errors='coerce').fillna(0)
            self.fixes_applied.append("Cleaned cost data formatting")
        
        return df
    
    def _fix_bom_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix BOM data quality issues"""
        if 'Percentage' in df.columns:
            # Convert percentage to decimal
            df['Percentage'] = pd.to_numeric(df['Percentage'], errors='coerce') / 100
            
            # Fix percentages > 0.99 to 1.0
            high_percentage_mask = df['Percentage'] > 0.99
            high_percentage_count = high_percentage_mask.sum()
            
            if high_percentage_count > 0:
                df.loc[high_percentage_mask, 'Percentage'] = 1.0
                self.fixes_applied.append(f"Fixed {high_percentage_count} BOM percentages > 0.99")
        
        return df
    
    def _fix_material_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix material master data quality issues"""
        # Clean cost data
        if 'Yarn_Cost' in df.columns:
            df['Yarn_Cost'] = df['Yarn_Cost'].astype(str).str.replace('$', '').str.replace(',', '')
            df['Yarn_Cost'] = pd.to_numeric(df['Yarn_Cost'], errors='coerce').fillna(0)
            
            zero_cost_count = (df['Yarn_Cost'] == 0).sum()
            if zero_cost_count > 0:
                self.quality_issues.append(f"{zero_cost_count} materials have zero cost")
        
        return df
    
    def _fix_supplier_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix supplier data quality issues"""
        # Remove suppliers marked for removal
        if 'Remove' in df.columns:
            original_count = len(df)
            df = df[df['Remove'].isna() | (df['Remove'] == '')]
            removed_count = original_count - len(df)
            
            if removed_count > 0:
                self.fixes_applied.append(f"Removed {removed_count} suppliers marked for removal")
        
        return df
    
    def _create_domain_objects(self, cleaned_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Create domain objects from cleaned data"""
        domain_objects = {}
        
        # Create materials
        if 'yarn_master' in cleaned_data:
            materials = self._create_materials(cleaned_data['yarn_master'])
            domain_objects['materials'] = materials
        
        # Create suppliers
        if 'suppliers' in cleaned_data:
            suppliers = self._create_suppliers(cleaned_data['suppliers'])
            domain_objects['suppliers'] = suppliers
        
        # Create supplier-material relationships
        if 'yarn_master' in cleaned_data and 'yarn_inventory' in cleaned_data:
            supplier_materials = self._create_supplier_materials(
                cleaned_data['yarn_master'], 
                cleaned_data['yarn_inventory']
            )
            domain_objects['supplier_materials'] = supplier_materials
        
        # Create inventory
        if 'yarn_inventory' in cleaned_data:
            inventory = self._create_inventory(cleaned_data['yarn_inventory'])
            domain_objects['inventory'] = inventory
        
        # Create BOMs
        if 'style_bom' in cleaned_data:
            boms = self._create_boms(cleaned_data['style_bom'])
            domain_objects['boms'] = boms
        
        # Create forecasts from sales data
        if 'sales_orders' in cleaned_data:
            forecasts = self._create_forecasts(cleaned_data['sales_orders'])
            domain_objects['forecasts'] = forecasts
        
        return domain_objects
    
    def _create_materials(self, df: pd.DataFrame) -> List[Material]:
        """Create Material objects from yarn master data"""
        materials = []
        
        for _, row in df.iterrows():
            material = Material(
                id=MaterialId(value=str(row['Yarn_ID'])),
                name=row.get('Yarn_Description', f"Yarn {row['Yarn_ID']}"),
                type=MaterialType.YARN,
                description=row.get('Description', ''),
                specifications={
                    'blend': row.get('Blend', ''),
                    'type': row.get('Type', ''),
                    'color': row.get('Color', '')
                }
            )
            materials.append(material)
        
        return materials
    
    def _create_suppliers(self, df: pd.DataFrame) -> List[Supplier]:
        """Create Supplier objects from supplier master data"""
        suppliers = []
        
        for _, row in df.iterrows():
            supplier = Supplier(
                id=SupplierId(value=str(row['Supplier_ID'])),
                name=row.get('Name', f"Supplier {row['Supplier_ID']}"),
                contact_info=row.get('Contact', ''),
                lead_time=LeadTime(days=int(row.get('Lead_time', 14))),
                reliability_score=0.85,  # Default value
                risk_level=RiskLevel.MEDIUM
            )
            suppliers.append(supplier)
        
        return suppliers
    
    def _create_supplier_materials(self, yarn_df: pd.DataFrame, inventory_df: pd.DataFrame) -> List[SupplierMaterial]:
        """Create SupplierMaterial relationships"""
        supplier_materials = []
        
        # Merge yarn master with inventory data
        merged_df = yarn_df.merge(inventory_df, on='Yarn_ID', how='inner')
        
        for _, row in merged_df.iterrows():
            supplier_material = SupplierMaterial(
                supplier_id=SupplierId(value=str(row.get('Supplier_ID', 'DEFAULT'))),
                material_id=MaterialId(value=str(row['Yarn_ID'])),
                cost_per_unit=Money(
                    amount=Decimal(str(row.get('Cost_Pound', 0))),
                    currency="USD"
                ),
                moq=Quantity(amount=Decimal("100"), unit="lb"),  # Default MOQ
                lead_time=LeadTime(days=14),  # Default lead time
                reliability_score=0.85
            )
            supplier_materials.append(supplier_material)
        
        return supplier_materials
    
    def _create_inventory(self, df: pd.DataFrame) -> List[Inventory]:
        """Create Inventory objects from inventory data"""
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
                    amount=Decimal(str(row.get('Safety_Stock', 0))),
                    unit="lb"
                )
            )
            inventory.append(inv)
        
        return inventory
    
    def _create_boms(self, df: pd.DataFrame) -> List[BOM]:
        """Create BOM objects from style BOM data"""
        boms = []
        
        for _, row in df.iterrows():
            bom = BOM(
                sku_id=SkuId(value=str(row['Style_ID'])),
                material_id=MaterialId(value=str(row['Yarn_ID'])),
                qty_per_unit=Quantity(
                    amount=Decimal(str(row.get('Percentage', 0))),
                    unit="ratio"
                ),
                unit="ratio"
            )
            boms.append(bom)
        
        return boms
    
    def _create_forecasts(self, df: pd.DataFrame) -> List[Forecast]:
        """Create Forecast objects from sales order data"""
        forecasts = []
        
        for _, row in df.iterrows():
            forecast = Forecast(
                sku_id=SkuId(value=str(row.get('Style_ID', 'UNKNOWN'))),
                forecast_qty=Quantity(
                    amount=Decimal(str(row.get('Quantity', 0))),
                    unit="unit"
                ),
                forecast_date=date.today(),
                source=ForecastSource.SALES_ORDER,
                confidence_score=0.9
            )
            forecasts.append(forecast)
        
        return forecasts
    
    def _generate_quality_report(self):
        """Generate data quality report"""
        report_path = self.data_path / "data_quality_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("Beverly Knits Data Quality Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("FIXES APPLIED:\n")
            for fix in self.fixes_applied:
                f.write(f"- {fix}\n")
            
            f.write("\nQUALITY ISSUES IDENTIFIED:\n")
            for issue in self.quality_issues:
                f.write(f"- {issue}\n")
        
        logger.info(f"Quality report generated: {report_path}")
    
    def _save_integrated_data(self, cleaned_data: Dict[str, pd.DataFrame]):
        """Save integrated data to CSV files"""
        for key, df in cleaned_data.items():
            output_path = self.data_path / f"integrated_{key}.csv"
            df.to_csv(output_path, index=False)
            logger.info(f"Saved integrated data: {output_path}")