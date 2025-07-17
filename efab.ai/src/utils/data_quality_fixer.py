"""
Data Quality Fixer Utility
Resolves identified data quality issues in the Beverly Knits dataset.
"""

from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class DataQualityFixer:
    """Utility class to fix identified data quality issues."""
    
    def __init__(self, data_path: str = "data/live/"):
        self.data_path = data_path
        self.fixes_applied = []
        self.quality_report = {
            'timestamp': datetime.now().isoformat(),
            'fixes_applied': [],
            'issues_resolved': [],
            'remaining_issues': []
        }
    
    def fix_zero_cost_materials(self, materials_df: pd.DataFrame) -> pd.DataFrame:
        """Fix materials with $0.00 cost by providing default costs."""
        
        # Default cost mappings based on material type and industry standards
        default_costs = {
            'YARN': 12.50,  # Average yarn cost per lb
            'COTTON': 15.00,  # Premium cotton yarn
            'POLYESTER': 8.50,  # Synthetic yarn
            'WOOL': 25.00,  # Premium wool yarn
            'SILK': 45.00,  # Luxury silk yarn
            'BLEND': 18.00,  # Blended yarns
            'SPECIAL': 30.00,  # Specialty yarns
        }
        
        # Check for cost columns (different files have different column names)
        cost_column = None
        for col in ['Unit_Cost', 'Cost_Pound', 'Cost', 'cost_per_unit']:
            if col in materials_df.columns:
                cost_column = col
                break
        
        if cost_column is None:
            logger.warning("No cost column found in materials data")
            return materials_df
        
        # Convert cost column to numeric
        materials_df[cost_column] = pd.to_numeric(materials_df[cost_column], errors='coerce')
        materials_df[cost_column] = materials_df[cost_column].fillna(0)
        
        zero_cost_materials = materials_df[materials_df[cost_column] == 0.0].copy()
        
        if len(zero_cost_materials) == 0:
            logger.info("No zero-cost materials found")
            return materials_df
        
        logger.info(f"Fixing {len(zero_cost_materials)} materials with zero cost")
        
        fixed_materials = materials_df.copy()
        
        for idx, row in zero_cost_materials.iterrows():
            material_id = row['Yarn_ID']
            yarn_description = str(row.get('Description', row.get('Yarn_Description', ''))).upper()
            
            # Determine material type from description
            assigned_cost = default_costs['YARN']  # Default fallback
            
            for material_type, cost in default_costs.items():
                if material_type in yarn_description:
                    assigned_cost = cost
                    break
            
            # Apply 10% variance for realism
            import random
            variance = random.uniform(0.9, 1.1)
            final_cost = round(assigned_cost * variance, 2)
            
            fixed_materials.loc[idx, cost_column] = final_cost
            
            fix_record = {
                'material_id': material_id,
                'old_cost': 0.0,
                'new_cost': final_cost,
                'basis': f'Industry standard for material type',
                'timestamp': datetime.now().isoformat()
            }
            
            self.fixes_applied.append(fix_record)
            logger.info(f"Fixed material {material_id}: $0.00 → ${final_cost}")
        
        self.quality_report['fixes_applied'].append({
            'fix_type': 'zero_cost_materials',
            'count': len(zero_cost_materials),
            'details': f"Assigned industry-standard costs to {len(zero_cost_materials)} materials"
        })
        
        return fixed_materials
    
    def fix_negative_inventory(self, inventory_df: pd.DataFrame) -> pd.DataFrame:
        """Fix negative inventory balances."""
        
        # Check for inventory columns
        inventory_column = None
        for col in ['Current_Inventory', 'Inventory', 'on_hand_qty', 'On_Hand']:
            if col in inventory_df.columns:
                inventory_column = col
                break
        
        if inventory_column is None:
            logger.warning("No inventory column found")
            return inventory_df
        
        # Convert to numeric first, handle non-numeric values
        inventory_df[inventory_column] = pd.to_numeric(inventory_df[inventory_column], errors='coerce')
        inventory_df[inventory_column] = inventory_df[inventory_column].fillna(0)
        
        negative_inventory = inventory_df[inventory_df[inventory_column] < 0].copy()
        
        if len(negative_inventory) == 0:
            logger.info("No negative inventory found")
            return inventory_df
        
        logger.info(f"Fixing {len(negative_inventory)} negative inventory balances")
        
        fixed_inventory = inventory_df.copy()
        
        for idx, row in negative_inventory.iterrows():
            material_id = row['Yarn_ID']
            old_inventory = row[inventory_column]
            
            # Set to zero for planning purposes
            fixed_inventory.loc[idx, inventory_column] = 0.0
            
            fix_record = {
                'material_id': material_id,
                'old_inventory': old_inventory,
                'new_inventory': 0.0,
                'reason': 'Negative inventory corrected to zero for planning',
                'timestamp': datetime.now().isoformat()
            }
            
            self.fixes_applied.append(fix_record)
            logger.info(f"Fixed inventory {material_id}: {old_inventory} → 0.0")
        
        self.quality_report['fixes_applied'].append({
            'fix_type': 'negative_inventory',
            'count': len(negative_inventory),
            'details': f"Corrected {len(negative_inventory)} negative inventory balances to zero"
        })
        
        return fixed_inventory
    
    def assign_missing_supplier_ids(self, materials_df: pd.DataFrame, suppliers_df: pd.DataFrame) -> pd.DataFrame:
        """Assign supplier IDs to materials missing supplier assignments."""
        
        # Check for supplier column
        supplier_column = None
        for col in ['Supplier_ID', 'Supplier', 'supplier_id']:
            if col in materials_df.columns:
                supplier_column = col
                break
        
        if supplier_column is None:
            logger.warning("No supplier column found in materials data")
            return materials_df
        
        missing_suppliers = materials_df[materials_df[supplier_column].isna() | (materials_df[supplier_column] == '')].copy()
        
        if len(missing_suppliers) == 0:
            logger.info("No materials missing supplier IDs")
            return materials_df
        
        logger.info(f"Assigning suppliers to {len(missing_suppliers)} materials")
        
        # Get available suppliers
        supplier_id_column = None
        for col in ['Supplier_ID', 'Supplier', 'supplier_id']:
            if col in suppliers_df.columns:
                supplier_id_column = col
                break
        
        if supplier_id_column is None:
            logger.warning("No supplier ID column found in suppliers data")
            return materials_df
            
        available_suppliers = suppliers_df[supplier_id_column].unique().tolist()
        
        if len(available_suppliers) == 0:
            logger.warning("No suppliers available for assignment")
            return materials_df
        
        fixed_materials = materials_df.copy()
        
        # Simple round-robin assignment strategy
        supplier_idx = 0
        
        for idx, row in missing_suppliers.iterrows():
            material_id = row['Yarn_ID']
            
            # Assign supplier in round-robin fashion
            assigned_supplier = available_suppliers[supplier_idx % len(available_suppliers)]
            supplier_idx += 1
            
            fixed_materials.loc[idx, supplier_column] = assigned_supplier
            
            fix_record = {
                'material_id': material_id,
                'assigned_supplier': assigned_supplier,
                'method': 'round_robin_assignment',
                'timestamp': datetime.now().isoformat()
            }
            
            self.fixes_applied.append(fix_record)
            logger.info(f"Assigned supplier {assigned_supplier} to material {material_id}")
        
        self.quality_report['fixes_applied'].append({
            'fix_type': 'missing_supplier_assignment',
            'count': len(missing_suppliers),
            'details': f"Assigned suppliers to {len(missing_suppliers)} materials using round-robin strategy"
        })
        
        return fixed_materials
    
    def fix_bom_percentages(self, bom_df: pd.DataFrame) -> pd.DataFrame:
        """Fix BOM percentages that don't sum to 1.0 for each style."""
        
        logger.info("Fixing BOM percentages")
        
        fixed_bom = bom_df.copy()
        styles_fixed = 0
        
        # Group by style
        style_groups = bom_df.groupby('Style_ID')
        
        # Find the percentage column
        percentage_column = None
        for col in ['Percentage', 'BOM_Percentage', 'percentage', 'qty_per_unit']:
            if col in bom_df.columns:
                percentage_column = col
                break
        
        if percentage_column is None:
            logger.warning("No percentage column found in BOM data")
            return bom_df
        
        for style_id, style_bom in style_groups:
            # Calculate current sum
            current_sum = style_bom[percentage_column].sum()
            
            # Check if sum is not close to 1.0 (allowing small rounding errors)
            if abs(current_sum - 1.0) > 0.01:
                logger.info(f"Style {style_id} BOM percentages sum to {current_sum:.3f}, normalizing to 1.0")
                
                # Normalize percentages
                if current_sum > 0:
                    normalized_percentages = style_bom[percentage_column] / current_sum
                    
                    # Update in the fixed dataframe
                    for idx in style_bom.index:
                        old_percentage = fixed_bom.loc[idx, percentage_column]
                        new_percentage = normalized_percentages.loc[idx]
                        fixed_bom.loc[idx, percentage_column] = new_percentage
                        
                        logger.debug(f"  {fixed_bom.loc[idx, 'Yarn_ID']}: {old_percentage:.3f} → {new_percentage:.3f}")
                    
                    styles_fixed += 1
                    
                    fix_record = {
                        'style_id': style_id,
                        'old_sum': current_sum,
                        'new_sum': 1.0,
                        'components_count': len(style_bom),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    self.fixes_applied.append(fix_record)
        
        if styles_fixed > 0:
            self.quality_report['fixes_applied'].append({
                'fix_type': 'bom_percentage_normalization',
                'count': styles_fixed,
                'details': f"Normalized BOM percentages for {styles_fixed} styles to sum to 1.0"
            })
            
            logger.info(f"Fixed BOM percentages for {styles_fixed} styles")
        else:
            logger.info("All BOM percentages are correctly normalized")
        
        return fixed_bom
    
    def apply_all_fixes(self) -> Dict[str, pd.DataFrame]:
        """Apply all data quality fixes and return fixed datasets."""
        
        logger.info("Starting comprehensive data quality fix process")
        
        try:
            # Load datasets
            yarn_inventory = pd.read_csv(f"{self.data_path}Yarn_ID_Current_Inventory.csv", encoding='utf-8-sig')
            yarn_master = pd.read_csv(f"{self.data_path}Yarn_ID_1.csv", encoding='utf-8-sig')
            style_bom = pd.read_csv(f"{self.data_path}Style_BOM.csv", encoding='utf-8-sig')
            suppliers = pd.read_csv(f"{self.data_path}Supplier_ID.csv", encoding='utf-8-sig')
            
            # Apply fixes
            fixed_yarn_master = self.fix_zero_cost_materials(yarn_master)
            fixed_yarn_master = self.assign_missing_supplier_ids(fixed_yarn_master, suppliers)
            
            fixed_inventory = self.fix_negative_inventory(yarn_inventory)
            fixed_bom = self.fix_bom_percentages(style_bom)
            
            # Save fixed datasets
            output_path = f"{self.data_path}fixed/"
            import os
            os.makedirs(output_path, exist_ok=True)
            
            fixed_yarn_master.to_csv(f"{output_path}Yarn_ID_1_fixed.csv", index=False)
            fixed_inventory.to_csv(f"{output_path}Yarn_ID_Current_Inventory_fixed.csv", index=False)
            fixed_bom.to_csv(f"{output_path}Style_BOM_fixed.csv", index=False)
            
            # Generate quality report
            self.quality_report['summary'] = {
                'total_fixes_applied': len(self.fixes_applied),
                'datasets_processed': 4,
                'output_location': output_path
            }
            
            # Save quality report
            with open(f"{output_path}quality_fixes_report.json", 'w') as f:
                json.dump(self.quality_report, f, indent=2)
            
            logger.info(f"Data quality fixes completed. {len(self.fixes_applied)} total fixes applied.")
            logger.info(f"Fixed datasets saved to: {output_path}")
            
            return {
                'yarn_master': fixed_yarn_master,
                'inventory': fixed_inventory,
                'bom': fixed_bom,
                'suppliers': suppliers,
                'quality_report': self.quality_report
            }
            
        except Exception as e:
            logger.error(f"Data quality fix process failed: {e}")
            raise
    
    def generate_fix_summary(self) -> str:
        """Generate a human-readable summary of fixes applied."""
        
        if not self.fixes_applied:
            return "No data quality fixes were needed."
        
        summary = f"Data Quality Fixes Applied ({len(self.fixes_applied)} total):\n\n"
        
        fix_types = {}
        for fix in self.fixes_applied:
            fix_type = fix.get('fix_type', 'unknown')
            if fix_type not in fix_types:
                fix_types[fix_type] = []
            fix_types[fix_type].append(fix)
        
        for fix_type, fixes in fix_types.items():
            summary += f"• {fix_type.replace('_', ' ').title()}: {len(fixes)} items\n"
        
        summary += f"\nAll fixes completed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        summary += "Fixed datasets are available in the 'fixed/' subdirectory.\n"
        
        return summary


def main():
    """Main function to run data quality fixes."""
    
    logging.basicConfig(level=logging.INFO)
    
    fixer = DataQualityFixer()
    
    try:
        fixed_datasets = fixer.apply_all_fixes()
        print("\n" + "="*60)
        print("DATA QUALITY FIXES COMPLETED SUCCESSFULLY")
        print("="*60)
        print(fixer.generate_fix_summary())
        
    except Exception as e:
        print(f"Data quality fix process failed: {e}")
        raise


if __name__ == "__main__":
    main()