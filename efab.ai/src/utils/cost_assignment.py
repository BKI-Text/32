"""
Cost Assignment Utility
Assigns realistic costs to materials that are missing cost information.
"""

from decimal import Decimal
from typing import List, Dict, Any, Optional
import pandas as pd
import logging
from datetime import datetime
import json
import random

logger = logging.getLogger(__name__)

class CostAssigner:
    """Utility to assign realistic costs to materials missing cost information."""
    
    def __init__(self, data_path: str = "data/live/"):
        self.data_path = data_path
        self.cost_assignments = []
        
        # Industry-standard cost ranges by material type (USD per pound)
        self.cost_ranges = {
            'COTTON': (12.00, 18.00),      # Cotton yarns
            'POLYESTER': (6.00, 12.00),    # Synthetic yarns
            'WOOL': (20.00, 35.00),        # Wool yarns
            'SILK': (35.00, 60.00),        # Silk yarns
            'LINEN': (15.00, 25.00),       # Linen yarns
            'CASHMERE': (80.00, 150.00),   # Luxury cashmere
            'MOHAIR': (25.00, 45.00),      # Mohair yarns
            'ALPACA': (30.00, 50.00),      # Alpaca fibers
            'BAMBOO': (10.00, 16.00),      # Bamboo yarns
            'ACRYLIC': (4.00, 8.00),       # Acrylic yarns
            'NYLON': (8.00, 14.00),        # Nylon yarns
            'BLEND': (10.00, 20.00),       # Blended yarns
            'PREMIUM': (25.00, 45.00),     # Premium yarns
            'ORGANIC': (18.00, 28.00),     # Organic yarns
            'RECYCLED': (8.00, 15.00),     # Recycled materials
            'SPECIALTY': (20.00, 40.00),   # Specialty yarns
            'DEFAULT': (10.00, 18.00)      # Default range
        }
        
        # Quality multipliers
        self.quality_multipliers = {
            'PREMIUM': 1.4,
            'LUXURY': 1.6,
            'DELUXE': 1.3,
            'SUPERIOR': 1.2,
            'HIGH': 1.15,
            'FINE': 1.1,
            'BASIC': 0.8,
            'ECONOMY': 0.7,
            'BUDGET': 0.6
        }
    
    def analyze_material_description(self, description: str) -> Dict[str, Any]:
        """Analyze material description to determine appropriate cost range."""
        
        description_upper = str(description).upper()
        
        # Determine base material type
        base_material = 'DEFAULT'
        for material_type in self.cost_ranges:
            if material_type in description_upper:
                base_material = material_type
                break
        
        # Determine quality level
        quality_multiplier = 1.0
        for quality, multiplier in self.quality_multipliers.items():
            if quality in description_upper:
                quality_multiplier = multiplier
                break
        
        # Check for blend indicators
        is_blend = any(keyword in description_upper for keyword in ['BLEND', 'MIX', '/', '%'])
        
        return {
            'base_material': base_material,
            'quality_multiplier': quality_multiplier,
            'is_blend': is_blend,
            'description': description
        }
    
    def calculate_cost(self, material_analysis: Dict[str, Any]) -> float:
        """Calculate appropriate cost based on material analysis."""
        
        base_material = material_analysis['base_material']
        quality_multiplier = material_analysis['quality_multiplier']
        is_blend = material_analysis['is_blend']
        
        # Get base cost range
        cost_range = self.cost_ranges.get(base_material, self.cost_ranges['DEFAULT'])
        min_cost, max_cost = cost_range
        
        # Apply blend adjustment (blends typically cost less than pure materials)
        if is_blend and base_material != 'BLEND':
            min_cost *= 0.85
            max_cost *= 0.85
        
        # Apply quality multiplier
        min_cost *= quality_multiplier
        max_cost *= quality_multiplier
        
        # Generate random cost within range
        cost = random.uniform(min_cost, max_cost)
        
        # Round to 2 decimal places
        return round(cost, 2)
    
    def assign_costs_to_materials(self, materials_df: pd.DataFrame) -> pd.DataFrame:
        """Assign costs to materials that are missing cost information."""
        
        # Find cost column
        cost_column = None
        for col in ['Cost_Pound', 'Unit_Cost', 'Cost', 'cost_per_unit']:
            if col in materials_df.columns:
                cost_column = col
                break
        
        if cost_column is None:
            logger.error("No cost column found in materials data")
            return materials_df
        
        # Convert cost column to numeric
        materials_df[cost_column] = pd.to_numeric(materials_df[cost_column], errors='coerce')
        materials_df[cost_column] = materials_df[cost_column].fillna(0)
        
        # Find materials with zero or missing costs
        missing_cost_materials = materials_df[materials_df[cost_column] == 0.0].copy()
        
        if len(missing_cost_materials) == 0:
            logger.info("All materials already have cost information")
            return materials_df
        
        logger.info(f"Assigning costs to {len(missing_cost_materials)} materials")
        
        updated_materials = materials_df.copy()
        
        for idx, row in missing_cost_materials.iterrows():
            material_id = row['Yarn_ID']
            description = row.get('Description', '')
            blend = row.get('Blend', '')
            type_info = row.get('Type', '')
            
            # Combine description fields for analysis
            full_description = f"{description} {blend} {type_info}".strip()
            
            # Analyze material to determine appropriate cost
            material_analysis = self.analyze_material_description(full_description)
            assigned_cost = self.calculate_cost(material_analysis)
            
            # Update the dataframe
            updated_materials.loc[idx, cost_column] = assigned_cost
            
            # Record the assignment
            assignment_record = {
                'material_id': material_id,
                'description': full_description,
                'base_material': material_analysis['base_material'],
                'quality_multiplier': material_analysis['quality_multiplier'],
                'is_blend': material_analysis['is_blend'],
                'assigned_cost': assigned_cost,
                'cost_column': cost_column,
                'timestamp': datetime.now().isoformat()
            }
            
            self.cost_assignments.append(assignment_record)
            
            logger.info(f"Assigned cost to {material_id}: ${assigned_cost:.2f} (based on {material_analysis['base_material']})")
        
        return updated_materials
    
    def create_supplier_material_mappings(self, materials_df: pd.DataFrame) -> pd.DataFrame:
        """Create supplier-material mapping with the assigned costs."""
        
        supplier_mappings = []
        
        for _, row in materials_df.iterrows():
            material_id = row['Yarn_ID']
            supplier = row.get('Supplier', 'DEFAULT_SUPPLIER')
            cost = row.get('Cost_Pound', 0.0)
            
            if cost > 0:
                mapping = {
                    'material_id': material_id,
                    'supplier_id': supplier,
                    'cost_per_unit': cost,
                    'currency': 'USD',
                    'moq': 50,  # Default minimum order quantity
                    'lead_time_days': 14,  # Default lead time
                    'reliability_score': 0.85,  # Default reliability
                    'ordering_cost': 25.00,  # Default ordering cost
                    'holding_cost_rate': 0.20  # Default holding cost rate
                }
                supplier_mappings.append(mapping)
        
        return pd.DataFrame(supplier_mappings)
    
    def run_cost_assignment(self) -> Dict[str, Any]:
        """Run the complete cost assignment process."""
        
        logger.info("Starting cost assignment process")
        
        try:
            # Load materials data
            materials_file = f"{self.data_path}Yarn_ID_Current_Inventory.csv"
            materials_df = pd.read_csv(materials_file, encoding='utf-8-sig')
            
            logger.info(f"Loaded {len(materials_df)} materials from {materials_file}")
            
            # Assign costs
            updated_materials = self.assign_costs_to_materials(materials_df)
            
            # Create supplier mappings
            supplier_mappings = self.create_supplier_material_mappings(updated_materials)
            
            # Save updated data
            output_path = f"{self.data_path}cost_assigned/"
            import os
            os.makedirs(output_path, exist_ok=True)
            
            updated_materials.to_csv(f"{output_path}materials_with_costs.csv", index=False)
            supplier_mappings.to_csv(f"{output_path}supplier_material_mappings.csv", index=False)
            
            # Save assignment report
            assignment_report = {
                'timestamp': datetime.now().isoformat(),
                'total_materials': len(materials_df),
                'materials_assigned_costs': len(self.cost_assignments),
                'supplier_mappings_created': len(supplier_mappings),
                'assignments': self.cost_assignments
            }
            
            with open(f"{output_path}cost_assignment_report.json", 'w') as f:
                json.dump(assignment_report, f, indent=2)
            
            logger.info(f"Cost assignment completed. Assigned costs to {len(self.cost_assignments)} materials")
            logger.info(f"Created {len(supplier_mappings)} supplier-material mappings")
            logger.info(f"Output saved to: {output_path}")
            
            return {
                'updated_materials': updated_materials,
                'supplier_mappings': supplier_mappings,
                'assignment_report': assignment_report
            }
            
        except Exception as e:
            logger.error(f"Cost assignment process failed: {e}")
            raise
    
    def generate_assignment_summary(self) -> str:
        """Generate a summary of cost assignments."""
        
        if not self.cost_assignments:
            return "No cost assignments were made."
        
        summary = f"Cost Assignment Summary ({len(self.cost_assignments)} materials):\n\n"
        
        # Group by base material type
        material_types = {}
        total_cost = 0
        
        for assignment in self.cost_assignments:
            base_material = assignment['base_material']
            cost = assignment['assigned_cost']
            
            if base_material not in material_types:
                material_types[base_material] = {'count': 0, 'total_cost': 0, 'costs': []}
            
            material_types[base_material]['count'] += 1
            material_types[base_material]['total_cost'] += cost
            material_types[base_material]['costs'].append(cost)
            total_cost += cost
        
        for material_type, data in material_types.items():
            avg_cost = data['total_cost'] / data['count']
            min_cost = min(data['costs'])
            max_cost = max(data['costs'])
            
            summary += f"• {material_type}: {data['count']} materials\n"
            summary += f"  Average cost: ${avg_cost:.2f}, Range: ${min_cost:.2f} - ${max_cost:.2f}\n\n"
        
        summary += f"Total estimated value: ${total_cost:.2f}\n"
        summary += f"Average cost per material: ${total_cost / len(self.cost_assignments):.2f}\n"
        
        return summary


def main():
    """Main function to run cost assignment."""
    
    logging.basicConfig(level=logging.INFO)
    random.seed(42)  # For reproducible results
    
    assigner = CostAssigner()
    
    try:
        result = assigner.run_cost_assignment()
        
        print("\n" + "="*60)
        print("COST ASSIGNMENT COMPLETED SUCCESSFULLY")
        print("="*60)
        print(assigner.generate_assignment_summary())
        
    except Exception as e:
        print(f"Cost assignment process failed: {e}")
        raise


if __name__ == "__main__":
    main()