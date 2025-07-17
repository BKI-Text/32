import pandas as pd
import numpy as np
from decimal import Decimal
from datetime import datetime, date, timedelta
import random
from typing import List, Dict, Any
from pathlib import Path

from ..core.domain import (
    Material, Supplier, SupplierMaterial, Inventory, BOM, Forecast,
    ProcurementRecommendation, MaterialType, ForecastSource, RiskLevel,
    Money, Quantity, MaterialId, SupplierId, SkuId, LeadTime
)

class SampleDataGenerator:
    """Generate sample data for testing and demonstration purposes"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Predefined data for realistic generation
        self.yarn_types = [
            {"name": "Cotton Yarn", "type": MaterialType.YARN, "blend": "Cotton", "color": "Natural"},
            {"name": "Polyester Yarn", "type": MaterialType.YARN, "blend": "Polyester", "color": "White"},
            {"name": "Wool Yarn", "type": MaterialType.YARN, "blend": "Wool", "color": "Gray"},
            {"name": "Silk Yarn", "type": MaterialType.YARN, "blend": "Silk", "color": "Cream"},
            {"name": "Acrylic Yarn", "type": MaterialType.YARN, "blend": "Acrylic", "color": "Blue"},
            {"name": "Bamboo Yarn", "type": MaterialType.YARN, "blend": "Bamboo", "color": "Green"},
            {"name": "Linen Yarn", "type": MaterialType.YARN, "blend": "Linen", "color": "Beige"},
            {"name": "Cashmere Yarn", "type": MaterialType.YARN, "blend": "Cashmere", "color": "Ivory"}
        ]
        
        self.suppliers = [
            {"name": "ABC Textiles", "type": "Import", "reliability": 0.85, "lead_time": 14},
            {"name": "XYZ Materials", "type": "Domestic", "reliability": 0.90, "lead_time": 10},
            {"name": "Global Yarn Co", "type": "Import", "reliability": 0.80, "lead_time": 21},
            {"name": "Premium Fibers", "type": "Domestic", "reliability": 0.95, "lead_time": 7},
            {"name": "EcoTextile Solutions", "type": "Import", "reliability": 0.75, "lead_time": 28},
            {"name": "Synthetic Specialists", "type": "Domestic", "reliability": 0.88, "lead_time": 12}
        ]
        
        self.style_names = [
            "Classic Crew Neck", "V-Neck Pullover", "Cardigan Sweater", 
            "Turtleneck", "Hoodie", "Zip-Up Jacket", "Sleeveless Vest",
            "Cable Knit Sweater", "Ribbed Pullover", "Cashmere Blend"
        ]
        
        self.forecast_sources = [
            ForecastSource.SALES_ORDER,
            ForecastSource.PROD_PLAN,
            ForecastSource.PROJECTION,
            ForecastSource.SALES_HISTORY
        ]
    
    def generate_materials(self, count: int = 20) -> List[Material]:
        """Generate sample materials"""
        materials = []
        
        for i in range(count):
            yarn_type = random.choice(self.yarn_types)
            
            material = Material(
                id=MaterialId(value=f"YARN-{i+1:03d}"),
                name=f"{yarn_type['name']} {i+1}",
                type=yarn_type['type'],
                description=f"High-quality {yarn_type['blend'].lower()} yarn for textile manufacturing",
                specifications={
                    "blend": yarn_type['blend'],
                    "color": yarn_type['color'],
                    "weight": f"{random.randint(150, 300)}g",
                    "fiber_content": f"{random.randint(95, 100)}% {yarn_type['blend']}"
                },
                is_critical=random.choice([True, False])
            )
            
            materials.append(material)
        
        return materials
    
    def generate_suppliers(self, count: int = 10) -> List[Supplier]:
        """Generate sample suppliers"""
        suppliers = []
        
        for i in range(count):
            supplier_data = random.choice(self.suppliers)
            
            supplier = Supplier(
                id=SupplierId(value=f"SUP-{i+1:03d}"),
                name=f"{supplier_data['name']} {i+1}",
                contact_info=f"contact@{supplier_data['name'].lower().replace(' ', '')}.com",
                lead_time=LeadTime(days=supplier_data['lead_time'] + random.randint(-3, 3)),
                reliability_score=supplier_data['reliability'] + random.uniform(-0.05, 0.05),
                risk_level=self._calculate_risk_level(supplier_data['reliability']),
                is_active=True
            )
            
            suppliers.append(supplier)
        
        return suppliers
    
    def generate_supplier_materials(
        self, 
        materials: List[Material], 
        suppliers: List[Supplier]
    ) -> List[SupplierMaterial]:
        """Generate supplier-material relationships"""
        supplier_materials = []
        
        for material in materials:
            # Each material has 1-3 suppliers
            num_suppliers = random.randint(1, min(3, len(suppliers)))
            selected_suppliers = random.sample(suppliers, num_suppliers)
            
            base_cost = Decimal(str(random.uniform(3.0, 10.0)))
            
            for i, supplier in enumerate(selected_suppliers):
                # Vary cost by supplier
                cost_variation = Decimal(str(random.uniform(0.8, 1.2)))
                cost = base_cost * cost_variation
                
                supplier_material = SupplierMaterial(
                    supplier_id=supplier.id,
                    material_id=material.id,
                    cost_per_unit=Money(amount=cost, currency="USD"),
                    moq=Quantity(
                        amount=Decimal(str(random.randint(50, 200))),
                        unit="lb"
                    ),
                    lead_time=LeadTime(days=supplier.lead_time.days + random.randint(-2, 2)),
                    contract_qty_limit=Quantity(
                        amount=Decimal(str(random.randint(1000, 5000))),
                        unit="lb"
                    ) if random.random() < 0.3 else None,
                    reliability_score=supplier.reliability_score + random.uniform(-0.05, 0.05),
                    ordering_cost=Money(
                        amount=Decimal(str(random.randint(75, 150))),
                        currency="USD"
                    ),
                    holding_cost_rate=random.uniform(0.15, 0.35)
                )
                
                supplier_materials.append(supplier_material)
        
        return supplier_materials
    
    def generate_inventory(self, materials: List[Material]) -> List[Inventory]:
        """Generate sample inventory data"""
        inventory = []
        
        for material in materials:
            # Generate realistic inventory levels
            on_hand = Decimal(str(random.randint(0, 1000)))
            open_po = Decimal(str(random.randint(0, 500)))
            safety_stock = Decimal(str(random.randint(50, 200)))
            
            # Occasionally have zero or negative inventory
            if random.random() < 0.1:
                on_hand = Decimal("0")
            
            inventory_item = Inventory(
                material_id=material.id,
                on_hand_qty=Quantity(amount=on_hand, unit="lb"),
                open_po_qty=Quantity(amount=open_po, unit="lb"),
                po_expected_date=date.today() + timedelta(days=random.randint(5, 30)),
                safety_stock=Quantity(amount=safety_stock, unit="lb"),
                last_updated=datetime.now() - timedelta(days=random.randint(0, 7))
            )
            
            inventory.append(inventory_item)
        
        return inventory
    
    def generate_boms(self, materials: List[Material], num_styles: int = 15) -> List[BOM]:
        """Generate sample Bill of Materials"""
        boms = []
        
        for i in range(num_styles):
            style_id = SkuId(value=f"STYLE-{i+1:03d}")
            
            # Each style uses 1-4 materials
            num_materials = random.randint(1, min(4, len(materials)))
            selected_materials = random.sample(materials, num_materials)
            
            # Generate percentages that sum to 1.0
            percentages = [random.uniform(0.1, 0.8) for _ in range(num_materials)]
            total = sum(percentages)
            percentages = [p/total for p in percentages]
            
            for material, percentage in zip(selected_materials, percentages):
                bom = BOM(
                    sku_id=style_id,
                    material_id=material.id,
                    qty_per_unit=Quantity(
                        amount=Decimal(str(round(percentage, 3))),
                        unit="ratio"
                    ),
                    unit="ratio"
                )
                
                boms.append(bom)
        
        return boms
    
    def generate_forecasts(self, boms: List[BOM], weeks: int = 12) -> List[Forecast]:
        """Generate sample forecasts"""
        forecasts = []
        
        # Get unique style IDs
        style_ids = list(set(bom.sku_id.value for bom in boms))
        
        for style_id in style_ids:
            for week in range(weeks):
                forecast_date = date.today() + timedelta(weeks=week)
                
                # Generate multiple forecast sources for some styles
                num_sources = random.randint(1, 3)
                selected_sources = random.sample(self.forecast_sources, num_sources)
                
                for source in selected_sources:
                    base_qty = random.randint(50, 500)
                    
                    # Vary quantity by source reliability
                    if source == ForecastSource.SALES_ORDER:
                        qty_variation = 1.0
                        confidence = random.uniform(0.85, 0.95)
                    elif source == ForecastSource.PROD_PLAN:
                        qty_variation = random.uniform(0.8, 1.2)
                        confidence = random.uniform(0.80, 0.90)
                    elif source == ForecastSource.PROJECTION:
                        qty_variation = random.uniform(0.6, 1.4)
                        confidence = random.uniform(0.60, 0.80)
                    else:  # SALES_HISTORY
                        qty_variation = random.uniform(0.7, 1.3)
                        confidence = random.uniform(0.70, 0.85)
                    
                    forecast_qty = int(base_qty * qty_variation)
                    
                    forecast = Forecast(
                        sku_id=SkuId(value=style_id),
                        forecast_qty=Quantity(amount=Decimal(str(forecast_qty)), unit="unit"),
                        forecast_date=forecast_date,
                        source=source,
                        confidence_score=confidence,
                        created_at=datetime.now() - timedelta(days=random.randint(0, 7))
                    )
                    
                    forecasts.append(forecast)
        
        return forecasts
    
    def generate_complete_dataset(self) -> Dict[str, Any]:
        """Generate a complete dataset with all related objects"""
        print("Generating sample data for Beverly Knits AI Supply Chain Planner...")
        
        # Generate base entities
        materials = self.generate_materials(25)
        suppliers = self.generate_suppliers(12)
        
        # Generate relationships
        supplier_materials = self.generate_supplier_materials(materials, suppliers)
        inventory = self.generate_inventory(materials)
        boms = self.generate_boms(materials, 20)
        forecasts = self.generate_forecasts(boms, 16)
        
        dataset = {
            'materials': materials,
            'suppliers': suppliers,
            'supplier_materials': supplier_materials,
            'inventory': inventory,
            'boms': boms,
            'forecasts': forecasts
        }
        
        # Print summary
        print(f"Generated dataset summary:")
        print(f"  - Materials: {len(materials)}")
        print(f"  - Suppliers: {len(suppliers)}")
        print(f"  - Supplier-Material relationships: {len(supplier_materials)}")
        print(f"  - Inventory items: {len(inventory)}")
        print(f"  - BOM entries: {len(boms)}")
        print(f"  - Forecasts: {len(forecasts)}")
        
        return dataset
    
    def save_sample_csvs(self, dataset: Dict[str, Any], output_dir: str = "data/sample/"):
        """Save sample data as CSV files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrames and save
        self._save_materials_csv(dataset['materials'], output_path / "materials.csv")
        self._save_suppliers_csv(dataset['suppliers'], output_path / "suppliers.csv")
        self._save_supplier_materials_csv(dataset['supplier_materials'], output_path / "supplier_materials.csv")
        self._save_inventory_csv(dataset['inventory'], output_path / "inventory.csv")
        self._save_boms_csv(dataset['boms'], output_path / "boms.csv")
        self._save_forecasts_csv(dataset['forecasts'], output_path / "forecasts.csv")
        
        print(f"Sample CSV files saved to {output_path}")
    
    def _save_materials_csv(self, materials: List[Material], filepath: Path):
        """Save materials as CSV"""
        data = []
        for material in materials:
            data.append({
                'material_id': material.id.value,
                'name': material.name,
                'type': material.type.value,
                'description': material.description,
                'blend': material.specifications.get('blend', ''),
                'color': material.specifications.get('color', ''),
                'weight': material.specifications.get('weight', ''),
                'is_critical': material.is_critical
            })
        
        pd.DataFrame(data).to_csv(filepath, index=False)
    
    def _save_suppliers_csv(self, suppliers: List[Supplier], filepath: Path):
        """Save suppliers as CSV"""
        data = []
        for supplier in suppliers:
            data.append({
                'supplier_id': supplier.id.value,
                'name': supplier.name,
                'contact_info': supplier.contact_info,
                'lead_time_days': supplier.lead_time.days,
                'reliability_score': supplier.reliability_score,
                'risk_level': supplier.risk_level.value,
                'is_active': supplier.is_active
            })
        
        pd.DataFrame(data).to_csv(filepath, index=False)
    
    def _save_supplier_materials_csv(self, supplier_materials: List[SupplierMaterial], filepath: Path):
        """Save supplier materials as CSV"""
        data = []
        for sm in supplier_materials:
            data.append({
                'supplier_id': sm.supplier_id.value,
                'material_id': sm.material_id.value,
                'cost_per_unit': float(sm.cost_per_unit.amount),
                'currency': sm.cost_per_unit.currency,
                'moq': float(sm.moq.amount),
                'moq_unit': sm.moq.unit,
                'lead_time_days': sm.lead_time.days,
                'reliability_score': sm.reliability_score,
                'ordering_cost': float(sm.ordering_cost.amount),
                'holding_cost_rate': sm.holding_cost_rate
            })
        
        pd.DataFrame(data).to_csv(filepath, index=False)
    
    def _save_inventory_csv(self, inventory: List[Inventory], filepath: Path):
        """Save inventory as CSV"""
        data = []
        for inv in inventory:
            data.append({
                'material_id': inv.material_id.value,
                'on_hand_qty': float(inv.on_hand_qty.amount),
                'unit': inv.on_hand_qty.unit,
                'open_po_qty': float(inv.open_po_qty.amount),
                'po_expected_date': inv.po_expected_date.isoformat() if inv.po_expected_date else None,
                'safety_stock': float(inv.safety_stock.amount),
                'last_updated': inv.last_updated.isoformat()
            })
        
        pd.DataFrame(data).to_csv(filepath, index=False)
    
    def _save_boms_csv(self, boms: List[BOM], filepath: Path):
        """Save BOMs as CSV"""
        data = []
        for bom in boms:
            data.append({
                'sku_id': bom.sku_id.value,
                'material_id': bom.material_id.value,
                'qty_per_unit': float(bom.qty_per_unit.amount),
                'unit': bom.unit
            })
        
        pd.DataFrame(data).to_csv(filepath, index=False)
    
    def _save_forecasts_csv(self, forecasts: List[Forecast], filepath: Path):
        """Save forecasts as CSV"""
        data = []
        for forecast in forecasts:
            data.append({
                'sku_id': forecast.sku_id.value,
                'forecast_qty': float(forecast.forecast_qty.amount),
                'unit': forecast.forecast_qty.unit,
                'forecast_date': forecast.forecast_date.isoformat(),
                'source': forecast.source.value,
                'confidence_score': forecast.confidence_score,
                'created_at': forecast.created_at.isoformat()
            })
        
        pd.DataFrame(data).to_csv(filepath, index=False)
    
    def _calculate_risk_level(self, reliability: float) -> RiskLevel:
        """Calculate risk level based on reliability score"""
        if reliability >= 0.9:
            return RiskLevel.LOW
        elif reliability >= 0.8:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.HIGH

# Convenience function for external use
def generate_sample_data(save_csv: bool = True, output_dir: str = "data/sample/") -> Dict[str, Any]:
    """Generate and optionally save sample data"""
    generator = SampleDataGenerator()
    dataset = generator.generate_complete_dataset()
    
    if save_csv:
        generator.save_sample_csvs(dataset, output_dir)
    
    return dataset

if __name__ == "__main__":
    # Generate sample data when run directly
    dataset = generate_sample_data(save_csv=True)
    print("Sample data generation completed!")