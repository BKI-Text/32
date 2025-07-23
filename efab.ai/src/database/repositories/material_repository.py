"""Material Repository for Beverly Knits AI Supply Chain Planner"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import and_, or_
from decimal import Decimal

from .base_repository import BaseRepository
from ..models.material import MaterialModel
from ..models.supplier import SupplierMaterialModel
from ..models.inventory import InventoryModel
from ..models.bom import BOMModel

class MaterialRepository(BaseRepository[MaterialModel]):
    """Repository for Material operations"""
    
    def __init__(self):
        super().__init__(MaterialModel)
    
    def get_by_name(self, name: str) -> Optional[MaterialModel]:
        """Get material by name"""
        with self.get_session() as session:
            try:
                return session.query(MaterialModel).filter(
                    MaterialModel.name == name
                ).first()
            except SQLAlchemyError as e:
                raise e
    
    def get_by_type(self, material_type: str, skip: int = 0, limit: int = 100) -> List[MaterialModel]:
        """Get materials by type"""
        with self.get_session() as session:
            try:
                return session.query(MaterialModel).filter(
                    MaterialModel.type == material_type
                ).offset(skip).limit(limit).all()
            except SQLAlchemyError as e:
                raise e
    
    def get_critical_materials(self, skip: int = 0, limit: int = 100) -> List[MaterialModel]:
        """Get critical materials"""
        with self.get_session() as session:
            try:
                return session.query(MaterialModel).filter(
                    MaterialModel.is_critical == True
                ).offset(skip).limit(limit).all()
            except SQLAlchemyError as e:
                raise e
    
    def search_materials(self, search_term: str, skip: int = 0, limit: int = 100) -> List[MaterialModel]:
        """Search materials by name or description"""
        with self.get_session() as session:
            try:
                return session.query(MaterialModel).filter(
                    or_(
                        MaterialModel.name.ilike(f"%{search_term}%"),
                        MaterialModel.description.ilike(f"%{search_term}%")
                    )
                ).offset(skip).limit(limit).all()
            except SQLAlchemyError as e:
                raise e
    
    def get_materials_with_suppliers(self, skip: int = 0, limit: int = 100) -> List[MaterialModel]:
        """Get materials with their suppliers loaded"""
        with self.get_session() as session:
            try:
                return session.query(MaterialModel).options(
                    joinedload(MaterialModel.supplier_materials)
                ).offset(skip).limit(limit).all()
            except SQLAlchemyError as e:
                raise e
    
    def get_materials_with_inventory(self, skip: int = 0, limit: int = 100) -> List[MaterialModel]:
        """Get materials with their inventory loaded"""
        with self.get_session() as session:
            try:
                return session.query(MaterialModel).options(
                    joinedload(MaterialModel.inventory_records)
                ).offset(skip).limit(limit).all()
            except SQLAlchemyError as e:
                raise e
    
    def get_materials_without_suppliers(self, skip: int = 0, limit: int = 100) -> List[MaterialModel]:
        """Get materials that have no suppliers"""
        with self.get_session() as session:
            try:
                return session.query(MaterialModel).filter(
                    ~MaterialModel.supplier_materials.any()
                ).offset(skip).limit(limit).all()
            except SQLAlchemyError as e:
                raise e
    
    def get_materials_low_stock(self, skip: int = 0, limit: int = 100) -> List[MaterialModel]:
        """Get materials with low stock"""
        with self.get_session() as session:
            try:
                return session.query(MaterialModel).join(
                    InventoryModel
                ).filter(
                    InventoryModel.on_hand_qty < InventoryModel.safety_stock
                ).offset(skip).limit(limit).all()
            except SQLAlchemyError as e:
                raise e
    
    def get_materials_out_of_stock(self, skip: int = 0, limit: int = 100) -> List[MaterialModel]:
        """Get materials that are out of stock"""
        with self.get_session() as session:
            try:
                return session.query(MaterialModel).join(
                    InventoryModel
                ).filter(
                    InventoryModel.on_hand_qty <= 0
                ).offset(skip).limit(limit).all()
            except SQLAlchemyError as e:
                raise e
    
    def get_materials_by_supplier(self, supplier_id: str, skip: int = 0, limit: int = 100) -> List[MaterialModel]:
        """Get materials supplied by a specific supplier"""
        with self.get_session() as session:
            try:
                return session.query(MaterialModel).join(
                    SupplierMaterialModel
                ).filter(
                    SupplierMaterialModel.supplier_id == supplier_id
                ).offset(skip).limit(limit).all()
            except SQLAlchemyError as e:
                raise e
    
    def get_materials_used_in_bom(self, sku_id: str) -> List[MaterialModel]:
        """Get materials used in a specific BOM"""
        with self.get_session() as session:
            try:
                return session.query(MaterialModel).join(
                    BOMModel
                ).filter(
                    BOMModel.sku_id == sku_id
                ).all()
            except SQLAlchemyError as e:
                raise e
    
    def get_materials_with_specifications(self, spec_key: str, spec_value: str = None) -> List[MaterialModel]:
        """Get materials with specific specifications"""
        with self.get_session() as session:
            try:
                if spec_value:
                    return session.query(MaterialModel).filter(
                        MaterialModel.specifications[spec_key].astext == spec_value
                    ).all()
                else:
                    return session.query(MaterialModel).filter(
                        MaterialModel.specifications.has_key(spec_key)
                    ).all()
            except SQLAlchemyError as e:
                raise e
    
    def get_material_costs(self, material_id: str) -> List[Dict[str, Any]]:
        """Get all supplier costs for a material"""
        with self.get_session() as session:
            try:
                costs = session.query(SupplierMaterialModel).filter(
                    SupplierMaterialModel.material_id == material_id
                ).all()
                
                return [
                    {
                        "supplier_id": cost.supplier_id,
                        "cost_per_unit": cost.cost_per_unit,
                        "currency": cost.currency,
                        "moq_amount": cost.moq_amount,
                        "moq_unit": cost.moq_unit,
                        "lead_time_days": cost.lead_time_days,
                        "reliability_score": cost.reliability_score
                    }
                    for cost in costs
                ]
            except SQLAlchemyError as e:
                raise e
    
    def get_lowest_cost_materials(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get materials with their lowest costs"""
        with self.get_session() as session:
            try:
                # This is a complex query that would need proper SQL
                # For now, we'll implement a simplified version
                results = []
                materials = session.query(MaterialModel).limit(limit).all()
                
                for material in materials:
                    if material.supplier_materials:
                        lowest_cost = min(
                            material.supplier_materials, 
                            key=lambda sm: sm.cost_per_unit
                        )
                        results.append({
                            "material": material,
                            "lowest_cost": lowest_cost.cost_per_unit,
                            "supplier_id": lowest_cost.supplier_id,
                            "currency": lowest_cost.currency
                        })
                
                return sorted(results, key=lambda x: x["lowest_cost"])
            except SQLAlchemyError as e:
                raise e
    
    def get_materials_by_filters(self, filters: Dict[str, Any]) -> List[MaterialModel]:
        """Get materials by complex filters"""
        with self.get_session() as session:
            try:
                query = session.query(MaterialModel)
                
                # Apply material-specific filters
                if 'name' in filters:
                    query = query.filter(MaterialModel.name.ilike(f"%{filters['name']}%"))
                
                if 'type' in filters:
                    query = query.filter(MaterialModel.type == filters['type'])
                
                if 'is_critical' in filters:
                    query = query.filter(MaterialModel.is_critical == filters['is_critical'])
                
                if 'has_suppliers' in filters:
                    if filters['has_suppliers']:
                        query = query.filter(MaterialModel.supplier_materials.any())
                    else:
                        query = query.filter(~MaterialModel.supplier_materials.any())
                
                if 'max_cost' in filters:
                    query = query.join(SupplierMaterialModel).filter(
                        SupplierMaterialModel.cost_per_unit <= filters['max_cost']
                    )
                
                if 'min_inventory' in filters:
                    query = query.join(InventoryModel).filter(
                        InventoryModel.on_hand_qty >= filters['min_inventory']
                    )
                
                return query.all()
            except SQLAlchemyError as e:
                raise e
    
    def update_material_criticality(self, material_id: str, is_critical: bool) -> bool:
        """Update material criticality status"""
        with self.get_session() as session:
            try:
                material = session.query(MaterialModel).filter(
                    MaterialModel.id == material_id
                ).first()
                
                if material:
                    material.is_critical = is_critical
                    session.commit()
                    return True
                return False
            except SQLAlchemyError as e:
                session.rollback()
                raise e
    
    def add_material_specification(self, material_id: str, spec_key: str, spec_value: Any) -> bool:
        """Add specification to material"""
        with self.get_session() as session:
            try:
                material = session.query(MaterialModel).filter(
                    MaterialModel.id == material_id
                ).first()
                
                if material:
                    material.add_specification(spec_key, spec_value)
                    session.commit()
                    return True
                return False
            except SQLAlchemyError as e:
                session.rollback()
                raise e
    
    def remove_material_specification(self, material_id: str, spec_key: str) -> bool:
        """Remove specification from material"""
        with self.get_session() as session:
            try:
                material = session.query(MaterialModel).filter(
                    MaterialModel.id == material_id
                ).first()
                
                if material:
                    material.remove_specification(spec_key)
                    session.commit()
                    return True
                return False
            except SQLAlchemyError as e:
                session.rollback()
                raise e
    
    def get_material_statistics(self) -> Dict[str, Any]:
        """Get material statistics"""
        with self.get_session() as session:
            try:
                total_materials = session.query(MaterialModel).count()
                critical_materials = session.query(MaterialModel).filter(
                    MaterialModel.is_critical == True
                ).count()
                
                # Count by type
                type_counts = {}
                types = session.query(MaterialModel.type).distinct().all()
                for (material_type,) in types:
                    count = session.query(MaterialModel).filter(
                        MaterialModel.type == material_type
                    ).count()
                    type_counts[material_type] = count
                
                # Materials without suppliers
                no_suppliers = session.query(MaterialModel).filter(
                    ~MaterialModel.supplier_materials.any()
                ).count()
                
                return {
                    "total_materials": total_materials,
                    "critical_materials": critical_materials,
                    "critical_percentage": (critical_materials / total_materials * 100) if total_materials > 0 else 0,
                    "type_distribution": type_counts,
                    "materials_without_suppliers": no_suppliers
                }
            except SQLAlchemyError as e:
                raise e