"""Supplier Repository for Beverly Knits AI Supply Chain Planner"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import and_, or_, func
from decimal import Decimal

from .base_repository import BaseRepository
from ..models.supplier import SupplierModel, SupplierMaterialModel
from ..models.material import MaterialModel

class SupplierRepository(BaseRepository[SupplierModel]):
    """Repository for Supplier operations"""
    
    def __init__(self):
        super().__init__(SupplierModel)
    
    def get_by_name(self, name: str) -> Optional[SupplierModel]:
        """Get supplier by name"""
        with self.get_session() as session:
            try:
                return session.query(SupplierModel).filter(
                    SupplierModel.name == name
                ).first()
            except SQLAlchemyError as e:
                raise e
    
    def get_active_suppliers(self, skip: int = 0, limit: int = 100) -> List[SupplierModel]:
        """Get active suppliers"""
        with self.get_session() as session:
            try:
                return session.query(SupplierModel).filter(
                    SupplierModel.is_active == True
                ).offset(skip).limit(limit).all()
            except SQLAlchemyError as e:
                raise e
    
    def get_by_risk_level(self, risk_level: str, skip: int = 0, limit: int = 100) -> List[SupplierModel]:
        """Get suppliers by risk level"""
        with self.get_session() as session:
            try:
                return session.query(SupplierModel).filter(
                    SupplierModel.risk_level == risk_level
                ).offset(skip).limit(limit).all()
            except SQLAlchemyError as e:
                raise e
    
    def get_high_reliability_suppliers(self, threshold: float = 0.8, skip: int = 0, limit: int = 100) -> List[SupplierModel]:
        """Get suppliers with high reliability scores"""
        with self.get_session() as session:
            try:
                return session.query(SupplierModel).filter(
                    SupplierModel.reliability_score >= threshold
                ).offset(skip).limit(limit).all()
            except SQLAlchemyError as e:
                raise e
    
    def get_suppliers_with_materials(self, skip: int = 0, limit: int = 100) -> List[SupplierModel]:
        """Get suppliers with their materials loaded"""
        with self.get_session() as session:
            try:
                return session.query(SupplierModel).options(
                    joinedload(SupplierModel.supplier_materials)
                ).offset(skip).limit(limit).all()
            except SQLAlchemyError as e:
                raise e
    
    def get_suppliers_for_material(self, material_id: str) -> List[SupplierModel]:
        """Get suppliers for a specific material"""
        with self.get_session() as session:
            try:
                return session.query(SupplierModel).join(
                    SupplierMaterialModel
                ).filter(
                    SupplierMaterialModel.material_id == material_id
                ).all()
            except SQLAlchemyError as e:
                raise e
    
    def get_suppliers_by_lead_time(self, max_lead_time: int, skip: int = 0, limit: int = 100) -> List[SupplierModel]:
        """Get suppliers with lead time below threshold"""
        with self.get_session() as session:
            try:
                return session.query(SupplierModel).filter(
                    SupplierModel.lead_time_days <= max_lead_time
                ).offset(skip).limit(limit).all()
            except SQLAlchemyError as e:
                raise e
    
    def search_suppliers(self, search_term: str, skip: int = 0, limit: int = 100) -> List[SupplierModel]:
        """Search suppliers by name or contact info"""
        with self.get_session() as session:
            try:
                return session.query(SupplierModel).filter(
                    or_(
                        SupplierModel.name.ilike(f"%{search_term}%"),
                        SupplierModel.contact_info.ilike(f"%{search_term}%")
                    )
                ).offset(skip).limit(limit).all()
            except SQLAlchemyError as e:
                raise e
    
    def get_best_suppliers_for_material(self, material_id: str, criteria: str = "cost") -> List[Dict[str, Any]]:
        """Get best suppliers for a material based on criteria"""
        with self.get_session() as session:
            try:
                query = session.query(SupplierModel, SupplierMaterialModel).join(
                    SupplierMaterialModel
                ).filter(
                    SupplierMaterialModel.material_id == material_id,
                    SupplierModel.is_active == True
                )
                
                results = query.all()
                
                supplier_data = []
                for supplier, supplier_material in results:
                    supplier_data.append({
                        "supplier": supplier,
                        "supplier_material": supplier_material,
                        "cost_per_unit": supplier_material.cost_per_unit,
                        "reliability_score": supplier_material.reliability_score,
                        "lead_time_days": supplier_material.lead_time_days,
                        "combined_score": self._calculate_supplier_score(supplier, supplier_material)
                    })
                
                # Sort by criteria
                if criteria == "cost":
                    supplier_data.sort(key=lambda x: x["cost_per_unit"])
                elif criteria == "reliability":
                    supplier_data.sort(key=lambda x: x["reliability_score"], reverse=True)
                elif criteria == "lead_time":
                    supplier_data.sort(key=lambda x: x["lead_time_days"])
                elif criteria == "combined":
                    supplier_data.sort(key=lambda x: x["combined_score"], reverse=True)
                
                return supplier_data
            except SQLAlchemyError as e:
                raise e
    
    def _calculate_supplier_score(self, supplier: SupplierModel, supplier_material: SupplierMaterialModel) -> float:
        """Calculate combined supplier score"""
        # Normalize reliability (0-1)
        reliability_score = supplier_material.reliability_score
        
        # Normalize lead time (inverse, shorter is better)
        lead_time_score = max(0, 1 - (supplier_material.lead_time_days / 365))
        
        # Normalize cost (inverse, lower is better) - simplified
        cost_score = max(0, 1 - (float(supplier_material.cost_per_unit) / 1000))
        
        # Risk factor
        risk_factors = {"low": 1.0, "medium": 0.8, "high": 0.5}
        risk_score = risk_factors.get(supplier.risk_level, 0.5)
        
        # Weighted average
        return (reliability_score * 0.3 + lead_time_score * 0.2 + cost_score * 0.3 + risk_score * 0.2)
    
    def get_supplier_performance_metrics(self, supplier_id: str) -> Dict[str, Any]:
        """Get performance metrics for a supplier"""
        with self.get_session() as session:
            try:
                supplier = session.query(SupplierModel).filter(
                    SupplierModel.id == supplier_id
                ).first()
                
                if not supplier:
                    return {}
                
                # Get supplier materials
                supplier_materials = session.query(SupplierMaterialModel).filter(
                    SupplierMaterialModel.supplier_id == supplier_id
                ).all()
                
                if not supplier_materials:
                    return {
                        "supplier_id": supplier_id,
                        "material_count": 0,
                        "avg_cost": 0,
                        "avg_reliability": supplier.reliability_score,
                        "avg_lead_time": supplier.lead_time_days,
                        "risk_level": supplier.risk_level
                    }
                
                # Calculate averages
                avg_cost = sum(sm.cost_per_unit for sm in supplier_materials) / len(supplier_materials)
                avg_reliability = sum(sm.reliability_score for sm in supplier_materials) / len(supplier_materials)
                avg_lead_time = sum(sm.lead_time_days for sm in supplier_materials) / len(supplier_materials)
                
                return {
                    "supplier_id": supplier_id,
                    "supplier_name": supplier.name,
                    "material_count": len(supplier_materials),
                    "avg_cost": float(avg_cost),
                    "avg_reliability": avg_reliability,
                    "avg_lead_time": avg_lead_time,
                    "risk_level": supplier.risk_level,
                    "is_active": supplier.is_active,
                    "overall_score": self._calculate_overall_supplier_score(supplier, supplier_materials)
                }
            except SQLAlchemyError as e:
                raise e
    
    def _calculate_overall_supplier_score(self, supplier: SupplierModel, supplier_materials: List[SupplierMaterialModel]) -> float:
        """Calculate overall supplier score"""
        if not supplier_materials:
            return 0.0
        
        scores = [
            self._calculate_supplier_score(supplier, sm) 
            for sm in supplier_materials
        ]
        
        return sum(scores) / len(scores)
    
    def get_supplier_material_relationship(self, supplier_id: str, material_id: str) -> Optional[SupplierMaterialModel]:
        """Get supplier-material relationship"""
        with self.get_session() as session:
            try:
                return session.query(SupplierMaterialModel).filter(
                    SupplierMaterialModel.supplier_id == supplier_id,
                    SupplierMaterialModel.material_id == material_id
                ).first()
            except SQLAlchemyError as e:
                raise e
    
    def update_supplier_risk_level(self, supplier_id: str, new_risk_level: str) -> bool:
        """Update supplier risk level"""
        with self.get_session() as session:
            try:
                supplier = session.query(SupplierModel).filter(
                    SupplierModel.id == supplier_id
                ).first()
                
                if supplier:
                    supplier.update_risk_level(new_risk_level)
                    session.commit()
                    return True
                return False
            except SQLAlchemyError as e:
                session.rollback()
                raise e
    
    def update_supplier_reliability(self, supplier_id: str, new_reliability: float) -> bool:
        """Update supplier reliability score"""
        with self.get_session() as session:
            try:
                supplier = session.query(SupplierModel).filter(
                    SupplierModel.id == supplier_id
                ).first()
                
                if supplier:
                    supplier.update_reliability_score(new_reliability)
                    session.commit()
                    return True
                return False
            except SQLAlchemyError as e:
                session.rollback()
                raise e
    
    def activate_supplier(self, supplier_id: str) -> bool:
        """Activate supplier"""
        with self.get_session() as session:
            try:
                supplier = session.query(SupplierModel).filter(
                    SupplierModel.id == supplier_id
                ).first()
                
                if supplier:
                    supplier.activate()
                    session.commit()
                    return True
                return False
            except SQLAlchemyError as e:
                session.rollback()
                raise e
    
    def deactivate_supplier(self, supplier_id: str) -> bool:
        """Deactivate supplier"""
        with self.get_session() as session:
            try:
                supplier = session.query(SupplierModel).filter(
                    SupplierModel.id == supplier_id
                ).first()
                
                if supplier:
                    supplier.deactivate()
                    session.commit()
                    return True
                return False
            except SQLAlchemyError as e:
                session.rollback()
                raise e
    
    def get_supplier_statistics(self) -> Dict[str, Any]:
        """Get supplier statistics"""
        with self.get_session() as session:
            try:
                total_suppliers = session.query(SupplierModel).count()
                active_suppliers = session.query(SupplierModel).filter(
                    SupplierModel.is_active == True
                ).count()
                
                # Count by risk level
                risk_counts = {}
                for risk_level in ["low", "medium", "high"]:
                    count = session.query(SupplierModel).filter(
                        SupplierModel.risk_level == risk_level
                    ).count()
                    risk_counts[risk_level] = count
                
                # Average reliability
                avg_reliability = session.query(func.avg(SupplierModel.reliability_score)).scalar() or 0
                
                # Average lead time
                avg_lead_time = session.query(func.avg(SupplierModel.lead_time_days)).scalar() or 0
                
                return {
                    "total_suppliers": total_suppliers,
                    "active_suppliers": active_suppliers,
                    "inactive_suppliers": total_suppliers - active_suppliers,
                    "risk_distribution": risk_counts,
                    "avg_reliability_score": float(avg_reliability),
                    "avg_lead_time_days": float(avg_lead_time)
                }
            except SQLAlchemyError as e:
                raise e


class SupplierMaterialRepository(BaseRepository[SupplierMaterialModel]):
    """Repository for SupplierMaterial operations"""
    
    def __init__(self):
        super().__init__(SupplierMaterialModel)
    
    def get_by_supplier_and_material(self, supplier_id: str, material_id: str) -> Optional[SupplierMaterialModel]:
        """Get supplier-material relationship"""
        with self.get_session() as session:
            try:
                return session.query(SupplierMaterialModel).filter(
                    SupplierMaterialModel.supplier_id == supplier_id,
                    SupplierMaterialModel.material_id == material_id
                ).first()
            except SQLAlchemyError as e:
                raise e
    
    def get_by_supplier(self, supplier_id: str, skip: int = 0, limit: int = 100) -> List[SupplierMaterialModel]:
        """Get all materials for a supplier"""
        with self.get_session() as session:
            try:
                return session.query(SupplierMaterialModel).filter(
                    SupplierMaterialModel.supplier_id == supplier_id
                ).offset(skip).limit(limit).all()
            except SQLAlchemyError as e:
                raise e
    
    def get_by_material(self, material_id: str, skip: int = 0, limit: int = 100) -> List[SupplierMaterialModel]:
        """Get all suppliers for a material"""
        with self.get_session() as session:
            try:
                return session.query(SupplierMaterialModel).filter(
                    SupplierMaterialModel.material_id == material_id
                ).offset(skip).limit(limit).all()
            except SQLAlchemyError as e:
                raise e
    
    def get_lowest_cost_for_material(self, material_id: str) -> Optional[SupplierMaterialModel]:
        """Get lowest cost supplier for a material"""
        with self.get_session() as session:
            try:
                return session.query(SupplierMaterialModel).filter(
                    SupplierMaterialModel.material_id == material_id
                ).order_by(SupplierMaterialModel.cost_per_unit.asc()).first()
            except SQLAlchemyError as e:
                raise e
    
    def get_highest_reliability_for_material(self, material_id: str) -> Optional[SupplierMaterialModel]:
        """Get highest reliability supplier for a material"""
        with self.get_session() as session:
            try:
                return session.query(SupplierMaterialModel).filter(
                    SupplierMaterialModel.material_id == material_id
                ).order_by(SupplierMaterialModel.reliability_score.desc()).first()
            except SQLAlchemyError as e:
                raise e
    
    def update_cost(self, supplier_id: str, material_id: str, new_cost: Decimal) -> bool:
        """Update cost for supplier-material relationship"""
        with self.get_session() as session:
            try:
                supplier_material = session.query(SupplierMaterialModel).filter(
                    SupplierMaterialModel.supplier_id == supplier_id,
                    SupplierMaterialModel.material_id == material_id
                ).first()
                
                if supplier_material:
                    supplier_material.update_cost(new_cost)
                    session.commit()
                    return True
                return False
            except SQLAlchemyError as e:
                session.rollback()
                raise e
    
    def update_moq(self, supplier_id: str, material_id: str, new_moq: Decimal) -> bool:
        """Update MOQ for supplier-material relationship"""
        with self.get_session() as session:
            try:
                supplier_material = session.query(SupplierMaterialModel).filter(
                    SupplierMaterialModel.supplier_id == supplier_id,
                    SupplierMaterialModel.material_id == material_id
                ).first()
                
                if supplier_material:
                    supplier_material.update_moq(new_moq)
                    session.commit()
                    return True
                return False
            except SQLAlchemyError as e:
                session.rollback()
                raise e