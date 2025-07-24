"""Suppliers Router for Beverly Knits AI Supply Chain Planner API"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import List, Optional, Dict, Any
from datetime import datetime
from decimal import Decimal

from ..models.request_models import SupplierCreateRequest, SupplierUpdateRequest, SupplierMaterialRequest
from ..models.response_models import SupplierResponse, SupplierListResponse, SupplierMaterialResponse, BaseResponse
from src.auth.dependencies import get_current_user

router = APIRouter()

# Temporary in-memory storage (replace with database in CORE-002)
temp_suppliers = {
    "SUP001": {
        "id": "SUP001",
        "name": "Premium Textile Suppliers Inc.",
        "contact_info": "contact@premiumtextile.com | +1-555-0123",
        "lead_time_days": 14,
        "reliability_score": 0.95,
        "risk_level": "low",
        "is_active": True,
        "created_at": datetime.now(),
        "material_count": 5
    },
    "SUP002": {
        "id": "SUP002",
        "name": "Global Yarn Trading Co.",
        "contact_info": "orders@globalyarn.com | +1-555-0456",
        "lead_time_days": 21,
        "reliability_score": 0.88,
        "risk_level": "medium",
        "is_active": True,
        "created_at": datetime.now(),
        "material_count": 8
    },
    "SUP003": {
        "id": "SUP003",
        "name": "Eco-Friendly Fibers Ltd.",
        "contact_info": "info@ecofibers.com | +1-555-0789",
        "lead_time_days": 10,
        "reliability_score": 0.92,
        "risk_level": "low",
        "is_active": True,
        "created_at": datetime.now(),
        "material_count": 3
    }
}

temp_supplier_materials = {
    "SUP001_MAT001": {
        "supplier_id": "SUP001",
        "material_id": "MAT001",
        "cost_per_unit": Decimal("15.50"),
        "currency": "USD",
        "moq_amount": Decimal("500"),
        "moq_unit": "pounds",
        "lead_time_days": 14,
        "reliability_score": 0.95,
        "ordering_cost": Decimal("100.00"),
        "holding_cost_rate": 0.25
    },
    "SUP002_MAT001": {
        "supplier_id": "SUP002",
        "material_id": "MAT001",
        "cost_per_unit": Decimal("14.80"),
        "currency": "USD",
        "moq_amount": Decimal("1000"),
        "moq_unit": "pounds",
        "lead_time_days": 21,
        "reliability_score": 0.88,
        "ordering_cost": Decimal("150.00"),
        "holding_cost_rate": 0.30
    },
    "SUP001_MAT002": {
        "supplier_id": "SUP001",
        "material_id": "MAT002",
        "cost_per_unit": Decimal("8.25"),
        "currency": "USD",
        "moq_amount": Decimal("100"),
        "moq_unit": "spools",
        "lead_time_days": 10,
        "reliability_score": 0.93,
        "ordering_cost": Decimal("75.00"),
        "holding_cost_rate": 0.20
    }
}

@router.get("/", response_model=SupplierListResponse)
async def get_suppliers(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    risk_level: Optional[str] = Query(None, description="Filter by risk level"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    min_reliability: Optional[float] = Query(None, ge=0, le=1, description="Minimum reliability score"),
    search: Optional[str] = Query(None, description="Search in name and contact info"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get suppliers with pagination and filtering
    
    Returns a paginated list of suppliers with optional filtering by risk level,
    active status, reliability score, and text search.
    """
    suppliers = list(temp_suppliers.values())
    
    # Apply filters
    if risk_level:
        suppliers = [s for s in suppliers if s["risk_level"] == risk_level]
    
    if is_active is not None:
        suppliers = [s for s in suppliers if s["is_active"] == is_active]
    
    if min_reliability is not None:
        suppliers = [s for s in suppliers if s["reliability_score"] >= min_reliability]
    
    if search:
        search_lower = search.lower()
        suppliers = [
            s for s in suppliers 
            if search_lower in s["name"].lower() or 
               search_lower in s.get("contact_info", "").lower()
        ]
    
    # Pagination
    total = len(suppliers)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated_suppliers = suppliers[start_idx:end_idx]
    
    # Convert to response models
    supplier_responses = [
        SupplierResponse(
            id=s["id"],
            name=s["name"],
            contact_info=s.get("contact_info"),
            lead_time_days=s["lead_time_days"],
            reliability_score=s["reliability_score"],
            risk_level=s["risk_level"],
            is_active=s["is_active"],
            created_at=s["created_at"],
            material_count=s.get("material_count", 0)
        ) for s in paginated_suppliers
    ]
    
    return SupplierListResponse(
        success=True,
        message=f"Retrieved {len(supplier_responses)} suppliers",
        suppliers=supplier_responses,
        total=total
    )

@router.get("/risk-levels", response_model=Dict[str, List[str]])
async def get_risk_levels(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get available risk levels
    
    Returns a list of available risk levels in the system.
    """
    return {
        "levels": ["low", "medium", "high"],
        "description": "Available risk levels for supplier assessment"
    }

@router.get("/high-risk", response_model=SupplierListResponse)
async def get_high_risk_suppliers(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get high-risk suppliers
    
    Returns all suppliers marked as high-risk.
    """
    high_risk_suppliers = [s for s in temp_suppliers.values() if s["risk_level"] == "high"]
    
    supplier_responses = [
        SupplierResponse(
            id=s["id"],
            name=s["name"],
            contact_info=s.get("contact_info"),
            lead_time_days=s["lead_time_days"],
            reliability_score=s["reliability_score"],
            risk_level=s["risk_level"],
            is_active=s["is_active"],
            created_at=s["created_at"],
            material_count=s.get("material_count", 0)
        ) for s in high_risk_suppliers
    ]
    
    return SupplierListResponse(
        success=True,
        message=f"Retrieved {len(supplier_responses)} high-risk suppliers",
        suppliers=supplier_responses,
        total=len(supplier_responses)
    )

@router.get("/{supplier_id}", response_model=SupplierResponse)
async def get_supplier(
    supplier_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get supplier by ID
    
    Returns detailed information about a specific supplier.
    """
    supplier = temp_suppliers.get(supplier_id)
    
    if not supplier:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Supplier with ID {supplier_id} not found"
        )
    
    return SupplierResponse(
        id=supplier["id"],
        name=supplier["name"],
        contact_info=supplier.get("contact_info"),
        lead_time_days=supplier["lead_time_days"],
        reliability_score=supplier["reliability_score"],
        risk_level=supplier["risk_level"],
        is_active=supplier["is_active"],
        created_at=supplier["created_at"],
        material_count=supplier.get("material_count", 0)
    )

@router.post("/", response_model=SupplierResponse, status_code=status.HTTP_201_CREATED)
async def create_supplier(
    supplier_request: SupplierCreateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Create new supplier
    
    Creates a new supplier with the provided information.
    
    **Note:** Requires appropriate permissions in production implementation.
    """
    # Generate new supplier ID
    supplier_id = f"SUP{len(temp_suppliers) + 1:03d}"
    
    # Check if supplier with similar name exists
    existing_names = [s["name"].lower() for s in temp_suppliers.values()]
    if supplier_request.name.lower() in existing_names:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Supplier with similar name already exists"
        )
    
    # Create supplier
    supplier = {
        "id": supplier_id,
        "name": supplier_request.name,
        "contact_info": supplier_request.contact_info,
        "lead_time_days": supplier_request.lead_time_days,
        "reliability_score": supplier_request.reliability_score,
        "risk_level": supplier_request.risk_level.value,
        "is_active": supplier_request.is_active,
        "created_at": datetime.now(),
        "material_count": 0
    }
    
    temp_suppliers[supplier_id] = supplier
    
    return SupplierResponse(
        id=supplier["id"],
        name=supplier["name"],
        contact_info=supplier.get("contact_info"),
        lead_time_days=supplier["lead_time_days"],
        reliability_score=supplier["reliability_score"],
        risk_level=supplier["risk_level"],
        is_active=supplier["is_active"],
        created_at=supplier["created_at"],
        material_count=supplier.get("material_count", 0)
    )

@router.put("/{supplier_id}", response_model=SupplierResponse)
async def update_supplier(
    supplier_id: str,
    supplier_request: SupplierUpdateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Update supplier
    
    Updates an existing supplier with the provided information.
    
    **Note:** Requires appropriate permissions in production implementation.
    """
    supplier = temp_suppliers.get(supplier_id)
    
    if not supplier:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Supplier with ID {supplier_id} not found"
        )
    
    # Update fields if provided
    if supplier_request.name is not None:
        supplier["name"] = supplier_request.name
    if supplier_request.contact_info is not None:
        supplier["contact_info"] = supplier_request.contact_info
    if supplier_request.lead_time_days is not None:
        supplier["lead_time_days"] = supplier_request.lead_time_days
    if supplier_request.reliability_score is not None:
        supplier["reliability_score"] = supplier_request.reliability_score
    if supplier_request.risk_level is not None:
        supplier["risk_level"] = supplier_request.risk_level.value
    if supplier_request.is_active is not None:
        supplier["is_active"] = supplier_request.is_active
    
    return SupplierResponse(
        id=supplier["id"],
        name=supplier["name"],
        contact_info=supplier.get("contact_info"),
        lead_time_days=supplier["lead_time_days"],
        reliability_score=supplier["reliability_score"],
        risk_level=supplier["risk_level"],
        is_active=supplier["is_active"],
        created_at=supplier["created_at"],
        material_count=supplier.get("material_count", 0)
    )

@router.delete("/{supplier_id}", response_model=BaseResponse)
async def delete_supplier(
    supplier_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Delete supplier
    
    Deletes a supplier from the system.
    
    **Note:** Requires appropriate permissions in production implementation.
    **Warning:** This will also affect related supplier materials and recommendations.
    """
    supplier = temp_suppliers.get(supplier_id)
    
    if not supplier:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Supplier with ID {supplier_id} not found"
        )
    
    # Check if supplier has materials
    supplier_materials = [sm for sm in temp_supplier_materials.values() if sm["supplier_id"] == supplier_id]
    if supplier_materials:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot delete supplier with {len(supplier_materials)} associated materials. Remove materials first."
        )
    
    # Delete supplier
    del temp_suppliers[supplier_id]
    
    return BaseResponse(
        success=True,
        message=f"Supplier {supplier_id} deleted successfully"
    )

@router.get("/{supplier_id}/materials", response_model=List[SupplierMaterialResponse])
async def get_supplier_materials(
    supplier_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get materials for a specific supplier
    
    Returns all materials associated with a specific supplier.
    """
    supplier = temp_suppliers.get(supplier_id)
    
    if not supplier:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Supplier with ID {supplier_id} not found"
        )
    
    supplier_materials = [
        sm for sm in temp_supplier_materials.values() 
        if sm["supplier_id"] == supplier_id
    ]
    
    return [
        SupplierMaterialResponse(
            supplier_id=sm["supplier_id"],
            material_id=sm["material_id"],
            cost_per_unit=sm["cost_per_unit"],
            currency=sm["currency"],
            moq_amount=sm["moq_amount"],
            moq_unit=sm["moq_unit"],
            lead_time_days=sm["lead_time_days"],
            reliability_score=sm["reliability_score"],
            ordering_cost=sm["ordering_cost"],
            holding_cost_rate=sm["holding_cost_rate"]
        ) for sm in supplier_materials
    ]

@router.post("/{supplier_id}/materials", response_model=SupplierMaterialResponse, status_code=status.HTTP_201_CREATED)
async def create_supplier_material(
    supplier_id: str,
    material_request: SupplierMaterialRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Add material to supplier
    
    Associates a material with a supplier including cost and lead time information.
    
    **Note:** Requires appropriate permissions in production implementation.
    """
    supplier = temp_suppliers.get(supplier_id)
    
    if not supplier:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Supplier with ID {supplier_id} not found"
        )
    
    # Check if material exists (in production, validate against materials table)
    material_id = material_request.material_id
    
    # Check if supplier-material combination already exists
    key = f"{supplier_id}_{material_id}"
    if key in temp_supplier_materials:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Material {material_id} already associated with supplier {supplier_id}"
        )
    
    # Create supplier material
    supplier_material = {
        "supplier_id": supplier_id,
        "material_id": material_id,
        "cost_per_unit": material_request.cost_per_unit,
        "currency": material_request.currency,
        "moq_amount": material_request.moq_amount,
        "moq_unit": material_request.moq_unit,
        "lead_time_days": material_request.lead_time_days,
        "reliability_score": material_request.reliability_score,
        "ordering_cost": material_request.ordering_cost,
        "holding_cost_rate": material_request.holding_cost_rate
    }
    
    temp_supplier_materials[key] = supplier_material
    
    # Update supplier material count
    temp_suppliers[supplier_id]["material_count"] += 1
    
    return SupplierMaterialResponse(
        supplier_id=supplier_material["supplier_id"],
        material_id=supplier_material["material_id"],
        cost_per_unit=supplier_material["cost_per_unit"],
        currency=supplier_material["currency"],
        moq_amount=supplier_material["moq_amount"],
        moq_unit=supplier_material["moq_unit"],
        lead_time_days=supplier_material["lead_time_days"],
        reliability_score=supplier_material["reliability_score"],
        ordering_cost=supplier_material["ordering_cost"],
        holding_cost_rate=supplier_material["holding_cost_rate"]
    )

@router.put("/{supplier_id}/materials/{material_id}", response_model=SupplierMaterialResponse)
async def update_supplier_material(
    supplier_id: str,
    material_id: str,
    material_request: SupplierMaterialRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Update supplier material
    
    Updates cost and lead time information for a supplier-material combination.
    
    **Note:** Requires appropriate permissions in production implementation.
    """
    key = f"{supplier_id}_{material_id}"
    supplier_material = temp_supplier_materials.get(key)
    
    if not supplier_material:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Supplier material combination {supplier_id}-{material_id} not found"
        )
    
    # Update supplier material
    supplier_material.update({
        "cost_per_unit": material_request.cost_per_unit,
        "currency": material_request.currency,
        "moq_amount": material_request.moq_amount,
        "moq_unit": material_request.moq_unit,
        "lead_time_days": material_request.lead_time_days,
        "reliability_score": material_request.reliability_score,
        "ordering_cost": material_request.ordering_cost,
        "holding_cost_rate": material_request.holding_cost_rate
    })
    
    return SupplierMaterialResponse(
        supplier_id=supplier_material["supplier_id"],
        material_id=supplier_material["material_id"],
        cost_per_unit=supplier_material["cost_per_unit"],
        currency=supplier_material["currency"],
        moq_amount=supplier_material["moq_amount"],
        moq_unit=supplier_material["moq_unit"],
        lead_time_days=supplier_material["lead_time_days"],
        reliability_score=supplier_material["reliability_score"],
        ordering_cost=supplier_material["ordering_cost"],
        holding_cost_rate=supplier_material["holding_cost_rate"]
    )

@router.delete("/{supplier_id}/materials/{material_id}", response_model=BaseResponse)
async def delete_supplier_material(
    supplier_id: str,
    material_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Remove material from supplier
    
    Removes the association between a supplier and a material.
    
    **Note:** Requires appropriate permissions in production implementation.
    """
    key = f"{supplier_id}_{material_id}"
    supplier_material = temp_supplier_materials.get(key)
    
    if not supplier_material:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Supplier material combination {supplier_id}-{material_id} not found"
        )
    
    # Delete supplier material
    del temp_supplier_materials[key]
    
    # Update supplier material count
    if supplier_id in temp_suppliers:
        temp_suppliers[supplier_id]["material_count"] -= 1
    
    return BaseResponse(
        success=True,
        message=f"Material {material_id} removed from supplier {supplier_id}"
    )