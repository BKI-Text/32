"""Materials Router for Beverly Knits AI Supply Chain Planner API"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import List, Optional, Dict, Any
from datetime import datetime
from decimal import Decimal

from ..models.request_models import MaterialCreateRequest, MaterialUpdateRequest, PaginationRequest
from ..models.response_models import MaterialResponse, MaterialListResponse, BaseResponse
from src.auth.dependencies import (
    get_current_user, 
    require_view_materials, 
    require_edit_materials,
    require_permission
)

router = APIRouter()

# Temporary in-memory storage (replace with database in CORE-002)
temp_materials = {
    "MAT001": {
        "id": "MAT001",
        "name": "Cotton Yarn - Natural",
        "type": "yarn",
        "description": "100% organic cotton yarn for premium textile production",
        "specifications": {
            "weight": "Worsted",
            "fiber_content": "100% Cotton",
            "color": "Natural",
            "twist": "Z-twist"
        },
        "is_critical": True,
        "created_at": datetime.now(),
        "updated_at": datetime.now()
    },
    "MAT002": {
        "id": "MAT002",
        "name": "Polyester Thread",
        "type": "thread",
        "description": "High-strength polyester thread for industrial sewing",
        "specifications": {
            "weight": "40 wt",
            "fiber_content": "100% Polyester",
            "color": "Black",
            "strength": "High"
        },
        "is_critical": False,
        "created_at": datetime.now(),
        "updated_at": datetime.now()
    },
    "MAT003": {
        "id": "MAT003",
        "name": "Wool Blend Fabric",
        "type": "fabric",
        "description": "Wool-polyester blend fabric for outerwear",
        "specifications": {
            "weight": "Medium",
            "fiber_content": "70% Wool, 30% Polyester",
            "width": "60 inches",
            "finish": "Brushed"
        },
        "is_critical": False,
        "created_at": datetime.now(),
        "updated_at": datetime.now()
    }
}

@router.get("/", response_model=MaterialListResponse)
async def get_materials(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    material_type: Optional[str] = Query(None, description="Filter by material type"),
    is_critical: Optional[bool] = Query(None, description="Filter by critical status"),
    search: Optional[str] = Query(None, description="Search in name and description"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get materials with pagination and filtering
    
    Returns a paginated list of materials with optional filtering by type,
    critical status, and text search.
    """
    materials = list(temp_materials.values())
    
    # Apply filters
    if material_type:
        materials = [m for m in materials if m["type"] == material_type]
    
    if is_critical is not None:
        materials = [m for m in materials if m["is_critical"] == is_critical]
    
    if search:
        search_lower = search.lower()
        materials = [
            m for m in materials 
            if search_lower in m["name"].lower() or 
               search_lower in m.get("description", "").lower()
        ]
    
    # Pagination
    total = len(materials)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated_materials = materials[start_idx:end_idx]
    
    # Convert to response models
    material_responses = [
        MaterialResponse(
            id=m["id"],
            name=m["name"],
            type=m["type"],
            description=m.get("description"),
            specifications=m.get("specifications", {}),
            is_critical=m["is_critical"],
            created_at=m["created_at"],
            updated_at=m["updated_at"]
        ) for m in paginated_materials
    ]
    
    return MaterialListResponse(
        success=True,
        message=f"Retrieved {len(material_responses)} materials",
        materials=material_responses,
        total=total
    )

@router.get("/types", response_model=Dict[str, List[str]])
async def get_material_types(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get available material types
    
    Returns a list of available material types in the system.
    """
    return {
        "types": ["yarn", "fabric", "thread", "accessory", "trim"],
        "description": "Available material types in the system"
    }

@router.get("/critical", response_model=MaterialListResponse)
async def get_critical_materials(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get critical materials
    
    Returns all materials marked as critical for production.
    """
    critical_materials = [m for m in temp_materials.values() if m["is_critical"]]
    
    material_responses = [
        MaterialResponse(
            id=m["id"],
            name=m["name"],
            type=m["type"],
            description=m.get("description"),
            specifications=m.get("specifications", {}),
            is_critical=m["is_critical"],
            created_at=m["created_at"],
            updated_at=m["updated_at"]
        ) for m in critical_materials
    ]
    
    return MaterialListResponse(
        success=True,
        message=f"Retrieved {len(material_responses)} critical materials",
        materials=material_responses,
        total=len(material_responses)
    )

@router.get("/{material_id}", response_model=MaterialResponse)
async def get_material(
    material_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get material by ID
    
    Returns detailed information about a specific material.
    """
    material = temp_materials.get(material_id)
    
    if not material:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Material with ID {material_id} not found"
        )
    
    return MaterialResponse(
        id=material["id"],
        name=material["name"],
        type=material["type"],
        description=material.get("description"),
        specifications=material.get("specifications", {}),
        is_critical=material["is_critical"],
        created_at=material["created_at"],
        updated_at=material["updated_at"]
    )

@router.post("/", response_model=MaterialResponse, status_code=status.HTTP_201_CREATED)
async def create_material(
    material_request: MaterialCreateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Create new material
    
    Creates a new material with the provided information.
    
    **Note:** Requires appropriate permissions in production implementation.
    """
    # Generate new material ID
    material_id = f"MAT{len(temp_materials) + 1:03d}"
    
    # Check if material with similar name exists
    existing_names = [m["name"].lower() for m in temp_materials.values()]
    if material_request.name.lower() in existing_names:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Material with similar name already exists"
        )
    
    # Create material
    now = datetime.now()
    material = {
        "id": material_id,
        "name": material_request.name,
        "type": material_request.type.value,
        "description": material_request.description,
        "specifications": material_request.specifications,
        "is_critical": material_request.is_critical,
        "created_at": now,
        "updated_at": now
    }
    
    temp_materials[material_id] = material
    
    return MaterialResponse(
        id=material["id"],
        name=material["name"],
        type=material["type"],
        description=material.get("description"),
        specifications=material.get("specifications", {}),
        is_critical=material["is_critical"],
        created_at=material["created_at"],
        updated_at=material["updated_at"]
    )

@router.put("/{material_id}", response_model=MaterialResponse)
async def update_material(
    material_id: str,
    material_request: MaterialUpdateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Update material
    
    Updates an existing material with the provided information.
    
    **Note:** Requires appropriate permissions in production implementation.
    """
    material = temp_materials.get(material_id)
    
    if not material:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Material with ID {material_id} not found"
        )
    
    # Update fields if provided
    if material_request.name is not None:
        material["name"] = material_request.name
    if material_request.type is not None:
        material["type"] = material_request.type.value
    if material_request.description is not None:
        material["description"] = material_request.description
    if material_request.specifications is not None:
        material["specifications"] = material_request.specifications
    if material_request.is_critical is not None:
        material["is_critical"] = material_request.is_critical
    
    material["updated_at"] = datetime.now()
    
    return MaterialResponse(
        id=material["id"],
        name=material["name"],
        type=material["type"],
        description=material.get("description"),
        specifications=material.get("specifications", {}),
        is_critical=material["is_critical"],
        created_at=material["created_at"],
        updated_at=material["updated_at"]
    )

@router.delete("/{material_id}", response_model=BaseResponse)
async def delete_material(
    material_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Delete material
    
    Deletes a material from the system.
    
    **Note:** Requires appropriate permissions in production implementation.
    **Warning:** This will also affect related BOMs and supplier materials.
    """
    material = temp_materials.get(material_id)
    
    if not material:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Material with ID {material_id} not found"
        )
    
    # Check if material is critical
    if material["is_critical"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete critical material. Update critical status first."
        )
    
    # Delete material
    del temp_materials[material_id]
    
    return BaseResponse(
        success=True,
        message=f"Material {material_id} deleted successfully"
    )

@router.get("/{material_id}/specifications", response_model=Dict[str, Any])
async def get_material_specifications(
    material_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get material specifications
    
    Returns detailed specifications for a specific material.
    """
    material = temp_materials.get(material_id)
    
    if not material:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Material with ID {material_id} not found"
        )
    
    return {
        "material_id": material_id,
        "specifications": material.get("specifications", {}),
        "last_updated": material["updated_at"]
    }

@router.put("/{material_id}/specifications", response_model=Dict[str, Any])
async def update_material_specifications(
    material_id: str,
    specifications: Dict[str, str],
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Update material specifications
    
    Updates the specifications for a specific material.
    
    **Note:** Requires appropriate permissions in production implementation.
    """
    material = temp_materials.get(material_id)
    
    if not material:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Material with ID {material_id} not found"
        )
    
    # Update specifications
    material["specifications"].update(specifications)
    material["updated_at"] = datetime.now()
    
    return {
        "material_id": material_id,
        "specifications": material["specifications"],
        "updated_at": material["updated_at"]
    }