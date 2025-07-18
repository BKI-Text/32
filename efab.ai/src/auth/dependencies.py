"""Authentication Dependencies for FastAPI"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, List

from .auth_service import auth_service
from ..database.models.user import UserModel

# Security scheme
security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> UserModel:
    """Dependency to get current authenticated user"""
    return auth_service.get_current_user(credentials.credentials)

def get_current_active_user(current_user: UserModel = Depends(get_current_user)) -> UserModel:
    """Dependency to get current active user"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated"
        )
    return current_user

def get_current_verified_user(current_user: UserModel = Depends(get_current_active_user)) -> UserModel:
    """Dependency to get current verified user"""
    if not current_user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is not verified"
        )
    return current_user

def require_role(required_role: str):
    """Dependency factory to require specific role"""
    def role_dependency(current_user: UserModel = Depends(get_current_active_user)) -> UserModel:
        auth_service.require_role(current_user, required_role)
        return current_user
    return role_dependency

def require_permission(required_permission: str):
    """Dependency factory to require specific permission"""
    def permission_dependency(current_user: UserModel = Depends(get_current_active_user)) -> UserModel:
        auth_service.require_permission(current_user, required_permission)
        return current_user
    return permission_dependency

def require_permissions(required_permissions: List[str]):
    """Dependency factory to require multiple permissions"""
    def permissions_dependency(current_user: UserModel = Depends(get_current_active_user)) -> UserModel:
        for permission in required_permissions:
            auth_service.require_permission(current_user, permission)
        return current_user
    return permissions_dependency

def require_admin(current_user: UserModel = Depends(get_current_active_user)) -> UserModel:
    """Dependency to require admin role"""
    auth_service.require_role(current_user, "admin")
    return current_user

def require_manager_or_admin(current_user: UserModel = Depends(get_current_active_user)) -> UserModel:
    """Dependency to require manager or admin role"""
    if not (current_user.is_admin() or current_user.is_manager()):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Manager or admin role required"
        )
    return current_user

# Specific permission dependencies
def require_view_materials(current_user: UserModel = Depends(get_current_active_user)) -> UserModel:
    """Dependency to require view_materials permission"""
    auth_service.require_permission(current_user, "view_materials")
    return current_user

def require_edit_materials(current_user: UserModel = Depends(get_current_active_user)) -> UserModel:
    """Dependency to require edit_materials permission"""
    auth_service.require_permission(current_user, "edit_materials")
    return current_user

def require_manage_suppliers(current_user: UserModel = Depends(get_current_active_user)) -> UserModel:
    """Dependency to require manage_suppliers permission"""
    auth_service.require_permission(current_user, "manage_suppliers")
    return current_user

def require_execute_planning(current_user: UserModel = Depends(get_current_active_user)) -> UserModel:
    """Dependency to require execute_planning permission"""
    auth_service.require_permission(current_user, "execute_planning")
    return current_user

def require_view_analytics(current_user: UserModel = Depends(get_current_active_user)) -> UserModel:
    """Dependency to require view_analytics permission"""
    auth_service.require_permission(current_user, "view_analytics")
    return current_user

def require_manage_users(current_user: UserModel = Depends(get_current_active_user)) -> UserModel:
    """Dependency to require manage_users permission"""
    auth_service.require_permission(current_user, "manage_users")
    return current_user

# Optional authentication (for endpoints that work with or without auth)
def get_current_user_optional(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[UserModel]:
    """Dependency to get current user (optional)"""
    if credentials is None:
        return None
    
    try:
        return auth_service.get_current_user(credentials.credentials)
    except HTTPException:
        return None