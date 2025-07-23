"""Authentication Package for Beverly Knits AI Supply Chain Planner"""

from .auth_service import auth_service, AuthService
from .dependencies import (
    get_current_user,
    get_current_active_user,
    get_current_verified_user,
    require_role,
    require_permission,
    require_permissions,
    require_admin,
    require_manager_or_admin,
    require_view_materials,
    require_edit_materials,
    require_manage_suppliers,
    require_execute_planning,
    require_view_analytics,
    require_manage_users,
    get_current_user_optional
)

__all__ = [
    "auth_service",
    "AuthService",
    "get_current_user",
    "get_current_active_user", 
    "get_current_verified_user",
    "require_role",
    "require_permission",
    "require_permissions",
    "require_admin",
    "require_manager_or_admin",
    "require_view_materials",
    "require_edit_materials",
    "require_manage_suppliers",
    "require_execute_planning",
    "require_view_analytics",
    "require_manage_users",
    "get_current_user_optional"
]