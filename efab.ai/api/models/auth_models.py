"""Authentication Models for Beverly Knits AI Supply Chain Planner API"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

# User Role Enum
class UserRole(str, Enum):
    ADMIN = "admin"
    MANAGER = "manager"
    PLANNER = "planner"
    VIEWER = "viewer"

# Permission Enum
class Permission(str, Enum):
    READ_MATERIALS = "read_materials"
    WRITE_MATERIALS = "write_materials"
    READ_SUPPLIERS = "read_suppliers"
    WRITE_SUPPLIERS = "write_suppliers"
    READ_INVENTORY = "read_inventory"
    WRITE_INVENTORY = "write_inventory"
    READ_PLANNING = "read_planning"
    EXECUTE_PLANNING = "execute_planning"
    READ_FORECASTING = "read_forecasting"
    EXECUTE_FORECASTING = "execute_forecasting"
    READ_ANALYTICS = "read_analytics"
    SYSTEM_ADMIN = "system_admin"

# Token Type Enum
class TokenType(str, Enum):
    ACCESS = "access"
    REFRESH = "refresh"

# Authentication Models
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: str = Field(..., description="Email address")
    password: str = Field(..., min_length=8, description="Password")
    role: UserRole = Field(default=UserRole.VIEWER, description="User role")
    is_active: bool = Field(default=True, description="User active status")
    
    @validator('username')
    def validate_username(cls, v):
        if not v.isalnum() and '_' not in v:
            raise ValueError('Username must contain only alphanumeric characters and underscores')
        return v
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v

class UserUpdate(BaseModel):
    email: Optional[str] = Field(None, description="Email address")
    role: Optional[UserRole] = Field(None, description="User role")
    is_active: Optional[bool] = Field(None, description="User active status")

class UserInDB(BaseModel):
    id: str = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    email: str = Field(..., description="Email address")
    hashed_password: str = Field(..., description="Hashed password")
    role: UserRole = Field(..., description="User role")
    is_active: bool = Field(..., description="User active status")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")
    failed_login_attempts: int = Field(default=0, description="Failed login attempts")
    account_locked_until: Optional[datetime] = Field(None, description="Account lock expiration")

class TokenData(BaseModel):
    user_id: str = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    role: UserRole = Field(..., description="User role")
    permissions: List[Permission] = Field(..., description="User permissions")
    token_type: TokenType = Field(..., description="Token type")
    expires_at: datetime = Field(..., description="Token expiration")
    issued_at: datetime = Field(..., description="Token issue time")
    jti: str = Field(..., description="JWT ID")

class RefreshTokenInDB(BaseModel):
    id: str = Field(..., description="Token ID")
    user_id: str = Field(..., description="User ID")
    token: str = Field(..., description="Refresh token")
    expires_at: datetime = Field(..., description="Token expiration")
    created_at: datetime = Field(..., description="Creation timestamp")
    is_revoked: bool = Field(default=False, description="Token revoked status")
    revoked_at: Optional[datetime] = Field(None, description="Revocation timestamp")

# Session Models
class SessionCreate(BaseModel):
    user_id: str = Field(..., description="User ID")
    ip_address: str = Field(..., description="IP address")
    user_agent: str = Field(..., description="User agent")
    expires_at: datetime = Field(..., description="Session expiration")

class SessionInDB(BaseModel):
    id: str = Field(..., description="Session ID")
    user_id: str = Field(..., description="User ID")
    ip_address: str = Field(..., description="IP address")
    user_agent: str = Field(..., description="User agent")
    created_at: datetime = Field(..., description="Creation timestamp")
    expires_at: datetime = Field(..., description="Session expiration")
    last_activity: datetime = Field(..., description="Last activity timestamp")
    is_active: bool = Field(default=True, description="Session active status")

# Security Models
class SecurityEvent(BaseModel):
    event_type: str = Field(..., description="Event type")
    user_id: Optional[str] = Field(None, description="User ID")
    username: Optional[str] = Field(None, description="Username")
    ip_address: str = Field(..., description="IP address")
    user_agent: str = Field(..., description="User agent")
    timestamp: datetime = Field(..., description="Event timestamp")
    details: Dict[str, Any] = Field(default_factory=dict, description="Event details")
    risk_score: float = Field(default=0.0, description="Risk score")
    status: str = Field(..., description="Event status")

class LoginAttempt(BaseModel):
    username: str = Field(..., description="Username")
    ip_address: str = Field(..., description="IP address")
    user_agent: str = Field(..., description="User agent")
    timestamp: datetime = Field(..., description="Attempt timestamp")
    success: bool = Field(..., description="Login success status")
    failure_reason: Optional[str] = Field(None, description="Failure reason")
    risk_score: float = Field(default=0.0, description="Risk score")

# API Key Models
class APIKeyCreate(BaseModel):
    name: str = Field(..., description="API key name")
    description: Optional[str] = Field(None, description="API key description")
    permissions: List[Permission] = Field(..., description="API key permissions")
    expires_at: Optional[datetime] = Field(None, description="API key expiration")
    rate_limit: Optional[int] = Field(None, description="Rate limit per minute")

class APIKeyInDB(BaseModel):
    id: str = Field(..., description="API key ID")
    name: str = Field(..., description="API key name")
    description: Optional[str] = Field(None, description="API key description")
    key_hash: str = Field(..., description="API key hash")
    permissions: List[Permission] = Field(..., description="API key permissions")
    created_by: str = Field(..., description="Creator user ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    expires_at: Optional[datetime] = Field(None, description="API key expiration")
    last_used: Optional[datetime] = Field(None, description="Last used timestamp")
    usage_count: int = Field(default=0, description="Usage count")
    rate_limit: Optional[int] = Field(None, description="Rate limit per minute")
    is_active: bool = Field(default=True, description="API key active status")

# Role Permission Mapping
ROLE_PERMISSIONS = {
    UserRole.ADMIN: [
        Permission.READ_MATERIALS, Permission.WRITE_MATERIALS,
        Permission.READ_SUPPLIERS, Permission.WRITE_SUPPLIERS,
        Permission.READ_INVENTORY, Permission.WRITE_INVENTORY,
        Permission.READ_PLANNING, Permission.EXECUTE_PLANNING,
        Permission.READ_FORECASTING, Permission.EXECUTE_FORECASTING,
        Permission.READ_ANALYTICS, Permission.SYSTEM_ADMIN
    ],
    UserRole.MANAGER: [
        Permission.READ_MATERIALS, Permission.WRITE_MATERIALS,
        Permission.READ_SUPPLIERS, Permission.WRITE_SUPPLIERS,
        Permission.READ_INVENTORY, Permission.WRITE_INVENTORY,
        Permission.READ_PLANNING, Permission.EXECUTE_PLANNING,
        Permission.READ_FORECASTING, Permission.EXECUTE_FORECASTING,
        Permission.READ_ANALYTICS
    ],
    UserRole.PLANNER: [
        Permission.READ_MATERIALS,
        Permission.READ_SUPPLIERS,
        Permission.READ_INVENTORY, Permission.WRITE_INVENTORY,
        Permission.READ_PLANNING, Permission.EXECUTE_PLANNING,
        Permission.READ_FORECASTING, Permission.EXECUTE_FORECASTING,
        Permission.READ_ANALYTICS
    ],
    UserRole.VIEWER: [
        Permission.READ_MATERIALS,
        Permission.READ_SUPPLIERS,
        Permission.READ_INVENTORY,
        Permission.READ_PLANNING,
        Permission.READ_FORECASTING,
        Permission.READ_ANALYTICS
    ]
}

# Utility Functions
def get_user_permissions(role: UserRole) -> List[Permission]:
    """Get permissions for a user role"""
    return ROLE_PERMISSIONS.get(role, [])

def has_permission(user_role: UserRole, required_permission: Permission) -> bool:
    """Check if user role has required permission"""
    user_permissions = get_user_permissions(user_role)
    return required_permission in user_permissions