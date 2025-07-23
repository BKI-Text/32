"""User Model for Beverly Knits AI Supply Chain Planner"""

from sqlalchemy import Column, String, Boolean, DateTime, JSON, Index
from sqlalchemy.sql import func
from datetime import datetime
from typing import Dict, Any, List

from .base import BaseModel

class UserModel(BaseModel):
    """User database model"""
    __tablename__ = "users"
    
    # Core user fields
    username = Column(String(255), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    full_name = Column(String(255), nullable=False)
    
    # Authentication fields
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    is_verified = Column(Boolean, default=False, nullable=False)
    
    # Role and permissions
    role = Column(String(50), nullable=False, default="user", index=True)  # admin, manager, user, viewer
    permissions = Column(JSON, nullable=True, default=list)  # List of permission strings
    
    # Profile information
    department = Column(String(100), nullable=True)
    position = Column(String(100), nullable=True)
    phone = Column(String(20), nullable=True)
    
    # Session and security
    last_login = Column(DateTime, nullable=True)
    password_changed_at = Column(DateTime, nullable=True)
    failed_login_attempts = Column(String, nullable=False, default="0")
    locked_until = Column(DateTime, nullable=True)
    
    # Preferences and settings
    preferences = Column(JSON, nullable=True, default=dict)  # User preferences
    timezone = Column(String(50), nullable=True, default="UTC")
    
    # Indexes
    __table_args__ = (
        Index('idx_user_username', 'username'),
        Index('idx_user_email', 'email'),
        Index('idx_user_role_active', 'role', 'is_active'),
        Index('idx_user_last_login', 'last_login'),
    )
    
    def __repr__(self):
        return f"<UserModel(id={self.id}, username={self.username}, role={self.role})>"
    
    def to_domain_entity(self):
        """Convert to domain entity"""
        from ...core.domain.entities import User, UserRole
        from ...core.domain.value_objects import UserId, Email
        
        return User(
            id=UserId(value=self.id),
            username=self.username,
            email=Email(value=self.email),
            full_name=self.full_name,
            role=UserRole(self.role),
            permissions=self.permissions or [],
            is_active=self.is_active,
            is_verified=self.is_verified,
            department=self.department,
            position=self.position,
            preferences=self.preferences or {},
            timezone=self.timezone,
            last_login=self.last_login,
            created_at=self.created_at,
            updated_at=self.updated_at
        )
    
    @classmethod
    def from_domain_entity(cls, user):
        """Create from domain entity"""
        return cls(
            id=user.id.value,
            username=user.username,
            email=user.email.value,
            full_name=user.full_name,
            role=user.role.value,
            permissions=user.permissions,
            is_active=user.is_active,
            is_verified=user.is_verified,
            department=user.department,
            position=user.position,
            preferences=user.preferences,
            timezone=user.timezone,
            last_login=user.last_login,
            created_at=user.created_at,
            updated_at=user.updated_at
        )
    
    def is_admin(self) -> bool:
        """Check if user is admin"""
        return self.role == "admin"
    
    def is_manager(self) -> bool:
        """Check if user is manager"""
        return self.role == "manager"
    
    def is_viewer(self) -> bool:
        """Check if user is viewer"""
        return self.role == "viewer"
    
    def can_view_materials(self) -> bool:
        """Check if user can view materials"""
        return self.is_active and ("view_materials" in self.permissions or self.is_admin())
    
    def can_edit_materials(self) -> bool:
        """Check if user can edit materials"""
        return self.is_active and ("edit_materials" in self.permissions or self.is_admin())
    
    def can_manage_suppliers(self) -> bool:
        """Check if user can manage suppliers"""
        return self.is_active and ("manage_suppliers" in self.permissions or self.is_admin())
    
    def can_execute_planning(self) -> bool:
        """Check if user can execute planning"""
        return self.is_active and ("execute_planning" in self.permissions or self.is_admin())
    
    def can_view_analytics(self) -> bool:
        """Check if user can view analytics"""
        return self.is_active and ("view_analytics" in self.permissions or self.is_admin())
    
    def can_manage_users(self) -> bool:
        """Check if user can manage users"""
        return self.is_active and self.is_admin()
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission"""
        return self.is_active and (permission in self.permissions or self.is_admin())
    
    def add_permission(self, permission: str):
        """Add permission to user"""
        if self.permissions is None:
            self.permissions = []
        if permission not in self.permissions:
            self.permissions.append(permission)
    
    def remove_permission(self, permission: str):
        """Remove permission from user"""
        if self.permissions and permission in self.permissions:
            self.permissions.remove(permission)
    
    def set_permissions(self, permissions: List[str]):
        """Set user permissions"""
        self.permissions = permissions
    
    def update_last_login(self):
        """Update last login timestamp"""
        self.last_login = datetime.utcnow()
    
    def update_password_changed(self):
        """Update password changed timestamp"""
        self.password_changed_at = datetime.utcnow()
    
    def increment_failed_login(self):
        """Increment failed login attempts"""
        current_attempts = int(self.failed_login_attempts)
        self.failed_login_attempts = str(current_attempts + 1)
    
    def reset_failed_login(self):
        """Reset failed login attempts"""
        self.failed_login_attempts = "0"
        self.locked_until = None
    
    def lock_account(self, minutes: int = 30):
        """Lock account for specified minutes"""
        from datetime import timedelta
        self.locked_until = datetime.utcnow() + timedelta(minutes=minutes)
    
    def is_locked(self) -> bool:
        """Check if account is locked"""
        if self.locked_until is None:
            return False
        return datetime.utcnow() < self.locked_until
    
    def unlock_account(self):
        """Unlock account"""
        self.locked_until = None
        self.reset_failed_login()
    
    def deactivate(self):
        """Deactivate user"""
        self.is_active = False
    
    def activate(self):
        """Activate user"""
        self.is_active = True
    
    def verify(self):
        """Verify user account"""
        self.is_verified = True
    
    def get_preference(self, key: str, default=None):
        """Get user preference"""
        if self.preferences is None:
            return default
        return self.preferences.get(key, default)
    
    def set_preference(self, key: str, value: Any):
        """Set user preference"""
        if self.preferences is None:
            self.preferences = {}
        self.preferences[key] = value
    
    def remove_preference(self, key: str):
        """Remove user preference"""
        if self.preferences and key in self.preferences:
            del self.preferences[key]
    
    def update_profile(self, full_name: str = None, department: str = None, 
                      position: str = None, phone: str = None):
        """Update user profile"""
        if full_name is not None:
            self.full_name = full_name
        if department is not None:
            self.department = department
        if position is not None:
            self.position = position
        if phone is not None:
            self.phone = phone
    
    def get_display_name(self) -> str:
        """Get display name for user"""
        return self.full_name or self.username
    
    def get_role_display(self) -> str:
        """Get role display name"""
        role_names = {
            "admin": "Administrator",
            "manager": "Manager",
            "user": "User",
            "viewer": "Viewer"
        }
        return role_names.get(self.role, self.role.title())
    
    def get_permission_count(self) -> int:
        """Get number of permissions"""
        return len(self.permissions) if self.permissions else 0
    
    def days_since_last_login(self) -> int:
        """Get days since last login"""
        if not self.last_login:
            return -1
        return (datetime.utcnow() - self.last_login).days
    
    def password_age_days(self) -> int:
        """Get password age in days"""
        if not self.password_changed_at:
            return -1
        return (datetime.utcnow() - self.password_changed_at).days
    
    def needs_password_change(self, max_age_days: int = 90) -> bool:
        """Check if password needs to be changed"""
        age = self.password_age_days()
        return age > max_age_days if age >= 0 else False
    
    def is_inactive_user(self, days_threshold: int = 90) -> bool:
        """Check if user is inactive"""
        days_since_login = self.days_since_last_login()
        return days_since_login > days_threshold if days_since_login >= 0 else True
    
    def get_user_summary(self) -> dict:
        """Get summary information about user"""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "full_name": self.full_name,
            "display_name": self.get_display_name(),
            "role": self.role,
            "role_display": self.get_role_display(),
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "is_locked": self.is_locked(),
            "department": self.department,
            "position": self.position,
            "permission_count": self.get_permission_count(),
            "days_since_login": self.days_since_last_login(),
            "password_age_days": self.password_age_days(),
            "needs_password_change": self.needs_password_change(),
            "is_inactive": self.is_inactive_user(),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None
        }
    
    def validate_user(self) -> list:
        """Validate user and return list of issues"""
        issues = []
        
        if not self.username:
            issues.append("Username is required")
        elif len(self.username) < 3:
            issues.append("Username must be at least 3 characters")
        
        if not self.email:
            issues.append("Email is required")
        elif "@" not in self.email:
            issues.append("Invalid email format")
        
        if not self.full_name:
            issues.append("Full name is required")
        
        if not self.password_hash:
            issues.append("Password hash is required")
        
        if self.role not in ["admin", "manager", "user", "viewer"]:
            issues.append("Invalid role")
        
        if self.permissions is not None and not isinstance(self.permissions, list):
            issues.append("Permissions must be a list")
        
        return issues