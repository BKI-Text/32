"""Authentication Service for Beverly Knits AI Supply Chain Planner"""

import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status
from pydantic import BaseModel

from ..database.models.user import UserModel
from ..database.repositories.user_repository import UserRepository

# Import settings
from ..config.settings import settings

# JWT Configuration from settings
SECRET_KEY = settings.security.secret_key
ALGORITHM = settings.security.algorithm
ACCESS_TOKEN_EXPIRE_MINUTES = settings.security.access_token_expire_minutes
REFRESH_TOKEN_EXPIRE_DAYS = settings.security.refresh_token_expire_days

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# User repository
user_repository = UserRepository()

class TokenData(BaseModel):
    username: Optional[str] = None
    user_id: Optional[str] = None
    permissions: Optional[list] = None
    role: Optional[str] = None

class AuthService:
    """Authentication service with JWT support"""
    
    def __init__(self):
        self.pwd_context = pwd_context
        self.user_repository = user_repository
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Generate password hash"""
        return self.pwd_context.hash(password)
    
    def authenticate_user(self, username: str, password: str) -> Optional[UserModel]:
        """Authenticate user with username and password"""
        try:
            # Get user by username
            user = self.user_repository.get_by_username(username)
            if not user:
                return None
            
            # Check if account is locked
            if user.is_locked():
                raise HTTPException(
                    status_code=status.HTTP_423_LOCKED,
                    detail="Account is temporarily locked due to failed login attempts"
                )
            
            # Verify password
            if not self.verify_password(password, user.password_hash):
                # Increment failed login attempts
                user.increment_failed_login()
                
                # Lock account if too many failed attempts
                if int(user.failed_login_attempts) >= 5:
                    user.lock_account(minutes=30)
                
                self.user_repository.update(user)
                return None
            
            # Check if user is active
            if not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="User account is deactivated"
                )
            
            # Reset failed login attempts on successful login
            user.reset_failed_login()
            user.update_last_login()
            self.user_repository.update(user)
            
            return user
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Authentication error: {str(e)}"
            )
    
    def create_access_token(self, user: UserModel, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode = {
            "sub": user.username,
            "user_id": user.id,
            "role": user.role,
            "permissions": user.permissions or [],
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        }
        
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    def create_refresh_token(self, user: UserModel, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT refresh token"""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        
        to_encode = {
            "sub": user.username,
            "user_id": user.id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        }
        
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    def verify_token(self, token: str) -> TokenData:
        """Verify JWT token and return token data"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            user_id: str = payload.get("user_id")
            
            if username is None or user_id is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token"
                )
            
            token_data = TokenData(
                username=username,
                user_id=user_id,
                permissions=payload.get("permissions", []),
                role=payload.get("role")
            )
            
            return token_data
            
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    def verify_refresh_token(self, refresh_token: str) -> TokenData:
        """Verify refresh token"""
        try:
            payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
            
            # Check if it's a refresh token
            if payload.get("type") != "refresh":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid refresh token"
                )
            
            username: str = payload.get("sub")
            user_id: str = payload.get("user_id")
            
            if username is None or user_id is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid refresh token"
                )
            
            token_data = TokenData(
                username=username,
                user_id=user_id
            )
            
            return token_data
            
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
    
    def get_current_user(self, token: str) -> UserModel:
        """Get current user from token"""
        token_data = self.verify_token(token)
        
        user = self.user_repository.get_by_username(token_data.username)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is deactivated"
            )
        
        return user
    
    def refresh_access_token(self, refresh_token: str) -> Dict[str, str]:
        """Refresh access token using refresh token"""
        token_data = self.verify_refresh_token(refresh_token)
        
        user = self.user_repository.get_by_username(token_data.username)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is deactivated"
            )
        
        # Create new access token
        access_token = self.create_access_token(user)
        
        return {
            "access_token": access_token,
            "token_type": "bearer"
        }
    
    def create_user(self, username: str, email: str, password: str, 
                   full_name: str, role: str = "user") -> UserModel:
        """Create new user"""
        try:
            # Check if username already exists
            existing_user = self.user_repository.get_by_username(username)
            if existing_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username already exists"
                )
            
            # Check if email already exists
            existing_email = self.user_repository.get_by_email(email)
            if existing_email:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already exists"
                )
            
            # Create user
            password_hash = self.get_password_hash(password)
            
            user = UserModel(
                username=username,
                email=email,
                full_name=full_name,
                password_hash=password_hash,
                role=role,
                is_active=True,
                is_verified=False,
                password_changed_at=datetime.utcnow()
            )
            
            # Set default permissions based on role
            user.permissions = self._get_default_permissions(role)
            
            created_user = self.user_repository.create(user)
            return created_user
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"User creation error: {str(e)}"
            )
    
    def change_password(self, user_id: str, current_password: str, new_password: str) -> bool:
        """Change user password"""
        try:
            user = self.user_repository.get_by_id(user_id)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
            # Verify current password
            if not self.verify_password(current_password, user.password_hash):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Current password is incorrect"
                )
            
            # Update password
            user.password_hash = self.get_password_hash(new_password)
            user.update_password_changed()
            
            self.user_repository.update(user)
            return True
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Password change error: {str(e)}"
            )
    
    def reset_password(self, user_id: str, new_password: str) -> bool:
        """Reset user password (admin only)"""
        try:
            user = self.user_repository.get_by_id(user_id)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
            # Update password
            user.password_hash = self.get_password_hash(new_password)
            user.update_password_changed()
            
            self.user_repository.update(user)
            return True
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Password reset error: {str(e)}"
            )
    
    def update_user_permissions(self, user_id: str, permissions: list) -> bool:
        """Update user permissions"""
        try:
            user = self.user_repository.get_by_id(user_id)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
            user.set_permissions(permissions)
            self.user_repository.update(user)
            return True
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Permission update error: {str(e)}"
            )
    
    def update_user_role(self, user_id: str, new_role: str) -> bool:
        """Update user role"""
        try:
            user = self.user_repository.get_by_id(user_id)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
            user.role = new_role
            user.permissions = self._get_default_permissions(new_role)
            
            self.user_repository.update(user)
            return True
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Role update error: {str(e)}"
            )
    
    def _get_default_permissions(self, role: str) -> list:
        """Get default permissions for role"""
        permission_sets = {
            "admin": [
                "view_materials", "edit_materials", "delete_materials",
                "view_suppliers", "edit_suppliers", "delete_suppliers",
                "view_inventory", "edit_inventory",
                "view_bom", "edit_bom", "delete_bom",
                "view_forecasts", "edit_forecasts", "delete_forecasts",
                "execute_planning", "view_planning_results",
                "view_analytics", "export_data",
                "manage_users", "manage_permissions",
                "system_admin"
            ],
            "manager": [
                "view_materials", "edit_materials",
                "view_suppliers", "edit_suppliers",
                "view_inventory", "edit_inventory",
                "view_bom", "edit_bom",
                "view_forecasts", "edit_forecasts",
                "execute_planning", "view_planning_results",
                "view_analytics", "export_data"
            ],
            "user": [
                "view_materials", "edit_materials",
                "view_suppliers",
                "view_inventory",
                "view_bom",
                "view_forecasts",
                "execute_planning", "view_planning_results",
                "view_analytics"
            ],
            "viewer": [
                "view_materials",
                "view_suppliers",
                "view_inventory",
                "view_bom",
                "view_forecasts",
                "view_planning_results",
                "view_analytics"
            ]
        }
        
        return permission_sets.get(role, permission_sets["viewer"])
    
    def has_permission(self, user: UserModel, permission: str) -> bool:
        """Check if user has specific permission"""
        return user.has_permission(permission)
    
    def require_permission(self, user: UserModel, permission: str):
        """Require specific permission or raise exception"""
        if not self.has_permission(user, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission required: {permission}"
            )
    
    def require_role(self, user: UserModel, required_role: str):
        """Require specific role or raise exception"""
        role_hierarchy = {
            "admin": 4,
            "manager": 3,
            "user": 2,
            "viewer": 1
        }
        
        user_level = role_hierarchy.get(user.role, 0)
        required_level = role_hierarchy.get(required_role, 0)
        
        if user_level < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role required: {required_role}"
            )

# Create singleton instance
auth_service = AuthService()