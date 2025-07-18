"""Authentication Router for Beverly Knits AI Supply Chain Planner API"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, Any
from datetime import datetime, timedelta

from ..models.request_models import LoginRequest, RefreshTokenRequest, UserCreateRequest, PasswordChangeRequest
from ..models.response_models import TokenResponse, UserResponse, BaseResponse
from ..models.auth_models import UserRole, TokenData
from src.auth.auth_service import auth_service
from src.auth.dependencies import get_current_user, require_admin, require_manage_users

router = APIRouter()
security = HTTPBearer()

@router.post("/login", response_model=TokenResponse)
async def login(login_request: LoginRequest):
    """
    User login endpoint
    
    Authenticates user credentials and returns JWT access and refresh tokens.
    """
    # Authenticate user
    user = auth_service.authenticate_user(login_request.username, login_request.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Generate JWT tokens
    access_token = auth_service.create_access_token(user)
    refresh_token = auth_service.create_refresh_token(user)
    
    return TokenResponse(
        success=True,
        message="Login successful",
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=30 * 60  # 30 minutes
    )

@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(refresh_request: RefreshTokenRequest):
    """
    Refresh access token endpoint
    
    Generates a new access token using a valid refresh token.
    """
    # Refresh access token
    token_data = auth_service.refresh_access_token(refresh_request.refresh_token)
    
    return TokenResponse(
        success=True,
        message="Token refreshed successfully",
        access_token=token_data["access_token"],
        refresh_token=refresh_request.refresh_token,  # Keep same refresh token
        token_type="bearer",
        expires_in=30 * 60  # 30 minutes
    )

@router.post("/logout", response_model=BaseResponse)
async def logout(current_user = Depends(get_current_user)):
    """
    User logout endpoint
    
    Invalidates the current access token.
    
    **Note:** In production, tokens should be blacklisted.
    For JWT, tokens expire naturally but blacklisting provides immediate revocation.
    """
    # In production, add token to blacklist/revocation list
    # For now, just return success since JWT tokens expire naturally
    
    return BaseResponse(
        success=True,
        message="Logout successful"
    )

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user = Depends(get_current_user)):
    """
    Get current user information
    
    Returns information about the currently authenticated user.
    """
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        role=current_user.role,
        is_active=current_user.is_active,
        created_at=current_user.created_at,
        last_login=current_user.last_login
    )

@router.get("/verify", response_model=BaseResponse)
async def verify_token(current_user = Depends(get_current_user)):
    """
    Verify access token
    
    Validates the current access token and returns verification status.
    """
    return BaseResponse(
        success=True,
        message="Token is valid"
    )

@router.post("/register", response_model=UserResponse)
async def register_user(user_request: UserCreateRequest, admin_user = Depends(require_admin)):
    """
    Register new user endpoint (Admin only)
    
    Creates a new user account with specified role and permissions.
    """
    user = auth_service.create_user(
        username=user_request.username,
        email=user_request.email,
        password=user_request.password,
        full_name=user_request.full_name,
        role=user_request.role
    )
    
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        role=user.role,
        is_active=user.is_active,
        created_at=user.created_at,
        last_login=user.last_login
    )

@router.post("/change-password", response_model=BaseResponse)
async def change_password(password_request: PasswordChangeRequest, current_user = Depends(get_current_user)):
    """
    Change user password endpoint
    
    Allows authenticated users to change their own password.
    """
    success = auth_service.change_password(
        user_id=current_user.id,
        current_password=password_request.current_password,
        new_password=password_request.new_password
    )
    
    if success:
        return BaseResponse(
            success=True,
            message="Password changed successfully"
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to change password"
        )

@router.post("/reset-password/{user_id}", response_model=BaseResponse)
async def reset_user_password(user_id: str, new_password: str, admin_user = Depends(require_admin)):
    """
    Reset user password endpoint (Admin only)
    
    Allows administrators to reset any user's password.
    """
    success = auth_service.reset_password(user_id, new_password)
    
    if success:
        return BaseResponse(
            success=True,
            message="Password reset successfully"
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to reset password"
        )

@router.put("/users/{user_id}/permissions", response_model=BaseResponse)
async def update_user_permissions(user_id: str, permissions: list, admin_user = Depends(require_manage_users)):
    """
    Update user permissions endpoint (Admin only)
    
    Updates the permissions for a specific user.
    """
    success = auth_service.update_user_permissions(user_id, permissions)
    
    if success:
        return BaseResponse(
            success=True,
            message="User permissions updated successfully"
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to update user permissions"
        )

@router.put("/users/{user_id}/role", response_model=BaseResponse)
async def update_user_role(user_id: str, new_role: str, admin_user = Depends(require_admin)):
    """
    Update user role endpoint (Admin only)
    
    Updates the role for a specific user.
    """
    success = auth_service.update_user_role(user_id, new_role)
    
    if success:
        return BaseResponse(
            success=True,
            message="User role updated successfully"
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to update user role"
        )