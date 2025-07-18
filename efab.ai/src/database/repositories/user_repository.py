"""User Repository for Beverly Knits AI Supply Chain Planner"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import and_, or_, func
from datetime import datetime, timedelta

from .base_repository import BaseRepository
from ..models.user import UserModel

class UserRepository(BaseRepository[UserModel]):
    """Repository for User operations"""
    
    def __init__(self):
        super().__init__(UserModel)
    
    def get_by_username(self, username: str) -> Optional[UserModel]:
        """Get user by username"""
        with self.get_session() as session:
            try:
                return session.query(UserModel).filter(
                    UserModel.username == username
                ).first()
            except SQLAlchemyError as e:
                raise e
    
    def get_by_email(self, email: str) -> Optional[UserModel]:
        """Get user by email"""
        with self.get_session() as session:
            try:
                return session.query(UserModel).filter(
                    UserModel.email == email
                ).first()
            except SQLAlchemyError as e:
                raise e
    
    def get_active_users(self, skip: int = 0, limit: int = 100) -> List[UserModel]:
        """Get active users"""
        with self.get_session() as session:
            try:
                return session.query(UserModel).filter(
                    UserModel.is_active == True
                ).offset(skip).limit(limit).all()
            except SQLAlchemyError as e:
                raise e
    
    def get_by_role(self, role: str, skip: int = 0, limit: int = 100) -> List[UserModel]:
        """Get users by role"""
        with self.get_session() as session:
            try:
                return session.query(UserModel).filter(
                    UserModel.role == role
                ).offset(skip).limit(limit).all()
            except SQLAlchemyError as e:
                raise e
    
    def get_verified_users(self, skip: int = 0, limit: int = 100) -> List[UserModel]:
        """Get verified users"""
        with self.get_session() as session:
            try:
                return session.query(UserModel).filter(
                    UserModel.is_verified == True
                ).offset(skip).limit(limit).all()
            except SQLAlchemyError as e:
                raise e
    
    def get_unverified_users(self, skip: int = 0, limit: int = 100) -> List[UserModel]:
        """Get unverified users"""
        with self.get_session() as session:
            try:
                return session.query(UserModel).filter(
                    UserModel.is_verified == False
                ).offset(skip).limit(limit).all()
            except SQLAlchemyError as e:
                raise e
    
    def get_locked_users(self, skip: int = 0, limit: int = 100) -> List[UserModel]:
        """Get locked users"""
        with self.get_session() as session:
            try:
                return session.query(UserModel).filter(
                    UserModel.locked_until > datetime.utcnow()
                ).offset(skip).limit(limit).all()
            except SQLAlchemyError as e:
                raise e
    
    def get_inactive_users(self, days_threshold: int = 90, skip: int = 0, limit: int = 100) -> List[UserModel]:
        """Get inactive users (no login for X days)"""
        with self.get_session() as session:
            try:
                cutoff_date = datetime.utcnow() - timedelta(days=days_threshold)
                return session.query(UserModel).filter(
                    or_(
                        UserModel.last_login < cutoff_date,
                        UserModel.last_login.is_(None)
                    )
                ).offset(skip).limit(limit).all()
            except SQLAlchemyError as e:
                raise e
    
    def get_users_with_expired_passwords(self, max_age_days: int = 90, skip: int = 0, limit: int = 100) -> List[UserModel]:
        """Get users with expired passwords"""
        with self.get_session() as session:
            try:
                cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)
                return session.query(UserModel).filter(
                    or_(
                        UserModel.password_changed_at < cutoff_date,
                        UserModel.password_changed_at.is_(None)
                    )
                ).offset(skip).limit(limit).all()
            except SQLAlchemyError as e:
                raise e
    
    def search_users(self, search_term: str, skip: int = 0, limit: int = 100) -> List[UserModel]:
        """Search users by username, email, or full name"""
        with self.get_session() as session:
            try:
                return session.query(UserModel).filter(
                    or_(
                        UserModel.username.ilike(f"%{search_term}%"),
                        UserModel.email.ilike(f"%{search_term}%"),
                        UserModel.full_name.ilike(f"%{search_term}%")
                    )
                ).offset(skip).limit(limit).all()
            except SQLAlchemyError as e:
                raise e
    
    def get_users_by_department(self, department: str, skip: int = 0, limit: int = 100) -> List[UserModel]:
        """Get users by department"""
        with self.get_session() as session:
            try:
                return session.query(UserModel).filter(
                    UserModel.department == department
                ).offset(skip).limit(limit).all()
            except SQLAlchemyError as e:
                raise e
    
    def get_users_with_permission(self, permission: str, skip: int = 0, limit: int = 100) -> List[UserModel]:
        """Get users with specific permission"""
        with self.get_session() as session:
            try:
                # This is a simplified version - in production, you'd want to use JSON operations
                users = session.query(UserModel).filter(
                    UserModel.is_active == True
                ).all()
                
                filtered_users = []
                for user in users:
                    if user.has_permission(permission):
                        filtered_users.append(user)
                
                return filtered_users[skip:skip+limit]
            except SQLAlchemyError as e:
                raise e
    
    def get_recent_logins(self, days: int = 7, skip: int = 0, limit: int = 100) -> List[UserModel]:
        """Get users with recent logins"""
        with self.get_session() as session:
            try:
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                return session.query(UserModel).filter(
                    UserModel.last_login >= cutoff_date
                ).order_by(UserModel.last_login.desc()).offset(skip).limit(limit).all()
            except SQLAlchemyError as e:
                raise e
    
    def get_failed_login_users(self, min_attempts: int = 3, skip: int = 0, limit: int = 100) -> List[UserModel]:
        """Get users with failed login attempts"""
        with self.get_session() as session:
            try:
                # Convert string to int for comparison
                users = session.query(UserModel).all()
                filtered_users = []
                
                for user in users:
                    if int(user.failed_login_attempts) >= min_attempts:
                        filtered_users.append(user)
                
                return filtered_users[skip:skip+limit]
            except SQLAlchemyError as e:
                raise e
    
    def activate_user(self, user_id: str) -> bool:
        """Activate user"""
        with self.get_session() as session:
            try:
                user = session.query(UserModel).filter(
                    UserModel.id == user_id
                ).first()
                
                if user:
                    user.activate()
                    session.commit()
                    return True
                return False
            except SQLAlchemyError as e:
                session.rollback()
                raise e
    
    def deactivate_user(self, user_id: str) -> bool:
        """Deactivate user"""
        with self.get_session() as session:
            try:
                user = session.query(UserModel).filter(
                    UserModel.id == user_id
                ).first()
                
                if user:
                    user.deactivate()
                    session.commit()
                    return True
                return False
            except SQLAlchemyError as e:
                session.rollback()
                raise e
    
    def verify_user(self, user_id: str) -> bool:
        """Verify user"""
        with self.get_session() as session:
            try:
                user = session.query(UserModel).filter(
                    UserModel.id == user_id
                ).first()
                
                if user:
                    user.verify()
                    session.commit()
                    return True
                return False
            except SQLAlchemyError as e:
                session.rollback()
                raise e
    
    def lock_user(self, user_id: str, minutes: int = 30) -> bool:
        """Lock user account"""
        with self.get_session() as session:
            try:
                user = session.query(UserModel).filter(
                    UserModel.id == user_id
                ).first()
                
                if user:
                    user.lock_account(minutes)
                    session.commit()
                    return True
                return False
            except SQLAlchemyError as e:
                session.rollback()
                raise e
    
    def unlock_user(self, user_id: str) -> bool:
        """Unlock user account"""
        with self.get_session() as session:
            try:
                user = session.query(UserModel).filter(
                    UserModel.id == user_id
                ).first()
                
                if user:
                    user.unlock_account()
                    session.commit()
                    return True
                return False
            except SQLAlchemyError as e:
                session.rollback()
                raise e
    
    def update_user_role(self, user_id: str, new_role: str) -> bool:
        """Update user role"""
        with self.get_session() as session:
            try:
                user = session.query(UserModel).filter(
                    UserModel.id == user_id
                ).first()
                
                if user and new_role in ["admin", "manager", "user", "viewer"]:
                    user.role = new_role
                    session.commit()
                    return True
                return False
            except SQLAlchemyError as e:
                session.rollback()
                raise e
    
    def add_user_permission(self, user_id: str, permission: str) -> bool:
        """Add permission to user"""
        with self.get_session() as session:
            try:
                user = session.query(UserModel).filter(
                    UserModel.id == user_id
                ).first()
                
                if user:
                    user.add_permission(permission)
                    session.commit()
                    return True
                return False
            except SQLAlchemyError as e:
                session.rollback()
                raise e
    
    def remove_user_permission(self, user_id: str, permission: str) -> bool:
        """Remove permission from user"""
        with self.get_session() as session:
            try:
                user = session.query(UserModel).filter(
                    UserModel.id == user_id
                ).first()
                
                if user:
                    user.remove_permission(permission)
                    session.commit()
                    return True
                return False
            except SQLAlchemyError as e:
                session.rollback()
                raise e
    
    def update_user_profile(self, user_id: str, profile_data: Dict[str, Any]) -> bool:
        """Update user profile"""
        with self.get_session() as session:
            try:
                user = session.query(UserModel).filter(
                    UserModel.id == user_id
                ).first()
                
                if user:
                    user.update_profile(
                        full_name=profile_data.get('full_name'),
                        department=profile_data.get('department'),
                        position=profile_data.get('position'),
                        phone=profile_data.get('phone')
                    )
                    session.commit()
                    return True
                return False
            except SQLAlchemyError as e:
                session.rollback()
                raise e
    
    def set_user_preference(self, user_id: str, key: str, value: Any) -> bool:
        """Set user preference"""
        with self.get_session() as session:
            try:
                user = session.query(UserModel).filter(
                    UserModel.id == user_id
                ).first()
                
                if user:
                    user.set_preference(key, value)
                    session.commit()
                    return True
                return False
            except SQLAlchemyError as e:
                session.rollback()
                raise e
    
    def get_user_statistics(self) -> Dict[str, Any]:
        """Get user statistics"""
        with self.get_session() as session:
            try:
                total_users = session.query(UserModel).count()
                active_users = session.query(UserModel).filter(
                    UserModel.is_active == True
                ).count()
                verified_users = session.query(UserModel).filter(
                    UserModel.is_verified == True
                ).count()
                
                # Count by role
                role_counts = {}
                for role in ["admin", "manager", "user", "viewer"]:
                    count = session.query(UserModel).filter(
                        UserModel.role == role
                    ).count()
                    role_counts[role] = count
                
                # Recent logins (last 7 days)
                recent_cutoff = datetime.utcnow() - timedelta(days=7)
                recent_logins = session.query(UserModel).filter(
                    UserModel.last_login >= recent_cutoff
                ).count()
                
                # Locked users
                locked_users = session.query(UserModel).filter(
                    UserModel.locked_until > datetime.utcnow()
                ).count()
                
                return {
                    "total_users": total_users,
                    "active_users": active_users,
                    "inactive_users": total_users - active_users,
                    "verified_users": verified_users,
                    "unverified_users": total_users - verified_users,
                    "role_distribution": role_counts,
                    "recent_logins": recent_logins,
                    "locked_users": locked_users
                }
            except SQLAlchemyError as e:
                raise e
    
    def cleanup_expired_locks(self) -> int:
        """Clean up expired user locks"""
        with self.get_session() as session:
            try:
                expired_locks = session.query(UserModel).filter(
                    UserModel.locked_until < datetime.utcnow()
                ).all()
                
                count = 0
                for user in expired_locks:
                    user.unlock_account()
                    count += 1
                
                session.commit()
                return count
            except SQLAlchemyError as e:
                session.rollback()
                raise e