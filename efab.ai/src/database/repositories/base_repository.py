"""Base Repository for Beverly Knits AI Supply Chain Planner"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Generic, TypeVar
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import and_, or_, desc, asc
from datetime import datetime, date, timedelta

from ..connection import get_session
from ..models.base import BaseModel

T = TypeVar('T', bound=BaseModel)

class BaseRepository(ABC, Generic[T]):
    """Base repository with common CRUD operations"""
    
    def __init__(self, model_class: type[T]):
        self.model_class = model_class
    
    def get_session(self) -> Session:
        """Get database session"""
        return get_session()
    
    def create(self, entity: T) -> T:
        """Create new entity"""
        with self.get_session() as session:
            try:
                session.add(entity)
                session.commit()
                session.refresh(entity)
                return entity
            except SQLAlchemyError as e:
                session.rollback()
                raise e
    
    def get_by_id(self, entity_id: str) -> Optional[T]:
        """Get entity by ID"""
        with self.get_session() as session:
            try:
                return session.query(self.model_class).filter(
                    self.model_class.id == entity_id
                ).first()
            except SQLAlchemyError as e:
                raise e
    
    def get_all(self, skip: int = 0, limit: int = 100) -> List[T]:
        """Get all entities with pagination"""
        with self.get_session() as session:
            try:
                return session.query(self.model_class).offset(skip).limit(limit).all()
            except SQLAlchemyError as e:
                raise e
    
    def update(self, entity: T) -> T:
        """Update existing entity"""
        with self.get_session() as session:
            try:
                entity.updated_at = datetime.utcnow()
                session.merge(entity)
                session.commit()
                return entity
            except SQLAlchemyError as e:
                session.rollback()
                raise e
    
    def delete(self, entity_id: str) -> bool:
        """Delete entity by ID"""
        with self.get_session() as session:
            try:
                entity = session.query(self.model_class).filter(
                    self.model_class.id == entity_id
                ).first()
                if entity:
                    session.delete(entity)
                    session.commit()
                    return True
                return False
            except SQLAlchemyError as e:
                session.rollback()
                raise e
    
    def count(self, filters: Dict[str, Any] = None) -> int:
        """Count entities with optional filters"""
        with self.get_session() as session:
            try:
                query = session.query(self.model_class)
                if filters:
                    query = self._apply_filters(query, filters)
                return query.count()
            except SQLAlchemyError as e:
                raise e
    
    def exists(self, entity_id: str) -> bool:
        """Check if entity exists"""
        with self.get_session() as session:
            try:
                return session.query(session.query(self.model_class).filter(
                    self.model_class.id == entity_id
                ).exists()).scalar()
            except SQLAlchemyError as e:
                raise e
    
    def find_by_filters(self, filters: Dict[str, Any], 
                       skip: int = 0, limit: int = 100,
                       order_by: str = None, order_desc: bool = False) -> List[T]:
        """Find entities by filters"""
        with self.get_session() as session:
            try:
                query = session.query(self.model_class)
                query = self._apply_filters(query, filters)
                
                if order_by:
                    order_column = getattr(self.model_class, order_by, None)
                    if order_column:
                        query = query.order_by(desc(order_column) if order_desc else asc(order_column))
                
                return query.offset(skip).limit(limit).all()
            except SQLAlchemyError as e:
                raise e
    
    def find_one_by_filters(self, filters: Dict[str, Any]) -> Optional[T]:
        """Find single entity by filters"""
        with self.get_session() as session:
            try:
                query = session.query(self.model_class)
                query = self._apply_filters(query, filters)
                return query.first()
            except SQLAlchemyError as e:
                raise e
    
    def bulk_create(self, entities: List[T]) -> List[T]:
        """Create multiple entities in bulk"""
        with self.get_session() as session:
            try:
                session.add_all(entities)
                session.commit()
                for entity in entities:
                    session.refresh(entity)
                return entities
            except SQLAlchemyError as e:
                session.rollback()
                raise e
    
    def bulk_update(self, entities: List[T]) -> List[T]:
        """Update multiple entities in bulk"""
        with self.get_session() as session:
            try:
                for entity in entities:
                    entity.updated_at = datetime.utcnow()
                    session.merge(entity)
                session.commit()
                return entities
            except SQLAlchemyError as e:
                session.rollback()
                raise e
    
    def bulk_delete(self, entity_ids: List[str]) -> int:
        """Delete multiple entities by IDs"""
        with self.get_session() as session:
            try:
                deleted_count = session.query(self.model_class).filter(
                    self.model_class.id.in_(entity_ids)
                ).delete(synchronize_session=False)
                session.commit()
                return deleted_count
            except SQLAlchemyError as e:
                session.rollback()
                raise e
    
    def get_recent(self, days: int = 30, limit: int = 100) -> List[T]:
        """Get recently created entities"""
        with self.get_session() as session:
            try:
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                return session.query(self.model_class).filter(
                    self.model_class.created_at >= cutoff_date
                ).order_by(desc(self.model_class.created_at)).limit(limit).all()
            except SQLAlchemyError as e:
                raise e
    
    def get_updated_since(self, since_date: datetime, limit: int = 100) -> List[T]:
        """Get entities updated since specific date"""
        with self.get_session() as session:
            try:
                return session.query(self.model_class).filter(
                    self.model_class.updated_at >= since_date
                ).order_by(desc(self.model_class.updated_at)).limit(limit).all()
            except SQLAlchemyError as e:
                raise e
    
    def _apply_filters(self, query, filters: Dict[str, Any]):
        """Apply filters to query"""
        for key, value in filters.items():
            if hasattr(self.model_class, key):
                column = getattr(self.model_class, key)
                
                if isinstance(value, dict):
                    # Handle complex filters
                    if 'eq' in value:
                        query = query.filter(column == value['eq'])
                    elif 'ne' in value:
                        query = query.filter(column != value['ne'])
                    elif 'gt' in value:
                        query = query.filter(column > value['gt'])
                    elif 'gte' in value:
                        query = query.filter(column >= value['gte'])
                    elif 'lt' in value:
                        query = query.filter(column < value['lt'])
                    elif 'lte' in value:
                        query = query.filter(column <= value['lte'])
                    elif 'in' in value:
                        query = query.filter(column.in_(value['in']))
                    elif 'not_in' in value:
                        query = query.filter(~column.in_(value['not_in']))
                    elif 'like' in value:
                        query = query.filter(column.like(f"%{value['like']}%"))
                    elif 'ilike' in value:
                        query = query.filter(column.ilike(f"%{value['ilike']}%"))
                elif isinstance(value, list):
                    # Handle list as 'in' filter
                    query = query.filter(column.in_(value))
                else:
                    # Handle simple equality
                    query = query.filter(column == value)
        
        return query
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get basic statistics about the repository"""
        with self.get_session() as session:
            try:
                total_count = session.query(self.model_class).count()
                
                # Get creation date range
                oldest = session.query(self.model_class).order_by(
                    asc(self.model_class.created_at)
                ).first()
                newest = session.query(self.model_class).order_by(
                    desc(self.model_class.created_at)
                ).first()
                
                return {
                    "total_count": total_count,
                    "oldest_created": oldest.created_at if oldest else None,
                    "newest_created": newest.created_at if newest else None,
                    "table_name": self.model_class.__tablename__
                }
            except SQLAlchemyError as e:
                raise e
    
    def truncate(self) -> bool:
        """Truncate table (delete all records)"""
        with self.get_session() as session:
            try:
                session.query(self.model_class).delete()
                session.commit()
                return True
            except SQLAlchemyError as e:
                session.rollback()
                raise e
    
    def get_paginated(self, page: int = 1, per_page: int = 20, 
                     filters: Dict[str, Any] = None,
                     order_by: str = None, order_desc: bool = False) -> Dict[str, Any]:
        """Get paginated results with metadata"""
        skip = (page - 1) * per_page
        
        with self.get_session() as session:
            try:
                query = session.query(self.model_class)
                
                if filters:
                    query = self._apply_filters(query, filters)
                
                total = query.count()
                
                if order_by:
                    order_column = getattr(self.model_class, order_by, None)
                    if order_column:
                        query = query.order_by(desc(order_column) if order_desc else asc(order_column))
                
                items = query.offset(skip).limit(per_page).all()
                
                return {
                    "items": items,
                    "total": total,
                    "page": page,
                    "per_page": per_page,
                    "pages": (total + per_page - 1) // per_page,
                    "has_prev": page > 1,
                    "has_next": skip + per_page < total
                }
            except SQLAlchemyError as e:
                raise e