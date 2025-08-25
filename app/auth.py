#!/usr/bin/env python3
"""
Authentication and authorization module for the Hospital Readmission Risk Predictor.
Implements JWT token handling and Role-Based Access Control (RBAC).
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Security configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Password hashing
try:
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
except Exception:
    # Fallback to sha256 if bcrypt fails
    pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto")

# JWT token security
security = HTTPBearer()

# User roles and permissions
ROLES = {
    "clinician": {
        "permissions": ["read", "predict", "explain"],
        "description": "Healthcare provider who can view predictions and explanations"
    },
    "nurse": {
        "permissions": ["read", "predict"],
        "description": "Nursing staff who can view predictions"
    },
    "admin": {
        "permissions": ["read", "predict", "explain", "admin", "monitor"],
        "description": "System administrator with full access"
    },
    "researcher": {
        "permissions": ["read", "predict", "explain", "research"],
        "description": "Research staff with extended access"
    }
}


class User(BaseModel):
    """User model for authentication."""
    username: str
    email: str
    full_name: str
    role: str
    permissions: list
    is_active: bool = True


class Token(BaseModel):
    """Token model for JWT responses."""
    access_token: str
    token_type: str
    expires_in: int


class TokenData(BaseModel):
    """Token data model for JWT payload."""
    username: Optional[str] = None
    role: Optional[str] = None
    permissions: Optional[list] = None


# Mock user database (in production, use a real database)
USERS_DB = {
    "doctor.smith": {
        "username": "doctor.smith",
        "email": "doctor.smith@hospital.com",
        "full_name": "Dr. John Smith",
        "role": "clinician",
        "permissions": ROLES["clinician"]["permissions"],
        "hashed_password": pwd_context.hash("password123"),
        "is_active": True
    },
    "nurse.jones": {
        "username": "nurse.jones",
        "username": "nurse.jones",
        "email": "nurse.jones@hospital.com",
        "full_name": "Sarah Jones, RN",
        "role": "nurse",
        "permissions": ROLES["nurse"]["permissions"],
        "hashed_password": pwd_context.hash("password123"),
        "is_active": True
    },
    "admin.user": {
        "username": "admin.user",
        "email": "admin@hospital.com",
        "full_name": "System Administrator",
        "role": "admin",
        "permissions": ROLES["admin"]["permissions"],
        "hashed_password": pwd_context.hash("admin123"),
        "is_active": True
    }
}


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def get_user(username: str) -> Optional[Dict[str, Any]]:
    """Get user from database."""
    if username in USERS_DB:
        user_dict = USERS_DB[username]
        return User(**user_dict)
    return None


def authenticate_user(username: str, password: str) -> Optional[User]:
    """Authenticate a user with username and password."""
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, USERS_DB[username]["hashed_password"]):
        return None
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> Optional[TokenData]:
    """Verify and decode a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        permissions: list = payload.get("permissions", [])
        if username is None:
            return None
        token_data = TokenData(username=username, role=role, permissions=permissions)
        return token_data
    except JWTError:
        return None


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Get the current authenticated user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        token = credentials.credentials
        token_data = verify_token(token)
        if token_data is None:
            raise credentials_exception
        
        username = token_data.username
        if username is None:
            raise credentials_exception
        
        user = get_user(username)
        if user is None:
            raise credentials_exception
        
        return user
    except Exception:
        raise credentials_exception


def require_permission(required_permission: str):
    """Decorator to require a specific permission."""
    def permission_checker(current_user: User = Depends(get_current_user)):
        if required_permission not in current_user.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{required_permission}' required"
            )
        return current_user
    return permission_checker


def require_role(required_role: str):
    """Decorator to require a specific role."""
    def role_checker(current_user: User = Depends(get_current_user)):
        if current_user.role != required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{required_role}' required"
            )
        return current_user
    return role_checker


# Permission checkers for different endpoints
def can_predict(current_user: User = Depends(get_current_user)):
    """Check if user can make predictions."""
    if "predict" not in current_user.permissions:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Prediction permission required"
        )
    return current_user


def can_explain(current_user: User = Depends(get_current_user)):
    """Check if user can get explanations."""
    if "explain" not in current_user.permissions:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Explanation permission required"
        )
    return current_user


def can_admin(current_user: User = Depends(get_current_user)):
    """Check if user has admin permissions."""
    if "admin" not in current_user.permissions:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin permission required"
        )
    return current_user


def can_monitor(current_user: User = Depends(get_current_user)):
    """Check if user can access monitoring data."""
    if "monitor" not in current_user.permissions:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Monitoring permission required"
        )
    return current_user


# Utility functions for user management
def create_user(username: str, email: str, full_name: str, role: str, password: str) -> User:
    """Create a new user (admin only)."""
    if role not in ROLES:
        raise ValueError(f"Invalid role: {role}")
    
    if username in USERS_DB:
        raise ValueError(f"Username {username} already exists")
    
    user_data = {
        "username": username,
        "email": email,
        "full_name": full_name,
        "role": role,
        "permissions": ROLES[role]["permissions"],
        "hashed_password": get_password_hash(password),
        "is_active": True
    }
    
    USERS_DB[username] = user_data
    return User(**user_data)


def update_user_permissions(username: str, new_permissions: list) -> User:
    """Update user permissions (admin only)."""
    if username not in USERS_DB:
        raise ValueError(f"User {username} not found")
    
    USERS_DB[username]["permissions"] = new_permissions
    return User(**USERS_DB[username])


def deactivate_user(username: str) -> bool:
    """Deactivate a user (admin only)."""
    if username not in USERS_DB:
        return False
    
    USERS_DB[username]["is_active"] = False
    return True


def get_all_users() -> list:
    """Get all users (admin only)."""
    return [User(**user_data) for user_data in USERS_DB.values()]


def get_user_stats() -> Dict[str, Any]:
    """Get user statistics (admin only)."""
    total_users = len(USERS_DB)
    active_users = sum(1 for user in USERS_DB.values() if user["is_active"])
    role_counts = {}
    
    for user in USERS_DB.values():
        role = user["role"]
        role_counts[role] = role_counts.get(role, 0) + 1
    
    return {
        "total_users": total_users,
        "active_users": active_users,
        "inactive_users": total_users - active_users,
        "role_distribution": role_counts,
        "timestamp": datetime.utcnow().isoformat()
    }
