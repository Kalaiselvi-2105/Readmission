#!/usr/bin/env python3
"""
Rate limiting module for the Hospital Readmission Risk Predictor.
Implements token bucket algorithm for API rate limiting.
"""

import time
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPAuthorizationCredentials
from app.auth import security, verify_token

logger = logging.getLogger(__name__)


class TokenBucket:
    """Token bucket implementation for rate limiting."""
    
    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket.
        
        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens per second to refill
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Consume tokens from the bucket.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False otherwise
        """
        now = time.time()
        
        # Refill tokens
        time_passed = now - self.last_refill
        tokens_to_add = time_passed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
        
        # Check if we can consume tokens
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False


class RateLimiter:
    """Rate limiter for API endpoints."""
    
    def __init__(self):
        """Initialize rate limiter with different limits for different user types."""
        # Rate limits per user type (requests per minute)
        self.rate_limits = {
            "clinician": {"requests": 60, "burst": 10},      # 1 request/second, burst of 10
            "nurse": {"requests": 30, "burst": 5},           # 1 request/2 seconds, burst of 5
            "admin": {"requests": 120, "burst": 20},         # 2 requests/second, burst of 20
            "researcher": {"requests": 90, "burst": 15},     # 1.5 requests/second, burst of 15
            "default": {"requests": 20, "burst": 3}          # Default limit
        }
        
        # User buckets
        self.user_buckets: Dict[str, TokenBucket] = {}
        
        # IP-based rate limiting for unauthenticated requests
        self.ip_buckets: Dict[str, TokenBucket] = {}
        self.ip_rate_limit = {"requests": 10, "burst": 2}  # 1 request/6 seconds, burst of 2
        
        # Cleanup interval
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # 5 minutes
    
    def _get_user_bucket(self, username: str, role: str) -> TokenBucket:
        """Get or create a token bucket for a user."""
        if username not in self.user_buckets:
            limits = self.rate_limits.get(role, self.rate_limits["default"])
            bucket = TokenBucket(
                capacity=limits["burst"],
                refill_rate=limits["requests"] / 60.0  # Convert per minute to per second
            )
            self.user_buckets[username] = bucket
        
        return self.user_buckets[username]
    
    def _get_ip_bucket(self, ip: str) -> TokenBucket:
        """Get or create a token bucket for an IP address."""
        if ip not in self.ip_buckets:
            bucket = TokenBucket(
                capacity=self.ip_rate_limit["burst"],
                refill_rate=self.ip_rate_limit["requests"] / 60.0
            )
            self.ip_buckets[ip] = bucket
        
        return self.ip_buckets[ip]
    
    def _cleanup_old_buckets(self):
        """Clean up old buckets to prevent memory leaks."""
        now = time.time()
        if now - self.last_cleanup > self.cleanup_interval:
            # Remove buckets that haven't been used in the last hour
            cutoff_time = now - 3600
            
            # Clean user buckets
            user_buckets_to_remove = []
            for username, bucket in self.user_buckets.items():
                if bucket.last_refill < cutoff_time:
                    user_buckets_to_remove.append(username)
            
            for username in user_buckets_to_remove:
                del self.user_buckets[username]
            
            # Clean IP buckets
            ip_buckets_to_remove = []
            for ip, bucket in self.ip_buckets.items():
                if bucket.last_refill < cutoff_time:
                    ip_buckets_to_remove.append(ip)
            
            for ip in ip_buckets_to_remove:
                del self.ip_buckets[ip]
            
            self.last_cleanup = now
            logger.info(f"Cleaned up {len(user_buckets_to_remove)} user buckets and {len(ip_buckets_to_remove)} IP buckets")
    
    def check_rate_limit(self, username: Optional[str] = None, 
                        role: Optional[str] = None, 
                        ip: Optional[str] = None,
                        endpoint_cost: int = 1) -> bool:
        """
        Check if a request is within rate limits.
        
        Args:
            username: Username for authenticated requests
            role: User role for rate limit determination
            ip: IP address for unauthenticated requests
            endpoint_cost: Cost of the endpoint (higher for expensive operations)
            
        Returns:
            True if request is allowed, False otherwise
        """
        try:
            self._cleanup_old_buckets()
            
            if username and role:
                # Authenticated request
                bucket = self._get_user_bucket(username, role)
                allowed = bucket.consume(endpoint_cost)
                
                if not allowed:
                    logger.warning(f"Rate limit exceeded for user {username} ({role})")
                    return False
                
                return True
            
            elif ip:
                # Unauthenticated request
                bucket = self._get_ip_bucket(ip)
                allowed = bucket.consume(endpoint_cost)
                
                if not allowed:
                    logger.warning(f"Rate limit exceeded for IP {ip}")
                    return False
                
                return True
            
            else:
                # No identification available, deny by default
                logger.warning("Rate limit check failed: no user or IP provided")
                return False
                
        except Exception as e:
            logger.error(f"Error in rate limit check: {e}")
            # In case of error, allow the request (fail open)
            return True
    
    def get_rate_limit_info(self, username: Optional[str] = None, 
                           role: Optional[str] = None, 
                           ip: Optional[str] = None) -> Dict:
        """Get rate limit information for a user or IP."""
        try:
            if username and role:
                bucket = self._get_user_bucket(username, role)
                limits = self.rate_limits.get(role, self.rate_limits["default"])
            elif ip:
                bucket = self._get_ip_bucket(ip)
                limits = self.ip_rate_limit
            else:
                return {"error": "No user or IP provided"}
            
            return {
                "current_tokens": bucket.tokens,
                "max_tokens": bucket.capacity,
                "refill_rate_per_second": bucket.refill_rate,
                "refill_rate_per_minute": bucket.refill_rate * 60,
                "last_refill": datetime.fromtimestamp(bucket.last_refill).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting rate limit info: {e}")
            return {"error": str(e)}


# Global rate limiter instance
rate_limiter = RateLimiter()


async def check_rate_limit_dependency(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
                                    ip: Optional[str] = None) -> bool:
    """
    FastAPI dependency for checking rate limits.
    
    Args:
        credentials: JWT credentials for authenticated requests
        ip: IP address (should be extracted from request)
        
    Returns:
        True if rate limit check passes
    """
    try:
        if credentials:
            # Authenticated request
            token_data = verify_token(credentials.credentials)
            if token_data and token_data.username and token_data.role:
                allowed = rate_limiter.check_rate_limit(
                    username=token_data.username,
                    role=token_data.role,
                    endpoint_cost=1
                )
                
                if not allowed:
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail="Rate limit exceeded. Please try again later.",
                        headers={"Retry-After": "60"}
                    )
                
                return True
        
        elif ip:
            # Unauthenticated request
            allowed = rate_limiter.check_rate_limit(ip=ip, endpoint_cost=1)
            
            if not allowed:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded. Please try again later.",
                    headers={"Retry-After": "60"}
                )
            
            return True
        
        else:
            # No identification available
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unable to identify request source for rate limiting"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in rate limit dependency: {e}")
        # In case of error, allow the request (fail open)
        return True


def get_rate_limit_info_dependency(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
                                 ip: Optional[str] = None) -> Dict:
    """FastAPI dependency for getting rate limit information."""
    try:
        if credentials:
            token_data = verify_token(credentials.credentials)
            if token_data and token_data.username and token_data.role:
                return rate_limiter.get_rate_limit_info(
                    username=token_data.username,
                    role=token_data.role
                )
        
        elif ip:
            return rate_limiter.get_rate_limit_info(ip=ip)
        
        return {"error": "No user or IP provided"}
        
    except Exception as e:
        logger.error(f"Error getting rate limit info: {e}")
        return {"error": str(e)}


# Rate limit configuration for different endpoints
ENDPOINT_COSTS = {
    "/health": 0,           # Health check is free
    "/predict": 1,          # Single prediction
    "/batch_predict": 5,    # Batch prediction costs more
    "/explain": 2,          # Explanation is more expensive
    "/metrics": 1,          # Metrics endpoint
    "/admin/users": 1,      # Admin endpoints
    "/admin/stats": 1       # Admin statistics
}


def get_endpoint_cost(endpoint: str) -> int:
    """Get the cost for a specific endpoint."""
    return ENDPOINT_COSTS.get(endpoint, 1)






