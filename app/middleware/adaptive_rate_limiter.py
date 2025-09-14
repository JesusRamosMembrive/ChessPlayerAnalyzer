"""
Adaptive Rate Limiter
Intelligent rate limiting that adapts based on performance optimizations and system capacity.
"""

import time
import asyncio
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
import redis.asyncio as redis
import hashlib
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    # Base limits (pre-optimization baseline)
    base_requests_per_minute: int = 60
    base_analysis_requests_per_minute: int = 10
    base_batch_requests_per_hour: int = 5

    # Performance multipliers based on 4.6x speedup
    performance_multiplier: float = 4.6
    burst_allowance: float = 1.5

    # Adaptive scaling factors
    system_health_factor: float = 1.0
    user_tier_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "free": 1.0,
        "premium": 3.0,
        "enterprise": 10.0
    })

    # Window configurations
    rate_window_seconds: int = 60
    burst_window_seconds: int = 10
    cooldown_period_seconds: int = 300

@dataclass
class UserMetrics:
    """User-specific performance metrics."""
    requests_count: int = 0
    analysis_requests_count: int = 0
    batch_requests_count: int = 0
    avg_response_time_ms: float = 0.0
    success_rate: float = 100.0
    last_request_timestamp: float = 0.0
    tier: str = "free"

class AdaptiveRateLimiter:
    """Intelligent rate limiter that adapts based on performance and system health."""

    def __init__(self, redis_client: redis.Redis, config: RateLimitConfig = None):
        self.redis_client = redis_client
        self.config = config or RateLimitConfig()
        self.performance_metrics = {}

        # System health monitoring
        self.system_health = {
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "active_analyses": 0,
            "queue_length": 0,
            "avg_speedup_factor": 4.6
        }

    async def get_user_identifier(self, request: Request) -> str:
        """Get user identifier for rate limiting."""
        # Try different identification methods
        user_id = None

        # Check for API key
        api_key = request.headers.get("X-API-Key")
        if api_key:
            user_id = f"api_key:{hashlib.md5(api_key.encode()).hexdigest()[:8]}"

        # Check for user authentication
        auth_header = request.headers.get("Authorization")
        if auth_header and not user_id:
            user_id = f"auth:{hashlib.md5(auth_header.encode()).hexdigest()[:8]}"

        # Fall back to IP address
        if not user_id:
            client_ip = request.client.host
            user_id = f"ip:{client_ip}"

        return user_id

    async def get_user_tier(self, user_id: str, request: Request) -> str:
        """Determine user tier for adaptive limits."""
        # Check for tier information in headers
        tier = request.headers.get("X-User-Tier", "free").lower()

        if tier not in self.config.user_tier_multipliers:
            tier = "free"

        # Could also check database or cache for user subscription info
        cached_tier = await self.redis_client.get(f"user_tier:{user_id}")
        if cached_tier:
            tier = cached_tier.decode("utf-8")

        return tier

    async def get_system_health(self) -> Dict[str, float]:
        """Get current system health metrics."""
        try:
            # In production, this would pull from monitoring systems
            # For now, simulate based on queue lengths and active processes

            # Check Redis queue lengths
            queue_length = await self.redis_client.llen("analysis_queue") or 0

            # Estimate system load based on queue
            cpu_usage = min(80.0, queue_length * 2.0)  # Rough estimation
            memory_usage = min(90.0, queue_length * 1.5)

            # Get current performance metrics from cache
            perf_data = await self.redis_client.get("system:performance:current")
            if perf_data:
                perf_metrics = json.loads(perf_data)
                avg_speedup = perf_metrics.get("avg_speedup_factor", 4.6)
            else:
                avg_speedup = 4.6

            self.system_health.update({
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "queue_length": queue_length,
                "avg_speedup_factor": avg_speedup
            })

        except Exception as e:
            logger.warning(f"Failed to get system health: {e}")

        return self.system_health

    async def calculate_adaptive_limits(self, user_id: str, tier: str, endpoint_type: str) -> Tuple[int, int]:
        """Calculate adaptive rate limits based on performance and system health."""
        system_health = await self.get_system_health()

        # Base limits by endpoint type
        base_limits = {
            "general": self.config.base_requests_per_minute,
            "analysis": self.config.base_analysis_requests_per_minute,
            "batch": self.config.base_batch_requests_per_hour // 60  # Convert to per-minute
        }

        base_limit = base_limits.get(endpoint_type, self.config.base_requests_per_minute)

        # Apply performance multiplier (4.6x speedup = more capacity)
        performance_adjusted_limit = int(base_limit * self.config.performance_multiplier)

        # Apply user tier multiplier
        tier_multiplier = self.config.user_tier_multipliers.get(tier, 1.0)
        tier_adjusted_limit = int(performance_adjusted_limit * tier_multiplier)

        # Apply system health factor
        health_factor = 1.0
        cpu_usage = system_health.get("cpu_usage", 0)
        memory_usage = system_health.get("memory_usage", 0)
        queue_length = system_health.get("queue_length", 0)

        # Reduce limits if system is under stress
        if cpu_usage > 70 or memory_usage > 80:
            health_factor = 0.7  # Reduce by 30%
        elif queue_length > 50:
            health_factor = 0.5  # Reduce by 50% if queue is very long
        elif cpu_usage < 30 and memory_usage < 50 and queue_length < 10:
            health_factor = 1.3  # Increase by 30% if system is healthy

        # Apply speedup factor (if system is performing better than expected, allow more)
        actual_speedup = system_health.get("avg_speedup_factor", 4.6)
        speedup_bonus = min(1.5, actual_speedup / 4.6)  # Cap bonus at 50%

        # Calculate final limits
        standard_limit = int(tier_adjusted_limit * health_factor)
        burst_limit = int(standard_limit * self.config.burst_allowance * speedup_bonus)

        logger.debug(f"Adaptive limits for {user_id} ({tier}, {endpoint_type}): "
                    f"standard={standard_limit}, burst={burst_limit}, "
                    f"health_factor={health_factor}, speedup_bonus={speedup_bonus}")

        return standard_limit, burst_limit

    async def get_user_metrics(self, user_id: str) -> UserMetrics:
        """Get user-specific metrics from cache."""
        try:
            cached_metrics = await self.redis_client.get(f"rate_limit:metrics:{user_id}")
            if cached_metrics:
                data = json.loads(cached_metrics)
                return UserMetrics(**data)
        except Exception as e:
            logger.warning(f"Failed to get user metrics for {user_id}: {e}")

        return UserMetrics()

    async def update_user_metrics(self, user_id: str, metrics: UserMetrics):
        """Update user metrics in cache."""
        try:
            await self.redis_client.set(
                f"rate_limit:metrics:{user_id}",
                json.dumps(metrics.__dict__),
                ex=3600  # Expire after 1 hour
            )
        except Exception as e:
            logger.warning(f"Failed to update user metrics for {user_id}: {e}")

    async def check_rate_limit(self, request: Request, endpoint_type: str = "general") -> Tuple[bool, Dict[str, Any]]:
        """Check if request should be rate limited."""
        user_id = await self.get_user_identifier(request)
        tier = await self.get_user_tier(user_id, request)
        current_time = time.time()

        # Get current user metrics
        user_metrics = await self.get_user_metrics(user_id)
        user_metrics.tier = tier

        # Calculate adaptive limits
        standard_limit, burst_limit = await self.calculate_adaptive_limits(user_id, tier, endpoint_type)

        # Check different rate limit windows
        rate_limit_keys = [
            f"rate_limit:{user_id}:standard:{int(current_time // self.config.rate_window_seconds)}",
            f"rate_limit:{user_id}:burst:{int(current_time // self.config.burst_window_seconds)}"
        ]

        # Get current counts
        try:
            pipe = self.redis_client.pipeline()
            for key in rate_limit_keys:
                pipe.get(key)
            counts = await pipe.execute()

            standard_count = int(counts[0] or 0)
            burst_count = int(counts[1] or 0)

        except Exception as e:
            logger.warning(f"Failed to get rate limit counts: {e}")
            # Default to allowing the request if Redis fails
            return True, {"error": "rate_limit_check_failed"}

        # Check limits
        rate_limit_info = {
            "user_id": user_id,
            "tier": tier,
            "endpoint_type": endpoint_type,
            "standard_limit": standard_limit,
            "burst_limit": burst_limit,
            "standard_count": standard_count,
            "burst_count": burst_count,
            "standard_remaining": max(0, standard_limit - standard_count),
            "burst_remaining": max(0, burst_limit - burst_count),
            "reset_time": int((current_time // self.config.rate_window_seconds + 1) * self.config.rate_window_seconds),
            "performance_multiplier": self.config.performance_multiplier,
            "system_health": await self.get_system_health()
        }

        # Check if limits exceeded
        if standard_count >= standard_limit:
            return False, {**rate_limit_info, "limit_type": "standard", "exceeded": True}

        if burst_count >= burst_limit:
            return False, {**rate_limit_info, "limit_type": "burst", "exceeded": True}

        # Increment counters
        try:
            pipe = self.redis_client.pipeline()
            pipe.incr(rate_limit_keys[0])
            pipe.expire(rate_limit_keys[0], self.config.rate_window_seconds)
            pipe.incr(rate_limit_keys[1])
            pipe.expire(rate_limit_keys[1], self.config.burst_window_seconds)
            await pipe.execute()
        except Exception as e:
            logger.warning(f"Failed to update rate limit counters: {e}")

        # Update user metrics
        user_metrics.requests_count += 1
        if endpoint_type == "analysis":
            user_metrics.analysis_requests_count += 1
        elif endpoint_type == "batch":
            user_metrics.batch_requests_count += 1

        user_metrics.last_request_timestamp = current_time
        await self.update_user_metrics(user_id, user_metrics)

        return True, rate_limit_info

async def rate_limit_middleware(request: Request, call_next, rate_limiter: AdaptiveRateLimiter):
    """Rate limiting middleware with adaptive behavior."""
    # Determine endpoint type based on path
    path = request.url.path
    endpoint_type = "general"

    if "/analysis" in path or "/analyze" in path:
        endpoint_type = "analysis"
    elif "/batch" in path:
        endpoint_type = "batch"

    # Check rate limit
    allowed, rate_info = await rate_limiter.check_rate_limit(request, endpoint_type)

    if not allowed:
        # Return rate limit exceeded response
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "Rate limit exceeded",
                "message": f"Too many {endpoint_type} requests",
                "rate_limit_info": rate_info,
                "retry_after": rate_info.get("reset_time", 60) - int(time.time()),
                "optimization_info": {
                    "performance_multiplier": rate_info.get("performance_multiplier", 4.6),
                    "adaptive_scaling": "enabled",
                    "system_health": rate_info.get("system_health", {})
                }
            },
            headers={
                "X-RateLimit-Limit": str(rate_info.get("standard_limit", 0)),
                "X-RateLimit-Remaining": str(rate_info.get("standard_remaining", 0)),
                "X-RateLimit-Reset": str(rate_info.get("reset_time", 0)),
                "X-RateLimit-Type": "adaptive",
                "X-Performance-Multiplier": str(rate_info.get("performance_multiplier", 4.6)),
                "Retry-After": str(rate_info.get("reset_time", 60) - int(time.time()))
            }
        )

    # Add rate limit headers to successful responses
    response = await call_next(request)

    response.headers["X-RateLimit-Limit"] = str(rate_info.get("standard_limit", 0))
    response.headers["X-RateLimit-Remaining"] = str(rate_info.get("standard_remaining", 0))
    response.headers["X-RateLimit-Reset"] = str(rate_info.get("reset_time", 0))
    response.headers["X-RateLimit-Type"] = "adaptive"
    response.headers["X-Performance-Multiplier"] = str(rate_info.get("performance_multiplier", 4.6))

    return response