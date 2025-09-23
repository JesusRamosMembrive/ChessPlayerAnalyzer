"""
Performance optimizations and configuration tuning.
Centralized performance settings for the clean architecture.
"""
from typing import Dict, Any
import asyncio
from functools import lru_cache
from datetime import datetime, timedelta

from .config import get_config


class PerformanceOptimizer:
    """Centralized performance optimization manager."""

    def __init__(self):
        self.config = get_config()
        self._cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0
        }

    @lru_cache(maxsize=1000)
    def get_player_cache_key(self, username: str, include_analysis: bool = False) -> str:
        """Generate optimized cache key for player data."""
        suffix = "_with_analysis" if include_analysis else ""
        return f"player:{username.lower()}{suffix}"

    @lru_cache(maxsize=500)
    def get_game_cache_key(self, game_id: int, include_moves: bool = False) -> str:
        """Generate optimized cache key for game data."""
        suffix = "_with_moves" if include_moves else ""
        return f"game:{game_id}{suffix}"

    def get_database_optimization_settings(self) -> Dict[str, Any]:
        """Get optimized database connection settings."""
        return {
            # Connection pool optimization
            "pool_size": 20,  # Increased from default 5
            "max_overflow": 30,  # Increased from default 10
            "pool_timeout": 30,
            "pool_recycle": 3600,  # 1 hour
            "pool_pre_ping": True,

            # Query optimization
            "echo": False,  # Disable in production
            "future": True,

            # Connection optimization
            "connect_args": {
                "check_same_thread": False,
                "timeout": 20,
                # SQLite specific optimizations
                "pragmas": {
                    "journal_mode": "WAL",
                    "cache_size": -1 * 64000,  # 64MB cache
                    "foreign_keys": 1,
                    "ignore_check_constraints": 0,
                    "synchronous": 0,
                    "temp_store": 2,
                    "mmap_size": 268435456,  # 256MB mmap
                }
            }
        }

    def get_redis_optimization_settings(self) -> Dict[str, Any]:
        """Get optimized Redis connection settings."""
        return {
            "max_connections": 50,  # Increased from default 10
            "retry_on_timeout": True,
            "health_check_interval": 30,
            "socket_timeout": 5,
            "socket_connect_timeout": 5,
            "socket_keepalive": True,
            "socket_keepalive_options": {}
        }

    def get_celery_optimization_settings(self) -> Dict[str, Any]:
        """Get optimized Celery configuration."""
        return {
            # Task execution optimization
            "task_serializer": "json",
            "result_serializer": "json",
            "accept_content": ["json"],
            "result_accept_content": ["json"],

            # Worker optimization
            "worker_prefetch_multiplier": 4,  # Increased from 1
            "worker_max_tasks_per_child": 1000,
            "worker_disable_rate_limits": False,
            "worker_send_task_events": True,

            # Result backend optimization
            "result_expires": 3600,  # 1 hour
            "result_persistent": True,
            "result_compression": "gzip",

            # Task execution settings
            "task_time_limit": 30 * 60,  # 30 minutes
            "task_soft_time_limit": 25 * 60,  # 25 minutes
            "task_acks_late": True,
            "task_reject_on_worker_lost": True,

            # Concurrency optimization
            "worker_concurrency": 4,  # Adjust based on CPU cores
            "broker_connection_retry_on_startup": True,
            "broker_connection_retry": True,

            # Monitoring
            "worker_send_task_events": True,
            "task_send_sent_event": True,
        }

    def get_api_optimization_settings(self) -> Dict[str, Any]:
        """Get optimized FastAPI settings."""
        return {
            # Response optimization
            "default_response_class": "fastapi.responses.JSONResponse",
            "response_model_exclude_unset": True,
            "response_model_exclude_none": True,

            # Request optimization
            "max_request_size": 10 * 1024 * 1024,  # 10MB
            "timeout_keep_alive": 75,

            # Concurrency
            "limit_concurrency": 100,
            "limit_max_requests": 1000,

            # CORS optimization
            "cors_allow_origins": ["*"],  # Configure for production
            "cors_allow_methods": ["GET", "POST", "PUT", "DELETE"],
            "cors_allow_headers": ["*"],
            "cors_max_age": 600,  # 10 minutes
        }

    async def warm_up_services(self) -> Dict[str, bool]:
        """Warm up critical services during startup."""
        results = {}

        # Warm up database connections
        try:
            from ..database import engine
            with engine.connect() as conn:
                conn.execute("SELECT 1")
            results["database"] = True
        except Exception:
            results["database"] = False

        # Warm up Redis connections
        try:
            from ..utils import redis_client
            redis_client.ping()
            results["redis"] = True
        except Exception:
            results["redis"] = False

        # Warm up DI container
        try:
            from ..application.container import get_container
            container = get_container()
            container.get_get_player_status_use_case()
            results["container"] = True
        except Exception:
            results["container"] = False

        return results

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self._cache_stats["hits"] + self._cache_stats["misses"]
        hit_rate = (
            self._cache_stats["hits"] / total_requests * 100
            if total_requests > 0 else 0
        )

        return {
            "hits": self._cache_stats["hits"],
            "misses": self._cache_stats["misses"],
            "evictions": self._cache_stats["evictions"],
            "total_requests": total_requests,
            "hit_rate_percentage": round(hit_rate, 2)
        }

    async def optimize_query_performance(self, query_type: str, **kwargs) -> Dict[str, Any]:
        """Apply query-specific optimizations."""
        optimizations = {
            "player_lookup": {
                "use_index": "idx_player_username",
                "limit_results": True,
                "cache_duration": 300  # 5 minutes
            },
            "game_analysis": {
                "use_index": "idx_game_player_id",
                "batch_size": 50,
                "cache_duration": 600  # 10 minutes
            },
            "player_analysis": {
                "use_index": "idx_analysis_player_id",
                "preload_relations": ["games", "metrics"],
                "cache_duration": 900  # 15 minutes
            }
        }

        return optimizations.get(query_type, {})


# Global performance optimizer instance
performance_optimizer = PerformanceOptimizer()


# Performance monitoring decorators
def monitor_performance(func_name: str):
    """Decorator to monitor function performance."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            start_time = datetime.utcnow()
            try:
                result = await func(*args, **kwargs)
                success = True
                error = None
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                duration = (datetime.utcnow() - start_time).total_seconds()
                # Log performance metrics
                print(f"PERF: {func_name} - {duration:.3f}s - {'SUCCESS' if success else 'ERROR'}")
                if error:
                    print(f"ERROR: {error}")
            return result

        def sync_wrapper(*args, **kwargs):
            start_time = datetime.utcnow()
            try:
                result = func(*args, **kwargs)
                success = True
                error = None
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                duration = (datetime.utcnow() - start_time).total_seconds()
                print(f"PERF: {func_name} - {duration:.3f}s - {'SUCCESS' if success else 'ERROR'}")
                if error:
                    print(f"ERROR: {error}")
            return result

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


# Cache utilities
class SimpleCache:
    """Simple in-memory cache with TTL support."""

    def __init__(self, default_ttl: int = 300):
        self._cache = {}
        self._expires = {}
        self.default_ttl = default_ttl

    def get(self, key: str):
        """Get value from cache."""
        if key in self._cache:
            if datetime.utcnow() < self._expires[key]:
                performance_optimizer._cache_stats["hits"] += 1
                return self._cache[key]
            else:
                # Expired
                del self._cache[key]
                del self._expires[key]
                performance_optimizer._cache_stats["evictions"] += 1

        performance_optimizer._cache_stats["misses"] += 1
        return None

    def set(self, key: str, value, ttl: int = None):
        """Set value in cache with TTL."""
        ttl = ttl or self.default_ttl
        self._cache[key] = value
        self._expires[key] = datetime.utcnow() + timedelta(seconds=ttl)

    def clear(self):
        """Clear all cache."""
        self._cache.clear()
        self._expires.clear()


# Global simple cache instance
simple_cache = SimpleCache()