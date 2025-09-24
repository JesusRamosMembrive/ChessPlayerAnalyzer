# app/infrastructure/redis_service.py
"""
Redis Service - Centralized Redis operations.

This service encapsulates all Redis operations including caching,
pub/sub notifications, and locking mechanisms.
"""
import json
import logging
import os
import hashlib
from typing import Optional, Any, Dict, List, Union
from contextlib import contextmanager

import redis
from redis.lock import Lock


class RedisService:
    """
    Centralized Redis operations service.

    Provides caching, pub/sub, and locking functionality
    with proper error handling and connection management.
    """

    def __init__(self, redis_url: Optional[str] = None, decode_responses: bool = True):
        """
        Initialize Redis service.

        Args:
            redis_url: Redis connection URL (defaults to env var)
            decode_responses: Whether to decode responses as strings
        """
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://redis:6379/0")
        self.decode_responses = decode_responses
        self._client: Optional[redis.Redis] = None
        self.logger = logging.getLogger(__name__)

    @property
    def client(self) -> redis.Redis:
        """Get Redis client, creating if needed."""
        if self._client is None:
            self._client = redis.Redis.from_url(
                self.redis_url,
                decode_responses=self.decode_responses
            )
        return self._client

    def ping(self) -> bool:
        """Test Redis connection."""
        try:
            self.client.ping()
            return True
        except Exception as e:
            self.logger.error(f"Redis ping failed: {e}")
            return False

    # ── Caching Operations ──

    def cache_get(self, task_name: str, args: Union[list, tuple], kwargs: dict) -> Optional[dict]:
        """
        Get cached result for task with given arguments.

        Args:
            task_name: Name of the task
            args: Task positional arguments
            kwargs: Task keyword arguments

        Returns:
            Cached result dict or None if not found
        """
        try:
            key = self._generate_cache_key(task_name, args, kwargs)
            cached = self.client.get(key)
            if cached:
                return json.loads(cached)
            return None
        except Exception as exc:
            self.logger.warning(f"cache_get: could not decode cached value for {task_name}: {exc}")
            return None

    def cache_set(self, task_name: str, args: Union[list, tuple], kwargs: dict,
                  result: dict, ttl: int = 86400) -> bool:
        """
        Cache result for task with given arguments.

        Args:
            task_name: Name of the task
            args: Task positional arguments
            kwargs: Task keyword arguments
            result: Result to cache
            ttl: Time to live in seconds (default: 1 day)

        Returns:
            True if successfully cached, False otherwise
        """
        try:
            key = self._generate_cache_key(task_name, args, kwargs)
            self.client.setex(key, ttl, json.dumps(result, default=str))
            return True
        except Exception as exc:
            self.logger.error(f"cache_set: could not store result for {task_name}: {exc}")
            return False

    def cache_delete(self, task_name: str, args: Union[list, tuple], kwargs: dict) -> bool:
        """Delete cached result for task."""
        try:
            key = self._generate_cache_key(task_name, args, kwargs)
            return bool(self.client.delete(key))
        except Exception as e:
            self.logger.error(f"cache_delete failed: {e}")
            return False

    def cache_clear_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern."""
        try:
            keys = self.client.keys(pattern)
            if keys:
                return self.client.delete(*keys)
            return 0
        except Exception as e:
            self.logger.error(f"cache_clear_pattern failed: {e}")
            return 0

    # ── Pub/Sub Operations ──

    def publish(self, channel: str, message: dict) -> bool:
        """
        Publish message to Redis channel.

        Args:
            channel: Channel name
            message: Message dict to publish

        Returns:
            True if published successfully, False otherwise
        """
        try:
            payload = json.dumps(message)
            self.client.publish(channel, payload)
            return True
        except Exception as e:
            self.logger.error(f"publish failed: {e}")
            return False

    def subscribe(self, channels: Union[str, List[str]]):
        """
        Subscribe to Redis channels.

        Args:
            channels: Channel name or list of channel names

        Returns:
            Redis PubSub instance
        """
        pubsub = self.client.pubsub()
        if isinstance(channels, str):
            channels = [channels]
        pubsub.subscribe(*channels)
        return pubsub

    # ── Lock Operations ──

    @contextmanager
    def lock(self, name: str, timeout: int = 300, blocking_timeout: Optional[int] = None):
        """
        Context manager for Redis distributed lock.

        Args:
            name: Lock name
            timeout: Lock timeout in seconds
            blocking_timeout: Time to wait for lock acquisition

        Yields:
            Lock instance

        Raises:
            redis.exceptions.LockError: If lock cannot be acquired
        """
        lock = self.client.lock(name, timeout=timeout)
        try:
            acquired = lock.acquire(blocking_timeout=blocking_timeout)
            if not acquired:
                raise redis.exceptions.LockError(f"Could not acquire lock: {name}")
            yield lock
        finally:
            try:
                lock.release()
            except Exception as e:
                self.logger.warning(f"Error releasing lock {name}: {e}")

    def is_locked(self, name: str) -> bool:
        """Check if lock exists."""
        try:
            return bool(self.client.get(name))
        except Exception:
            return False

    # ── Key Operations ──

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set key-value pair."""
        try:
            if ttl:
                return self.client.setex(key, ttl, value)
            else:
                return self.client.set(key, value)
        except Exception as e:
            self.logger.error(f"set failed: {e}")
            return False

    def get(self, key: str) -> Optional[str]:
        """Get value by key."""
        try:
            return self.client.get(key)
        except Exception as e:
            self.logger.error(f"get failed: {e}")
            return None

    def delete(self, *keys: str) -> int:
        """Delete keys."""
        try:
            return self.client.delete(*keys)
        except Exception as e:
            self.logger.error(f"delete failed: {e}")
            return 0

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        try:
            return bool(self.client.exists(key))
        except Exception as e:
            self.logger.error(f"exists failed: {e}")
            return False

    # ── Private Methods ──

    def _generate_cache_key(self, task_name: str, args: Union[list, tuple], kwargs: dict) -> str:
        """Generate cache key from task name and arguments."""
        # Convert args and kwargs to a consistent string representation
        args_str = str(sorted(args)) if args else ""
        kwargs_str = str(sorted(kwargs.items())) if kwargs else ""
        combined = f"{task_name}:{args_str}:{kwargs_str}"

        # Hash to avoid very long keys
        key_hash = hashlib.sha256(combined.encode()).hexdigest()[:16]
        return f"cache:{task_name}:{key_hash}"


# Global instance (for backward compatibility with utils.py)
_redis_service: Optional[RedisService] = None


def get_redis_service() -> RedisService:
    """Get global Redis service instance."""
    global _redis_service
    if _redis_service is None:
        _redis_service = RedisService()
    return _redis_service


def create_redis_service(redis_url: Optional[str] = None) -> RedisService:
    """Create new Redis service instance (for testing)."""
    return RedisService(redis_url=redis_url)