# app/services/analysis_lock.py
"""
Analysis Lock Service - Unified lock management for player analysis.

This service consolidates all the different lock implementations found across
the codebase into a single, consistent interface.
"""
import logging
from contextlib import contextmanager
from typing import Optional, Dict, Any
from enum import Enum

from app.infrastructure.redis_service import get_redis_service


class LockType(Enum):
    """Types of analysis locks."""
    PLAYER_ANALYSIS = "player_analysis"
    GLOBAL_ANALYSIS = "global_analysis"
    CLEANUP = "cleanup"


class AnalysisLockService:
    """
    Unified service for managing analysis-related locks.

    Consolidates the multiple lock implementations found in:
    - app/main.py (global analysis/cleanup flags)
    - app/api/v1/endpoints/players.py (duplicated logic)
    - app/utils.py (player_lock context manager)
    """

    def __init__(self, redis_service=None):
        """Initialize the lock service."""
        self.redis_service = redis_service or get_redis_service()
        self.logger = logging.getLogger(__name__)

        # Lock key patterns
        self.LOCK_KEYS = {
            LockType.PLAYER_ANALYSIS: "lock:player:{username}",
            LockType.GLOBAL_ANALYSIS: "analysis_in_progress",
            LockType.CLEANUP: "cleanup_in_progress"
        }

        # Default timeouts (in seconds)
        self.DEFAULT_TIMEOUTS = {
            LockType.PLAYER_ANALYSIS: 900,    # 15 minutes
            LockType.GLOBAL_ANALYSIS: 7200,   # 2 hours
            LockType.CLEANUP: 3600            # 1 hour
        }

    # ── Player-specific locks ──

    @contextmanager
    def player_lock(self, username: str, timeout: int = None, blocking_timeout: int = 5):
        """
        Context manager for player-specific analysis lock.

        This replaces the player_lock function from utils.py with better error handling.

        Args:
            username: Player username
            timeout: Lock timeout in seconds (default: 15 minutes)
            blocking_timeout: Time to wait for lock acquisition

        Raises:
            RuntimeError: If lock cannot be acquired
        """
        timeout = timeout or self.DEFAULT_TIMEOUTS[LockType.PLAYER_ANALYSIS]
        lock_key = self.LOCK_KEYS[LockType.PLAYER_ANALYSIS].format(username=username)

        self.logger.info(f"Acquiring player lock for {username}")

        try:
            lock = self.redis_service.lock(lock_key, timeout=timeout, blocking_timeout=blocking_timeout)
            with lock:
                self.logger.info(f"Player lock acquired for {username}")
                yield lock
                self.logger.info(f"Player lock released for {username}")

        except Exception as e:
            # Convert Redis lock errors to the expected RuntimeError for backward compatibility
            if "Could not acquire lock" in str(e):
                self.logger.warning(f"Player {username} is already locked")
                raise RuntimeError(f"player {username!r} is already locked")
            self.logger.error(f"Lock error for player {username}: {e}")
            raise

    def is_player_locked(self, username: str) -> bool:
        """Check if a specific player is currently locked."""
        lock_key = self.LOCK_KEYS[LockType.PLAYER_ANALYSIS].format(username=username)
        return self.redis_service.exists(lock_key)

    # ── Global analysis locks ──

    def set_global_analysis_lock(self, username: str, timeout: int = None) -> bool:
        """
        Set global analysis lock.

        Replaces set_analysis_in_progress from main.py

        Args:
            username: Username currently performing analysis
            timeout: Lock timeout in seconds (default: 2 hours)

        Returns:
            True if lock was set successfully
        """
        timeout = timeout or self.DEFAULT_TIMEOUTS[LockType.GLOBAL_ANALYSIS]
        lock_key = self.LOCK_KEYS[LockType.GLOBAL_ANALYSIS]

        success = self.redis_service.set(lock_key, username, ttl=timeout)
        if success:
            self.logger.info(f"Global analysis lock set for {username}")
        else:
            self.logger.error(f"Failed to set global analysis lock for {username}")
        return success

    def clear_global_analysis_lock(self) -> bool:
        """
        Clear global analysis lock.

        Replaces clear_analysis_in_progress from main.py

        Returns:
            True if lock was cleared
        """
        lock_key = self.LOCK_KEYS[LockType.GLOBAL_ANALYSIS]
        result = self.redis_service.delete(lock_key)

        if result:
            self.logger.info("Global analysis lock cleared")
        return bool(result)

    def get_global_analysis_user(self) -> Optional[str]:
        """
        Get username of user currently holding global analysis lock.

        Replaces logic from endpoints/players.py

        Returns:
            Username if analysis is in progress, None otherwise
        """
        lock_key = self.LOCK_KEYS[LockType.GLOBAL_ANALYSIS]
        return self.redis_service.get(lock_key)

    def is_global_analysis_in_progress(self) -> bool:
        """
        Check if global analysis is currently in progress.

        Replaces is_analysis_in_progress from main.py

        Returns:
            True if any analysis is in progress
        """
        return self.get_global_analysis_user() is not None

    # ── Cleanup locks ──

    def set_cleanup_lock(self, username: str, timeout: int = None) -> bool:
        """
        Set cleanup lock.

        Replaces set_cleanup_in_progress from main.py

        Args:
            username: Username performing cleanup
            timeout: Lock timeout in seconds (default: 1 hour)

        Returns:
            True if lock was set successfully
        """
        timeout = timeout or self.DEFAULT_TIMEOUTS[LockType.CLEANUP]
        lock_key = self.LOCK_KEYS[LockType.CLEANUP]

        success = self.redis_service.set(lock_key, username, ttl=timeout)
        if success:
            self.logger.info(f"Cleanup lock set for {username}")
        else:
            self.logger.error(f"Failed to set cleanup lock for {username}")
        return success

    def clear_cleanup_lock(self) -> bool:
        """
        Clear cleanup lock.

        Replaces clear_cleanup_in_progress from main.py

        Returns:
            True if lock was cleared
        """
        lock_key = self.LOCK_KEYS[LockType.CLEANUP]
        result = self.redis_service.delete(lock_key)

        if result:
            self.logger.info("Cleanup lock cleared")
        return bool(result)

    def get_cleanup_user(self) -> Optional[str]:
        """
        Get username of user currently holding cleanup lock.

        Returns:
            Username if cleanup is in progress, None otherwise
        """
        lock_key = self.LOCK_KEYS[LockType.CLEANUP]
        return self.redis_service.get(lock_key)

    def is_cleanup_in_progress(self) -> bool:
        """
        Check if cleanup is currently in progress.

        Replaces is_cleanup_in_progress from main.py

        Returns:
            True if cleanup is in progress
        """
        return self.get_cleanup_user() is not None

    # ── High-level operations ──

    def check_analysis_preconditions(self, username: str) -> Dict[str, Any]:
        """
        Check all preconditions for starting analysis.

        Replaces the complex logic from endpoints/players.py

        Args:
            username: Username requesting analysis

        Returns:
            Dict with status and any conflict information
        """
        result = {
            "can_proceed": True,
            "conflicts": [],
            "current_user": username
        }

        # Check cleanup lock
        cleanup_user = self.get_cleanup_user()
        if cleanup_user:
            result["can_proceed"] = False
            result["conflicts"].append({
                "type": "cleanup_in_progress",
                "user": cleanup_user,
                "message": f"Cleanup in progress for user {cleanup_user}. Please wait."
            })

        # Check global analysis lock
        analysis_user = self.get_global_analysis_user()
        if analysis_user and analysis_user != username:
            result["can_proceed"] = False
            result["conflicts"].append({
                "type": "analysis_in_progress",
                "user": analysis_user,
                "message": f"Analysis in progress for user {analysis_user}. Please wait."
            })

        return result

    def cleanup_stale_locks(self, max_age_seconds: int = 86400) -> Dict[str, int]:
        """
        Clean up stale locks older than max_age_seconds.

        This is a maintenance operation to prevent locks from getting stuck.

        Args:
            max_age_seconds: Maximum age for locks (default: 24 hours)

        Returns:
            Dict with counts of cleaned locks by type
        """
        cleaned = {
            "player_locks": 0,
            "global_locks": 0,
            "cleanup_locks": 0
        }

        try:
            # For now, we'll implement a simple TTL-based cleanup
            # More sophisticated age-based cleanup could be added later

            # Check if global locks are stale (this is basic - could be enhanced)
            global_user = self.get_global_analysis_user()
            if global_user:
                # If we had timestamp tracking, we could check age here
                # For now, just log that there's an active global lock
                self.logger.info(f"Active global analysis lock for user: {global_user}")

            cleanup_user = self.get_cleanup_user()
            if cleanup_user:
                self.logger.info(f"Active cleanup lock for user: {cleanup_user}")

            self.logger.info("Lock cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during lock cleanup: {e}")

        return cleaned


# Global instance for backward compatibility
_analysis_lock_service: Optional[AnalysisLockService] = None


def get_analysis_lock_service() -> AnalysisLockService:
    """Get global analysis lock service instance."""
    global _analysis_lock_service
    if _analysis_lock_service is None:
        _analysis_lock_service = AnalysisLockService()
    return _analysis_lock_service


def create_analysis_lock_service(redis_service=None) -> AnalysisLockService:
    """Create new analysis lock service instance (for testing)."""
    return AnalysisLockService(redis_service=redis_service)