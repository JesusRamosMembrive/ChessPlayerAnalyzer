# tests/refactor/unit/test_analysis_lock_service.py
"""
Unit tests for AnalysisLockService.

These tests validate the unified lock service behavior using mocks,
ensuring it properly consolidates the 3 different lock implementations.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from contextlib import contextmanager

from app.services.analysis_lock import (
    AnalysisLockService,
    LockType,
    get_analysis_lock_service,
    create_analysis_lock_service
)


@pytest.fixture
def mock_redis_service():
    """Mock Redis service for testing."""
    mock = Mock()
    # Setup default behaviors
    mock.set.return_value = True
    mock.delete.return_value = 1
    mock.get.return_value = None
    mock.exists.return_value = False
    return mock


@pytest.fixture
def lock_service(mock_redis_service):
    """AnalysisLockService with mocked Redis service."""
    return AnalysisLockService(redis_service=mock_redis_service)


class TestAnalysisLockService:
    """Test cases for AnalysisLockService."""

    def test_initialization(self, mock_redis_service):
        """Test that service initializes correctly."""
        service = AnalysisLockService(redis_service=mock_redis_service)

        assert service.redis_service == mock_redis_service
        assert LockType.PLAYER_ANALYSIS in service.LOCK_KEYS
        assert LockType.GLOBAL_ANALYSIS in service.LOCK_KEYS
        assert LockType.CLEANUP in service.LOCK_KEYS

    def test_player_lock_success(self, lock_service, mock_redis_service):
        """Test successful player lock acquisition and release."""
        mock_lock = Mock()
        mock_lock.__enter__ = Mock(return_value=mock_lock)
        mock_lock.__exit__ = Mock(return_value=None)
        mock_redis_service.lock.return_value = mock_lock

        with lock_service.player_lock("test_player") as lock:
            assert lock == mock_lock

        # Verify lock was created with correct parameters
        mock_redis_service.lock.assert_called_once_with(
            "lock:player:test_player",
            timeout=900,  # Default timeout
            blocking_timeout=5
        )

    def test_player_lock_failure(self, lock_service, mock_redis_service):
        """Test player lock acquisition failure."""
        mock_redis_service.lock.side_effect = Exception("Could not acquire lock")

        with pytest.raises(RuntimeError, match="player 'test_player' is already locked"):
            with lock_service.player_lock("test_player"):
                pass

    def test_is_player_locked(self, lock_service, mock_redis_service):
        """Test checking if player is locked."""
        mock_redis_service.exists.return_value = True

        result = lock_service.is_player_locked("test_player")

        assert result is True
        mock_redis_service.exists.assert_called_once_with("lock:player:test_player")

    def test_set_global_analysis_lock(self, lock_service, mock_redis_service):
        """Test setting global analysis lock."""
        mock_redis_service.set.return_value = True

        result = lock_service.set_global_analysis_lock("test_user")

        assert result is True
        mock_redis_service.set.assert_called_once_with(
            "analysis_in_progress",
            "test_user",
            ttl=7200  # Default timeout
        )

    def test_clear_global_analysis_lock(self, lock_service, mock_redis_service):
        """Test clearing global analysis lock."""
        mock_redis_service.delete.return_value = 1

        result = lock_service.clear_global_analysis_lock()

        assert result is True
        mock_redis_service.delete.assert_called_once_with("analysis_in_progress")

    def test_get_global_analysis_user(self, lock_service, mock_redis_service):
        """Test getting current global analysis user."""
        mock_redis_service.get.return_value = "current_user"

        result = lock_service.get_global_analysis_user()

        assert result == "current_user"
        mock_redis_service.get.assert_called_once_with("analysis_in_progress")

    def test_is_global_analysis_in_progress(self, lock_service, mock_redis_service):
        """Test checking if global analysis is in progress."""
        mock_redis_service.get.return_value = "some_user"

        result = lock_service.is_global_analysis_in_progress()

        assert result is True

        # Test when no analysis is in progress
        mock_redis_service.get.return_value = None
        result = lock_service.is_global_analysis_in_progress()
        assert result is False

    def test_set_cleanup_lock(self, lock_service, mock_redis_service):
        """Test setting cleanup lock."""
        mock_redis_service.set.return_value = True

        result = lock_service.set_cleanup_lock("cleanup_user")

        assert result is True
        mock_redis_service.set.assert_called_once_with(
            "cleanup_in_progress",
            "cleanup_user",
            ttl=3600  # Default timeout
        )

    def test_clear_cleanup_lock(self, lock_service, mock_redis_service):
        """Test clearing cleanup lock."""
        mock_redis_service.delete.return_value = 1

        result = lock_service.clear_cleanup_lock()

        assert result is True
        mock_redis_service.delete.assert_called_once_with("cleanup_in_progress")

    def test_get_cleanup_user(self, lock_service, mock_redis_service):
        """Test getting current cleanup user."""
        mock_redis_service.get.return_value = "cleanup_user"

        result = lock_service.get_cleanup_user()

        assert result == "cleanup_user"
        mock_redis_service.get.assert_called_once_with("cleanup_in_progress")

    def test_is_cleanup_in_progress(self, lock_service, mock_redis_service):
        """Test checking if cleanup is in progress."""
        mock_redis_service.get.return_value = "cleanup_user"

        result = lock_service.is_cleanup_in_progress()

        assert result is True

        # Test when no cleanup is in progress
        mock_redis_service.get.return_value = None
        result = lock_service.is_cleanup_in_progress()
        assert result is False

    def test_check_analysis_preconditions_success(self, lock_service, mock_redis_service):
        """Test analysis preconditions when no conflicts exist."""
        mock_redis_service.get.return_value = None  # No locks active

        result = lock_service.check_analysis_preconditions("test_user")

        assert result["can_proceed"] is True
        assert len(result["conflicts"]) == 0
        assert result["current_user"] == "test_user"

    def test_check_analysis_preconditions_cleanup_conflict(self, lock_service, mock_redis_service):
        """Test analysis preconditions when cleanup is in progress."""
        def mock_get_side_effect(key):
            if key == "cleanup_in_progress":
                return "other_user"
            return None

        mock_redis_service.get.side_effect = mock_get_side_effect

        result = lock_service.check_analysis_preconditions("test_user")

        assert result["can_proceed"] is False
        assert len(result["conflicts"]) == 1
        assert result["conflicts"][0]["type"] == "cleanup_in_progress"
        assert result["conflicts"][0]["user"] == "other_user"

    def test_check_analysis_preconditions_analysis_conflict(self, lock_service, mock_redis_service):
        """Test analysis preconditions when analysis is in progress for different user."""
        def mock_get_side_effect(key):
            if key == "analysis_in_progress":
                return "other_user"
            return None

        mock_redis_service.get.side_effect = mock_get_side_effect

        result = lock_service.check_analysis_preconditions("test_user")

        assert result["can_proceed"] is False
        assert len(result["conflicts"]) == 1
        assert result["conflicts"][0]["type"] == "analysis_in_progress"
        assert result["conflicts"][0]["user"] == "other_user"

    def test_check_analysis_preconditions_same_user_analysis(self, lock_service, mock_redis_service):
        """Test analysis preconditions when same user has analysis in progress."""
        def mock_get_side_effect(key):
            if key == "analysis_in_progress":
                return "test_user"
            return None

        mock_redis_service.get.side_effect = mock_get_side_effect

        result = lock_service.check_analysis_preconditions("test_user")

        # Same user should be able to proceed
        assert result["can_proceed"] is True
        assert len(result["conflicts"]) == 0

    def test_cleanup_stale_locks(self, lock_service, mock_redis_service):
        """Test cleanup of stale locks."""
        mock_redis_service.get.return_value = None

        result = lock_service.cleanup_stale_locks()

        # Should return counts structure
        assert "player_locks" in result
        assert "global_locks" in result
        assert "cleanup_locks" in result

    def test_custom_timeouts(self, lock_service, mock_redis_service):
        """Test that custom timeouts are respected."""
        mock_lock = Mock()
        mock_lock.__enter__ = Mock(return_value=mock_lock)
        mock_lock.__exit__ = Mock(return_value=None)
        mock_redis_service.lock.return_value = mock_lock

        # Test player lock with custom timeout
        with lock_service.player_lock("test_player", timeout=300, blocking_timeout=10):
            pass

        mock_redis_service.lock.assert_called_with(
            "lock:player:test_player",
            timeout=300,
            blocking_timeout=10
        )

        # Test global analysis lock with custom timeout
        lock_service.set_global_analysis_lock("test_user", timeout=1800)
        mock_redis_service.set.assert_called_with(
            "analysis_in_progress",
            "test_user",
            ttl=1800
        )


class TestModuleFunctions:
    """Test module-level functions."""

    @patch('app.services.analysis_lock._analysis_lock_service', None)
    def test_get_analysis_lock_service_singleton(self):
        """Test that get_analysis_lock_service returns singleton."""
        service1 = get_analysis_lock_service()
        service2 = get_analysis_lock_service()

        assert service1 is service2
        assert isinstance(service1, AnalysisLockService)

    def test_create_analysis_lock_service(self):
        """Test create_analysis_lock_service function."""
        mock_redis = Mock()
        service = create_analysis_lock_service(redis_service=mock_redis)

        assert isinstance(service, AnalysisLockService)
        assert service.redis_service is mock_redis