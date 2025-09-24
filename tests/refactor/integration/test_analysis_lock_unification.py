# tests/refactor/integration/test_analysis_lock_unification.py
"""
Integration tests for Analysis Lock Service unification.

These tests validate that the AnalysisLockService properly consolidates
all the different lock implementations and maintains backward compatibility.
"""
import pytest
from unittest.mock import Mock, patch

from app.services.analysis_lock import get_analysis_lock_service, create_analysis_lock_service


class TestAnalysisLockUnification:
    """Test that all lock implementations are properly unified."""

    @pytest.fixture
    def mock_redis_service(self):
        """Mock Redis service for testing."""
        mock = Mock()
        mock.set.return_value = True
        mock.delete.return_value = 1
        mock.get.return_value = None
        mock.exists.return_value = False

        # Mock lock context manager
        mock_lock = Mock()
        mock_lock.__enter__ = Mock(return_value=mock_lock)
        mock_lock.__exit__ = Mock(return_value=None)
        mock.lock.return_value = mock_lock

        return mock

    @pytest.fixture
    def lock_service(self, mock_redis_service):
        """AnalysisLockService with mocked Redis."""
        return create_analysis_lock_service(redis_service=mock_redis_service)

    def test_main_py_functions_compatibility(self, lock_service, mock_redis_service):
        """Test that main.py functions work through AnalysisLockService."""
        # Mock the service being used in main.py
        with patch('app.main._analysis_lock_service', lock_service):
            from app.main import (
                is_cleanup_in_progress,
                is_analysis_in_progress,
                set_cleanup_in_progress,
                clear_cleanup_in_progress,
                set_analysis_in_progress,
                clear_analysis_in_progress
            )

            # Test cleanup functions
            result = set_cleanup_in_progress("test_user")
            assert result is True
            mock_redis_service.set.assert_called_with("cleanup_in_progress", "test_user", ttl=3600)

            mock_redis_service.get.return_value = "test_user"
            assert is_cleanup_in_progress() is True

            result = clear_cleanup_in_progress()
            assert result is True
            mock_redis_service.delete.assert_called_with("cleanup_in_progress")

            # Test analysis functions
            mock_redis_service.reset_mock()
            result = set_analysis_in_progress("test_user")
            assert result is True
            mock_redis_service.set.assert_called_with("analysis_in_progress", "test_user", ttl=7200)

            mock_redis_service.get.return_value = "test_user"
            assert is_analysis_in_progress() is True

            result = clear_analysis_in_progress()
            assert result is True
            mock_redis_service.delete.assert_called_with("analysis_in_progress")

    def test_utils_py_player_lock_compatibility(self, lock_service, mock_redis_service):
        """Test that utils.py player_lock works through AnalysisLockService."""
        with patch('app.utils._analysis_lock_service', lock_service):
            from app.utils import player_lock

            # Test successful lock
            with player_lock("test_player", timeout=300, block=10):
                pass

            mock_redis_service.lock.assert_called_once_with(
                "lock:player:test_player",
                timeout=300,
                blocking_timeout=10
            )

    def test_endpoints_preconditions_integration(self, lock_service, mock_redis_service):
        """Test that endpoints can use check_analysis_preconditions."""
        # Test no conflicts
        result = lock_service.check_analysis_preconditions("test_user")
        assert result["can_proceed"] is True
        assert len(result["conflicts"]) == 0

        # Test cleanup conflict
        def mock_get_cleanup(key):
            if key == "cleanup_in_progress":
                return "other_user"
            return None

        mock_redis_service.get.side_effect = mock_get_cleanup
        result = lock_service.check_analysis_preconditions("test_user")
        assert result["can_proceed"] is False
        assert len(result["conflicts"]) == 1
        assert "Cleanup in progress" in result["conflicts"][0]["message"]

    def test_lock_key_consistency(self, lock_service):
        """Test that all lock keys are consistent across implementations."""
        # Check that the service uses the same keys that were in the original code
        assert "lock:player:{username}" == lock_service.LOCK_KEYS.get(lock_service.LockType.PLAYER_ANALYSIS)
        assert "analysis_in_progress" == lock_service.LOCK_KEYS.get(lock_service.LockType.GLOBAL_ANALYSIS)
        assert "cleanup_in_progress" == lock_service.LOCK_KEYS.get(lock_service.LockType.CLEANUP)

    def test_timeout_consistency(self, lock_service):
        """Test that timeouts match the original implementations."""
        # Check default timeouts match original implementations
        assert lock_service.DEFAULT_TIMEOUTS.get(lock_service.LockType.PLAYER_ANALYSIS) == 900   # 15 min
        assert lock_service.DEFAULT_TIMEOUTS.get(lock_service.LockType.GLOBAL_ANALYSIS) == 7200  # 2 hours
        assert lock_service.DEFAULT_TIMEOUTS.get(lock_service.LockType.CLEANUP) == 3600          # 1 hour

    def test_error_handling_compatibility(self, lock_service, mock_redis_service):
        """Test that error handling maintains backward compatibility."""
        # Setup lock to fail
        mock_redis_service.lock.side_effect = Exception("Could not acquire lock")

        with patch('app.utils._analysis_lock_service', lock_service):
            from app.utils import player_lock

            with pytest.raises(RuntimeError, match="player 'test_player' is already locked"):
                with player_lock("test_player"):
                    pass

    def test_singleton_behavior(self):
        """Test that get_analysis_lock_service returns singleton."""
        service1 = get_analysis_lock_service()
        service2 = get_analysis_lock_service()
        assert service1 is service2

    def test_api_endpoints_integration_flow(self, lock_service, mock_redis_service):
        """Test the complete flow that would happen in API endpoints."""
        # Simulate the flow from app/api/v1/endpoints/players.py

        # 1. Check preconditions (should pass)
        preconditions = lock_service.check_analysis_preconditions("test_user")
        assert preconditions["can_proceed"] is True

        # 2. Set global analysis lock
        result = lock_service.set_global_analysis_lock("test_user")
        assert result is True

        # 3. Simulate that analysis is now in progress for same user (should still pass)
        mock_redis_service.get.return_value = "test_user"
        preconditions = lock_service.check_analysis_preconditions("test_user")
        assert preconditions["can_proceed"] is True

        # 4. Simulate different user trying (should fail)
        preconditions = lock_service.check_analysis_preconditions("other_user")
        assert preconditions["can_proceed"] is False
        assert len(preconditions["conflicts"]) == 1

        # 5. Clear lock (cleanup)
        mock_redis_service.reset_mock()
        result = lock_service.clear_global_analysis_lock()
        assert result is True