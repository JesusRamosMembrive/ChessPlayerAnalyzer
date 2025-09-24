# tests/refactor/integration/test_redis_backward_compatibility.py
"""
Integration tests for Redis Service backward compatibility.

These tests validate that the new RedisService maintains exact compatibility
with the old utils.py functions, ensuring no breaking changes.
"""
import json
import pytest
from unittest.mock import Mock, patch

from app.infrastructure.redis_service import get_redis_service


class TestBackwardCompatibility:
    """Test backward compatibility with utils.py functions."""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client for testing."""
        with patch('redis.Redis') as mock_redis_class:
            mock_client = Mock()
            mock_redis_class.from_url.return_value = mock_client
            yield mock_client

    def test_cache_functions_compatibility(self, mock_redis):
        """Test that cache functions work exactly like utils.py versions."""
        service = get_redis_service()
        # Force client creation with mock
        _ = service.client

        # Test data
        task_name = "test_task"
        args = ["player1", 12]
        kwargs = {"depth": 10}
        result_data = {"games": 100, "rating": 1500}

        # Mock Redis responses
        mock_redis.setex.return_value = True
        mock_redis.get.return_value = json.dumps(result_data)

        # Test cache_set (should behave like utils.cache_set)
        success = service.cache_set(task_name, args, kwargs, result_data)
        assert success is True

        # Verify setex was called with correct parameters
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        key, ttl, value = call_args[0]

        assert ttl == 86400  # Default TTL
        assert json.loads(value) == result_data
        assert key.startswith("cache:test_task:")

        # Test cache_get (should behave like utils.cache_get)
        cached_result = service.cache_get(task_name, args, kwargs)
        assert cached_result == result_data

        # Verify get was called
        mock_redis.get.assert_called_once()

    def test_publish_compatibility(self, mock_redis):
        """Test that publish works exactly like utils.notify_ws."""
        service = get_redis_service()
        _ = service.client  # Force client creation

        mock_redis.publish.return_value = 1

        # Test message (similar to notify_ws format)
        channel = "player:test_user"
        message = {
            "type": "progress",
            "username": "test_user",
            "progress": 50,
            "message": "Processing games..."
        }

        # Test publish
        success = service.publish(channel, message)
        assert success is True

        # Verify publish was called correctly
        mock_redis.publish.assert_called_once_with(
            channel,
            json.dumps(message)
        )

    def test_lock_compatibility(self, mock_redis):
        """Test that lock works like utils.player_lock."""
        service = get_redis_service()
        _ = service.client  # Force client creation

        mock_lock = Mock()
        mock_lock.acquire.return_value = True
        mock_redis.lock.return_value = mock_lock

        username = "test_player"
        lock_name = f"lock:player:{username}"

        # Test lock usage (similar to player_lock context manager)
        with service.lock(lock_name, timeout=300) as lock:
            assert lock == mock_lock

        # Verify lock was created and used correctly
        mock_redis.lock.assert_called_once_with(lock_name, timeout=300)
        mock_lock.acquire.assert_called_once()
        mock_lock.release.assert_called_once()

    def test_key_operations_compatibility(self, mock_redis):
        """Test basic key operations for compatibility."""
        service = get_redis_service()
        _ = service.client

        # Test operations that might be used elsewhere
        mock_redis.setex.return_value = True
        mock_redis.get.return_value = "test_value"
        mock_redis.delete.return_value = 1

        # Test setex (used for temporary flags)
        result = service.set("analysis_flag", "active", ttl=7200)
        assert result is True

        # Test get
        value = service.get("analysis_flag")
        assert value == "test_value"

        # Test delete
        deleted = service.delete("analysis_flag")
        assert deleted == 1

    def test_error_handling_compatibility(self, mock_redis):
        """Test that error handling matches utils.py behavior."""
        service = get_redis_service()
        _ = service.client

        # Test cache_get with Redis error (should return None like utils.py)
        mock_redis.get.side_effect = Exception("Redis error")
        result = service.cache_get("test_task", [], {})
        assert result is None

        # Test cache_set with Redis error (should return False)
        mock_redis.setex.side_effect = Exception("Redis error")
        result = service.cache_set("test_task", [], {}, {})
        assert result is False

        # Test publish with Redis error (should return False)
        mock_redis.publish.side_effect = Exception("Redis error")
        result = service.publish("test_channel", {})
        assert result is False

    def test_argument_types_compatibility(self, mock_redis):
        """Test that different argument types work correctly."""
        service = get_redis_service()
        _ = service.client

        mock_redis.setex.return_value = True
        mock_redis.get.return_value = '{"result": "ok"}'

        # Test with different argument types (like utils.py handles)
        test_cases = [
            ("task1", [], {}),
            ("task2", ["string_arg"], {}),
            ("task3", [123, "mixed"], {"int_kw": 456}),
            ("task4", ("tuple_args",), {"nested": {"dict": "value"}}),
        ]

        for task_name, args, kwargs in test_cases:
            # Should not raise exceptions
            service.cache_set(task_name, args, kwargs, {"old_tests_with_real_data": "data"})
            service.cache_get(task_name, args, kwargs)

        # Verify all calls succeeded
        assert mock_redis.setex.call_count == len(test_cases)
        assert mock_redis.get.call_count == len(test_cases)