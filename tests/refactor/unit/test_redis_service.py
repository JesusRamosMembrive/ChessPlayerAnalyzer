# tests/refactor/unit/test_redis_service.py
"""
Unit tests for RedisService.

These tests validate that RedisService works correctly in isolation
using mocks, without requiring a real Redis instance.
"""
import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from contextlib import contextmanager

from app.infrastructure.redis_service import RedisService, get_redis_service, create_redis_service


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    with patch('redis.Redis') as mock_redis_class:
        mock_client = Mock()
        mock_redis_class.from_url.return_value = mock_client
        yield mock_client


@pytest.fixture
def redis_service(mock_redis):
    """RedisService with mocked Redis client."""
    service = RedisService("redis://localhost:6379")
    # Force client creation
    _ = service.client
    return service


class TestRedisService:
    """Test cases for RedisService."""

    def test_client_property_lazy_initialization(self, mock_redis):
        """Test that Redis client is created lazily."""
        service = RedisService()

        # Client should not be created yet
        assert service._client is None

        # Accessing client should create it
        client = service.client
        assert client is not None
        assert service._client is client

    def test_ping_success(self, redis_service, mock_redis):
        """Test successful ping."""
        mock_redis.ping.return_value = True

        result = redis_service.ping()

        assert result is True
        mock_redis.ping.assert_called_once()

    def test_ping_failure(self, redis_service, mock_redis):
        """Test ping failure handling."""
        mock_redis.ping.side_effect = Exception("Connection failed")

        result = redis_service.ping()

        assert result is False

    def test_cache_set_get(self, redis_service, mock_redis):
        """Test cache set and get operations."""
        # Setup mock
        test_data = {"result": "test_value"}
        mock_redis.get.return_value = json.dumps(test_data)
        mock_redis.setex.return_value = True

        # Test cache_set
        success = redis_service.cache_set("test_task", ["arg1"], {"kwarg1": "value1"}, test_data)
        assert success is True
        mock_redis.setex.assert_called_once()

        # Test cache_get
        result = redis_service.cache_get("test_task", ["arg1"], {"kwarg1": "value1"})
        assert result == test_data
        mock_redis.get.assert_called_once()

    def test_cache_get_miss(self, redis_service, mock_redis):
        """Test cache miss."""
        mock_redis.get.return_value = None

        result = redis_service.cache_get("test_task", [], {})

        assert result is None

    def test_cache_get_invalid_json(self, redis_service, mock_redis):
        """Test cache get with invalid JSON."""
        mock_redis.get.return_value = "invalid json"

        result = redis_service.cache_get("test_task", [], {})

        assert result is None

    def test_cache_delete(self, redis_service, mock_redis):
        """Test cache delete operation."""
        mock_redis.delete.return_value = 1

        result = redis_service.cache_delete("test_task", ["arg1"], {})

        assert result is True
        mock_redis.delete.assert_called_once()

    def test_cache_clear_pattern(self, redis_service, mock_redis):
        """Test clearing cache by pattern."""
        mock_redis.keys.return_value = ["key1", "key2"]
        mock_redis.delete.return_value = 2

        result = redis_service.cache_clear_pattern("test:*")

        assert result == 2
        mock_redis.keys.assert_called_once_with("test:*")
        mock_redis.delete.assert_called_once_with("key1", "key2")

    def test_publish(self, redis_service, mock_redis):
        """Test message publishing."""
        mock_redis.publish.return_value = 1

        message = {"type": "test", "data": "value"}
        result = redis_service.publish("test_channel", message)

        assert result is True
        mock_redis.publish.assert_called_once_with("test_channel", json.dumps(message))

    def test_subscribe(self, redis_service, mock_redis):
        """Test channel subscription."""
        mock_pubsub = Mock()
        mock_redis.pubsub.return_value = mock_pubsub

        # Test single channel
        result = redis_service.subscribe("test_channel")

        assert result == mock_pubsub
        mock_pubsub.subscribe.assert_called_once_with("test_channel")

    def test_subscribe_multiple_channels(self, redis_service, mock_redis):
        """Test multiple channel subscription."""
        mock_pubsub = Mock()
        mock_redis.pubsub.return_value = mock_pubsub

        # Test multiple channels
        result = redis_service.subscribe(["channel1", "channel2"])

        assert result == mock_pubsub
        mock_pubsub.subscribe.assert_called_once_with("channel1", "channel2")

    def test_lock_context_manager(self, redis_service, mock_redis):
        """Test lock context manager."""
        mock_lock = Mock()
        mock_lock.acquire.return_value = True
        mock_redis.lock.return_value = mock_lock

        with redis_service.lock("test_lock") as lock:
            assert lock == mock_lock

        mock_redis.lock.assert_called_once_with("test_lock", timeout=300)
        mock_lock.acquire.assert_called_once_with(blocking_timeout=None)
        mock_lock.release.assert_called_once()

    def test_lock_acquisition_failure(self, redis_service, mock_redis):
        """Test lock acquisition failure."""
        mock_lock = Mock()
        mock_lock.acquire.return_value = False
        mock_redis.lock.return_value = mock_lock

        with pytest.raises(Exception):  # Should raise LockError
            with redis_service.lock("test_lock"):
                pass

    def test_basic_key_operations(self, redis_service, mock_redis):
        """Test basic key operations."""
        # Test set
        mock_redis.set.return_value = True
        result = redis_service.set("key", "value")
        assert result is True

        # Test get
        mock_redis.get.return_value = "value"
        result = redis_service.get("key")
        assert result == "value"

        # Test exists
        mock_redis.exists.return_value = 1
        result = redis_service.exists("key")
        assert result is True

        # Test delete
        mock_redis.delete.return_value = 1
        result = redis_service.delete("key")
        assert result == 1

    def test_cache_key_generation(self, redis_service):
        """Test cache key generation is consistent."""
        key1 = redis_service._generate_cache_key("task", ["arg1"], {"k": "v"})
        key2 = redis_service._generate_cache_key("task", ["arg1"], {"k": "v"})
        key3 = redis_service._generate_cache_key("task", ["arg2"], {"k": "v"})

        assert key1 == key2  # Same args should produce same key
        assert key1 != key3  # Different args should produce different keys
        assert key1.startswith("cache:task:")


class TestModuleFunctions:
    """Test module-level functions."""

    @patch('app.infrastructure.redis_service._redis_service', None)
    def test_get_redis_service_singleton(self):
        """Test that get_redis_service returns singleton."""
        service1 = get_redis_service()
        service2 = get_redis_service()

        assert service1 is service2
        assert isinstance(service1, RedisService)

    def test_create_redis_service(self):
        """Test create_redis_service function."""
        service = create_redis_service("redis://test:6379")

        assert isinstance(service, RedisService)
        assert service.redis_url == "redis://test:6379"