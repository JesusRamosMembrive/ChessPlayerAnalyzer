# tests/refactor/test_dependencies.py
"""
Test that all required dependencies are available and working.
This helps validate that requirements.txt and requirements-dev.txt are correct.
"""
import pytest
import sys
import importlib


def test_core_dependencies():
    """Test that core production dependencies can be imported."""
    required_modules = [
        'fastapi',
        'redis',
        'celery',
        'sqlmodel',
        'requests',
        'pandas',
        'numpy',
        'alembic',
        'prometheus_fastapi_instrumentator',
    ]

    for module_name in required_modules:
        try:
            importlib.import_module(module_name)
        except ImportError as e:
            pytest.fail(f"Required module '{module_name}' not available: {e}")


def test_testing_dependencies():
    """Test that testing dependencies are available."""
    testing_modules = [
        'pytest',
        'httpx',
    ]

    for module_name in testing_modules:
        try:
            importlib.import_module(module_name)
        except ImportError as e:
            pytest.fail(f"Testing module '{module_name}' not available: {e}")


def test_refactor_modules():
    """Test that our refactor modules can be imported."""
    try:
        from app.infrastructure.redis_service import RedisService, get_redis_service
        from app.factories import create_app, create_worker

        # Basic smoke test
        redis_service = get_redis_service()
        assert redis_service is not None

    except ImportError as e:
        pytest.fail(f"Refactor modules not available: {e}")


def test_python_version():
    """Test that we're running on a supported Python version."""
    version_info = sys.version_info

    # We require Python 3.8+
    assert version_info.major == 3, f"Python 3 required, got {version_info.major}"
    assert version_info.minor >= 8, f"Python 3.8+ required, got {version_info.major}.{version_info.minor}"


if __name__ == "__main__":
    # Allow running this test directly
    pytest.main([__file__, "-v"])