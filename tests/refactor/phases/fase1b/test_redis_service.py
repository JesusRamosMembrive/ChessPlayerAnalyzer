#!/usr/bin/env python3
"""
Tests for FASE 1B - RedisService Extraction.

This test validates that Redis operations were successfully extracted
to a dedicated service with backward compatibility.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def test_redis_service_creation():
    """Test that RedisService can be created and used."""
    print("🧪 Testing FASE 1B - RedisService...")

    try:
        from app.infrastructure.redis_service import RedisService, get_redis_service
        from unittest.mock import Mock
        print("✅ 1. RedisService imports work")

        # Test service creation
        mock_redis = Mock()
        service = RedisService(redis_connection=mock_redis)
        assert service is not None, "RedisService should be created"
        print("✅ 2. RedisService creation works")

        # Test singleton pattern
        service1 = get_redis_service()
        service2 = get_redis_service()
        assert service1 is service2, "get_redis_service should return singleton"
        print("✅ 3. Singleton pattern works")

        print("🎉 FASE 1B basic validation passed!")
        return True

    except Exception as e:
        print(f"❌ Error during FASE 1B test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_utils_uses_redis_service():
    """Test that app/utils.py was updated to use RedisService."""
    print("\n🔄 Testing utils.py RedisService integration...")

    try:
        utils_file = project_root / "app" / "utils.py"
        with open(utils_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for RedisService usage
        expected_patterns = [
            "from app.infrastructure.redis_service import",
            "_redis_service"
        ]

        for pattern in expected_patterns:
            if pattern not in content:
                print(f"❌ Missing pattern in utils.py: {pattern}")
                return False

        print("✅ utils.py properly uses RedisService")
        return True

    except Exception as e:
        print(f"❌ Error checking utils.py: {e}")
        return False


def test_backward_compatibility():
    """Test that backward compatibility is maintained."""
    print("\n🔄 Testing backward compatibility...")

    try:
        # These should still work as before
        from app.utils import cache_get, cache_set
        print("✅ 1. cache_get/cache_set still available")

        # Test that they are callable
        assert callable(cache_get), "cache_get should be callable"
        assert callable(cache_set), "cache_set should be callable"
        print("✅ 2. Backward compatibility functions are callable")

        return True

    except Exception as e:
        print(f"❌ Error testing backward compatibility: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 VALIDACIÓN FASE 1B - REDIS SERVICE")
    print("=" * 60)

    all_passed = True

    if not test_redis_service_creation():
        all_passed = False

    if not test_utils_uses_redis_service():
        all_passed = False

    if not test_backward_compatibility():
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ¡FASE 1B VALIDADA EXITOSAMENTE!")
        print("✅ RedisService funciona correctamente")
        print("✅ utils.py usa RedisService internamente")
        print("✅ Backward compatibility mantenida")
    else:
        print("❌ Algunos tests de FASE 1B fallaron")
    print("=" * 60)

    sys.exit(0 if all_passed else 1)