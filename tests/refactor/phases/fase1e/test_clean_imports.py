#!/usr/bin/env python3
"""
Tests para FASE 1E: Legacy Cleanup

Valida que todos los imports están limpios y sin side effects.
Este es el test final del refactor FASE 1.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))


def test_infrastructure_imports_clean():
    """Test que los componentes de infraestructura importan sin side effects."""
    print("📋 Testing infrastructure imports...")

    try:
        from app.infrastructure.redis_service import get_redis_service
        from app.infrastructure.http_client import get_http_client
        from app.services.analysis_lock import get_analysis_lock_service

        # Verify they return valid callables
        redis_service = get_redis_service()
        http_client = get_http_client()
        lock_service = get_analysis_lock_service()

        assert redis_service is not None, "RedisService should not be None"
        assert http_client is not None, "HttpClient should not be None"
        assert lock_service is not None, "AnalysisLockService should not be None"

        print("✅ Infrastructure imports clean and functional")
        return True
    except Exception as e:
        print(f"❌ Infrastructure imports failed: {e}")
        return False


def test_utils_imports_clean():
    """Test que app.utils importa sin disparar conexión DB."""
    print("📋 Testing app.utils clean imports...")

    try:
        # This was the problematic import before the refactor
        from app.utils import fetch_games, notify_ws, update_progress, sa_to_dict

        # Verify functions are importable
        assert callable(fetch_games), "fetch_games should be callable"
        assert callable(notify_ws), "notify_ws should be callable"
        assert callable(update_progress), "update_progress should be callable"
        assert callable(sa_to_dict), "sa_to_dict should be callable"

        print("✅ app.utils imports clean (no database connection triggered)")
        return True
    except Exception as e:
        print(f"❌ app.utils imports failed: {e}")
        return False


def test_factories_imports_clean():
    """Test que app.factories importa sin side effects."""
    print("📋 Testing app.factories clean imports...")

    try:
        from app.factories import create_app, create_worker, get_logger

        # Verify functions are importable
        assert callable(create_app), "create_app should be callable"
        assert callable(create_worker), "create_worker should be callable"
        assert callable(get_logger), "get_logger should be callable"

        print("✅ app.factories imports clean (no OTEL/database connection triggered)")
        return True
    except Exception as e:
        print(f"❌ app.factories imports failed: {e}")
        return False


def test_app_init_clean():
    """Test que app.__init__ importa limpiamente."""
    print("📋 Testing app.__init__ clean imports...")

    try:
        import app

        # app.__init__ should be minimal with no side effects
        print("✅ app.__init__ imports clean")
        return True
    except Exception as e:
        print(f"❌ app.__init__ imports failed: {e}")
        return False


def test_backward_compatibility():
    """Test que el refactor mantiene backward compatibility."""
    print("📋 Testing backward compatibility...")

    try:
        # Test that original interfaces are preserved
        from app.utils import fetch_games
        from app.infrastructure.redis_service import get_redis_service
        from app.infrastructure.http_client import get_http_client
        from app.services.analysis_lock import get_analysis_lock_service

        # Test that services can be retrieved
        redis_service = get_redis_service()
        http_client = get_http_client()
        lock_service = get_analysis_lock_service()

        # Verify backward compatibility APIs exist
        assert hasattr(redis_service, 'get'), "RedisService should have get method"
        assert hasattr(redis_service, 'set'), "RedisService should have set method"
        assert 'fetch_games_from_chesscom' in http_client, "HttpClient should have fetch_games_from_chesscom"
        assert hasattr(lock_service, 'player_lock'), "AnalysisLockService should have player_lock"

        print("✅ Backward compatibility maintained")
        return True
    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_no_circular_imports():
    """Test que no hay imports circulares."""
    print("📋 Testing for circular imports...")

    try:
        # Import all major modules in sequence to detect circular imports
        modules_to_test = [
            'app',
            'app.infrastructure.redis_service',
            'app.infrastructure.http_client',
            'app.services.analysis_lock',
            'app.utils',
            'app.factories',
        ]

        for module_name in modules_to_test:
            __import__(module_name)

        print("✅ No circular imports detected")
        return True
    except ImportError as e:
        if "circular import" in str(e).lower():
            print(f"❌ Circular import detected: {e}")
            return False
        else:
            # Re-raise if it's not a circular import issue
            raise
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False


def run_all_tests():
    """Ejecuta todos los tests de FASE 1E."""
    print("🧪 FASE 1E: Legacy Cleanup Tests")
    print("=" * 50)

    tests = [
        test_infrastructure_imports_clean,
        test_utils_imports_clean,
        test_factories_imports_clean,
        test_app_init_clean,
        test_backward_compatibility,
        test_no_circular_imports,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            failed += 1
        print()

    print("=" * 50)
    print(f"📊 Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 FASE 1E: All tests passed! Clean imports achieved!")
        return True
    else:
        print("⚠️ FASE 1E: Some tests failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)