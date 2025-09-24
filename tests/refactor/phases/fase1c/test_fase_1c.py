#!/usr/bin/env python3
"""
Test directo para FASE 1C - AnalysisLockService.

Este old_tests_with_real_data se puede ejecutar independientemente sin dependencias externas
para validar que la FASE 1C funcione correctamente.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def test_analysis_lock_service_basic():
    """Test básico de AnalysisLockService sin dependencias externas."""
    print("🧪 Testing FASE 1C - AnalysisLockService...")

    try:
        # 1. Test que el módulo se puede importar
        from app.services.analysis_lock import AnalysisLockService, LockType, create_analysis_lock_service
        print("✅ 1. AnalysisLockService se puede importar")

        # 2. Test que se puede crear con mock Redis
        from unittest.mock import Mock
        mock_redis = Mock()
        mock_redis.set.return_value = True
        mock_redis.delete.return_value = 1
        mock_redis.get.return_value = None
        mock_redis.exists.return_value = False

        # Mock para locks
        mock_lock = Mock()
        mock_lock.__enter__ = Mock(return_value=mock_lock)
        mock_lock.__exit__ = Mock(return_value=None)
        mock_redis.lock.return_value = mock_lock

        service = create_analysis_lock_service(redis_service=mock_redis)
        print("✅ 2. AnalysisLockService se puede crear con mock Redis")

        # 3. Test operaciones básicas de global analysis lock
        result = service.set_global_analysis_lock("test_user")
        assert result is True, "set_global_analysis_lock should return True"
        mock_redis.set.assert_called_with("analysis_in_progress", "test_user", ttl=7200)
        print("✅ 3. set_global_analysis_lock funciona correctamente")

        # 4. Test clear global analysis lock
        result = service.clear_global_analysis_lock()
        assert result is True, "clear_global_analysis_lock should return True"
        mock_redis.delete.assert_called_with("analysis_in_progress")
        print("✅ 4. clear_global_analysis_lock funciona correctamente")

        # 5. Test operaciones de cleanup lock
        result = service.set_cleanup_lock("cleanup_user")
        assert result is True, "set_cleanup_lock should return True"
        print("✅ 5. set_cleanup_lock funciona correctamente")

        result = service.clear_cleanup_lock()
        assert result is True, "clear_cleanup_lock should return True"
        print("✅ 6. clear_cleanup_lock funciona correctamente")

        # 6. Test check_analysis_preconditions - caso exitoso
        preconditions = service.check_analysis_preconditions("test_user")
        assert preconditions["can_proceed"] is True, "Should be able to proceed when no conflicts"
        assert len(preconditions["conflicts"]) == 0, "Should have no conflicts"
        print("✅ 7. check_analysis_preconditions (sin conflictos) funciona correctamente")

        # 7. Test check_analysis_preconditions - conflicto por cleanup
        def mock_get_with_cleanup(key):
            if key == "cleanup_in_progress":
                return "other_user"
            return None

        mock_redis.get.side_effect = mock_get_with_cleanup
        preconditions = service.check_analysis_preconditions("test_user")
        assert preconditions["can_proceed"] is False, "Should not proceed when cleanup in progress"
        assert len(preconditions["conflicts"]) == 1, "Should have one conflict"
        assert preconditions["conflicts"][0]["type"] == "cleanup_in_progress"
        print("✅ 8. check_analysis_preconditions (con conflicto) funciona correctamente")

        # 8. Test player lock context manager
        mock_redis.reset_mock()
        mock_redis.get.side_effect = None  # Reset side effect

        with service.player_lock("test_player", timeout=300, blocking_timeout=10):
            pass  # Test that context manager works

        mock_redis.lock.assert_called_with("lock:player:test_player", timeout=300, blocking_timeout=10)
        print("✅ 9. player_lock context manager funciona correctamente")

        # 9. Test que los tipos de lock están definidos correctamente
        assert LockType.PLAYER_ANALYSIS in service.LOCK_KEYS
        assert LockType.GLOBAL_ANALYSIS in service.LOCK_KEYS
        assert LockType.CLEANUP in service.LOCK_KEYS
        print("✅ 10. Tipos de lock definidos correctamente")

        # 10. Test que las keys coinciden con las implementaciones originales
        assert service.LOCK_KEYS[LockType.PLAYER_ANALYSIS] == "lock:player:{username}"
        assert service.LOCK_KEYS[LockType.GLOBAL_ANALYSIS] == "analysis_in_progress"
        assert service.LOCK_KEYS[LockType.CLEANUP] == "cleanup_in_progress"
        print("✅ 11. Keys de lock coinciden con implementaciones originales")

        print("\n🎉 ¡FASE 1C VALIDADA EXITOSAMENTE!")
        print("   AnalysisLockService funciona correctamente y unifica las 3 implementaciones")
        return True

    except Exception as e:
        print(f"\n❌ Error durante old_tests_with_real_data de FASE 1C: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backward_compatibility():
    """Test que backward compatibility funciona."""
    print("\n🔄 Testing backward compatibility...")

    try:
        # Test singleton pattern
        from app.services.analysis_lock import get_analysis_lock_service
        service1 = get_analysis_lock_service()
        service2 = get_analysis_lock_service()
        assert service1 is service2, "get_analysis_lock_service should return singleton"
        print("✅ 1. Singleton pattern funciona")

        # Test que los enums están correctos
        from app.services.analysis_lock import LockType
        assert hasattr(LockType, 'PLAYER_ANALYSIS')
        assert hasattr(LockType, 'GLOBAL_ANALYSIS')
        assert hasattr(LockType, 'CLEANUP')
        print("✅ 2. LockType enum está completo")

        print("✅ Backward compatibility validada")
        return True

    except Exception as e:
        print(f"❌ Error en backward compatibility: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Ejecutar todos los tests de FASE 1C."""
    print("=" * 60)
    print("🚀 VALIDACIÓN FASE 1C - ANALYSIS LOCK SERVICE")
    print("=" * 60)

    all_passed = True

    # Test básico
    if not test_analysis_lock_service_basic():
        all_passed = False

    # Test backward compatibility
    if not test_backward_compatibility():
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ¡TODOS LOS TESTS DE FASE 1C PASARON!")
        print("✅ AnalysisLockService está funcionando correctamente")
        print("✅ Las 3 implementaciones de locks fueron unificadas exitosamente")
    else:
        print("❌ Algunos tests fallaron")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())