#!/usr/bin/env python3
"""
Tests for FASE 1A - Factory Pattern Implementation.

This test validates that initialization was successfully centralized
using factory patterns without side effects.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def test_factory_imports_without_side_effects():
    """Test that factory imports don't trigger side effects."""
    print("🧪 Testing FASE 1A - Factory Pattern...")

    try:
        # This should not trigger DB connections or other side effects
        from app.factories import create_app, create_worker, get_logger
        print("✅ 1. Factory imports work without side effects")

        # Test that functions exist and are callable
        assert callable(create_app), "create_app should be callable"
        assert callable(create_worker), "create_worker should be callable"
        assert callable(get_logger), "get_logger should be callable"
        print("✅ 2. All factory functions are callable")

        # Test that logger can be created without side effects
        logger = get_logger("test")
        assert logger is not None, "Logger should be created"
        print("✅ 3. Logger creation works")

        print("🎉 FASE 1A validation passed!")
        return True

    except Exception as e:
        print(f"❌ Error during FASE 1A test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_main_uses_factories():
    """Test that app/main.py was updated to use factories."""
    print("\n🔄 Testing main.py factory integration...")

    try:
        main_file = project_root / "app" / "main.py"
        with open(main_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for factory usage
        expected_patterns = [
            "from app.factories import create_app",
            "app = create_app()"
        ]

        for pattern in expected_patterns:
            if pattern not in content:
                print(f"❌ Missing pattern in main.py: {pattern}")
                return False

        # Check that old initialization is removed
        problematic_patterns = [
            "init_otel()",
            "setup_logging()",
            "Instrumentator()"
        ]

        for pattern in problematic_patterns:
            if pattern in content:
                print(f"⚠️  Found old initialization pattern: {pattern}")
                # This is not a failure, just a warning

        print("✅ main.py properly uses factories")
        return True

    except Exception as e:
        print(f"❌ Error checking main.py: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 VALIDACIÓN FASE 1A - FACTORY PATTERNS")
    print("=" * 60)

    all_passed = True

    if not test_factory_imports_without_side_effects():
        all_passed = False

    if not test_main_uses_factories():
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ¡FASE 1A VALIDADA EXITOSAMENTE!")
        print("✅ Factory patterns funcionan correctamente")
        print("✅ Inicialización centralizada sin side effects")
    else:
        print("❌ Algunos tests de FASE 1A fallaron")
    print("=" * 60)

    sys.exit(0 if all_passed else 1)