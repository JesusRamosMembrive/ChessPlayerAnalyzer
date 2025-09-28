#!/usr/bin/env python3
"""
Tests para FASE 2A: Real Data Validation

Valida que el nuevo sistema fail-fast funciona correctamente
y que los datos reales son preservados sin sanitización.
"""
import math
import sys
import os
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from app.validation import (
    validate_analysis_metrics,
    validate_stockfish_evaluation,
    validate_timing_data,
    ensure_real_data_or_none,
    InvalidAnalysisDataError
)


def test_validate_analysis_metrics_with_good_data():
    """Test que datos válidos pasan la validación."""
    print("📋 Testing validate_analysis_metrics with good data...")

    good_data = {
        'acpl': 45.6,
        'accuracy': 87.3,
        'blunders': 2,
        'mistakes': 5,
        'quality': {
            'avg_loss': 12.4,
            'match_rate': 0.85
        },
        'timing': {
            'avg_time': 23.7,
            'moves': [12.3, 15.6, 8.9]
        }
    }

    try:
        result = validate_analysis_metrics(good_data, "test_context")
        assert result == good_data, "Good data should pass unchanged"
        print("✅ Good data validation passed")
        return True
    except Exception as e:
        print(f"❌ Good data validation failed: {e}")
        return False


def test_validate_analysis_metrics_fails_on_nan():
    """Test que NaN values causan fallo inmediato."""
    print("📋 Testing validate_analysis_metrics fails on NaN...")

    bad_data = {
        'acpl': float('nan'),  # This should cause failure
        'accuracy': 87.3,
        'blunders': 2
    }

    try:
        validate_analysis_metrics(bad_data, "test_context")
        print("❌ NaN validation should have failed but didn't")
        return False
    except InvalidAnalysisDataError as e:
        expected_msg = "NaN detected in acpl"
        if expected_msg in str(e):
            print("✅ NaN validation correctly failed")
            return True
        else:
            print(f"❌ Wrong error message: {e}")
            return False
    except Exception as e:
        print(f"❌ Unexpected exception: {e}")
        return False


def test_validate_analysis_metrics_fails_on_infinity():
    """Test que Infinity values causan fallo inmediato."""
    print("📋 Testing validate_analysis_metrics fails on Infinity...")

    bad_data = {
        'timing': {
            'avg_time': float('inf'),  # This should cause failure
            'moves': [12.3, 15.6]
        }
    }

    try:
        validate_analysis_metrics(bad_data, "test_context")
        print("❌ Infinity validation should have failed but didn't")
        return False
    except InvalidAnalysisDataError as e:
        expected_msg = "Infinity detected in timing.avg_time"
        if expected_msg in str(e):
            print("✅ Infinity validation correctly failed")
            return True
        else:
            print(f"❌ Wrong error message: {e}")
            return False


def test_validate_analysis_metrics_nested_data():
    """Test validación en estructuras anidadas."""
    print("📋 Testing validate_analysis_metrics with nested structures...")

    nested_bad_data = {
        'quality': {
            'moves': [
                {'eval': 45.6},
                {'eval': float('nan')},  # Deep NaN should be caught
                {'eval': 23.1}
            ]
        }
    }

    try:
        validate_analysis_metrics(nested_bad_data, "nested_test")
        print("❌ Nested NaN validation should have failed but didn't")
        return False
    except InvalidAnalysisDataError as e:
        expected_msg = "NaN detected"
        if expected_msg in str(e):
            print("✅ Nested NaN validation correctly failed")
            return True
        else:
            print(f"❌ Wrong error message: {e}")
            return False


def test_validate_stockfish_evaluation():
    """Test validación de evaluaciones de Stockfish."""
    print("📋 Testing validate_stockfish_evaluation...")

    try:
        # Good evaluations
        assert validate_stockfish_evaluation(0.5, 1) == 0.5
        assert validate_stockfish_evaluation(-1.2, 2) == -1.2
        assert validate_stockfish_evaluation(None, 3) is None

        # Bad evaluation should fail
        try:
            validate_stockfish_evaluation(float('nan'), 4)
            print("❌ NaN stockfish evaluation should have failed")
            return False
        except InvalidAnalysisDataError:
            pass  # Expected

        print("✅ Stockfish evaluation validation works correctly")
        return True
    except Exception as e:
        print(f"❌ Stockfish evaluation validation failed: {e}")
        return False


def test_validate_timing_data():
    """Test validación de datos de timing."""
    print("📋 Testing validate_timing_data...")

    try:
        # Good timing data
        good_times = [12.3, 15.6, 8.9, 23.1]
        result = validate_timing_data(good_times)
        assert result == good_times, "Good timing data should pass unchanged"

        # Data with None values (acceptable)
        mixed_times = [12.3, None, 8.9, 23.1]
        result = validate_timing_data(mixed_times)
        expected = [12.3, 8.9, 23.1]  # None values filtered out
        assert result == expected, "None values should be filtered out"

        # Negative time should fail
        try:
            validate_timing_data([12.3, -5.0, 8.9])
            print("❌ Negative time should have failed")
            return False
        except InvalidAnalysisDataError:
            pass  # Expected

        # Suspiciously high time should fail
        try:
            validate_timing_data([12.3, 5000.0, 8.9])  # More than 1 hour
            print("❌ Suspiciously high time should have failed")
            return False
        except InvalidAnalysisDataError:
            pass  # Expected

        print("✅ Timing data validation works correctly")
        return True
    except Exception as e:
        print(f"❌ Timing data validation failed: {e}")
        return False


def test_ensure_real_data_or_none():
    """Test función que asegura datos reales o None explícito."""
    print("📋 Testing ensure_real_data_or_none...")

    try:
        # Good data
        assert ensure_real_data_or_none(45.6, "test_field") == 45.6
        assert ensure_real_data_or_none(None, "test_field") is None
        assert ensure_real_data_or_none("text", "test_field") == "text"

        # Bad data should fail
        try:
            ensure_real_data_or_none(float('nan'), "test_field")
            print("❌ NaN should have failed in ensure_real_data_or_none")
            return False
        except InvalidAnalysisDataError:
            pass  # Expected

        print("✅ ensure_real_data_or_none works correctly")
        return True
    except Exception as e:
        print(f"❌ ensure_real_data_or_none failed: {e}")
        return False


def test_no_clean_json_numbers_imports():
    """Test que clean_json_numbers ya no existe en el código."""
    print("📋 Testing that clean_json_numbers is completely removed...")

    try:
        # Should not be able to import clean_json_numbers anymore
        try:
            from app.utils import clean_json_numbers
            print("❌ clean_json_numbers still exists in app.utils")
            return False
        except ImportError:
            pass  # Expected

        try:
            from app.utils_sanitize import clean_json_numbers
            print("❌ utils_sanitize.py still exists")
            return False
        except ImportError:
            pass  # Expected

        print("✅ clean_json_numbers completely removed")
        return True
    except Exception as e:
        print(f"❌ Clean import test failed: {e}")
        return False


def test_engine_imports_validation():
    """Test que el engine ahora importa validation en lugar de sanitization."""
    print("📋 Testing engine imports validation functions...")

    try:
        # Should be able to import validation from engine context
        from app.validation import validate_analysis_metrics, InvalidAnalysisDataError

        # Verify the engine file doesn't have clean_json_numbers references
        with open('/home/jesusramos/Git/ChessPlayerAnalyzer/app/analysis/engine.py', 'r') as f:
            engine_content = f.read()

        if 'clean_json_numbers' in engine_content:
            print("❌ engine.py still contains clean_json_numbers references")
            return False

        if 'validate_analysis_metrics' not in engine_content:
            print("❌ engine.py doesn't import validation functions")
            return False

        print("✅ Engine correctly imports validation functions")
        return True
    except Exception as e:
        print(f"❌ Engine import test failed: {e}")
        return False


def run_all_tests():
    """Ejecuta todos los tests de FASE 2A."""
    print("🧪 FASE 2A: Real Data Validation Tests")
    print("=" * 50)

    tests = [
        test_validate_analysis_metrics_with_good_data,
        test_validate_analysis_metrics_fails_on_nan,
        test_validate_analysis_metrics_fails_on_infinity,
        test_validate_analysis_metrics_nested_data,
        test_validate_stockfish_evaluation,
        test_validate_timing_data,
        test_ensure_real_data_or_none,
        test_no_clean_json_numbers_imports,
        test_engine_imports_validation,
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
        print("🎉 FASE 2A: All tests passed! Real data validation working!")
        print("")
        print("🔥 BULLDOZER IMPACT:")
        print("   - Sanitization completely eliminated")
        print("   - Real data preserved or explicit failures")
        print("   - Mathematical problems will be caught immediately")
        print("   - No more hidden data corruption")
        return True
    else:
        print("⚠️ FASE 2A: Some tests failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)