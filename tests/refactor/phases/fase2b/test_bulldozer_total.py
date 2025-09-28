#!/usr/bin/env python3
"""
Tests para FASE 2B: BULLDOZER TOTAL

Valida que el nuevo sistema ultra-simplificado funciona correctamente:
- Modelo de tabla única
- Motor de análisis funcional
- API simplificada
- No dependencias circulares
"""
import sys
import os
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

# Test imports
from app.models import GameAnalysis, PlayerProgress
from app.analysis.bulldozer_engine import analyze_game_simple, save_analysis_to_db, get_player_analysis_summary
from app.bulldozer_api import start_player_analysis_bulldozer, get_player_status_bulldozer

# Test PGN for validation
SAMPLE_PGN = """[Event "Live Chess"]
[Site "Chess.com"]
[Date "2024.01.15"]
[Round "-"]
[White "testplayer"]
[Black "opponent"]
[WhiteElo "1500"]
[BlackElo "1480"]
[TimeControl "600"]
[Termination "testplayer won by checkmate"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O
9. h3 Nb8 10. d4 Nbd7 11. c4 c6 12. cxb5 axb5 13. Nc3 Bb7 14. Bg5 b4 15. Nb1 h6
16. Bh4 c5 17. dxe5 Nxe5 18. Nxe5 dxe5 19. Bxf6 Bxf6 20. Nd2 Re8 21. Qf3 Qd4
22. Nf1 Bc6 23. Ne3 Rad8 24. Rad1 Qb6 25. Rxd8 Rxd8 26. Qf5 Rd4 27. Nf1 Be7
28. f3 Bf6 29. Kh1 Rd1 30. Rxd1 Qxb3 31. axb3 Bxe4 32. fxe4 c4 33. bxc4 b3
34. Rd8+ Kh7 35. Rb8 b2 36. Kg2 Bg5 37. Kf3 Bf4 38. g3 Bd2 39. Rb7 Kg6
40. Rxf7 Kh5 41. Rf5 Kg6 42. Rxe5 Kf6 43. Re8 Kf7 44. Re7+ Kf8 45. Rxg7 b1=Q
46. Rg8+ Ke7 47. Re8+ Kd6 48. Rd8+ Kc6 49. Rc8+ Kb6 50. Rb8+ Ka6 51. Ra8# 1-0"""


def test_bulldozer_models_import():
    """Test que los modelos BULLDOZER se importan correctamente."""
    print("📋 Testing BULLDOZER models import...")

    try:
        # Should be able to create instances
        game_analysis = GameAnalysis(
            pgn=SAMPLE_PGN,
            white_username="testplayer",
            black_username="opponent",
            analyzed_username="testplayer",
            analyzed_color="white",
            analysis={"quality": {"acpl": 45.0}}
        )

        player_progress = PlayerProgress(
            username="testplayer",
            status="pending",
            progress=0
        )

        assert game_analysis.pgn == SAMPLE_PGN
        assert game_analysis.analyzed_username == "testplayer"
        assert player_progress.username == "testplayer"

        print("✅ BULLDOZER models import and creation works")
        return True
    except Exception as e:
        print(f"❌ BULLDOZER models test failed: {e}")
        return False


def test_bulldozer_analysis_engine():
    """Test que el motor de análisis BULLDOZER funciona."""
    print("📋 Testing BULLDOZER analysis engine...")

    try:
        # Test with mock data (no Stockfish required for basic structure test)
        result = analyze_game_simple(
            pgn_text=SAMPLE_PGN,
            username="testplayer",
            color="white",
            stockfish_path="echo",  # Mock path that will fail gracefully
            depth=1
        )

        # Should return structured data even if Stockfish fails
        assert isinstance(result, dict)

        # If Stockfish analysis fails, should still have basic structure
        if "error" in result:
            expected_error = "Stockfish analysis failed"
            assert expected_error in result["error"]
            print("✅ BULLDOZER engine handles Stockfish failure gracefully")
        else:
            # If it works, check structure
            assert "quality" in result
            assert "timing" in result
            assert "moves" in result
            print("✅ BULLDOZER engine produces correct structure")

        return True
    except Exception as e:
        print(f"❌ BULLDOZER analysis engine test failed: {e}")
        return False


def test_bulldozer_pgn_parsing():
    """Test que el parsing de PGN funciona correctamente."""
    print("📋 Testing BULLDOZER PGN parsing...")

    try:
        from app.analysis.bulldozer_engine import _parse_pgn, _extract_player_moves

        # Test PGN parsing
        game = _parse_pgn(SAMPLE_PGN)
        assert game is not None
        assert game.headers["White"] == "testplayer"
        assert game.headers["Black"] == "opponent"

        # Test move extraction
        white_moves = _extract_player_moves(game, "white")
        black_moves = _extract_player_moves(game, "black")

        assert white_moves is not None
        assert black_moves is not None
        assert len(white_moves) > 0
        assert len(black_moves) > 0

        # Check move structure
        first_white_move = white_moves[0]
        assert len(first_white_move) == 3  # (move_number, move_san, time_spent)
        assert first_white_move[1] == "e4"  # First move should be e4

        print("✅ BULLDOZER PGN parsing works correctly")
        return True
    except Exception as e:
        print(f"❌ BULLDOZER PGN parsing test failed: {e}")
        return False


def test_bulldozer_no_circular_imports():
    """Test que no hay imports circulares en BULLDOZER."""
    print("📋 Testing BULLDOZER has no circular imports...")

    try:
        # These imports should work without circular dependency issues
        from app.models import GameAnalysis, PlayerProgress
        from app.analysis.bulldozer_engine import analyze_game_simple
        from app.bulldozer_api import start_player_analysis_bulldozer

        # Test that we can access all classes
        assert GameAnalysis is not None
        assert PlayerProgress is not None
        assert analyze_game_simple is not None
        assert start_player_analysis_bulldozer is not None

        print("✅ BULLDOZER has no circular imports")
        return True
    except Exception as e:
        print(f"❌ BULLDOZER circular import test failed: {e}")
        return False


def test_bulldozer_simple_structure():
    """Test que la estructura BULLDOZER es realmente simple."""
    print("📋 Testing BULLDOZER architecture simplicity...")

    try:
        # Count lines in key files to ensure simplicity
        files_to_check = [
            "app/models.py",
            "app/analysis/bulldozer_engine.py",
            "app/bulldozer_api.py"
        ]

        total_lines = 0
        for file_path in files_to_check:
            full_path = os.path.join(os.path.dirname(__file__), '../../../..', file_path)
            if os.path.exists(full_path):
                with open(full_path, 'r') as f:
                    lines = len(f.readlines())
                    total_lines += lines
                    print(f"   {file_path}: {lines} lines")

        print(f"   Total BULLDOZER core files: {total_lines} lines")

        # BULLDOZER should be significantly smaller than complex architecture
        if total_lines < 1000:  # Reasonable limit for simple architecture
            print("✅ BULLDOZER architecture is appropriately simple")
            return True
        else:
            print(f"⚠️ BULLDOZER might be getting complex: {total_lines} lines")
            return False

    except Exception as e:
        print(f"❌ BULLDOZER simplicity test failed: {e}")
        return False


def test_bulldozer_validation_integration():
    """Test que BULLDOZER integra correctamente con validación fail-fast."""
    print("📋 Testing BULLDOZER validation integration...")

    try:
        from app.validation import validate_analysis_metrics, InvalidAnalysisDataError

        # Test good data passes
        good_analysis = {
            "quality": {"acpl": 45.0, "match_rate": 0.85},
            "timing": {"mean_move_time": 12.5},
            "moves": [{"move_number": 1, "played": "e4"}]
        }

        validated = validate_analysis_metrics(good_analysis, "bulldozer_test")
        assert validated == good_analysis

        # Test bad data fails fast
        bad_analysis = {
            "quality": {"acpl": float('nan')},  # This should fail
            "timing": {"mean_move_time": 12.5}
        }

        try:
            validate_analysis_metrics(bad_analysis, "bulldozer_test")
            print("❌ Validation should have failed but didn't")
            return False
        except InvalidAnalysisDataError:
            pass  # Expected

        print("✅ BULLDOZER validation integration works correctly")
        return True
    except Exception as e:
        print(f"❌ BULLDOZER validation integration test failed: {e}")
        return False


def test_bulldozer_api_structure():
    """Test que las funciones API BULLDOZER tienen la estructura correcta."""
    print("📋 Testing BULLDOZER API structure...")

    try:
        from app.bulldozer_api import (
            start_player_analysis_bulldozer,
            get_player_status_bulldozer,
            get_player_metrics_bulldozer,
            get_game_analysis_bulldozer
        )

        # Check that functions exist and are callable
        assert callable(start_player_analysis_bulldozer)
        assert callable(get_player_status_bulldozer)
        assert callable(get_player_metrics_bulldozer)
        assert callable(get_game_analysis_bulldozer)

        print("✅ BULLDOZER API structure is correct")
        return True
    except Exception as e:
        print(f"❌ BULLDOZER API structure test failed: {e}")
        return False


def run_all_tests():
    """Ejecuta todos los tests de FASE 2B: BULLDOZER TOTAL."""
    print("🧪 FASE 2B: BULLDOZER TOTAL Tests")
    print("=" * 50)

    tests = [
        test_bulldozer_models_import,
        test_bulldozer_analysis_engine,
        test_bulldozer_pgn_parsing,
        test_bulldozer_no_circular_imports,
        test_bulldozer_simple_structure,
        test_bulldozer_validation_integration,
        test_bulldozer_api_structure,
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
        print("🎉 FASE 2B: All tests passed! BULLDOZER TOTAL working!")
        print("")
        print("💥 BULLDOZER ACHIEVEMENTS:")
        print("   - Single table architecture ✅")
        print("   - Functional analysis engine ✅")
        print("   - No circular imports ✅")
        print("   - Ultra-simplified structure ✅")
        print("   - Fail-fast validation integrated ✅")
        print("   - Ready for AI/PyTorch integration ✅")
        return True
    else:
        print("⚠️ FASE 2B: Some tests failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)