#!/usr/bin/env python3
"""
Tests para FASE 1D: HttpClient extraction

Valida que el HttpClient es independiente y testeable sin conexiones externas.
"""
import json
import tempfile
import pathlib
from unittest.mock import patch, Mock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

def test_http_client_independent_import():
    """
    Test que HttpClient se puede importar independientemente
    sin disparar conexiones a DB o servicios externos.
    """
    print("📋 Testing HttpClient independent import...")

    try:
        from app.infrastructure.http_client import (
            fetch_games_from_chesscom,
            save_games_archive,
            get_http_client
        )
        print("✅ HttpClient imports successfully without side effects")
        return True
    except Exception as e:
        print(f"❌ HttpClient import failed: {e}")
        return False


def test_http_client_factory():
    """Test que el factory pattern funciona correctamente."""
    print("📋 Testing HttpClient factory pattern...")

    try:
        from app.infrastructure.http_client import get_http_client

        client = get_http_client()

        # Verify all expected functions are present
        expected_functions = ['fetch_games_from_chesscom', 'save_games_archive']
        for func_name in expected_functions:
            assert func_name in client, f"Missing function: {func_name}"
            assert callable(client[func_name]), f"Function {func_name} is not callable"

        print("✅ HttpClient factory working correctly")
        return True
    except Exception as e:
        print(f"❌ HttpClient factory test failed: {e}")
        return False


@patch('app.infrastructure.http_client.requests.Session')
def test_fetch_games_mocked(mock_session):
    """Test fetch_games_from_chesscom con mocks (sin conexiones reales)."""
    print("📋 Testing fetch_games_from_chesscom with mocks...")

    try:
        from app.infrastructure.http_client import fetch_games_from_chesscom

        # Mock session responses
        mock_session_instance = Mock()
        mock_session_instance.headers = {}  # Allow header assignment
        mock_session.return_value = mock_session_instance

        # Mock archives response
        archives_response = Mock()
        archives_response.json.return_value = {
            "archives": ["https://api.chess.com/pub/player/test/games/2024/01"]
        }
        archives_response.raise_for_status.return_value = None

        # Mock games response
        games_response = Mock()
        games_response.json.return_value = {
            "games": [
                {
                    "pgn": "[Event \"Test\"] 1. e4 e5 [%clk 0:10:00] 2. Nf3 Nc6 [%clk 0:09:58] *",
                    "white": {"username": "testwhite"},
                    "black": {"username": "testblack"},
                    "end_time": 1640995200  # 2022-01-01 timestamp
                }
            ]
        }

        # Configure session mock responses
        mock_session_instance.get.side_effect = [archives_response, games_response]

        # Test the function
        result = fetch_games_from_chesscom("testuser", months=1)

        # Verify result structure
        assert len(result) == 1
        game = result[0]
        assert "pgn" in game
        assert "move_times" in game
        assert "white" in game
        assert "black" in game
        assert "end_time" in game

        print("✅ fetch_games_from_chesscom works with mocks")
        return True
    except Exception as e:
        print(f"❌ fetch_games_from_chesscom mock test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_save_games_archive():
    """Test save_games_archive con archivo temporal."""
    print("📋 Testing save_games_archive...")

    try:
        from app.infrastructure.http_client import save_games_archive

        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set environment variable to use temp directory
            os.environ["FETCH_ARCHIVE_DIR"] = temp_dir

            # Test data
            test_games = [
                {
                    "pgn": "[Event \"Test\"] 1. e4 e5 *",
                    "move_times": [2, 1],
                    "white": "testwhite",
                    "black": "testblack",
                    "end_time": "2022-01-01T00:00:00"
                }
            ]

            # Save games
            save_games_archive(test_games, "testuser")

            # Verify file was created
            archive_path = pathlib.Path(temp_dir)
            json_files = list(archive_path.glob("testuser_*.json"))
            assert len(json_files) == 1, "Archive file not created"

            # Verify content
            with json_files[0].open('r') as f:
                saved_data = json.load(f)

            assert saved_data == test_games, "Saved data doesn't match"

            print("✅ save_games_archive works correctly")
            return True
    except Exception as e:
        print(f"❌ save_games_archive test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up environment
        if "FETCH_ARCHIVE_DIR" in os.environ:
            del os.environ["FETCH_ARCHIVE_DIR"]


def run_all_tests():
    """Ejecuta todos los tests de FASE 1D."""
    print("🧪 FASE 1D: HttpClient Extraction Tests")
    print("=" * 50)

    tests = [
        test_http_client_independent_import,
        test_http_client_factory,
        test_fetch_games_mocked,
        test_save_games_archive,
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
        print("🎉 FASE 1D: All tests passed!")
        return True
    else:
        print("⚠️ FASE 1D: Some tests failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)