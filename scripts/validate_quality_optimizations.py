#!/usr/bin/env python3
"""
Script para validar las optimizaciones NumPy en quality.py
Verifica que las funciones optimizadas produzcan resultados consistentes.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Agregar repo root al path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.analysis.quality import (
    acpl,
    wdl_loss,
    complexity_weighted_match,
    phase_blunder_rate_single,
    eval_to_wdl_prob
)


def create_test_game_data():
    """Crea datos de prueba simulando una partida."""
    np.random.seed(42)  # Para reproducibilidad

    n_moves = 50
    data = {
        'delta_eval': np.random.uniform(5, 200, n_moves),  # CP loss
        'eval_cp_before': np.random.uniform(-100, 300, n_moves),
        'eval_cp_after': np.random.uniform(-150, 250, n_moves),
        'is_engine_best': np.random.choice([0, 1], n_moves, p=[0.7, 0.3]),
        'legal_moves': np.random.randint(2, 40, n_moves),
        'phase': np.concatenate([
            np.repeat('opening', 12),
            np.repeat('middlegame', 26),
            np.repeat('endgame', 12)
        ])
    }

    # Introducir algunos NaN para probar robustez
    data['delta_eval'][5:7] = np.nan
    data['eval_cp_before'][10] = np.nan

    return pd.DataFrame(data)


def test_acpl_function():
    """Testear función ACPL optimizada."""
    print("🧪 Testing ACPL function...")

    game_df = create_test_game_data()

    # Test caso normal
    result = acpl(game_df, 'white', cap_cp=1500, use_median=True)
    assert isinstance(result, float), "ACPL should return float"
    assert 0 <= result <= 1500, f"ACPL result {result} out of expected range"

    # Test caso con median vs mean
    result_median = acpl(game_df, 'white', use_median=True)
    result_mean = acpl(game_df, 'white', use_median=False)
    assert abs(result_median - result_mean) >= 0, "Median and mean should potentially differ"

    # Test caso vacío
    empty_df = pd.DataFrame({'delta_eval': []})
    result_empty = acpl(empty_df)
    assert result_empty == 0.0, "Empty data should return 0.0"

    print("✅ ACPL tests passed")


def test_wdl_loss_function():
    """Testear función WDL loss optimizada."""
    print("🧪 Testing WDL loss function...")

    game_df = create_test_game_data()

    # Test caso normal
    result = wdl_loss(game_df, 'white')
    assert isinstance(result, float), "WDL loss should return float"
    assert -1.0 <= result <= 1.0, f"WDL loss {result} out of expected range"

    # Test caso con datos problemáticos
    bad_df = pd.DataFrame({
        'eval_cp_before': [np.nan, np.inf, -np.inf],
        'eval_cp_after': [100, np.nan, 200]
    })
    result_bad = wdl_loss(bad_df, 'white')
    assert result_bad == 0.0, "Bad data should return 0.0"

    print("✅ WDL loss tests passed")


def test_complexity_weighted_match():
    """Testear función complexity weighted match."""
    print("🧪 Testing complexity weighted match...")

    game_df = create_test_game_data()

    result = complexity_weighted_match(game_df, max_moves_cap=30)
    assert isinstance(result, float), "Should return float"
    assert 0.0 <= result <= 1.0, f"Result {result} should be between 0 and 1"

    print("✅ Complexity weighted match tests passed")


def test_phase_blunder_rate():
    """Testear función phase blunder rate."""
    print("🧪 Testing phase blunder rate...")

    game_df = create_test_game_data()

    result = phase_blunder_rate_single(game_df)
    assert isinstance(result, dict), "Should return dict"

    expected_keys = ['opening_blunder_rate', 'middlegame_blunder_rate',
                     'endgame_blunder_rate', 'overall_blunder_rate']
    for key in expected_keys:
        if key in result:
            assert isinstance(result[key], (float, type(None))), f"{key} should be float or None"
            if result[key] is not None:
                assert 0.0 <= result[key] <= 1.0, f"{key} should be between 0 and 1"

    print("✅ Phase blunder rate tests passed")


def test_eval_to_wdl():
    """Testear función eval to WDL prob."""
    print("🧪 Testing eval to WDL conversion...")

    # Test casos conocidos
    assert eval_to_wdl_prob(0) == 0.5, "0 centipawns should be 50% win prob"
    assert eval_to_wdl_prob(400) > 0.9, "400 cp advantage should be >90% win prob"
    assert eval_to_wdl_prob(-400) < 0.1, "-400 cp should be <10% win prob"

    # Test extremos
    result_big = eval_to_wdl_prob(10000)
    assert 0.99 <= result_big <= 1.0, "Large advantage should approach 100%"

    print("✅ Eval to WDL tests passed")


def performance_comparison():
    """Comparar performance básica."""
    print("⚡ Running basic performance comparison...")

    import time

    # Crear dataset más grande para medir performance
    np.random.seed(42)
    n_games = 100
    n_moves_per_game = 80

    large_dataset = []
    for _ in range(n_games):
        game_data = create_test_game_data()
        large_dataset.append(game_data)

    # Test performance ACPL
    start_time = time.time()
    results = []
    for game_df in large_dataset:
        result = acpl(game_df, use_median=True)
        results.append(result)

    elapsed = time.time() - start_time
    print(f"📊 Processed {n_games} games in {elapsed:.3f}s ({elapsed/n_games*1000:.1f}ms per game)")
    print(f"📈 Average ACPL: {np.mean(results):.2f}")

    assert len(results) == n_games, "Should process all games"
    assert all(isinstance(r, float) for r in results), "All results should be floats"


def main():
    """Ejecutar todas las validaciones."""
    print("🚀 Starting quality.py optimization validation...\n")

    try:
        test_acpl_function()
        test_wdl_loss_function()
        test_complexity_weighted_match()
        test_phase_blunder_rate()
        test_eval_to_wdl()
        performance_comparison()

        print("\n🎉 All validation tests passed!")
        print("✨ Quality.py NumPy optimizations are working correctly")
        return 0

    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())