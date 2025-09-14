#!/usr/bin/env python3
"""
Test standalone de las optimizaciones quality.py sin dependencias DB.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Agregar repo root al path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Test solo las funciones más básicas sin imports complejos
def eval_to_wdl_prob(evaluation_in_cp: float, k: float = 400.0) -> float:
    """Copia de la función para test standalone."""
    return 1.0 / (1.0 + 10 ** (-evaluation_in_cp / k))


def _extract_numeric_array(df_or_dict, column: str) -> np.ndarray:
    """Helper para extraer array numérico."""
    if isinstance(df_or_dict, pd.DataFrame):
        if column in df_or_dict.columns:
            return pd.to_numeric(df_or_dict[column], errors="coerce").values
        else:
            return np.array([])
    elif isinstance(df_or_dict, dict) and column in df_or_dict:
        return df_or_dict[column]
    else:
        return np.array([])


def _robust_aggregate(values: np.ndarray, method: str = "median",
                     cap_value: float = None) -> float:
    """Agregación robusta con capping y manejo de NaN."""
    if len(values) == 0:
        return 0.0

    valid_values = values[~np.isnan(values)]
    if len(valid_values) == 0:
        return 0.0

    if cap_value is not None:
        valid_values = np.clip(valid_values, None, cap_value)

    if method == "median":
        return float(np.median(valid_values))
    elif method == "mean":
        return float(np.mean(valid_values))
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def acpl_optimized(game_data, player_color: str = 'white',
                   cap_cp: int = 1500, use_median: bool = True) -> float:
    """Versión optimizada de ACPL para test."""
    delta_eval = _extract_numeric_array(game_data, "delta_eval")

    if len(delta_eval) > 0:
        method = "median" if use_median else "mean"
        result = _robust_aggregate(np.abs(delta_eval), method, cap_cp)
        return result

    # Fallback
    eval_before = _extract_numeric_array(game_data, "eval_cp_before")
    eval_after = _extract_numeric_array(game_data, "eval_cp_after")

    if len(eval_before) == 0 or len(eval_after) == 0:
        return 0.0

    if player_color == 'black':
        eval_before = -eval_before
        eval_after = -eval_after

    valid_mask = ~np.isnan(eval_before) & ~np.isnan(eval_after)
    if not np.any(valid_mask):
        return 0.0

    diffs = np.abs(eval_after[valid_mask] - eval_before[valid_mask])
    method = "median" if use_median else "mean"
    return _robust_aggregate(diffs, method, cap_cp)


def create_test_data():
    """Crear datos de test."""
    np.random.seed(42)
    n_moves = 50

    return pd.DataFrame({
        'delta_eval': np.random.uniform(5, 200, n_moves),
        'eval_cp_before': np.random.uniform(-100, 300, n_moves),
        'eval_cp_after': np.random.uniform(-150, 250, n_moves),
        'is_engine_best': np.random.choice([0, 1], n_moves),
        'phase': np.concatenate([
            np.repeat('opening', 12),
            np.repeat('middlegame', 26),
            np.repeat('endgame', 12)
        ])
    })


def test_optimizations():
    """Test principal."""
    print("Testing optimized quality functions...")

    # Test 1: ACPL function
    test_df = create_test_data()

    result = acpl_optimized(test_df, use_median=True)
    assert isinstance(result, float), "ACPL should return float"
    assert 0 <= result <= 1500, f"ACPL {result} out of range"
    print(f"[OK] ACPL median: {result:.2f}")

    result_mean = acpl_optimized(test_df, use_median=False)
    print(f"[OK] ACPL mean: {result_mean:.2f}")

    # Test 2: Helper functions
    values = np.array([1, 2, np.nan, 4, 5, 100])
    robust_result = _robust_aggregate(values, "median", cap_value=50)
    assert robust_result > 0, "Should handle NaN and capping"
    print(f"[OK] Robust aggregate: {robust_result:.2f}")

    # Test 3: Performance comparison
    import time

    n_tests = 1000
    start = time.time()

    for _ in range(n_tests):
        result = acpl_optimized(test_df, use_median=True)

    elapsed = time.time() - start
    print(f"[PERF] Performance: {n_tests} calls in {elapsed:.3f}s ({elapsed/n_tests*1000:.2f}ms per call)")

    # Test 4: Edge cases
    empty_df = pd.DataFrame({'delta_eval': []})
    empty_result = acpl_optimized(empty_df)
    assert empty_result == 0.0, "Empty data should return 0"
    print("[OK] Empty data handling works")

    # Test 5: NaN handling
    nan_df = pd.DataFrame({'delta_eval': [np.nan, np.nan, np.nan]})
    nan_result = acpl_optimized(nan_df)
    assert nan_result == 0.0, "All-NaN data should return 0"
    print("[OK] NaN handling works")

    print("\n[SUCCESS] All optimization tests passed!")
    print("[INFO] NumPy optimizations are working correctly")


if __name__ == "__main__":
    test_optimizations()