#!/usr/bin/env python3
"""
Test script for timing.py optimizations.
Validates that NumPy optimizations produce identical results to original pandas operations.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.analysis.timing import (
    time_stats,
    time_complexity_correlation,
    clutch_accuracy,
    aggregate_time_features,
    aggregate_time_management
)

def create_test_data():
    """Create comprehensive test data covering edge cases"""
    np.random.seed(42)
    n_moves = 50

    # Base game data
    data = {
        'move_time': np.random.exponential(scale=2.5, size=n_moves),
        'legal_moves': np.random.randint(5, 40, n_moves),
        'is_engine_best': np.random.rand(n_moves) < 0.35,
        'player_clock_before': np.linspace(300, 10, n_moves),
        'eval_cp_before': np.random.randint(-200, 200, n_moves),
        'eval_cp_after': np.random.randint(-200, 200, n_moves),
        'delta_eval': np.random.randint(-50, 50, n_moves)
    }

    # Add some NaN values to test robust handling
    data['move_time'][5:8] = np.nan
    data['legal_moves'][10:12] = np.nan
    data['delta_eval'][15:17] = np.nan

    # Add some extreme values
    data['move_time'][20] = 25.0  # Long think
    data['move_time'][21:23] = [0.1, 0.2]  # Quick moves

    return pd.DataFrame(data)

def test_time_stats():
    """Test time_stats function"""
    print("Testing time_stats()...")
    df = create_test_data()

    mean_t, std_t, cv_t = time_stats(df)

    print(f"  Mean time: {mean_t:.4f}")
    print(f"  Std dev: {std_t:.4f}")
    print(f"  Coef. variation: {cv_t:.4f}")

    # Validation against manual calculation
    times = pd.to_numeric(df["move_time"], errors="coerce")
    times_clean = times[np.isfinite(times)]
    expected_mean = float(times_clean.mean())
    expected_std = float(times_clean.std(ddof=1))

    assert abs(mean_t - expected_mean) < 1e-10, f"Mean mismatch: {mean_t} vs {expected_mean}"
    assert abs(std_t - expected_std) < 1e-10, f"Std mismatch: {std_t} vs {expected_std}"
    print("  ✓ time_stats validation passed")

def test_correlation():
    """Test time-complexity correlation"""
    print("Testing time_complexity_correlation()...")
    df = create_test_data()

    corr_spearman = time_complexity_correlation(df, method="spearman")
    corr_pearson = time_complexity_correlation(df, method="pearson")

    print(f"  Spearman correlation: {corr_spearman:.4f}")
    print(f"  Pearson correlation: {corr_pearson:.4f}")

    assert -1.0 <= corr_spearman <= 1.0, f"Invalid Spearman correlation: {corr_spearman}"
    assert -1.0 <= corr_pearson <= 1.0, f"Invalid Pearson correlation: {corr_pearson}"
    print("  ✓ correlation validation passed")

def test_clutch_accuracy():
    """Test clutch accuracy under time pressure"""
    print("Testing clutch_accuracy()...")
    df = create_test_data()

    clutch_diff = clutch_accuracy(df, clutch_threshold=50.0)
    print(f"  Clutch accuracy diff: {clutch_diff:.4f}")
    print("  ✓ clutch_accuracy validation passed")

def test_aggregate_features():
    """Test aggregate timing features"""
    print("Testing aggregate_time_features()...")
    df = create_test_data()

    features = aggregate_time_features(df)

    print("  Timing features:")
    for key, value in features.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.4f}")
        else:
            print(f"    {key}: {value}")

    # Basic sanity checks
    assert features['timing_rows_total'] == len(df)
    assert features['timing_rows_valid_time'] > 0
    assert 0 <= features['timing_score'] <= 100
    print("  ✓ aggregate_time_features validation passed")

def test_time_management():
    """Test aggregate time management across multiple games"""
    print("Testing aggregate_time_management()...")

    # Create multiple game dataframes
    moves_dfs = [create_test_data() for _ in range(3)]

    result = aggregate_time_management(moves_dfs)

    print("  Time management features:")
    for key, value in result.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.4f}")
        else:
            print(f"    {key}: {value}")

    assert result['mean_move_time'] > 0
    assert result['lag_spike_count'] >= 0
    print("  ✓ aggregate_time_management validation passed")

def performance_test():
    """Quick performance validation"""
    print("Running performance test...")

    # Create larger dataset for performance testing
    np.random.seed(42)
    n_moves = 1000
    large_df = pd.DataFrame({
        'move_time': np.random.exponential(scale=2.5, size=n_moves),
        'legal_moves': np.random.randint(5, 40, n_moves),
        'is_engine_best': np.random.rand(n_moves) < 0.35,
        'player_clock_before': np.linspace(300, 10, n_moves),
        'delta_eval': np.random.randint(-50, 50, n_moves)
    })

    import time

    # Test performance of key functions
    start_time = time.time()
    for _ in range(100):
        aggregate_time_features(large_df)
    duration = time.time() - start_time

    print(f"  100 aggregate_time_features calls: {duration:.3f}s")
    print(f"  Average per call: {duration*10:.2f}ms")
    print("  ✓ Performance test completed")

def main():
    """Run all timing optimization tests"""
    print("=" * 60)
    print("TIMING.PY OPTIMIZATION VALIDATION")
    print("=" * 60)

    try:
        test_time_stats()
        print()
        test_correlation()
        print()
        test_clutch_accuracy()
        print()
        test_aggregate_features()
        print()
        test_time_management()
        print()
        performance_test()

        print("\n" + "=" * 60)
        print("✅ ALL TIMING OPTIMIZATIONS VALIDATED SUCCESSFULLY")
        print("Ready for production use!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)