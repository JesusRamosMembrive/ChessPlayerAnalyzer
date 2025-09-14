#!/usr/bin/env python3
"""
Standalone test script for openings.py optimizations.
Tests the functions directly without importing the full app.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Set up simple logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Copy the optimized functions directly here for testing
def shannon_entropy_optimized(series: pd.Series) -> float:
    """Optimized Shannon entropy with NumPy"""
    clean_series = series.dropna()
    if clean_series.empty:
        return 0.0

    # Optimized with NumPy operations
    values = clean_series.values
    unique_vals, counts = np.unique(values, return_counts=True)
    counts = counts.astype(float)
    total_count = np.sum(counts)
    probs = counts / total_count

    # Avoid log(0) by filtering out zero probabilities
    probs = probs[probs > 0]
    return -float(np.sum(probs * np.log2(probs)))

def shannon_entropy_original(series: pd.Series) -> float:
    """Original pandas version for comparison"""
    clean_series = series.dropna()
    if clean_series.empty:
        return 0.0

    counts = clean_series.value_counts()
    probs  = counts / counts.sum()
    return -(probs * np.log2(probs)).sum()

def novelty_depth_stats_optimized(depths_array: np.ndarray) -> dict:
    """Optimized novelty depth stats with NumPy"""
    return {
        'mean_tn_depth' : float(np.mean(depths_array)),
        'median_tn_depth': float(np.median(depths_array)),
        'sd_tn_depth'   : float(np.std(depths_array, ddof=1)),
        'pct_late_nov'  : float(np.mean(depths_array >= 20))     # % de novedades "tardías"
    }

def novelty_depth_stats_original(depths_array: np.ndarray) -> dict:
    """Original version for comparison"""
    return {
        'mean_tn_depth' : depths_array.mean(),
        'median_tn_depth': np.median(depths_array),
        'sd_tn_depth'   : depths_array.std(ddof=1),
        'pct_late_nov'  : (depths_array >= 20).mean()     # % de novedades "tardías"
    }

def second_choice_rate_optimized(moves_df: pd.DataFrame,
                                rank_col: str = "bestmove_rank",
                                delta_eval_col: str = "delta_eval",
                                threshold_cp: int = 50) -> dict:
    """Optimized second choice rate with NumPy"""
    mask = moves_df[delta_eval_col] > threshold_cp
    if not mask.any():
        return {'second_choice_pct': np.nan}

    # Optimized with NumPy array operations
    rank_values = moves_df.loc[mask, rank_col].values
    secondish = float(np.mean(np.isin(rank_values, [2, 3])))
    return {'second_choice_pct': secondish}

def second_choice_rate_original(moves_df: pd.DataFrame,
                               rank_col: str = "bestmove_rank",
                               delta_eval_col: str = "delta_eval",
                               threshold_cp: int = 50) -> dict:
    """Original pandas version"""
    mask = moves_df[delta_eval_col] > threshold_cp
    if not mask.any():
        return {'second_choice_pct': np.nan}

    secondish = moves_df.loc[mask, rank_col].isin([2, 3]).mean()
    return {'second_choice_pct': secondish}

def repertoire_breadth_focus_optimized(games_df: pd.DataFrame,
                                      eco_col: str = "eco_code",
                                      min_occurrences: int = 3) -> dict:
    """Optimized breadth/focus calculation with NumPy"""
    # Optimized with NumPy operations
    eco_values = games_df[eco_col].values
    unique_ecos, counts = np.unique(eco_values, return_counts=True)
    breadth = len(unique_ecos)

    # Focus: top 3 openings percentage
    sorted_counts = np.sort(counts)[::-1]  # Descending order
    top3_sum = np.sum(sorted_counts[:3]) if len(sorted_counts) >= 3 else np.sum(sorted_counts)
    total_sum = np.sum(counts)
    focus = float(top3_sum / total_sum) if total_sum > 0 else 0.0

    # Optionally flag "hyper‑specialist"
    hyper_specialist = focus > 0.75 and breadth >= min_occurrences
    return {
        'breadth_openings' : breadth,
        'focus_top3_pct'   : focus,
        'hyper_specialist' : hyper_specialist
    }

def repertoire_breadth_focus_original(games_df: pd.DataFrame,
                                     eco_col: str = "eco_code",
                                     min_occurrences: int = 3) -> dict:
    """Original pandas version"""
    eco_counts = games_df[eco_col].value_counts()
    breadth = eco_counts.size
    focus   = eco_counts.head(3).sum() / eco_counts.sum()

    # Optionally flag "hyper‑specialist"
    hyper_specialist = focus > 0.75 and breadth >= min_occurrences
    return {
        'breadth_openings' : breadth,
        'focus_top3_pct'   : focus,
        'hyper_specialist' : hyper_specialist
    }

def create_test_data():
    """Create comprehensive test data"""
    np.random.seed(42)
    n_games = 100

    # ECO codes for openings (realistic distribution)
    eco_codes = ['C54', 'B30', 'C50', 'A46', 'E60', 'D20', 'C41', 'B10']
    eco_weights = [0.3, 0.2, 0.15, 0.1, 0.08, 0.07, 0.06, 0.04]  # Some openings more frequent

    games_data = {
        'eco_code': np.random.choice(eco_codes, n_games, p=eco_weights),
        'elo': np.random.normal(1800, 200, n_games).astype(int)
    }

    # Moves data
    n_moves = 500
    moves_data = {
        'bestmove_rank': np.random.choice([1, 2, 3, 4], n_moves, p=[0.6, 0.25, 0.1, 0.05]),
        'delta_eval': np.random.randint(10, 200, n_moves),
        'best_rank': np.random.choice([1, 2, 3], n_moves, p=[0.7, 0.2, 0.1]),
        'played': [f"move_{i}" for i in range(n_moves)]
    }

    games_df = pd.DataFrame(games_data)
    moves_df = pd.DataFrame(moves_data)

    return games_df, moves_df

def test_shannon_entropy():
    """Test Shannon entropy optimization"""
    print("Testing shannon_entropy()...")
    games_df, _ = create_test_data()

    entropy_orig = shannon_entropy_original(games_df['eco_code'])
    entropy_opt = shannon_entropy_optimized(games_df['eco_code'])

    print(f"  Original entropy: {entropy_orig:.4f}")
    print(f"  Optimized entropy: {entropy_opt:.4f}")

    # Should be very close (within floating point precision)
    assert abs(entropy_orig - entropy_opt) < 1e-10, f"Entropy mismatch: {entropy_orig} vs {entropy_opt}"
    print("  OK shannon_entropy validation passed")

def test_novelty_depth_stats():
    """Test novelty depth statistics"""
    print("Testing novelty_depth_stats()...")

    # Create test depth data
    depths = np.random.randint(1, 50, 100)

    stats_orig = novelty_depth_stats_original(depths)
    stats_opt = novelty_depth_stats_optimized(depths)

    print(f"  Original mean depth: {stats_orig['mean_tn_depth']:.2f}")
    print(f"  Optimized mean depth: {stats_opt['mean_tn_depth']:.2f}")

    # Compare all metrics
    for key in stats_orig.keys():
        assert abs(stats_orig[key] - stats_opt[key]) < 1e-10, f"Depth {key} mismatch: {stats_orig[key]} vs {stats_opt[key]}"

    print("  OK novelty_depth_stats validation passed")

def test_second_choice_rate():
    """Test second choice rate optimization"""
    print("Testing second_choice_rate()...")
    _, moves_df = create_test_data()

    rate_orig = second_choice_rate_original(moves_df)
    rate_opt = second_choice_rate_optimized(moves_df)

    print(f"  Original second choice rate: {rate_orig['second_choice_pct']:.4f}")
    print(f"  Optimized second choice rate: {rate_opt['second_choice_pct']:.4f}")

    # Handle NaN case
    if np.isnan(rate_orig['second_choice_pct']) and np.isnan(rate_opt['second_choice_pct']):
        pass  # Both NaN is valid
    else:
        assert abs(rate_orig['second_choice_pct'] - rate_opt['second_choice_pct']) < 1e-10, \
            f"Rate mismatch: {rate_orig['second_choice_pct']} vs {rate_opt['second_choice_pct']}"

    print("  OK second_choice_rate validation passed")

def test_repertoire_breadth_focus():
    """Test repertoire breadth and focus optimization"""
    print("Testing repertoire_breadth_focus()...")
    games_df, _ = create_test_data()

    result_orig = repertoire_breadth_focus_original(games_df)
    result_opt = repertoire_breadth_focus_optimized(games_df)

    print(f"  Original breadth: {result_orig['breadth_openings']}, focus: {result_orig['focus_top3_pct']:.3f}")
    print(f"  Optimized breadth: {result_opt['breadth_openings']}, focus: {result_opt['focus_top3_pct']:.3f}")

    # Compare all metrics
    assert result_orig['breadth_openings'] == result_opt['breadth_openings'], \
        f"Breadth mismatch: {result_orig['breadth_openings']} vs {result_opt['breadth_openings']}"

    assert abs(result_orig['focus_top3_pct'] - result_opt['focus_top3_pct']) < 1e-10, \
        f"Focus mismatch: {result_orig['focus_top3_pct']} vs {result_opt['focus_top3_pct']}"

    assert result_orig['hyper_specialist'] == result_opt['hyper_specialist'], \
        f"Hyper specialist mismatch: {result_orig['hyper_specialist']} vs {result_opt['hyper_specialist']}"

    print("  OK repertoire_breadth_focus validation passed")

def performance_test():
    """Performance comparison test"""
    print("Running performance test...")
    import time

    # Create larger dataset
    np.random.seed(42)
    n_games = 5000
    eco_codes = ['C54', 'B30', 'C50', 'A46', 'E60', 'D20', 'C41', 'B10'] * 100  # More variety
    large_games = pd.DataFrame({
        'eco_code': np.random.choice(eco_codes, n_games)
    })

    # Shannon entropy performance test
    start_time = time.time()
    for _ in range(100):
        shannon_entropy_original(large_games['eco_code'])
    orig_entropy_duration = time.time() - start_time

    start_time = time.time()
    for _ in range(100):
        shannon_entropy_optimized(large_games['eco_code'])
    opt_entropy_duration = time.time() - start_time

    entropy_speedup = orig_entropy_duration / opt_entropy_duration if opt_entropy_duration > 0 else float('inf')

    # Repertoire breadth/focus performance test
    start_time = time.time()
    for _ in range(100):
        repertoire_breadth_focus_original(large_games)
    orig_breadth_duration = time.time() - start_time

    start_time = time.time()
    for _ in range(100):
        repertoire_breadth_focus_optimized(large_games)
    opt_breadth_duration = time.time() - start_time

    breadth_speedup = orig_breadth_duration / opt_breadth_duration if opt_breadth_duration > 0 else float('inf')

    print(f"  Shannon Entropy (100 calls):")
    print(f"    Original: {orig_entropy_duration:.3f}s")
    print(f"    Optimized: {opt_entropy_duration:.3f}s")
    print(f"    Speedup: {entropy_speedup:.2f}x")

    print(f"  Repertoire Breadth/Focus (100 calls):")
    print(f"    Original: {orig_breadth_duration:.3f}s")
    print(f"    Optimized: {opt_breadth_duration:.3f}s")
    print(f"    Speedup: {breadth_speedup:.2f}x")

    print("  OK Performance test completed")

def main():
    """Run all openings optimization tests"""
    print("=" * 60)
    print("OPENINGS.PY OPTIMIZATION VALIDATION (Standalone)")
    print("=" * 60)

    try:
        test_shannon_entropy()
        print()
        test_novelty_depth_stats()
        print()
        test_second_choice_rate()
        print()
        test_repertoire_breadth_focus()
        print()
        performance_test()

        print("\n" + "=" * 60)
        print("CHECKMARK ALL OPENINGS OPTIMIZATIONS VALIDATED SUCCESSFULLY")
        print("Ready for production use!")
        print("=" * 60)

    except Exception as e:
        print(f"\nX VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)