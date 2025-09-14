#!/usr/bin/env python3
"""
Benchmark standalone de funciones optimizadas sin dependencias DB.
Compara performance pandas vs NumPy optimizado.
"""
import time
import numpy as np
import pandas as pd
import statistics
from typing import Dict, List, Union
import json


# ============================================================================
# FUNCIONES OPTIMIZADAS (COPIAS STANDALONE)
# ============================================================================

def _extract_numeric_array(df_or_dict: Union[pd.DataFrame, Dict[str, np.ndarray]],
                          column: str) -> np.ndarray:
    """Helper para extraer array numérico de DataFrame o dict de arrays."""
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

    # Filtrar NaN
    valid_values = values[~np.isnan(values)]
    if len(valid_values) == 0:
        return 0.0

    # Aplicar cap si se especifica
    if cap_value is not None:
        valid_values = np.clip(valid_values, None, cap_value)

    if method == "median":
        return float(np.median(valid_values))
    elif method == "mean":
        return float(np.mean(valid_values))
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def acpl_optimized(game_data: Union[pd.DataFrame, Dict[str, np.ndarray]],
                   player_color: str = 'white', cap_cp: int = 1500, use_median: bool = True) -> float:
    """Versión optimizada de ACPL usando NumPy."""
    delta_eval = _extract_numeric_array(game_data, "delta_eval")

    if len(delta_eval) > 0:
        method = "median" if use_median else "mean"
        result = _robust_aggregate(np.abs(delta_eval), method, cap_cp)
        return result

    # Fallback usando eval_before/after
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


def acpl_pandas_baseline(game_df: pd.DataFrame, player_color: str = 'white',
                        cap_cp: int = 1500, use_median: bool = True) -> float:
    """Versión baseline usando pandas tradicional."""
    if "delta_eval" in game_df.columns:
        vals = pd.to_numeric(game_df["delta_eval"], errors="coerce").abs().dropna()

        if cap_cp is not None:
            vals = vals.clip(upper=cap_cp)

        if vals.empty:
            return 0.0

        return float(vals.median() if use_median else vals.mean())

    # Fallback
    if not {"eval_cp_before", "eval_cp_after"}.issubset(game_df.columns):
        return 0.0

    eval_before = pd.to_numeric(game_df["eval_cp_before"], errors="coerce")
    eval_after = pd.to_numeric(game_df["eval_cp_after"], errors="coerce")

    if player_color == 'black':
        eval_before = -eval_before
        eval_after = -eval_after

    diffs = (eval_after - eval_before).abs().dropna()

    if diffs.empty:
        return 0.0

    if cap_cp is not None:
        diffs = diffs.clip(upper=cap_cp)

    return float(diffs.median() if use_median else diffs.mean())


def complexity_weighted_match_optimized(game_df: pd.DataFrame, max_moves_cap: int = 30) -> float:
    """Versión optimizada con NumPy."""
    legal_moves = _extract_numeric_array(game_df, "legal_moves")
    is_engine_best = _extract_numeric_array(game_df, "is_engine_best")

    if len(legal_moves) == 0 or len(is_engine_best) == 0:
        return 0.0

    # Crear pesos logarítmicos
    capped_moves = np.clip(legal_moves, 0, max_moves_cap)
    weights = np.log1p(max_moves_cap - capped_moves)

    total_w = np.sum(weights)
    if total_w == 0 or np.isnan(total_w):
        return float(np.mean(is_engine_best)) if len(is_engine_best) > 0 else 0.0

    return float(np.dot(is_engine_best, weights) / total_w)


def complexity_weighted_match_pandas(game_df: pd.DataFrame, max_moves_cap: int = 30) -> float:
    """Versión baseline con pandas."""
    if "legal_moves" not in game_df.columns or "is_engine_best" not in game_df.columns:
        return game_df.get("is_engine_best", pd.Series([0])).mean()

    weights = np.log1p(max_moves_cap - game_df.legal_moves.clip(0, max_moves_cap))

    total_w = weights.sum()
    if total_w == 0 or np.isnan(total_w):
        return game_df.is_engine_best.mean()

    return np.dot(game_df.is_engine_best, weights) / total_w


# ============================================================================
# BENCHMARK FRAMEWORK
# ============================================================================

class PerformanceBenchmark:
    """Framework de benchmarking standalone."""

    def __init__(self):
        self.results = {}
        self.datasets = {}

    def create_test_data(self, n_games: int = 100, moves_per_game: int = 60) -> List[pd.DataFrame]:
        """Crear datos de test realistas."""
        cache_key = f"{n_games}_{moves_per_game}"
        if cache_key in self.datasets:
            return self.datasets[cache_key]

        np.random.seed(42)  # Reproducibilidad
        games = []

        for game_idx in range(n_games):
            n_moves = np.random.randint(
                int(moves_per_game * 0.7),
                int(moves_per_game * 1.3)
            )

            # Datos realistas
            eval_before = np.random.normal(0, 150, n_moves)
            eval_noise = np.random.normal(0, 80, n_moves)
            eval_after = eval_before + eval_noise

            delta_eval = np.abs(eval_after - eval_before)

            # Blunders ocasionales
            blunder_indices = np.random.choice(
                n_moves,
                size=max(1, n_moves // 25),
                replace=False
            )
            delta_eval[blunder_indices] += np.random.uniform(200, 600, len(blunder_indices))

            # Engine match rate
            is_engine_best = np.random.choice([0, 1], n_moves, p=[0.68, 0.32])
            legal_moves = np.random.randint(2, 35, n_moves)

            # Introducir NaN para robustez
            if n_moves > 5:
                nan_indices = np.random.choice(n_moves, size=min(2, n_moves//10), replace=False)
                delta_eval[nan_indices] = np.nan

            game_df = pd.DataFrame({
                'game_id': game_idx,
                'delta_eval': delta_eval,
                'eval_cp_before': eval_before,
                'eval_cp_after': eval_after,
                'is_engine_best': is_engine_best,
                'legal_moves': legal_moves,
            })

            games.append(game_df)

        self.datasets[cache_key] = games
        return games

    def benchmark_function(self, func, args, n_iterations: int = 100, warmup: int = 5):
        """Benchmark una función específica."""
        # Warmup
        for _ in range(warmup):
            try:
                func(*args)
            except:
                pass

        times = []
        results = []
        errors = 0

        for _ in range(n_iterations):
            try:
                start = time.perf_counter()
                result = func(*args)
                end = time.perf_counter()

                times.append((end - start) * 1000)  # ms
                results.append(result)
            except Exception as e:
                errors += 1
                if errors <= 2:  # Solo mostrar primeros errores
                    print(f"Error: {e}")

        if not times:
            return None

        return {
            'mean_ms': statistics.mean(times),
            'median_ms': statistics.median(times),
            'std_ms': statistics.stdev(times) if len(times) > 1 else 0,
            'min_ms': min(times),
            'max_ms': max(times),
            'errors': errors,
            'success_rate': (n_iterations - errors) / n_iterations,
            'sample_result': results[0] if results else None
        }

    def compare_acpl_implementations(self):
        """Comparar ACPL optimizado vs baseline."""
        print("\n=== ACPL PERFORMANCE COMPARISON ===")

        test_sizes = [
            ("Small", 20, 40),
            ("Medium", 50, 60),
            ("Large", 100, 80)
        ]

        for size_name, n_games, moves_per_game in test_sizes:
            print(f"\n--- {size_name} dataset: {n_games} games, ~{moves_per_game} moves ---")

            games = self.create_test_data(n_games, moves_per_game)
            sample_games = games[:min(10, len(games))]  # Test subset

            # Benchmark ambas versiones
            optimized_times = []
            pandas_times = []

            for game in sample_games:
                # NumPy optimizado
                opt_result = self.benchmark_function(
                    acpl_optimized, (game, 'white', 1500, True),
                    n_iterations=50
                )
                if opt_result and opt_result['success_rate'] > 0.8:
                    optimized_times.append(opt_result['mean_ms'])

                # Pandas baseline
                pandas_result = self.benchmark_function(
                    acpl_pandas_baseline, (game, 'white', 1500, True),
                    n_iterations=50
                )
                if pandas_result and pandas_result['success_rate'] > 0.8:
                    pandas_times.append(pandas_result['mean_ms'])

            if optimized_times and pandas_times:
                opt_avg = statistics.mean(optimized_times)
                pandas_avg = statistics.mean(pandas_times)
                speedup = pandas_avg / opt_avg if opt_avg > 0 else 0

                print(f"NumPy optimized: {opt_avg:.3f}ms avg")
                print(f"Pandas baseline: {pandas_avg:.3f}ms avg")
                print(f"Speedup: {speedup:.1f}x faster")

                self.results[f'acpl_{size_name.lower()}'] = {
                    'optimized_ms': opt_avg,
                    'baseline_ms': pandas_avg,
                    'speedup': speedup,
                    'games_tested': len(optimized_times)
                }

    def compare_complexity_match_implementations(self):
        """Comparar complexity weighted match."""
        print("\n=== COMPLEXITY WEIGHTED MATCH COMPARISON ===")

        games = self.create_test_data(50, 60)
        sample_games = games[:15]

        optimized_times = []
        pandas_times = []

        for game in sample_games:
            # NumPy optimizado
            opt_result = self.benchmark_function(
                complexity_weighted_match_optimized, (game, 30),
                n_iterations=50
            )
            if opt_result and opt_result['success_rate'] > 0.8:
                optimized_times.append(opt_result['mean_ms'])

            # Pandas baseline
            pandas_result = self.benchmark_function(
                complexity_weighted_match_pandas, (game, 30),
                n_iterations=50
            )
            if pandas_result and pandas_result['success_rate'] > 0.8:
                pandas_times.append(pandas_result['mean_ms'])

        if optimized_times and pandas_times:
            opt_avg = statistics.mean(optimized_times)
            pandas_avg = statistics.mean(pandas_times)
            speedup = pandas_avg / opt_avg if opt_avg > 0 else 0

            print(f"NumPy optimized: {opt_avg:.3f}ms avg")
            print(f"Pandas baseline: {pandas_avg:.3f}ms avg")
            print(f"Speedup: {speedup:.1f}x faster")

            self.results['complexity_match'] = {
                'optimized_ms': opt_avg,
                'baseline_ms': pandas_avg,
                'speedup': speedup,
                'games_tested': len(optimized_times)
            }

    def integration_benchmark(self):
        """Test de integración - múltiples funciones."""
        print("\n=== INTEGRATION PERFORMANCE ===")

        games = self.create_test_data(30, 50)
        test_games = games[:10]

        def full_analysis_optimized(games_list):
            results = []
            for game in games_list:
                game_result = {
                    'acpl_median': acpl_optimized(game, use_median=True),
                    'acpl_mean': acpl_optimized(game, use_median=False),
                    'complexity_match': complexity_weighted_match_optimized(game)
                }
                results.append(game_result)
            return results

        def full_analysis_baseline(games_list):
            results = []
            for game in games_list:
                game_result = {
                    'acpl_median': acpl_pandas_baseline(game, use_median=True),
                    'acpl_mean': acpl_pandas_baseline(game, use_median=False),
                    'complexity_match': complexity_weighted_match_pandas(game)
                }
                results.append(game_result)
            return results

        opt_result = self.benchmark_function(
            full_analysis_optimized, (test_games,),
            n_iterations=20
        )

        baseline_result = self.benchmark_function(
            full_analysis_baseline, (test_games,),
            n_iterations=20
        )

        if opt_result and baseline_result:
            opt_total = opt_result['mean_ms']
            baseline_total = baseline_result['mean_ms']
            speedup = baseline_total / opt_total if opt_total > 0 else 0

            print(f"Optimized pipeline: {opt_total:.2f}ms total ({opt_total/len(test_games):.2f}ms per game)")
            print(f"Baseline pipeline: {baseline_total:.2f}ms total ({baseline_total/len(test_games):.2f}ms per game)")
            print(f"Integration speedup: {speedup:.1f}x faster")

            self.results['integration'] = {
                'optimized_total_ms': opt_total,
                'baseline_total_ms': baseline_total,
                'optimized_per_game_ms': opt_total / len(test_games),
                'baseline_per_game_ms': baseline_total / len(test_games),
                'speedup': speedup,
                'games_per_run': len(test_games)
            }

    def generate_summary(self):
        """Generar resumen de resultados."""
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)

        total_speedups = []

        for key, result in self.results.items():
            if 'speedup' in result:
                speedup = result['speedup']
                total_speedups.append(speedup)
                print(f"{key}: {speedup:.1f}x speedup")

        if total_speedups:
            avg_speedup = statistics.mean(total_speedups)
            print(f"\nAverage speedup: {avg_speedup:.1f}x")

            if avg_speedup > 5:
                performance_grade = "EXCELLENT"
            elif avg_speedup > 2:
                performance_grade = "GOOD"
            elif avg_speedup > 1.2:
                performance_grade = "MODERATE"
            else:
                performance_grade = "MINIMAL"

            print(f"Performance grade: {performance_grade}")

        return self.results


def main():
    """Ejecutar benchmark completo."""
    print("Starting standalone refactor benchmark...")
    print("Comparing NumPy optimizations vs pandas baseline\n")

    benchmark = PerformanceBenchmark()

    try:
        benchmark.compare_acpl_implementations()
        benchmark.compare_complexity_match_implementations()
        benchmark.integration_benchmark()

        results = benchmark.generate_summary()

        # Guardar resultados
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = f"benchmark_results_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {filename}")
        print("Benchmark completed successfully!")

        return 0

    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())