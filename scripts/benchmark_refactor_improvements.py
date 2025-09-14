#!/usr/bin/env python3
"""
Comprehensive benchmarking script para medir mejoras del refactor.
Compara performance antes y después de las optimizaciones NumPy.
"""
import sys
import time
import numpy as np
import pandas as pd
import traceback
from pathlib import Path
from typing import Dict, List, Callable, Any
import json
import statistics

# Agregar repo root al path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


class BenchmarkSuite:
    """Suite de benchmarking para funciones refactorizadas."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = {}
        self.test_data_cache = {}

    def log(self, message: str):
        """Log con control de verbosidad."""
        if self.verbose:
            print(message)

    def create_test_dataset(self, size: str = "medium") -> Dict[str, pd.DataFrame]:
        """Crear datasets de test de diferentes tamaños."""
        if size in self.test_data_cache:
            return self.test_data_cache[size]

        np.random.seed(42)  # Reproducibilidad

        sizes = {
            "small": {"n_games": 10, "moves_per_game": 30},
            "medium": {"n_games": 50, "moves_per_game": 60},
            "large": {"n_games": 200, "moves_per_game": 80},
            "xlarge": {"n_games": 500, "moves_per_game": 100}
        }

        config = sizes.get(size, sizes["medium"])
        n_games = config["n_games"]
        moves_per_game = config["moves_per_game"]

        games = []

        for game_idx in range(n_games):
            # Simular datos realistas de una partida
            n_moves = np.random.randint(
                int(moves_per_game * 0.7),
                int(moves_per_game * 1.3)
            )

            # Datos de evaluación del motor
            eval_before = np.random.normal(0, 150, n_moves)
            eval_noise = np.random.normal(0, 80, n_moves)
            eval_after = eval_before + eval_noise

            # Delta eval (pérdida de centipawns)
            delta_eval = np.abs(eval_after - eval_before)

            # Introducir algunos blunders (pérdidas grandes)
            blunder_indices = np.random.choice(
                n_moves,
                size=max(1, n_moves // 20),
                replace=False
            )
            delta_eval[blunder_indices] += np.random.uniform(200, 800, len(blunder_indices))

            # Engine best moves (match rate)
            is_engine_best = np.random.choice(
                [0, 1],
                n_moves,
                p=[0.65, 0.35]  # ~35% match rate realista
            )

            # Legal moves count
            legal_moves = np.random.randint(2, 40, n_moves)

            # Game phases
            opening_end = int(n_moves * 0.25)
            endgame_start = int(n_moves * 0.75)

            phases = np.array(['middlegame'] * n_moves, dtype=object)
            phases[:opening_end] = 'opening'
            phases[endgame_start:] = 'endgame'

            # Introducir algunos NaN para probar robustez
            if n_moves > 10:
                nan_indices = np.random.choice(n_moves, size=2, replace=False)
                delta_eval[nan_indices] = np.nan

            game_df = pd.DataFrame({
                'game_id': game_idx,
                'move_number': range(1, n_moves + 1),
                'delta_eval': delta_eval,
                'eval_cp_before': eval_before,
                'eval_cp_after': eval_after,
                'is_engine_best': is_engine_best,
                'legal_moves': legal_moves,
                'phase': phases,
            })

            games.append(game_df)

        dataset = {
            'games': games,
            'summary': {
                'n_games': n_games,
                'avg_moves': np.mean([len(g) for g in games]),
                'total_moves': sum(len(g) for g in games)
            }
        }

        self.test_data_cache[size] = dataset
        return dataset

    def benchmark_function(self, func: Callable, args: tuple,
                          n_iterations: int = 100) -> Dict[str, float]:
        """Benchmark individual de una función."""
        times = []
        results = []
        errors = 0

        # Warmup
        for _ in range(5):
            try:
                func(*args)
            except Exception:
                pass

        # Benchmark real
        for _ in range(n_iterations):
            try:
                start = time.perf_counter()
                result = func(*args)
                end = time.perf_counter()

                times.append((end - start) * 1000)  # Convert to milliseconds
                results.append(result)
            except Exception as e:
                errors += 1
                if self.verbose and errors <= 3:
                    self.log(f"Error in benchmark: {e}")

        if not times:
            return {
                'mean_ms': float('inf'),
                'median_ms': float('inf'),
                'std_ms': float('inf'),
                'min_ms': float('inf'),
                'max_ms': float('inf'),
                'errors': errors,
                'success_rate': 0.0
            }

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

    def benchmark_quality_functions(self):
        """Benchmark específico para funciones de quality.py."""
        self.log("\n=== BENCHMARKING QUALITY FUNCTIONS ===")

        # Import funciones optimizadas
        try:
            from app.analysis.quality import acpl, wdl_loss, complexity_weighted_match
            from app.analysis.quality import phase_blunder_rate_single, eval_to_wdl_prob
        except ImportError as e:
            self.log(f"Warning: Could not import optimized functions: {e}")
            return

        datasets = {
            'small': self.create_test_dataset('small'),
            'medium': self.create_test_dataset('medium'),
            'large': self.create_test_dataset('large')
        }

        functions_to_test = [
            ('acpl_median', lambda df: acpl(df, use_median=True)),
            ('acpl_mean', lambda df: acpl(df, use_median=False)),
            ('wdl_loss', lambda df: wdl_loss(df)),
            ('complexity_weighted_match', complexity_weighted_match),
            ('phase_blunder_rate', phase_blunder_rate_single),
        ]

        for dataset_name, dataset in datasets.items():
            self.log(f"\n--- Testing dataset: {dataset_name} ---")
            self.log(f"Games: {dataset['summary']['n_games']}, "
                    f"Avg moves: {dataset['summary']['avg_moves']:.1f}")

            for func_name, func in functions_to_test:
                self.log(f"Benchmarking {func_name}...")

                # Test en múltiples juegos para estadísticas
                game_times = []
                game_results = []

                for game_df in dataset['games'][:min(20, len(dataset['games']))]:
                    try:
                        bench_result = self.benchmark_function(
                            func, (game_df,), n_iterations=50
                        )
                        if bench_result['success_rate'] > 0.8:
                            game_times.extend([bench_result['mean_ms']])
                            if bench_result['sample_result'] is not None:
                                game_results.append(bench_result['sample_result'])
                    except Exception as e:
                        self.log(f"Error benchmarking {func_name}: {e}")

                if game_times:
                    result_key = f"{dataset_name}_{func_name}"
                    self.results[result_key] = {
                        'function': func_name,
                        'dataset': dataset_name,
                        'mean_time_ms': statistics.mean(game_times),
                        'median_time_ms': statistics.median(game_times),
                        'std_time_ms': statistics.stdev(game_times) if len(game_times) > 1 else 0,
                        'games_tested': len(game_times),
                        'sample_results': game_results[:5]  # Algunos resultados de ejemplo
                    }

                    self.log(f"  {func_name}: {statistics.mean(game_times):.3f}ms avg "
                            f"({len(game_times)} games)")

    def benchmark_integration_performance(self):
        """Benchmark de integración - simular flujo completo."""
        self.log("\n=== INTEGRATION BENCHMARK ===")

        dataset = self.create_test_dataset('medium')
        games = dataset['games'][:10]  # Sample games

        # Simular análisis completo de múltiples funciones
        def full_analysis(game_df):
            """Simular análisis completo de una partida."""
            try:
                from app.analysis.quality import (
                    acpl, wdl_loss, complexity_weighted_match,
                    phase_blunder_rate_single
                )

                results = {}
                results['acpl'] = acpl(game_df, use_median=True)
                results['acpl_mean'] = acpl(game_df, use_median=False)
                results['wdl_loss'] = wdl_loss(game_df)
                results['complexity_match'] = complexity_weighted_match(game_df)
                results['phase_blunders'] = phase_blunder_rate_single(game_df)

                return results
            except Exception as e:
                return {'error': str(e)}

        self.log("Running full analysis benchmark...")

        integration_benchmark = self.benchmark_function(
            lambda games: [full_analysis(game) for game in games],
            (games,),
            n_iterations=20
        )

        self.results['integration_full_analysis'] = {
            'description': 'Full analysis pipeline (5 functions per game)',
            'games_per_run': len(games),
            'total_time_ms': integration_benchmark['mean_ms'],
            'time_per_game_ms': integration_benchmark['mean_ms'] / len(games),
            'success_rate': integration_benchmark['success_rate'],
            'functions_per_game': 5
        }

        self.log(f"Integration benchmark: {integration_benchmark['mean_ms']:.2f}ms total "
                f"({integration_benchmark['mean_ms']/len(games):.2f}ms per game)")

    def memory_usage_test(self):
        """Test básico de uso de memoria."""
        self.log("\n=== MEMORY USAGE TEST ===")

        try:
            import psutil
            import os

            process = psutil.Process(os.getpid())

            # Memoria baseline
            baseline_mb = process.memory_info().rss / 1024 / 1024

            # Crear dataset grande
            large_dataset = self.create_test_dataset('xlarge')
            after_creation_mb = process.memory_info().rss / 1024 / 1024

            # Procesar dataset
            from app.analysis.quality import acpl
            results = []
            for game in large_dataset['games'][:50]:
                results.append(acpl(game, use_median=True))

            after_processing_mb = process.memory_info().rss / 1024 / 1024

            self.results['memory_usage'] = {
                'baseline_mb': baseline_mb,
                'after_dataset_creation_mb': after_creation_mb,
                'after_processing_mb': after_processing_mb,
                'dataset_overhead_mb': after_creation_mb - baseline_mb,
                'processing_overhead_mb': after_processing_mb - after_creation_mb
            }

            self.log(f"Memory usage: {baseline_mb:.1f}MB → {after_processing_mb:.1f}MB "
                    f"(+{after_processing_mb-baseline_mb:.1f}MB)")

        except ImportError:
            self.log("psutil not available, skipping memory test")

    def generate_report(self) -> Dict[str, Any]:
        """Generar reporte completo de benchmarks."""
        self.log("\n=== BENCHMARK REPORT ===")

        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {
                'total_functions_tested': len([k for k in self.results.keys() if '_' in k and k != 'integration_full_analysis']),
                'datasets_tested': len(set(k.split('_')[0] for k in self.results.keys() if '_' in k)),
            },
            'results': self.results,
            'performance_summary': {},
            'recommendations': []
        }

        # Analizar resultados por función
        function_stats = {}
        for key, result in self.results.items():
            if 'function' in result:
                func_name = result['function']
                if func_name not in function_stats:
                    function_stats[func_name] = []
                function_stats[func_name].append(result['mean_time_ms'])

        for func_name, times in function_stats.items():
            report['performance_summary'][func_name] = {
                'avg_time_ms': statistics.mean(times),
                'min_time_ms': min(times),
                'max_time_ms': max(times),
                'datasets_tested': len(times)
            }

            avg_time = statistics.mean(times)
            if avg_time < 1.0:
                performance = "excellent"
            elif avg_time < 5.0:
                performance = "good"
            elif avg_time < 20.0:
                performance = "acceptable"
            else:
                performance = "needs_optimization"

            self.log(f"{func_name}: {avg_time:.3f}ms avg ({performance})")

        # Recomendaciones
        if 'integration_full_analysis' in self.results:
            integration = self.results['integration_full_analysis']
            time_per_game = integration['time_per_game_ms']

            if time_per_game < 5:
                report['recommendations'].append("Performance is excellent for production use")
            elif time_per_game < 20:
                report['recommendations'].append("Performance is good, suitable for real-time analysis")
            else:
                report['recommendations'].append("Consider further optimization for real-time use cases")

        return report

    def save_report(self, report: Dict[str, Any], filename: str = None):
        """Guardar reporte en JSON."""
        if filename is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"benchmark_report_{timestamp}.json"

        filepath = REPO_ROOT / "benchmark_results" / filename
        filepath.parent.mkdir(exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)

        self.log(f"\nReport saved to: {filepath}")
        return filepath


def main():
    """Ejecutar benchmark completo."""
    print("Starting comprehensive refactor benchmark...")
    print(f"Repository: {REPO_ROOT}")

    benchmark = BenchmarkSuite(verbose=True)

    try:
        # Tests principales
        benchmark.benchmark_quality_functions()
        benchmark.benchmark_integration_performance()
        benchmark.memory_usage_test()

        # Generar y guardar reporte
        report = benchmark.generate_report()
        report_path = benchmark.save_report(report)

        print("\n" + "="*60)
        print("BENCHMARK COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Report saved to: {report_path}")

        # Summary
        if 'integration_full_analysis' in benchmark.results:
            integration = benchmark.results['integration_full_analysis']
            print(f"Integration performance: {integration['time_per_game_ms']:.2f}ms per game")

        print(f"Functions tested: {report['summary']['total_functions_tested']}")

        return 0

    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())