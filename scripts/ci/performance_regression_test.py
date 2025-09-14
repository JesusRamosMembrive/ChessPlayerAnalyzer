#!/usr/bin/env python3
"""
Performance Regression Testing for CI/CD Pipeline
Validates that performance optimizations (4.6x baseline speedup) are maintained.
"""

import json
import time
import sys
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

# Add app to Python path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import optimized modules
from app.analysis.quality import acpl, wdl_loss, complexity_weighted_match
from app.analysis.timing import time_stats, clutch_accuracy
from app.analysis.longitudinal import aggregate_roi, selectivity_score

# Performance baselines (from previous optimization work)
PERFORMANCE_BASELINES = {
    'quality_acpl': 0.32,  # ms - pandas baseline
    'quality_wdl_loss': 0.45,  # ms - pandas baseline
    'quality_complexity_match': 0.50,  # ms - pandas baseline
    'timing_time_stats': 0.80,  # ms - pandas baseline
    'timing_clutch_accuracy': 0.90,  # ms - pandas baseline
    'longitudinal_roi': 1.20,  # ms - pandas baseline
    'longitudinal_selectivity': 1.10,  # ms - pandas baseline
}

# Expected speedup ratios (from optimization work)
EXPECTED_SPEEDUPS = {
    'quality_acpl': 6.0,
    'quality_wdl_loss': 5.5,
    'quality_complexity_match': 8.1,
    'timing_time_stats': 2.3,
    'timing_clutch_accuracy': 2.1,
    'longitudinal_roi': 2.1,
    'longitudinal_selectivity': 2.6,
}

# Test data generators
def generate_test_moves(size: int = 100):
    """Generate test chess moves data."""
    import random
    moves = []
    for i in range(size):
        moves.append({
            'move': f'e{random.randint(1,8)}',
            'cp': random.randint(-500, 500),
            'mate': None,
            'time_spent': random.uniform(1.0, 30.0),
            'clock': random.uniform(60, 1800),
            'ply': i + 1
        })
    return moves

def generate_test_games(count: int = 50):
    """Generate test games data."""
    games = []
    for i in range(count):
        game = {
            'id': i,
            'moves': generate_test_moves(random.randint(50, 150)),
            'rating_white': random.randint(1200, 2800),
            'rating_black': random.randint(1200, 2800),
            'result': random.choice(['1-0', '0-1', '1/2-1/2']),
            'time_control': '10+0'
        }
        games.append(game)
    return games

def benchmark_function(func, args, iterations: int = 100) -> float:
    """Benchmark a function with given arguments."""
    times = []

    # Warmup
    for _ in range(5):
        try:
            func(*args)
        except Exception as e:
            logging.warning(f"Warmup failed for {func.__name__}: {e}")

    # Actual benchmark
    for _ in range(iterations):
        start = time.perf_counter()
        try:
            result = func(*args)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
        except Exception as e:
            logging.error(f"Benchmark failed for {func.__name__}: {e}")
            return float('inf')

    return sum(times) / len(times) if times else float('inf')

def run_performance_tests() -> Dict[str, Dict]:
    """Run all performance regression tests."""
    logging.info("Starting performance regression tests...")

    # Generate test data
    test_moves = generate_test_moves(100)
    test_games = generate_test_games(20)

    results = {}

    # Quality module tests
    logging.info("Testing quality module optimizations...")

    # ACPL test
    try:
        acpl_time = benchmark_function(acpl, (test_moves,), iterations=50)
        results['quality_acpl'] = {
            'current_time': acpl_time,
            'baseline_time': PERFORMANCE_BASELINES['quality_acpl'],
            'expected_speedup': EXPECTED_SPEEDUPS['quality_acpl'],
            'actual_speedup': PERFORMANCE_BASELINES['quality_acpl'] / acpl_time if acpl_time > 0 else 0,
            'status': 'PASS' if acpl_time < (PERFORMANCE_BASELINES['quality_acpl'] / 3.0) else 'FAIL'
        }
    except Exception as e:
        results['quality_acpl'] = {'status': 'ERROR', 'error': str(e)}

    # WDL Loss test
    try:
        wdl_time = benchmark_function(wdl_loss, (test_moves,), iterations=50)
        results['quality_wdl_loss'] = {
            'current_time': wdl_time,
            'baseline_time': PERFORMANCE_BASELINES['quality_wdl_loss'],
            'expected_speedup': EXPECTED_SPEEDUPS['quality_wdl_loss'],
            'actual_speedup': PERFORMANCE_BASELINES['quality_wdl_loss'] / wdl_time if wdl_time > 0 else 0,
            'status': 'PASS' if wdl_time < (PERFORMANCE_BASELINES['quality_wdl_loss'] / 3.0) else 'FAIL'
        }
    except Exception as e:
        results['quality_wdl_loss'] = {'status': 'ERROR', 'error': str(e)}

    # Complexity match test
    try:
        complexity_time = benchmark_function(complexity_weighted_match, (test_moves, 'middlegame'), iterations=50)
        results['quality_complexity_match'] = {
            'current_time': complexity_time,
            'baseline_time': PERFORMANCE_BASELINES['quality_complexity_match'],
            'expected_speedup': EXPECTED_SPEEDUPS['quality_complexity_match'],
            'actual_speedup': PERFORMANCE_BASELINES['quality_complexity_match'] / complexity_time if complexity_time > 0 else 0,
            'status': 'PASS' if complexity_time < (PERFORMANCE_BASELINES['quality_complexity_match'] / 4.0) else 'FAIL'
        }
    except Exception as e:
        results['quality_complexity_match'] = {'status': 'ERROR', 'error': str(e)}

    # Timing module tests
    logging.info("Testing timing module optimizations...")

    try:
        timing_time = benchmark_function(time_stats, (test_moves,), iterations=50)
        results['timing_time_stats'] = {
            'current_time': timing_time,
            'baseline_time': PERFORMANCE_BASELINES['timing_time_stats'],
            'expected_speedup': EXPECTED_SPEEDUPS['timing_time_stats'],
            'actual_speedup': PERFORMANCE_BASELINES['timing_time_stats'] / timing_time if timing_time > 0 else 0,
            'status': 'PASS' if timing_time < (PERFORMANCE_BASELINES['timing_time_stats'] / 1.5) else 'FAIL'
        }
    except Exception as e:
        results['timing_time_stats'] = {'status': 'ERROR', 'error': str(e)}

    # Longitudinal module tests
    logging.info("Testing longitudinal module optimizations...")

    try:
        roi_time = benchmark_function(aggregate_roi, (test_games,), iterations=20)
        results['longitudinal_roi'] = {
            'current_time': roi_time,
            'baseline_time': PERFORMANCE_BASELINES['longitudinal_roi'],
            'expected_speedup': EXPECTED_SPEEDUPS['longitudinal_roi'],
            'actual_speedup': PERFORMANCE_BASELINES['longitudinal_roi'] / roi_time if roi_time > 0 else 0,
            'status': 'PASS' if roi_time < (PERFORMANCE_BASELINES['longitudinal_roi'] / 1.5) else 'FAIL'
        }
    except Exception as e:
        results['longitudinal_roi'] = {'status': 'ERROR', 'error': str(e)}

    return results

def generate_performance_report(results: Dict) -> Dict:
    """Generate comprehensive performance report."""
    passed_tests = sum(1 for r in results.values() if r.get('status') == 'PASS')
    failed_tests = sum(1 for r in results.values() if r.get('status') == 'FAIL')
    error_tests = sum(1 for r in results.values() if r.get('status') == 'ERROR')
    total_tests = len(results)

    # Calculate overall speedup
    valid_speedups = [r['actual_speedup'] for r in results.values() if 'actual_speedup' in r and r['actual_speedup'] > 0]
    avg_speedup = sum(valid_speedups) / len(valid_speedups) if valid_speedups else 0

    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'summary': {
            'total_tests': total_tests,
            'passed': passed_tests,
            'failed': failed_tests,
            'errors': error_tests,
            'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            'average_speedup': avg_speedup,
            'baseline_maintained': avg_speedup >= 2.0  # Minimum acceptable speedup
        },
        'detailed_results': results,
        'performance_grade': 'EXCELLENT' if avg_speedup >= 4.0 else 'GOOD' if avg_speedup >= 2.5 else 'ACCEPTABLE' if avg_speedup >= 2.0 else 'POOR'
    }

    return report

def main():
    parser = argparse.ArgumentParser(description='Performance regression testing for CI/CD pipeline')
    parser.add_argument('--output', '-o', help='Output JSON file path', default='performance_test_results.json')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--fail-threshold', type=float, default=2.0, help='Minimum speedup threshold to pass')

    args = parser.parse_args()

    # Configure logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        # Run performance tests
        results = run_performance_tests()

        # Generate report
        report = generate_performance_report(results)

        # Save results
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)

        # Print summary
        print(f"Performance Regression Test Results:")
        print(f"Tests run: {report['summary']['total_tests']}")
        print(f"Passed: {report['summary']['passed']}")
        print(f"Failed: {report['summary']['failed']}")
        print(f"Errors: {report['summary']['errors']}")
        print(f"Success rate: {report['summary']['success_rate']:.1f}%")
        print(f"Average speedup: {report['summary']['average_speedup']:.2f}x")
        print(f"Performance grade: {report['performance_grade']}")

        # Determine exit code
        if report['summary']['average_speedup'] >= args.fail_threshold and report['summary']['errors'] == 0:
            print("✅ Performance regression test PASSED")
            return 0
        else:
            print("❌ Performance regression test FAILED")
            if args.verbose:
                for test_name, test_result in results.items():
                    if test_result.get('status') != 'PASS':
                        print(f"  - {test_name}: {test_result.get('status', 'UNKNOWN')} - {test_result.get('error', 'Low performance')}")
            return 1

    except Exception as e:
        logging.error(f"Performance testing failed: {e}")
        logging.error(traceback.format_exc())
        return 1

if __name__ == '__main__':
    import random
    sys.exit(main())