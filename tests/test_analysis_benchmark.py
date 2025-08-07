#!/usr/bin/env python3
"""
Unit tests for app.analysis.benchmark module.
"""
import pytest
import pandas as pd
import numpy as np
import logging
import sys
import os
import importlib.util

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

spec = importlib.util.spec_from_file_location("benchmark", "app/analysis/benchmark.py")
if spec is None or spec.loader is None:
    raise ImportError("Could not load benchmark module")
benchmark_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark_module)

_pct = benchmark_module._pct
compute_benchmark = benchmark_module.compute_benchmark
REFERENCE = benchmark_module.REFERENCE


class TestPct:
    """Test the percentile calculation function."""
    
    def test_pct_normal_values(self):
        """Test with normal quartile values."""
        quartiles = [100, 200, 300]  # 25th, 50th, 75th percentiles
        
        assert _pct(50, quartiles) == 10
        assert _pct(100, quartiles) == 10  # Equal to first quartile
        
        assert _pct(150, quartiles) == 35  # Between 1st and 2nd
        assert _pct(200, quartiles) == 35  # Equal to second quartile
        assert _pct(250, quartiles) == 65  # Between 2nd and 3rd
        assert _pct(300, quartiles) == 65  # Equal to third quartile
        
        assert _pct(400, quartiles) == 90
    
    def test_pct_zero_value(self):
        """Test with zero value."""
        quartiles = [100, 200, 300]
        result = _pct(0.0, quartiles)
        assert result == 5  # Special case for zero
    
    def test_pct_none_value(self):
        """Test with None value."""
        quartiles = [100, 200, 300]
        result = _pct(None, quartiles)
        assert result is None
    
    def test_pct_nan_value(self):
        """Test with NaN value."""
        quartiles = [100, 200, 300]
        result = _pct(np.nan, quartiles)
        assert result is None
    
    def test_pct_edge_case_quartiles(self):
        """Test with edge case quartile values."""
        quartiles = [100, 100, 100]
        assert _pct(50, quartiles) == 10
        assert _pct(100, quartiles) == 10
        assert _pct(150, quartiles) == 90
        
        quartiles = [0.1, 0.2, 0.3]
        assert _pct(0.05, quartiles) == 10
        assert _pct(0.15, quartiles) == 35
        assert _pct(0.25, quartiles) == 65
        assert _pct(0.5, quartiles) == 90
    
    def test_pct_negative_values(self):
        """Test with negative values."""
        quartiles = [-300, -200, -100]
        assert _pct(-400, quartiles) == 10
        assert _pct(-250, quartiles) == 35
        assert _pct(-150, quartiles) == 65
        assert _pct(-50, quartiles) == 90


class TestComputeBenchmark:
    """Test the benchmark computation function."""
    
    def test_compute_benchmark_normal_case(self):
        """Test with normal values and existing ELO bucket."""
        logger.info("Testing benchmark computation with normal values")
        result = compute_benchmark(avg_acpl=1500, mean_entropy=4.0, player_elo=1600)
        logger.info(f"Benchmark result: {result}")
        
        assert isinstance(result, dict)
        assert "percentile_acpl" in result
        assert "percentile_entropy" in result
        
        assert isinstance(result["percentile_acpl"], int)
        assert isinstance(result["percentile_entropy"], int)
        assert 0 <= result["percentile_acpl"] <= 100
        assert 0 <= result["percentile_entropy"] <= 100
        logger.info("✓ Benchmark normal case test passed")
    
    def test_compute_benchmark_exact_elo_buckets(self):
        """Test with exact ELO bucket values."""
        for elo in REFERENCE.keys():
            result = compute_benchmark(avg_acpl=1000, mean_entropy=3.0, player_elo=elo)
            
            assert isinstance(result, dict)
            assert "percentile_acpl" in result
            assert "percentile_entropy" in result
            assert result["percentile_acpl"] is not None
            assert result["percentile_entropy"] is not None
    
    def test_compute_benchmark_none_elo(self):
        """Test with None ELO (should use default 1600)."""
        result = compute_benchmark(avg_acpl=1500, mean_entropy=4.0, player_elo=None)
        
        assert isinstance(result, dict)
        assert "percentile_acpl" in result
        assert "percentile_entropy" in result
        
        expected = compute_benchmark(avg_acpl=1500, mean_entropy=4.0, player_elo=1600)
        assert result == expected
    
    def test_compute_benchmark_rounding_elo(self):
        """Test ELO rounding to nearest 200."""
        test_elos = [1550, 1600, 1650, 1699]
        results = []
        
        for elo in test_elos:
            result = compute_benchmark(avg_acpl=1000, mean_entropy=3.0, player_elo=elo)
            results.append(result)
        
        for result in results[1:]:
            assert result == results[0]
    
    def test_compute_benchmark_extreme_elo_values(self):
        """Test with ELO values outside reference range."""
        result_low = compute_benchmark(avg_acpl=2000, mean_entropy=2.0, player_elo=400)
        assert isinstance(result_low, dict)
        assert "percentile_acpl" in result_low
        assert "percentile_entropy" in result_low
        
        result_high = compute_benchmark(avg_acpl=500, mean_entropy=8.0, player_elo=3200)
        assert isinstance(result_high, dict)
        assert "percentile_acpl" in result_high
        assert "percentile_entropy" in result_high
    
    def test_compute_benchmark_none_metrics(self):
        """Test with None metric values."""
        result = compute_benchmark(avg_acpl=None, mean_entropy=None, player_elo=1600)
        
        assert isinstance(result, dict)
        assert "percentile_acpl" in result
        assert "percentile_entropy" in result
        assert result["percentile_acpl"] is None
        assert result["percentile_entropy"] is None
    
    def test_compute_benchmark_nan_metrics(self):
        """Test with NaN metric values."""
        result = compute_benchmark(avg_acpl=np.nan, mean_entropy=np.nan, player_elo=1600)
        
        assert isinstance(result, dict)
        assert "percentile_acpl" in result
        assert "percentile_entropy" in result
        assert result["percentile_acpl"] is None
        assert result["percentile_entropy"] is None
    
    def test_compute_benchmark_zero_metrics(self):
        """Test with zero metric values."""
        result = compute_benchmark(avg_acpl=0.0, mean_entropy=0.0, player_elo=1600)
        
        assert isinstance(result, dict)
        assert "percentile_acpl" in result
        assert "percentile_entropy" in result
        assert result["percentile_acpl"] == 5  # Special case for zero
        assert result["percentile_entropy"] == 5  # Special case for zero
    
    def test_compute_benchmark_extreme_metric_values(self):
        """Test with extreme metric values."""
        result_high_acpl = compute_benchmark(avg_acpl=10000, mean_entropy=3.0, player_elo=1600)
        assert result_high_acpl["percentile_acpl"] == 90  # Should be in highest percentile
        
        result_low_acpl = compute_benchmark(avg_acpl=50, mean_entropy=3.0, player_elo=1600)
        assert result_low_acpl["percentile_acpl"] == 10  # Should be in lowest percentile
        
        result_high_entropy = compute_benchmark(avg_acpl=1000, mean_entropy=15.0, player_elo=1600)
        assert result_high_entropy["percentile_entropy"] == 90
        
        result_low_entropy = compute_benchmark(avg_acpl=1000, mean_entropy=0.5, player_elo=1600)
        assert result_low_entropy["percentile_entropy"] == 10


class TestReferenceData:
    """Test the reference data structure."""
    
    def test_reference_structure(self):
        """Test that reference data has expected structure."""
        assert isinstance(REFERENCE, dict)
        assert len(REFERENCE) > 0
        
        for elo, data in REFERENCE.items():
            assert isinstance(elo, int)
            assert isinstance(data, dict)
            assert "acpl" in data
            assert "entropy" in data
            assert isinstance(data["acpl"], list)
            assert isinstance(data["entropy"], list)
            assert len(data["acpl"]) == 3  # 25th, 50th, 75th percentiles
            assert len(data["entropy"]) == 3
    
    def test_reference_data_ordering(self):
        """Test that reference quartiles are properly ordered."""
        for elo, data in REFERENCE.items():
            acpl_quartiles = data["acpl"]
            entropy_quartiles = data["entropy"]
            
            assert acpl_quartiles[0] <= acpl_quartiles[1] <= acpl_quartiles[2]
            
            assert entropy_quartiles[0] <= entropy_quartiles[1] <= entropy_quartiles[2]


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_pct_with_empty_quartiles(self):
        """Test _pct with empty quartiles list."""
        try:
            result = _pct(100, [])
            assert result is not None
        except (IndexError, ValueError):
            pass
    
    def test_compute_benchmark_with_invalid_types(self):
        """Test compute_benchmark with invalid input types."""
        try:
            result = compute_benchmark(avg_acpl="invalid", mean_entropy=3.0, player_elo=1600)
            assert isinstance(result, dict)
        except (TypeError, ValueError):
            pass
    
    def test_benchmark_consistency(self):
        """Test that benchmark results are consistent."""
        result1 = compute_benchmark(avg_acpl=1500, mean_entropy=4.0, player_elo=1600)
        result2 = compute_benchmark(avg_acpl=1500, mean_entropy=4.0, player_elo=1600)
        
        assert result1 == result2
    
    def test_benchmark_monotonicity(self):
        """Test that better metrics give better percentiles."""
        elo = 1600
        
        result_good_acpl = compute_benchmark(avg_acpl=500, mean_entropy=4.0, player_elo=elo)
        result_bad_acpl = compute_benchmark(avg_acpl=3000, mean_entropy=4.0, player_elo=elo)
        
        if result_good_acpl["percentile_acpl"] is not None and result_bad_acpl["percentile_acpl"] is not None:
            assert result_good_acpl["percentile_acpl"] <= result_bad_acpl["percentile_acpl"]
        
        result_good_entropy = compute_benchmark(avg_acpl=1500, mean_entropy=8.0, player_elo=elo)
        result_bad_entropy = compute_benchmark(avg_acpl=1500, mean_entropy=2.0, player_elo=elo)
        
        if result_good_entropy["percentile_entropy"] is not None and result_bad_entropy["percentile_entropy"] is not None:
            assert result_good_entropy["percentile_entropy"] >= result_bad_entropy["percentile_entropy"]
