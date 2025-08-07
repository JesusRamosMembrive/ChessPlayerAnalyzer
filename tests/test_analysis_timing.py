#!/usr/bin/env python3
"""
Unit tests for app.analysis.timing module.
Tests the timing analysis functions including time stats, time-complexity correlation,
lag spike detection, and feature aggregation.
"""

import pytest
import pandas as pd
import numpy as np
import json
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import importlib.util
spec = importlib.util.spec_from_file_location("timing", os.path.join(os.path.dirname(__file__), "..", "app", "analysis", "timing.py"))
if spec and spec.loader:
    timing_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(timing_module)
    
    time_stats = timing_module.time_stats
    time_complexity_correlation = timing_module.time_complexity_correlation
    detect_lag_spikes = timing_module.detect_lag_spikes
    aggregate_time_features = timing_module.aggregate_time_features
else:
    raise ImportError("Could not load timing module")


@pytest.fixture
def test_games_data():
    """Load test games subset for testing."""
    test_data_path = Path(__file__).parent / "data" / "test_games_subset.json"
    if test_data_path.exists():
        with open(test_data_path, 'r') as f:
            return json.load(f)
    else:
        return [{"pgn": "mock_game", "move_times": [10, 15, 20]}]


@pytest.fixture
def sample_timing_df():
    """Create a sample timing DataFrame for testing."""
    return pd.DataFrame({
        'move_number': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'time_taken': [10.5, 15.2, 30.8, 8.1, 45.3, 5.0, 20.7, 12.4, 60.9, 3.2],
        'legal_moves': [20, 25, 30, 18, 35, 15, 28, 22, 40, 12],
        'complexity': [0.1, 0.3, 0.5, 0.2, 0.8, 0.1, 0.4, 0.3, 0.9, 0.1],
        'is_critical': [False, False, True, False, True, False, False, False, True, False],
        'phase': ['opening', 'opening', 'middlegame', 'middlegame', 'middlegame', 
                 'middlegame', 'endgame', 'endgame', 'endgame', 'endgame']
    })


@pytest.fixture
def empty_timing_df():
    """Create an empty timing DataFrame for edge case testing."""
    return pd.DataFrame(columns=['move_number', 'time_taken', 'legal_moves', 'complexity'])


class TestTimeStats:
    """Test the basic time statistics calculation."""
    
    def test_time_stats_normal_case(self, sample_timing_df):
        """Test time stats calculation with normal data."""
        df_with_correct_col = sample_timing_df.rename(columns={'time_taken': 'move_time'})
        result = time_stats(df_with_correct_col)
        
        # time_stats returns (mean, std, cv) tuple
        assert isinstance(result, tuple)
        assert len(result) == 3
        
        mean_time, std_time, cv_time = result
        
        times = df_with_correct_col['move_time']
        expected_mean = times.mean()
        expected_std = times.std(ddof=1)  # time_stats uses ddof=1
        expected_cv = expected_std / expected_mean if expected_mean != 0 else np.nan
        
        assert abs(mean_time - expected_mean) < 1e-10
        assert abs(std_time - expected_std) < 1e-10
        if not np.isnan(expected_cv):
            assert abs(cv_time - expected_cv) < 1e-10
        else:
            assert np.isnan(cv_time)
    
    def test_time_stats_empty_dataframe(self, empty_timing_df):
        """Test time stats with empty DataFrame."""
        empty_df_with_col = empty_timing_df.assign(move_time=[])
        result = time_stats(empty_df_with_col)
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        
        mean_time, std_time, cv_time = result
        assert np.isnan(mean_time)
        assert np.isnan(std_time)
        assert np.isnan(cv_time)
    
    def test_time_stats_single_move(self):
        """Test time stats with single move."""
        df = pd.DataFrame({
            'move_time': [15.5],
            'move_number': [1]
        })
        result = time_stats(df)
        
        mean_time, std_time, cv_time = result
        assert mean_time == 15.5
        assert np.isnan(std_time)  # Single value with ddof=1 gives NaN
        assert np.isnan(cv_time)   # CV is NaN when std is NaN
    
    def test_time_stats_with_zeros(self):
        """Test time stats with zero time values."""
        df = pd.DataFrame({
            'move_time': [0, 10, 20, 0, 30],
            'move_number': [1, 2, 3, 4, 5]
        })
        result = time_stats(df)
        
        mean_time, std_time, cv_time = result
        assert mean_time == 12.0  # (0+10+20+0+30)/5
        assert std_time > 0  # Should have variation
        assert cv_time > 0


class TestTimeComplexityCorrelation:
    """Test the time-complexity correlation calculation."""
    
    def test_time_complexity_correlation_normal(self, sample_timing_df):
        """Test time-complexity correlation with normal data."""
        df_with_correct_col = sample_timing_df.rename(columns={'time_taken': 'move_time'})
        result = time_complexity_correlation(df_with_correct_col)
        
        assert isinstance(result, (int, float)) or result is None
        
        if result is not None:
            # Correlation should be between -1 and 1
            assert -1 <= result <= 1
    
    def test_time_complexity_correlation_perfect_positive(self):
        """Test with perfect positive correlation."""
        df = pd.DataFrame({
            'move_time': [1, 2, 3, 4, 5],
            'legal_moves': [10, 20, 30, 40, 50]
        })
        result = time_complexity_correlation(df)
        
        assert result is not None
        assert abs(result - 1.0) < 1e-10
    
    def test_time_complexity_correlation_no_correlation(self):
        """Test with no correlation."""
        df = pd.DataFrame({
            'move_time': [1, 2, 1, 2, 1],
            'legal_moves': [10, 10, 10, 10, 10]  # Constant complexity
        })
        result = time_complexity_correlation(df)
        
        assert result == 0.0
    
    def test_time_complexity_correlation_empty(self, empty_timing_df):
        """Test with empty DataFrame."""
        result = time_complexity_correlation(empty_timing_df)
        assert result == 0.0


class TestDetectLagSpikes:
    """Test the lag spike detection."""
    
    def test_detect_lag_spikes_normal(self, sample_timing_df):
        """Test lag spike detection with normal data."""
        df_with_correct_col = sample_timing_df.rename(columns={'time_taken': 'move_time'})
        result = detect_lag_spikes(df_with_correct_col, pause_sec=(5.0, 50.0), rapid_window=2, rapid_thresh=15)
        
        assert isinstance(result, (list, dict))
        
        if isinstance(result, list):
            for spike_idx in result:
                assert isinstance(spike_idx, int)
                assert 0 <= spike_idx < len(sample_timing_df)
        elif isinstance(result, dict):
            assert 'spike_count' in result or 'spike_indices' in result
    
    def test_detect_lag_spikes_no_spikes(self):
        """Test with data that has no lag spikes."""
        df = pd.DataFrame({
            'move_time': [5, 6, 7, 5, 6, 7, 5, 6],  # All below pause threshold
            'move_number': list(range(1, 9))
        })
        result = detect_lag_spikes(df, pause_sec=(10.0, 20.0), rapid_window=3, rapid_thresh=8)
        
        assert isinstance(result, list)
        assert len(result) == 0
    
    def test_detect_lag_spikes_with_clear_spike(self):
        """Test with clear lag spike pattern."""
        df = pd.DataFrame({
            'move_time': [5, 6, 50, 3, 4, 2, 7, 8],  # 50s pause followed by rapid moves
            'move_number': list(range(1, 9))
        })
        result = detect_lag_spikes(df, pause_sec=(40.0, 60.0), rapid_window=2, rapid_thresh=5)
        
        # Should detect the spike at index 2
        assert isinstance(result, list)
        assert len(result) > 0
        assert 2 in result  # Index of the 50s pause
    
    def test_detect_lag_spikes_edge_cases(self):
        """Test edge cases for lag spike detection."""
        df = pd.DataFrame({
            'move_time': [10, 20],
            'move_number': [1, 2]
        })
        result = detect_lag_spikes(df, pause_sec=(15.0, 25.0), rapid_window=3, rapid_thresh=5)
        
        assert isinstance(result, list)
        assert len(result) == 0  # Too short to have spikes with window=3


class TestAggregateTimeFeatures:
    """Test the main time features aggregation function."""
    
    def test_aggregate_time_features_normal(self, sample_timing_df):
        """Test time features aggregation with normal data."""
        result = aggregate_time_features(sample_timing_df)
        
        assert isinstance(result, dict)
        
        expected_keys = [
            'mean_move_time', 'time_variance', 'time_complexity_corr', 
            'lag_spike_count', 'uniformity_score', 'clutch_accuracy_diff', 'timing_score'
        ]
        
        present_keys = [key for key in expected_keys if key in result]
        assert len(present_keys) > 0
        
        for key, value in result.items():
            if key.endswith('_time') and not pd.isna(value):
                assert value >= 0  # Time values should be non-negative
            elif key.endswith('_corr') and not pd.isna(value):
                assert -1 <= value <= 1  # Correlation should be between -1 and 1
    
    def test_aggregate_time_features_empty(self, empty_timing_df):
        """Test time features aggregation with empty DataFrame."""
        result = aggregate_time_features(empty_timing_df)
        
        assert isinstance(result, dict)
        
        for key, value in result.items():
            assert pd.isna(value) or value == 0 or isinstance(value, dict)
    
    def test_aggregate_time_features_with_test_games(self, test_games_data):
        """Test time features with realistic game data structure."""
        realistic_df = pd.DataFrame({
            'move_number': list(range(1, 41)),  # 40 moves
            'time_taken': np.random.exponential(15, 40),  # Realistic time distribution
            'legal_moves': np.random.randint(5, 45, 40),  # Realistic legal moves count
            'complexity': np.random.uniform(0, 1, 40),
            'phase': ['opening'] * 10 + ['middlegame'] * 20 + ['endgame'] * 10,
            'is_critical': np.random.choice([True, False], 40, p=[0.2, 0.8])
        })
        
        result = aggregate_time_features(realistic_df)
        
        assert isinstance(result, dict)
        assert len(result) > 0
        
        if 'mean_move_time' in result and not pd.isna(result['mean_move_time']):
            assert result['mean_move_time'] > 0
        
        if 'timing_score' in result and not pd.isna(result['timing_score']):
            assert isinstance(result['timing_score'], (int, float))
    
    def test_aggregate_time_features_phase_analysis(self):
        """Test phase-specific time analysis."""
        df = pd.DataFrame({
            'move_number': list(range(1, 21)),
            'time_taken': [5] * 5 + [15] * 10 + [25] * 5,  # Different times per phase
            'phase': ['opening'] * 5 + ['middlegame'] * 10 + ['endgame'] * 5,
            'legal_moves': [20] * 20
        })
        
        result = aggregate_time_features(df)
        
        assert isinstance(result, dict)
        
        if 'phase_time_distribution' in result:
            phase_dist = result['phase_time_distribution']
            if isinstance(phase_dist, dict):
                if 'opening' in phase_dist and 'endgame' in phase_dist:
                    assert phase_dist['opening'] < phase_dist['endgame']


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_missing_columns(self):
        """Test behavior with missing required columns."""
        df = pd.DataFrame({
            'move_number': [1, 2, 3],
            'legal_moves': [20, 25, 30]
        })
        
        with pytest.raises((KeyError, AttributeError)):
            time_stats(df)
    
    def test_negative_times(self):
        """Test with negative time values."""
        df = pd.DataFrame({
            'move_time': [-5, 10, 15, -2, 20],
            'move_number': [1, 2, 3, 4, 5],
            'legal_moves': [20, 25, 30, 18, 35]
        })
        
        result = time_stats(df)
        assert isinstance(result, tuple)
        assert len(result) == 3
    
    def test_extreme_time_values(self):
        """Test with extreme time values."""
        df = pd.DataFrame({
            'move_time': [0.001, 1000000, 0.1, 500000, 0.01],  # Very small and very large
            'move_number': [1, 2, 3, 4, 5],
            'legal_moves': [20, 25, 30, 18, 35]
        })
        
        result = time_stats(df)
        assert isinstance(result, tuple)
        mean_time, std_time, cv_time = result
        assert not pd.isna(mean_time)
        
        if not pd.isna(cv_time):
            assert cv_time >= 0
    
    def test_all_same_times(self):
        """Test with all identical time values."""
        df = pd.DataFrame({
            'move_time': [10, 10, 10, 10, 10],
            'move_number': [1, 2, 3, 4, 5],
            'legal_moves': [20, 25, 30, 18, 35]
        })
        
        result = time_stats(df)
        mean_time, std_time, cv_time = result
        assert mean_time == 10
        assert std_time == 0
        assert cv_time == 0
        
        corr_result = time_complexity_correlation(df)
        assert corr_result == 0.0
