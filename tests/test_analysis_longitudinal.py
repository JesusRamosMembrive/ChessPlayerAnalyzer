#!/usr/bin/env python3
"""
Unit tests for app.analysis.longitudinal module.
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os
import importlib.util
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

spec = importlib.util.spec_from_file_location("longitudinal", "app/analysis/longitudinal.py")
if spec is None or spec.loader is None:
    raise ImportError("Could not load longitudinal module")
longitudinal_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(longitudinal_module)

roi_per_game = longitudinal_module.roi_per_game
detect_step_function = longitudinal_module.detect_step_function
compute_trends = longitudinal_module.compute_trends
performance_rating = longitudinal_module.performance_rating
aggregate_roi = longitudinal_module.aggregate_roi
longest_streak = longitudinal_module.longest_streak
selectivity_score = longitudinal_module.selectivity_score


@pytest.fixture
def sample_games_df():
    """Sample games DataFrame for testing."""
    return pd.DataFrame({
        'match_pct': [0.6, 0.7, 0.8, 0.5, 0.9, 0.4, 0.75, 0.85, 0.65, 0.55],
        'acpl': [100, 80, 60, 120, 40, 140, 70, 50, 90, 110],
        'match_rate': [0.6, 0.7, 0.8, 0.5, 0.9, 0.4, 0.75, 0.85, 0.65, 0.55],
        'weighted_match_rate': [0.62, 0.72, 0.82, 0.52, 0.92, 0.42, 0.77, 0.87, 0.67, 0.57],
        'game_id': [f'game_{i}' for i in range(10)],
        'date': pd.date_range('2024-01-01', periods=10, freq='D'),
        'created_at': pd.date_range('2024-01-01', periods=10, freq='D'),
        'roi': [2200, 2300, 2400, 2100, 2500, 2000, 2350, 2450, 2250, 2150]
    })


@pytest.fixture
def step_function_df():
    """DataFrame with clear step function pattern."""
    n = 20
    first_half = pd.DataFrame({
        'match_pct': [0.4] * (n//2),  # Constant poor performance
        'acpl': [150] * (n//2),       # Constant high error
        'game_id': [f'game_{i}' for i in range(n//2)]
    })
    second_half = pd.DataFrame({
        'match_pct': [0.8] * (n//2),  # Constant good performance
        'acpl': [60] * (n//2),        # Constant low error
        'game_id': [f'game_{i}' for i in range(n//2, n)]
    })
    return pd.concat([first_half, second_half], ignore_index=True)


@pytest.fixture
def trends_df():
    """DataFrame with trend data including dates."""
    base_date = datetime(2024, 1, 1)
    dates = [base_date + timedelta(days=i*7) for i in range(24)]  # Weekly games for 6 months
    
    acpl_values = [150 - i*2 + np.random.normal(0, 5) for i in range(24)]
    match_values = [0.4 + i*0.01 + np.random.normal(0, 0.02) for i in range(24)]
    roi_values = [2000 + i*10 + np.random.normal(0, 20) for i in range(24)]
    
    return pd.DataFrame({
        'date': dates,
        'acpl': acpl_values,
        'match_rate': match_values,
        'roi': roi_values,
        'game_id': [f'game_{i}' for i in range(24)]
    })


class TestRoiPerGame:
    """Test the ROI per game calculation."""
    
    def test_roi_per_game_normal_case(self, sample_games_df):
        """Test ROI calculation with normal data."""
        result = roi_per_game(sample_games_df)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_games_df)
        assert not result.empty
        
        expected_first = performance_rating(sample_games_df['match_pct'].iloc[0], 
                                          sample_games_df['acpl'].iloc[0])
        assert abs(result.iloc[0] - expected_first) < 1e-10
    
    def test_roi_per_game_with_explicit_columns(self, sample_games_df):
        """Test ROI calculation with explicitly specified columns."""
        result = roi_per_game(sample_games_df, match_col='match_rate', acpl_col='acpl')
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_games_df)
        
        expected_first = performance_rating(sample_games_df['match_rate'].iloc[0], 
                                          sample_games_df['acpl'].iloc[0])
        assert abs(result.iloc[0] - expected_first) < 1e-10
    
    def test_roi_per_game_missing_columns(self):
        """Test ROI calculation with missing required columns."""
        df = pd.DataFrame({
            'other_col': [1, 2, 3],
            'another_col': [4, 5, 6]
        })
        
        result = roi_per_game(df)
        
        assert isinstance(result, pd.Series)
        assert result.empty
        assert result.dtype == float
    
    def test_roi_per_game_partial_columns(self):
        """Test with only one of the required columns."""
        df = pd.DataFrame({
            'match_pct': [0.6, 0.7, 0.8],
            'other_col': [1, 2, 3]
        })
        
        result = roi_per_game(df)
        
        assert isinstance(result, pd.Series)
        assert result.empty
    
    def test_roi_per_game_alternative_column_names(self):
        """Test with alternative column names."""
        df = pd.DataFrame({
            'match_rate': [0.6, 0.7, 0.8],
            'acl': [100, 80, 60]
        })
        
        result = roi_per_game(df)
        
        assert isinstance(result, pd.Series)
        assert len(result) == 3
        assert not result.empty
    
    def test_roi_per_game_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        
        result = roi_per_game(df)
        
        assert isinstance(result, pd.Series)
        assert result.empty
    
    def test_roi_per_game_nan_values(self):
        """Test with NaN values in data."""
        df = pd.DataFrame({
            'match_pct': [0.6, np.nan, 0.8],
            'acpl': [100, 80, np.nan]
        })
        
        result = roi_per_game(df)
        
        assert isinstance(result, pd.Series)
        assert len(result) == 3
        assert pd.isna(result.iloc[1])
        assert pd.isna(result.iloc[2])


class TestDetectStepFunction:
    """Test the step function detection."""
    
    def test_detect_step_function_match_pct(self, step_function_df):
        """Test step function detection with match_pct."""
        result = detect_step_function(step_function_df, aliases=("match_pct",), min_delta=0.2)
        
        assert isinstance(result, dict)
        assert "step_match_pct_delta" in result
        assert "step_match_pct_index" in result
        assert "step_match_pct_flag" in result
        
        assert isinstance(result["step_match_pct_delta"], float)
        assert isinstance(result["step_match_pct_index"], int)
        assert isinstance(result["step_match_pct_flag"], bool)
        
    
    def test_detect_step_function_acpl(self, step_function_df):
        """Test step function detection with ACPL (error metric)."""
        result = detect_step_function(step_function_df, aliases=("acpl",), min_delta=50)
        
        assert isinstance(result, dict)
        assert "step_acpl_delta" in result
        assert "step_acpl_index" in result
        assert "step_acpl_flag" in result
        
    
    def test_detect_step_function_no_step(self, sample_games_df):
        """Test with data that has no clear step function."""
        result = detect_step_function(sample_games_df, aliases=("match_pct",), min_delta=0.5)
        
        assert isinstance(result, dict)
        assert "step_match_pct_delta" in result
        assert "step_match_pct_index" in result
        assert "step_match_pct_flag" in result
        
        assert result["step_match_pct_flag"] is False
        assert result["step_match_pct_index"] == -1
    
    def test_detect_step_function_missing_column(self):
        """Test with missing metric column."""
        df = pd.DataFrame({
            'other_col': [1, 2, 3, 4, 5],
            'another_col': [6, 7, 8, 9, 10]
        })
        
        result = detect_step_function(df, aliases=("match_pct", "acpl"), min_delta=10)
        
        assert isinstance(result, dict)
        assert "step_unknown_delta" in result
        assert "step_unknown_index" in result
        assert "step_unknown_flag" in result
        
        assert result["step_unknown_flag"] is False
        assert result["step_unknown_index"] == -1
        assert result["step_unknown_delta"] == 0.0
    
    def test_detect_step_function_custom_window(self, step_function_df):
        """Test with custom window size."""
        result = detect_step_function(step_function_df, aliases=("match_pct",), 
                                    min_delta=0.2, window=5)
        
        assert isinstance(result, dict)
        assert "step_match_pct_flag" in result
        assert isinstance(result["step_match_pct_flag"], bool)
    
    def test_detect_step_function_small_dataset(self):
        """Test with very small dataset."""
        df = pd.DataFrame({
            'match_pct': [0.4, 0.8],
            'game_id': ['game_1', 'game_2']
        })
        
        result = detect_step_function(df, aliases=("match_pct",), min_delta=0.2)
        
        assert isinstance(result, dict)
        assert "step_match_pct_flag" in result
        assert isinstance(result["step_match_pct_flag"], bool)
    
    def test_detect_step_function_all_nan(self):
        """Test with all NaN values."""
        df = pd.DataFrame({
            'match_pct': [np.nan, np.nan, np.nan, np.nan],
            'game_id': ['game_1', 'game_2', 'game_3', 'game_4']
        })
        
        result = detect_step_function(df, aliases=("match_pct",), min_delta=0.2)
        
        assert isinstance(result, dict)
        assert result["step_match_pct_flag"] is False
        assert result["step_match_pct_index"] == -1
        assert result["step_match_pct_delta"] == 0.0


class TestComputeTrends:
    """Test the trend computation function."""
    
    def test_compute_trends_normal_case(self, trends_df):
        """Test trend computation with normal data."""
        result = compute_trends(trends_df)
        
        assert isinstance(result, dict)
        assert "trend_acpl" in result
        assert "trend_match_rate" in result
        assert "roi_curve" in result
        
        assert isinstance(result["trend_acpl"], float)
        assert isinstance(result["trend_match_rate"], float)
        
        assert result["trend_acpl"] < 0
        
        assert result["trend_match_rate"] > 0
        
        if result["roi_curve"] is not None:
            assert isinstance(result["roi_curve"], list)
            assert len(result["roi_curve"]) <= 24
    
    def test_compute_trends_missing_date_column(self, sample_games_df):
        """Test with missing date column."""
        df = sample_games_df.drop('date', axis=1)
        
        result = compute_trends(df)
        
        assert isinstance(result, dict)
        assert result == {}  # Should return empty dict
    
    def test_compute_trends_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        
        result = compute_trends(df)
        
        assert isinstance(result, dict)
        assert result == {}
    
    def test_compute_trends_missing_required_columns(self):
        """Test with missing required columns."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10, freq='D'),
            'other_col': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })
        
        with pytest.raises(KeyError):
            compute_trends(df)
    
    def test_compute_trends_all_nan_values(self):
        """Test with all NaN values in metric columns."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10, freq='D'),
            'acpl': [np.nan] * 10,
            'match_rate': [np.nan] * 10,
            'roi': [np.nan] * 10
        })
        
        result = compute_trends(df)
        
        assert isinstance(result, dict)
        assert result["trend_acpl"] == 0.0
        assert result["trend_match_rate"] == 0.0
        if result["roi_curve"] is not None:
            assert result["roi_curve"] == []
    
    def test_compute_trends_single_data_point(self):
        """Test with single data point."""
        df = pd.DataFrame({
            'date': [datetime(2024, 1, 1)],
            'acpl': [100],
            'match_rate': [0.6],
            'roi': [2200]
        })
        
        with pytest.raises(np.linalg.LinAlgError):
            compute_trends(df)
    
    def test_compute_trends_constant_values(self):
        """Test with constant values (no trend)."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10, freq='D'),
            'acpl': [100] * 10,
            'match_rate': [0.6] * 10,
            'roi': [2200] * 10
        })
        
        result = compute_trends(df)
        
        assert isinstance(result, dict)
        assert abs(result["trend_acpl"]) < 1e-10
        assert abs(result["trend_match_rate"]) < 1e-10


class TestHelperFunctions:
    """Test helper functions used by main functions."""
    
    def test_performance_rating_calculation(self):
        """Test the performance rating calculation."""
        match_pct = 0.7
        acpl = 80
        
        result = performance_rating(match_pct, acpl)
        
        expected = 800 * 0.7 + (-0.5) * 80 + 2000
        assert abs(result - expected) < 1e-10
    
    def test_performance_rating_edge_cases(self):
        """Test performance rating with edge cases."""
        result_zero = performance_rating(0.0, 0.0)
        expected_zero = 800 * 0.0 + (-0.5) * 0.0 + 2000
        assert abs(result_zero - expected_zero) < 1e-10
        
        result_high = performance_rating(1.0, 200)
        expected_high = 800 * 1.0 + (-0.5) * 200 + 2000
        assert abs(result_high - expected_high) < 1e-10
    
    def test_aggregate_roi_normal_case(self, sample_games_df):
        """Test ROI aggregation."""
        result = aggregate_roi(sample_games_df)
        
        assert isinstance(result, dict)
        expected_keys = ['roi_mean', 'roi_max', 'roi_sd', 'roi_games>2']
        for key in expected_keys:
            assert key in result
        
        assert isinstance(result['roi_mean'], float)
        assert isinstance(result['roi_max'], float)
        assert isinstance(result['roi_sd'], float)
        assert isinstance(result['roi_games>2'], (int, np.integer))
    
    def test_aggregate_roi_empty_data(self):
        """Test ROI aggregation with empty data."""
        df = pd.DataFrame()
        result = aggregate_roi(df)
        
        assert isinstance(result, dict)
        assert result['roi_mean'] == 0.0
        assert result['roi_max'] == 0.0
        assert result['roi_std'] == 0.0
        assert result['roi_games>2'] == 0
    
    def test_longest_streak_calculation(self):
        """Test longest streak calculation."""
        roi_series = pd.Series([2.0, 2.8, 2.9, 1.5, 2.8, 2.9, 3.0, 3.1, 2.0])
        
        result = longest_streak(roi_series, threshold=2.75)
        
        assert isinstance(result, int)
        assert result >= 0
        assert result > 0
    
    def test_longest_streak_no_streak(self):
        """Test longest streak with no qualifying values."""
        roi_series = pd.Series([2.0, 2.5, 2.3, 2.1, 2.4])
        
        result = longest_streak(roi_series, threshold=2.75)
        
        assert isinstance(result, int)
        assert result == 0
    
    def test_selectivity_score_calculation(self, sample_games_df):
        """Test selectivity score calculation."""
        result = selectivity_score(sample_games_df)
        
        assert isinstance(result, dict)
        assert "selectivity_pct" in result
        assert isinstance(result["selectivity_pct"], float)
        assert 0 <= result["selectivity_pct"] <= 100


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_roi_per_game_with_series_input(self):
        """Test that performance_rating handles Series input correctly."""
        df = pd.DataFrame({
            'match_pct': [0.6, 0.7, 0.8],
            'acpl': [100, 80, 60]
        })
        
        result = roi_per_game(df)
        
        assert isinstance(result, pd.Series)
        assert len(result) == 3
    
    def test_detect_step_function_with_insufficient_data(self):
        """Test step function detection with insufficient data for window."""
        df = pd.DataFrame({
            'match_pct': [0.4, 0.5, 0.6],  # Only 3 points
            'game_id': ['game_1', 'game_2', 'game_3']
        })
        
        result = detect_step_function(df, aliases=("match_pct",), min_delta=0.1, window=10)
        
        assert isinstance(result, dict)
        assert "step_match_pct_flag" in result
        assert isinstance(result["step_match_pct_flag"], bool)
    
    def test_compute_trends_with_mixed_data_types(self):
        """Test trends computation with mixed data types."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5, freq='D'),
            'acpl': [100, 90, 80, 70, 60],
            'match_rate': [0.5, 0.6, 0.7, 0.8, 0.9],
            'roi': [2000, 2100, 2200, 2300, 2400]
        })
        
        result = compute_trends(df)
        
        assert isinstance(result, dict)
        assert "trend_acpl" in result
        assert "trend_match_rate" in result
    
    def test_functions_with_extreme_values(self):
        """Test functions with extreme values."""
        df = pd.DataFrame({
            'match_pct': [0.0, 1.0, 0.5],
            'acpl': [0, 10000, 50],
            'date': pd.date_range('2024-01-01', periods=3, freq='D'),
            'match_rate': [0.0, 1.0, 0.5],
            'roi': [0, 5000, 2500]
        })
        
        roi_result = roi_per_game(df)
        assert isinstance(roi_result, pd.Series)
        
        step_result = detect_step_function(df, aliases=("match_pct",), min_delta=0.1)
        assert isinstance(step_result, dict)
        
        trends_result = compute_trends(df)
        assert isinstance(trends_result, dict)
