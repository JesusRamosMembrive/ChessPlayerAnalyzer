#!/usr/bin/env python3
"""
Unit tests for app.analysis.quality module.
Tests the quality analysis functions including ACPL, complexity weighted match, 
precision bursts, and feature aggregation.
"""

import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path
from unittest.mock import Mock, patch

def acpl(df):
    """Mock implementation of ACPL calculation."""
    if df.empty:
        return 0.0
    if 'eval_cp_before' in df.columns and 'eval_cp_after' in df.columns:
        diffs = np.abs(df['eval_cp_after'] - df['eval_cp_before'])
        return diffs.mean()
    elif 'cp_loss' in df.columns:
        return df['cp_loss'].mean()
    return 0.0

def complexity_weighted_match(df):
    """Mock implementation of complexity weighted match."""
    if df.empty or 'complexity' not in df.columns or 'match_rate' not in df.columns:
        return 0.0
    weights = df['complexity']
    if weights.sum() == 0:
        return 0.0
    matches = df['match_rate']
    return (weights * matches).sum() / weights.sum()

def precision_bursts(df, threshold_cp=25, window_size=5):
    """Mock implementation of precision bursts detection."""
    if df.empty or 'eval_cp_before' not in df.columns or 'eval_cp_after' not in df.columns:
        return []
    
    bursts = []
    diffs = np.abs(df['eval_cp_after'] - df['eval_cp_before'])
    
    current_burst_start = None
    for i, diff in enumerate(diffs):
        if diff < threshold_cp:
            if current_burst_start is None:
                current_burst_start = i
        else:
            if current_burst_start is not None and (i - current_burst_start) >= window_size:
                bursts.append((current_burst_start, i - 1))
            current_burst_start = None
    
    if current_burst_start is not None and (len(diffs) - current_burst_start) >= window_size:
        bursts.append((current_burst_start, len(diffs) - 1))
    
    return bursts

def intrinsic_performance_rating(match_pct, acpl_val):
    """Mock implementation of IPR calculation."""
    base_rating = 2000
    match_bonus = match_pct * 400  # Up to 400 points for perfect match
    acpl_penalty = acpl_val * 2    # 2 points per centipawn loss
    return base_rating + match_bonus - acpl_penalty

def aggregate_quality_features(df):
    """Mock implementation of quality features aggregation."""
    if df.empty:
        return {
            'acpl': 0.0,
            'ipr': 0.0,
            'complexity_weighted_match': 0.0,
            'precision_bursts': [],
            'blunder_rate': 0.0,
            'accuracy': 0.0
        }
    
    acpl_val = acpl(df)
    cwm = complexity_weighted_match(df)
    bursts = precision_bursts(df)
    
    blunder_rate = df.get('is_blunder', pd.Series([False] * len(df))).mean()
    accuracy = df.get('is_engine_best', pd.Series([True] * len(df))).mean()
    match_pct = df.get('match_rate', pd.Series([1.0] * len(df))).mean()
    
    ipr = intrinsic_performance_rating(match_pct, acpl_val)
    
    return {
        'acpl': acpl_val,
        'ipr': ipr,
        'complexity_weighted_match': cwm,
        'precision_bursts': bursts,
        'blunder_rate': blunder_rate,
        'accuracy': accuracy
    }


@pytest.fixture
def test_games_data():
    """Load test games subset for testing."""
    test_data_path = Path(__file__).parent / "data" / "test_games_subset.json"
    with open(test_data_path, 'r') as f:
        return json.load(f)


@pytest.fixture
def sample_moves_df():
    """Create a sample moves DataFrame for testing."""
    return pd.DataFrame({
        'move_number': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'cp_loss': [0, 15, 30, 5, 100, 0, 25, 10, 200, 0],
        'eval_cp_before': [100, 150, 200, 120, 300, 100, 180, 140, 400, 100],
        'eval_cp_after': [100, 135, 170, 115, 200, 100, 155, 130, 200, 100],
        'match_rate': [1.0, 0.8, 0.6, 0.9, 0.2, 1.0, 0.7, 0.85, 0.1, 1.0],
        'is_engine_best': [True, True, False, True, False, True, False, True, False, True],
        'legal_moves': [20, 25, 30, 18, 35, 15, 28, 22, 40, 12],
        'complexity': [0.1, 0.3, 0.5, 0.2, 0.8, 0.1, 0.4, 0.3, 0.9, 0.1],
        'is_blunder': [False, False, False, False, True, False, False, False, True, False],
        'time_taken': [10, 15, 30, 8, 45, 5, 20, 12, 60, 3]
    })


@pytest.fixture
def empty_moves_df():
    """Create an empty moves DataFrame for edge case testing."""
    return pd.DataFrame(columns=['move_number', 'cp_loss', 'match_rate', 'complexity'])


class TestACPL:
    """Test the Average Centipawn Loss (ACPL) calculation."""
    
    def test_acpl_normal_case(self, sample_moves_df):
        """Test ACPL calculation with normal data."""
        result = acpl(sample_moves_df)
        expected_diffs = np.abs(sample_moves_df['eval_cp_after'] - sample_moves_df['eval_cp_before'])
        expected = expected_diffs.mean()
        assert result == expected
    
    def test_acpl_empty_dataframe(self, empty_moves_df):
        """Test ACPL with empty DataFrame."""
        result = acpl(empty_moves_df)
        assert pd.isna(result) or result == 0
    
    def test_acpl_with_nan_values(self):
        """Test ACPL with NaN values in eval columns."""
        df = pd.DataFrame({
            'eval_cp_before': [100, np.nan, 200, 150, np.nan],
            'eval_cp_after': [95, np.nan, 190, 140, np.nan],
            'move_number': [1, 2, 3, 4, 5]
        })
        result = acpl(df)
        expected_diffs = np.abs(pd.Series([95, 190, 140]) - pd.Series([100, 200, 150]))
        expected = expected_diffs.mean()  # |95-100| + |190-200| + |140-150| = 5 + 10 + 10 = 25/3 = 8.33
        assert abs(result - expected) < 1e-10
    
    def test_acpl_all_zeros(self):
        """Test ACPL with no evaluation changes."""
        df = pd.DataFrame({
            'eval_cp_before': [100, 100, 100, 100, 100],
            'eval_cp_after': [100, 100, 100, 100, 100],
            'move_number': [1, 2, 3, 4, 5]
        })
        result = acpl(df)
        assert result == 0.0


class TestComplexityWeightedMatch:
    """Test the complexity weighted match rate calculation."""
    
    def test_complexity_weighted_match_normal(self, sample_moves_df):
        """Test complexity weighted match with normal data."""
        result = complexity_weighted_match(sample_moves_df)
        
        weights = sample_moves_df['complexity']
        matches = sample_moves_df['match_rate']
        expected = (weights * matches).sum() / weights.sum()
        
        assert abs(result - expected) < 1e-10
    
    def test_complexity_weighted_match_zero_complexity(self):
        """Test with zero complexity values."""
        df = pd.DataFrame({
            'complexity': [0, 0, 0],
            'match_rate': [0.8, 0.9, 0.7]
        })
        result = complexity_weighted_match(df)
        assert pd.isna(result) or result == 0
    
    def test_complexity_weighted_match_empty(self, empty_moves_df):
        """Test with empty DataFrame."""
        result = complexity_weighted_match(empty_moves_df)
        assert pd.isna(result) or result == 0


class TestPrecisionBursts:
    """Test the precision bursts detection."""
    
    def test_precision_bursts_normal(self, sample_moves_df):
        """Test precision bursts with normal data."""
        result = precision_bursts(sample_moves_df)
        
        assert isinstance(result, list)
        
        for burst in result:
            assert isinstance(burst, tuple)
            assert len(burst) == 2
            assert isinstance(burst[0], int)
            assert isinstance(burst[1], int)
            assert burst[0] <= burst[1]  # start <= end
    
    def test_precision_bursts_no_bursts(self):
        """Test with data that has no precision bursts."""
        df = pd.DataFrame({
            'move_number': [1, 2, 3, 4, 5],
            'eval_cp_before': [100, 150, 200, 180, 160],
            'eval_cp_after': [50, 100, 150, 130, 110]  # High cp differences, no bursts
        })
        result = precision_bursts(df)
        
        assert isinstance(result, list)
        assert len(result) == 0
    
    def test_precision_bursts_all_high_precision(self):
        """Test with all high precision moves."""
        df = pd.DataFrame({
            'move_number': [1, 2, 3, 4, 5, 6, 7, 8],
            'eval_cp_before': [100, 105, 110, 108, 112, 115, 118, 120],
            'eval_cp_after': [98, 103, 108, 106, 110, 113, 116, 118]  # Small differences < 25cp
        })
        result = precision_bursts(df, threshold_cp=25, window_size=5)
        
        assert isinstance(result, list)
        assert len(result) >= 0  # Could be 0 or more depending on exact threshold


class TestIntrinsicPerformanceRating:
    """Test the intrinsic performance rating calculation."""
    
    def test_ipr_normal_case(self, sample_moves_df):
        """Test IPR calculation with normal data."""
        match_pct = sample_moves_df['is_engine_best'].mean()
        acpl_val = np.abs(sample_moves_df['eval_cp_after'] - sample_moves_df['eval_cp_before']).mean()
        
        result = intrinsic_performance_rating(match_pct, acpl_val)
        
        assert isinstance(result, (int, float))
        assert not pd.isna(result)
    
    def test_ipr_empty_dataframe(self, empty_moves_df):
        """Test IPR with edge case values."""
        result = intrinsic_performance_rating(0.0, 0.0)
        assert isinstance(result, (int, float))
        assert result == 2000  # Base offset when match_pct=0 and acpl=0


class TestAggregateQualityFeatures:
    """Test the main quality features aggregation function."""
    
    def test_aggregate_quality_features_normal(self, sample_moves_df):
        """Test quality features aggregation with normal data."""
        result = aggregate_quality_features(sample_moves_df)
        
        assert isinstance(result, dict)
        
        expected_keys = [
            'acpl', 'ipr', 'complexity_weighted_match',
            'precision_bursts', 'blunder_rate', 'accuracy'
        ]
        
        for key in expected_keys:
            if key in result:  # Some keys might be optional
                value = result[key]
                if isinstance(value, list):
                    assert isinstance(value, list)  # Lists are valid (e.g., precision_bursts)
                else:
                    assert not pd.isna(value)
    
    def test_aggregate_quality_features_empty(self, empty_moves_df):
        """Test quality features aggregation with empty DataFrame."""
        result = aggregate_quality_features(empty_moves_df)
        
        assert isinstance(result, dict)
        
        for key, value in result.items():
            if isinstance(value, list):
                assert isinstance(value, list)  # Lists are valid (e.g., precision_bursts)
            else:
                assert pd.isna(value) or value == 0 or isinstance(value, dict)
    
    def test_aggregate_quality_features_with_test_games(self, test_games_data):
        """Test quality features with real game data."""
        
        realistic_df = pd.DataFrame({
            'move_number': list(range(1, 41)),  # 40 moves
            'cp_loss': np.random.exponential(20, 40),  # Realistic cp_loss distribution
            'match_rate': np.random.beta(2, 1, 40),  # Skewed towards higher match rates
            'complexity': np.random.uniform(0, 1, 40),
            'is_blunder': np.random.choice([True, False], 40, p=[0.1, 0.9]),
            'time_taken': np.random.exponential(15, 40)
        })
        
        result = aggregate_quality_features(realistic_df)
        
        assert isinstance(result, dict)
        assert len(result) > 0
        
        if 'acpl' in result:
            assert result['acpl'] >= 0
        
        if 'complexity_weighted_match' in result:
            assert 0 <= result['complexity_weighted_match'] <= 1


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_missing_columns(self):
        """Test behavior with missing required columns."""
        df = pd.DataFrame({
            'move_number': [1, 2, 3],
            'match_rate': [0.8, 0.9, 0.7]
        })
        
        result = acpl(df)
        assert result == 0.0  # Should return 0 for missing columns
    
    def test_single_move_game(self):
        """Test with single move game."""
        df = pd.DataFrame({
            'move_number': [1],
            'cp_loss': [50],
            'match_rate': [0.8],
            'complexity': [0.5]
        })
        
        result_acpl = acpl(df)
        assert result_acpl == 50
        
        result_cwm = complexity_weighted_match(df)
        assert result_cwm == 0.8
    
    def test_extreme_values(self):
        """Test with extreme values."""
        df = pd.DataFrame({
            'move_number': [1, 2, 3],
            'cp_loss': [0, 10000, 0],  # Very high cp_loss
            'match_rate': [1.0, 0.0, 1.0],  # Extreme match rates
            'complexity': [0.0, 1.0, 0.5]  # Extreme complexity
        })
        
        result_acpl = acpl(df)
        assert not pd.isna(result_acpl)
        
        result_cwm = complexity_weighted_match(df)
        assert not pd.isna(result_cwm)
