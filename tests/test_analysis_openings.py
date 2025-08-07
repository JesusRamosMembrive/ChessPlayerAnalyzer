#!/usr/bin/env python3
"""
Unit tests for app.analysis.openings module.
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

spec = importlib.util.spec_from_file_location("openings", "app/analysis/openings.py")
if spec is None or spec.loader is None:
    raise ImportError("Could not load openings module")
openings_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(openings_module)

shannon_entropy = openings_module.shannon_entropy
opening_entropy = openings_module.opening_entropy
aggregate_opening_features = openings_module.aggregate_opening_features


@pytest.fixture
def sample_eco_series():
    """Sample ECO codes for testing."""
    return pd.Series(['C54', 'B30', 'C54', 'C54', 'B30', 'A46', 'C54', 'C50', 'C50', 'B30'])


@pytest.fixture
def empty_eco_series():
    """Empty ECO series for edge case testing."""
    return pd.Series([], dtype=str)


@pytest.fixture
def sample_games_df():
    """Sample games DataFrame for testing."""
    return pd.DataFrame({
        'eco_code': ['C54', 'B30', 'C54', 'C54', 'B30', 'A46', 'C54', 'C50', 'C50', 'B30'],
        'opening_key': ['e4 e5 Nf3 Nc6', 'e4 c5', 'e4 e5 Nf3 Nc6', 'e4 e5 Nf3 Nc6', 
                       'e4 c5', 'Nf3 Nf6', 'e4 e5 Nf3 Nc6', 'e4 e5 Nf3 Nc6', 
                       'e4 e5 Nf3 Nc6', 'e4 c5']
    })


@pytest.fixture
def sample_moves_df():
    """Sample moves DataFrame for testing."""
    return pd.DataFrame({
        'played': ['e4', 'e5', 'Nf3', 'Nc6', 'Bc4', 'Be7', 'c3', 'Bh4'],
        'best_rank': [1, 1, 1, 2, 1, 3, 1, 1],
        'move_number': [1, 2, 3, 4, 5, 6, 7, 8]
    })


class TestShannonEntropy:
    """Test the Shannon entropy calculation."""
    
    def test_shannon_entropy_normal_case(self, sample_eco_series):
        """Test entropy calculation with normal data."""
        logger.info("Testing Shannon entropy with normal ECO data")
        result = shannon_entropy(sample_eco_series)
        logger.info(f"Shannon entropy result: {result}")
        
        assert isinstance(result, float)
        assert result >= 0.0
        
        counts = sample_eco_series.value_counts()
        probs = counts / counts.sum()
        expected = -(probs * np.log2(probs)).sum()
        
        assert abs(result - expected) < 1e-10
        logger.info("✓ Shannon entropy normal case test passed")
    
    def test_shannon_entropy_empty_series(self, empty_eco_series):
        """Test with empty series."""
        result = shannon_entropy(empty_eco_series)
        assert result == 0.0
    
    def test_shannon_entropy_single_value(self):
        """Test with series containing only one unique value."""
        series = pd.Series(['C54', 'C54', 'C54', 'C54'])
        result = shannon_entropy(series)
        assert result == 0.0
    
    def test_shannon_entropy_with_nan(self):
        """Test with NaN values in series."""
        series = pd.Series(['C54', 'B30', np.nan, 'C54', np.nan, 'B30'])
        result = shannon_entropy(series)
        
        assert isinstance(result, float)
        assert result >= 0.0
        
        clean_series = series.dropna()
        counts = clean_series.value_counts()
        probs = counts / counts.sum()
        expected = -(probs * np.log2(probs)).sum()
        
        assert abs(result - expected) < 1e-10
    
    def test_shannon_entropy_uniform_distribution(self):
        """Test with perfectly uniform distribution."""
        series = pd.Series(['A', 'B', 'C', 'D'])
        result = shannon_entropy(series)
        
        expected = 2.0
        assert abs(result - expected) < 1e-10


class TestOpeningEntropy:
    """Test the opening entropy with ELO adjustment."""
    
    def test_opening_entropy_basic(self, sample_games_df):
        """Test basic opening entropy calculation."""
        result = opening_entropy(sample_games_df)
        
        assert isinstance(result, dict)
        assert 'H_opening' in result
        assert 'H_z' in result
        assert isinstance(result['H_opening'], float)
        assert result['H_z'] is None  # No ELO reference provided
        assert result['H_opening'] >= 0.0
    
    def test_opening_entropy_with_elo_reference(self, sample_games_df):
        """Test opening entropy with ELO reference."""
        reference_entropy = pd.Series({1400: 1.5, 1800: 2.2, 2200: 2.8})
        elo = 1800
        
        result = opening_entropy(sample_games_df, elo=elo, reference_entropy_by_elo=reference_entropy)
        
        assert isinstance(result, dict)
        assert 'H_opening' in result
        assert 'H_z' in result
        assert isinstance(result['H_opening'], float)
        assert isinstance(result['H_z'], float)
        assert result['H_opening'] >= 0.0
    
    def test_opening_entropy_custom_column(self):
        """Test with custom ECO column name."""
        df = pd.DataFrame({'custom_eco': ['C54', 'B30', 'C54']})
        result = opening_entropy(df, eco_col='custom_eco')
        
        assert isinstance(result, dict)
        assert 'H_opening' in result
        assert result['H_opening'] >= 0.0
    
    def test_opening_entropy_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame({'eco_code': []})
        result = opening_entropy(df)
        
        assert isinstance(result, dict)
        assert result['H_opening'] == 0.0
        assert result['H_z'] is None
    
    def test_opening_entropy_missing_column(self):
        """Test with missing ECO column."""
        df = pd.DataFrame({'other_col': ['A', 'B', 'C']})
        
        with pytest.raises(KeyError):
            opening_entropy(df)


class TestAggregateOpeningFeatures:
    """Test the aggregate opening features function."""
    
    def test_aggregate_opening_features_normal(self, sample_games_df, sample_moves_df):
        """Test with normal input data."""
        opening_key = "e4 e5 Nf3 Nc6"
        eco_code = "C54"
        
        result = aggregate_opening_features(opening_key, eco_code, sample_moves_df, sample_games_df)
        
        assert isinstance(result, dict)
        assert len(result) > 0
        
        expected_keys = ['opening_entropy', 'novelty_depth', 'second_choice_rate', 
                        'opening_breadth', 'opening_score']
        for key in expected_keys:
            assert key in result
            assert isinstance(result[key], (int, float))
    
    def test_aggregate_opening_features_missing_eco(self, sample_moves_df):
        """Test with DataFrame missing eco_code column."""
        games_df = pd.DataFrame({'other_col': [1, 2, 3]})
        opening_key = "e4 e5"
        eco_code = "C54"
        
        result = aggregate_opening_features(opening_key, eco_code, sample_moves_df, games_df)
        
        assert isinstance(result, dict)
        assert 'opening_entropy' in result
        assert result['opening_entropy'] == 0.0
    
    def test_aggregate_opening_features_empty_moves(self, sample_games_df):
        """Test with empty moves DataFrame."""
        empty_moves_df = pd.DataFrame({'played': [], 'best_rank': [], 'move_number': []})
        opening_key = "e4 e5"
        eco_code = "C54"
        
        result = aggregate_opening_features(opening_key, eco_code, empty_moves_df, sample_games_df)
        
        assert isinstance(result, dict)
        assert 'novelty_depth' in result
        assert 'second_choice_rate' in result
        assert 'opening_breadth' in result
    
    def test_aggregate_opening_features_novelty_calculation(self, sample_games_df):
        """Test novelty depth calculation."""
        moves_df = pd.DataFrame({
            'played': ['e4', 'e5', 'Nf3', 'Nc6', 'Bb5'],  # Bb5 not in opening_key
            'best_rank': [1, 1, 1, 1, 1],
            'move_number': [1, 2, 3, 4, 5]
        })
        opening_key = "e4 e5 Nf3 Nc6"
        eco_code = "C96"
        
        result = aggregate_opening_features(opening_key, eco_code, moves_df, sample_games_df)
        
        assert result['novelty_depth'] == 5
    
    def test_aggregate_opening_features_second_choice_rate(self, sample_games_df):
        """Test second choice rate calculation."""
        moves_df = pd.DataFrame({
            'played': ['e4', 'e5', 'Nf3', 'Nc6'],
            'best_rank': [1, 1, 1, 1],  # All best moves
            'move_number': [1, 2, 3, 4]
        })
        opening_key = "e4 e5 Nf3 Nc6"
        eco_code = "C54"
        
        result = aggregate_opening_features(opening_key, eco_code, moves_df, sample_games_df)
        
        assert result['second_choice_rate'] == 1.0
    
    def test_aggregate_opening_features_opening_breadth(self, sample_games_df):
        """Test opening breadth calculation."""
        moves_df = pd.DataFrame({
            'played': ['e4', 'e5', 'Nf3', 'Nc6', 'Bc4', 'Be7', 'c3', 'Bh4'],
            'best_rank': [1, 1, 1, 1, 1, 1, 1, 1],
            'move_number': [1, 2, 3, 4, 5, 6, 7, 8]
        })
        opening_key = "e4 e5 Nf3 Nc6"
        eco_code = "C54"
        
        result = aggregate_opening_features(opening_key, eco_code, moves_df, sample_games_df)
        
        assert result['opening_breadth'] == 8
    
    def test_aggregate_opening_features_opening_score(self, sample_games_df, sample_moves_df):
        """Test opening score calculation."""
        opening_key = "e4 e5 Nf3 Nc6"
        eco_code = "C54"
        
        result = aggregate_opening_features(opening_key, eco_code, sample_moves_df, sample_games_df)
        
        assert 'opening_score' in result
        assert isinstance(result['opening_score'], (int, float))
        assert 0 <= result['opening_score'] <= 100


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_shannon_entropy_all_nan(self):
        """Test Shannon entropy with all NaN values."""
        series = pd.Series([np.nan, np.nan, np.nan])
        result = shannon_entropy(series)
        assert result == 0.0
    
    def test_opening_entropy_interpolation_edge_cases(self):
        """Test opening entropy with edge case ELO values."""
        games_df = pd.DataFrame({'eco_code': ['C54', 'B30']})
        reference_entropy = pd.Series({1400: 1.5, 1800: 2.2, 2200: 2.8})
        
        result_low = opening_entropy(games_df, elo=1200, reference_entropy_by_elo=reference_entropy)
        assert isinstance(result_low['H_z'], float)
        
        result_high = opening_entropy(games_df, elo=2500, reference_entropy_by_elo=reference_entropy)
        assert isinstance(result_high['H_z'], float)
    
    def test_aggregate_opening_features_extreme_values(self, sample_games_df):
        """Test with extreme input values."""
        moves_df = pd.DataFrame({
            'played': ['e4'] * 100,  # Same move repeated
            'best_rank': [1] * 100,
            'move_number': list(range(1, 101))
        })
        opening_key = "e4"
        eco_code = "B00"
        
        result = aggregate_opening_features(opening_key, eco_code, moves_df, sample_games_df)
        
        assert isinstance(result, dict)
        assert all(isinstance(v, (int, float)) for v in result.values())
        
        assert result['opening_breadth'] == 1
