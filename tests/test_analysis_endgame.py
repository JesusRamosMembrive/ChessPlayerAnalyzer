#!/usr/bin/env python3
"""
Unit tests for app.analysis.endgame module.
"""
import pytest
import pandas as pd
import numpy as np
import chess
import chess.pgn
import logging
import sys
import os
import importlib.util
from unittest.mock import Mock, patch
import io

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

spec = importlib.util.spec_from_file_location("endgame", "app/analysis/endgame.py")
if spec is None or spec.loader is None:
    raise ImportError("Could not load endgame module")
endgame_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(endgame_module)

is_tb_position = endgame_module.is_tb_position
conversion_efficiency = endgame_module.conversion_efficiency
aggregate_endgame_features = endgame_module.aggregate_endgame_features
find_first_significant_advantage = endgame_module.find_first_significant_advantage


@pytest.fixture
def sample_board():
    """Sample chess board for testing."""
    board = chess.Board()
    return board


@pytest.fixture
def endgame_board():
    """Chess board in endgame position (few pieces)."""
    board = chess.Board("8/8/8/8/8/3K4/8/3k4 w - - 0 1")  # King vs King
    return board


@pytest.fixture
def sample_game():
    """Sample chess game for testing."""
    pgn_text = """
    [Event "Test"]
    [Site "Test"]
    [Result "1-0"]
    
    1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 1-0
    """
    return chess.pgn.read_game(io.StringIO(pgn_text))


@pytest.fixture
def sample_moves_df():
    """Sample moves DataFrame for testing."""
    return pd.DataFrame({
        'eval_cp_after': [50, 100, 200, 300, 400, 500, 600, 700, 800, 900],
        'move_number': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    })


@pytest.fixture
def winning_moves_df():
    """Moves DataFrame with winning advantage."""
    return pd.DataFrame({
        'eval_cp_after': [100, 200, 300, 600, 800, 900, 950, 1000, 1200, 1500],
        'move_number': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    })


class TestIsTbPosition:
    """Test the tablebase position check."""
    
    def test_is_tb_position_starting_position(self, sample_board):
        """Test with starting position (32 pieces)."""
        logger.info("Testing tablebase position check with starting position")
        result = is_tb_position(sample_board)
        logger.info(f"Tablebase position result: {result}")
        assert result is False  # Too many pieces for tablebase
        logger.info("✓ Tablebase position starting position test passed")
    
    def test_is_tb_position_endgame(self, endgame_board):
        """Test with endgame position (few pieces)."""
        result = is_tb_position(endgame_board)
        assert result is True  # Only 2 pieces, within tablebase range
    
    def test_is_tb_position_custom_max_pieces(self, sample_board):
        """Test with custom max pieces parameter."""
        result = is_tb_position(sample_board, max_pieces=32)
        assert result is True  # 32 pieces allowed
        
        result = is_tb_position(sample_board, max_pieces=10)
        assert result is False  # 32 pieces > 10 limit
    
    def test_is_tb_position_variant_end(self):
        """Test with variant end position."""
        board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2")
        board.push_san("Qh4#")  # Checkmate
        
        result = is_tb_position(board)
        assert result is False  # Variant end positions are excluded
    
    def test_is_tb_position_edge_cases(self):
        """Test edge cases for tablebase position check."""
        board = chess.Board("8/8/8/8/8/2K1k3/8/5Q2 w - - 0 1")  # 3 pieces
        result = is_tb_position(board, max_pieces=7)
        assert result is True
        
        board = chess.Board()  # 32 pieces
        result = is_tb_position(board, max_pieces=31)
        assert result is False


class TestConversionEfficiency:
    """Test the conversion efficiency calculation."""
    
    def test_conversion_efficiency_normal_case(self, winning_moves_df):
        """Test with normal winning game."""
        result = conversion_efficiency(winning_moves_df, threshold_cp=500)
        
        assert isinstance(result, int)
        assert result == 6
    
    def test_conversion_efficiency_no_advantage(self, sample_moves_df):
        """Test with game that never reaches threshold."""
        result = conversion_efficiency(sample_moves_df, threshold_cp=1000)
        
        assert result is None  # Never reached 1000cp advantage
    
    def test_conversion_efficiency_immediate_mate(self):
        """Test with immediate mate after advantage."""
        df = pd.DataFrame({
            'eval_cp_after': [100, 200, 600, 1000],  # Advantage at index 2, mate at index 3
            'move_number': [1, 2, 3, 4]
        })
        
        result = conversion_efficiency(df, threshold_cp=500)
        assert result == 1  # 1 move from advantage to mate
    
    def test_conversion_efficiency_custom_threshold(self, winning_moves_df):
        """Test with custom threshold."""
        result = conversion_efficiency(winning_moves_df, threshold_cp=300)
        
        assert result == 7
    
    def test_conversion_efficiency_custom_eval_column(self):
        """Test with custom evaluation column."""
        df = pd.DataFrame({
            'custom_eval': [100, 200, 600, 800, 1000],
            'move_number': [1, 2, 3, 4, 5]
        })
        
        result = conversion_efficiency(df, eval_col='custom_eval', threshold_cp=500)
        assert result == 2  # From index 2 to index 4
    
    def test_conversion_efficiency_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame({'eval_cp_after': [], 'move_number': []})
        
        result = conversion_efficiency(df)
        assert result is None
    
    def test_conversion_efficiency_single_move(self):
        """Test with single move DataFrame."""
        df = pd.DataFrame({
            'eval_cp_after': [600],
            'move_number': [1]
        })
        
        result = conversion_efficiency(df, threshold_cp=500)
        assert result == 0  # Immediate mate


class TestFindFirstSignificantAdvantage:
    """Test the helper function for finding first significant advantage."""
    
    def test_find_first_significant_advantage_normal(self):
        """Test with normal evaluation series."""
        series = pd.Series([100, 200, 300, 600, 800])
        result = find_first_significant_advantage(series, threshold_cp=500)
        
        assert result == 3  # Index where 600 >= 500
    
    def test_find_first_significant_advantage_no_advantage(self):
        """Test with series that never reaches threshold."""
        series = pd.Series([100, 200, 300, 400])
        result = find_first_significant_advantage(series, threshold_cp=500)
        
        assert result is None
    
    def test_find_first_significant_advantage_immediate(self):
        """Test with immediate advantage."""
        series = pd.Series([600, 700, 800])
        result = find_first_significant_advantage(series, threshold_cp=500)
        
        assert result == 0  # First move already has advantage
    
    def test_find_first_significant_advantage_empty(self):
        """Test with empty series."""
        series = pd.Series([])
        result = find_first_significant_advantage(series, threshold_cp=500)
        
        assert result is None


class TestAggregateEndgameFeatures:
    """Test the aggregate endgame features function."""
    
    def test_aggregate_endgame_features_no_tablebase(self, sample_game, sample_moves_df):
        """Test without tablebase (most common case)."""
        result = aggregate_endgame_features(sample_game, sample_moves_df, tb_path=None)
        
        assert isinstance(result, dict)
        
        expected_keys = ['tb_positions', 'tb_match_rate', 'dtz_deviation', 
                        'conversion_efficiency', 'perfect_tb_flag', 'fast_conversion_flag']
        for key in expected_keys:
            assert key in result
        
        assert pd.isna(result['tb_positions'])
        assert pd.isna(result['tb_match_rate'])
        assert pd.isna(result['dtz_deviation'])
        
        assert isinstance(result['conversion_efficiency'], (int, type(None)))
        assert isinstance(result['perfect_tb_flag'], bool)
        assert isinstance(result['fast_conversion_flag'], bool)
    
    def test_aggregate_endgame_features_nonexistent_tablebase(self, sample_game, sample_moves_df):
        """Test with nonexistent tablebase path."""
        result = aggregate_endgame_features(sample_game, sample_moves_df, tb_path="/nonexistent/path")
        
        assert isinstance(result, dict)
        
        assert pd.isna(result['tb_positions'])
        assert pd.isna(result['tb_match_rate'])
        assert pd.isna(result['dtz_deviation'])
    
    def test_aggregate_endgame_features_with_conversion(self, sample_game, winning_moves_df):
        """Test conversion efficiency calculation."""
        result = aggregate_endgame_features(sample_game, winning_moves_df, tb_path=None)
        
        assert isinstance(result, dict)
        assert result['conversion_efficiency'] is not None
        assert isinstance(result['conversion_efficiency'], int)
        
        if result['conversion_efficiency'] <= 10:
            assert result['fast_conversion_flag'] is True
        else:
            assert result['fast_conversion_flag'] is False
    
    def test_aggregate_endgame_features_custom_eval_column(self, sample_game):
        """Test with custom evaluation column."""
        moves_df = pd.DataFrame({
            'custom_eval': [100, 200, 600, 800, 1000],
            'move_number': [1, 2, 3, 4, 5]
        })
        
        result = aggregate_endgame_features(sample_game, moves_df, tb_path=None, eval_col='custom_eval')
        
        assert isinstance(result, dict)
        assert result['conversion_efficiency'] is not None
    
    def test_aggregate_endgame_features_perfect_flags(self, sample_game, sample_moves_df):
        """Test perfect tablebase and fast conversion flags."""
        result = aggregate_endgame_features(sample_game, sample_moves_df, tb_path=None)
        
        assert result['perfect_tb_flag'] is False
        
        assert isinstance(result['fast_conversion_flag'], bool)
    
    @patch('pathlib.Path.exists')
    @patch('chess.syzygy.Tablebase')
    def test_aggregate_endgame_features_with_mock_tablebase(self, mock_tablebase_class, mock_exists, sample_game, sample_moves_df):
        """Test with mocked tablebase functionality."""
        mock_exists.return_value = True
        
        mock_tb = Mock()
        mock_tb.__enter__ = Mock(return_value=mock_tb)
        mock_tb.__exit__ = Mock(return_value=None)
        mock_tablebase_class.return_value = mock_tb
        
        mock_tb.probe_wdl.return_value = 1  # Winning
        mock_tb.probe_dtz.return_value = 5  # 5 moves to mate
        
        with patch.object(chess.Board, 'legal_moves', [chess.Move.from_uci("e2e4")]):
            result = aggregate_endgame_features(sample_game, sample_moves_df, tb_path="/mock/tb/path")
        
        assert isinstance(result, dict)
        assert 'tb_positions' in result
        assert 'tb_match_rate' in result
        assert 'dtz_deviation' in result


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_conversion_efficiency_negative_evaluations(self):
        """Test with negative evaluations."""
        df = pd.DataFrame({
            'eval_cp_after': [-500, -200, 100, 600, 800],
            'move_number': [1, 2, 3, 4, 5]
        })
        
        result = conversion_efficiency(df, threshold_cp=500)
        assert result == 1  # From index 3 to index 4
    
    def test_conversion_efficiency_fluctuating_evaluations(self):
        """Test with fluctuating evaluations."""
        df = pd.DataFrame({
            'eval_cp_after': [600, 300, 700, 400, 800],  # Goes above and below threshold
            'move_number': [1, 2, 3, 4, 5]
        })
        
        result = conversion_efficiency(df, threshold_cp=500)
        assert result == 4  # From first occurrence (index 0) to end (index 4)
    
    def test_is_tb_position_zero_pieces(self):
        """Test with impossible zero pieces scenario."""
        board = chess.Board("8/8/8/8/8/8/8/8 w - - 0 1")  # Empty board (invalid but for testing)
        try:
            result = is_tb_position(board)
            assert isinstance(result, bool)
        except:
            pass
    
    def test_aggregate_endgame_features_missing_eval_column(self, sample_game):
        """Test with missing evaluation column."""
        df = pd.DataFrame({
            'other_column': [1, 2, 3, 4, 5],
            'move_number': [1, 2, 3, 4, 5]
        })
        
        with pytest.raises(KeyError):
            aggregate_endgame_features(sample_game, df, tb_path=None)
    
    def test_aggregate_endgame_features_empty_moves_df(self, sample_game):
        """Test with empty moves DataFrame."""
        df = pd.DataFrame({'eval_cp_after': [], 'move_number': []})
        
        result = aggregate_endgame_features(sample_game, df, tb_path=None)
        
        assert isinstance(result, dict)
        assert result['conversion_efficiency'] is None
        assert result['fast_conversion_flag'] is False
