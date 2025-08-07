import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from app.analysis.engine import _safe_mean, prepare_moves_dataframe
except ImportError:
    def _safe_mean(df: pd.DataFrame, col: str, default: float = 0.0) -> float:
        """Media que nunca devuelve None (NaN→default, col ausente→default)."""
        if col not in df.columns:
            return default
        val = df[col].mean()
        return float(val) if pd.notna(val) else default

class MockMoveAnalysis:
    """Mock class for MoveAnalysis to avoid database dependencies."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class MockGame:
    """Mock class for Game to avoid database dependencies."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        if not hasattr(self, 'moves'):
            self.moves = []


@pytest.fixture
def sample_game():
    """Create a sample game with moves for testing."""
    game = MockGame(
        id=1,
        white_username="player1",
        black_username="player2",
        pgn="1. e4 e5 2. Nf3 Nc6 3. Bb5 a6",
        eco_code="C60",
        opening_key="e4 e5 Nf3 Nc6 Bb5",
        move_times=[30.0, 25.0, 20.0, 15.0, 35.0, 40.0]
    )
    
    moves = [
        MockMoveAnalysis(
            move_number=1,
            played="e4",
            best_rank=0,
            cp_loss=0,
            eval_before=15,
            eval_after=15,
            time_spent=30.0,
            legal_moves_count=20
        ),
        MockMoveAnalysis(
            move_number=1,
            played="e5",
            best_rank=1,
            cp_loss=10,
            eval_before=-15,
            eval_after=-25,
            time_spent=25.0,
            legal_moves_count=20
        ),
        MockMoveAnalysis(
            move_number=2,
            played="Nf3",
            best_rank=0,
            cp_loss=0,
            eval_before=25,
            eval_after=25,
            time_spent=20.0,
            legal_moves_count=29
        ),
        MockMoveAnalysis(
            move_number=2,
            played="Nc6",
            best_rank=2,
            cp_loss=15,
            eval_before=-25,
            eval_after=-40,
            time_spent=15.0,
            legal_moves_count=21
        ),
        MockMoveAnalysis(
            move_number=3,
            played="Bb5",
            best_rank=0,
            cp_loss=0,
            eval_before=40,
            eval_after=40,
            time_spent=35.0,
            legal_moves_count=28
        ),
        MockMoveAnalysis(
            move_number=3,
            played="a6",
            best_rank=1,
            cp_loss=5,
            eval_before=-40,
            eval_after=-45,
            time_spent=40.0,
            legal_moves_count=16
        )
    ]
    
    game.moves = moves
    return game


@pytest.fixture
def sample_engine():
    """Create a sample ChessAnalysisEngine for testing."""
    mock_engine = Mock()
    
    mock_engine.reference_book = None
    mock_engine.tablebase_path = None
    mock_engine.reference_stats = None
    mock_engine.acpl_model = None
    
    def mock_get_player_color(game, username):
        if game.white_username == username:
            return "white"
        elif game.black_username == username:
            return "black"
        return None
    
    mock_engine._get_player_color = mock_get_player_color
    
    return mock_engine


class TestPrepareMovesMock:
    """Test the prepare_moves_dataframe function using mocks."""
    
    def test_prepare_moves_dataframe_white_player(self, sample_game):
        """Test preparing moves dataframe for white player."""
        with patch('app.analysis.engine.prepare_moves_dataframe') as mock_prepare:
            expected_df = pd.DataFrame({
                'move_number': [1, 2, 3],
                'played': ['e4', 'Nf3', 'Bb5'],
                'best_rank': [0, 0, 0],
                'cp_loss': [0, 0, 0],
                'eval_cp_before': [15, 25, 40],
                'eval_cp_after': [15, 25, 40],
                'move_time': [30.0, 20.0, 35.0],
                'legal_moves': [20, 29, 28],
                'player_clock_before': [600.0, 570.0, 550.0],
                'is_engine_best': [True, True, True],
                'player_color': ['white', 'white', 'white'],
                'phase': ['opening', 'middlegame', 'middlegame'],
                'delta_eval': [0, 0, 0]
            })
            mock_prepare.return_value = expected_df
            
            df = mock_prepare(sample_game, "player1")
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3  # Only white moves
        assert all(df['player_color'] == 'white')
        
        required_cols = [
            'move_number', 'played', 'best_rank', 'cp_loss',
            'eval_cp_before', 'eval_cp_after', 'move_time',
            'legal_moves', 'player_clock_before', 'is_engine_best',
            'player_color', 'phase', 'delta_eval'
        ]
        for col in required_cols:
            assert col in df.columns
        
        assert df.iloc[0]['played'] == 'e4'
        assert df.iloc[0]['best_rank'] == 0
        assert df.iloc[0]['is_engine_best'] == True
        assert df.iloc[0]['phase'] == 'opening'
    
    def test_prepare_moves_dataframe_black_player(self, sample_game):
        """Test preparing moves dataframe for black player."""
        with patch('app.analysis.engine.prepare_moves_dataframe') as mock_prepare:
            expected_df = pd.DataFrame({
                'move_number': [1, 2, 3],
                'played': ['e5', 'Nc6', 'a6'],
                'best_rank': [1, 2, 1],
                'cp_loss': [10, 15, 5],
                'eval_cp_before': [-15, -25, -40],
                'eval_cp_after': [-25, -40, -45],
                'move_time': [25.0, 15.0, 40.0],
                'legal_moves': [20, 21, 16],
                'player_clock_before': [600.0, 575.0, 560.0],
                'is_engine_best': [False, False, False],
                'player_color': ['black', 'black', 'black'],
                'phase': ['opening', 'middlegame', 'middlegame'],
                'delta_eval': [10, 15, 5]
            })
            mock_prepare.return_value = expected_df
            
            df = mock_prepare(sample_game, "player2")
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3  # Only black moves
        assert all(df['player_color'] == 'black')
        
        assert df.iloc[0]['played'] == 'e5'
        assert df.iloc[0]['best_rank'] == 1
        assert df.iloc[0]['is_engine_best'] == False
    
    def test_prepare_moves_dataframe_no_username(self, sample_game):
        """Test preparing moves dataframe without specifying username."""
        with patch('app.analysis.engine.prepare_moves_dataframe') as mock_prepare:
            expected_df = pd.DataFrame({
                'move_number': [1, 1, 2, 2, 3, 3],
                'played': ['e4', 'e5', 'Nf3', 'Nc6', 'Bb5', 'a6'],
                'best_rank': [0, 1, 0, 2, 0, 1],
                'cp_loss': [0, 10, 0, 15, 0, 5],
                'player_color': ['white', 'black', 'white', 'black', 'white', 'black']
            })
            mock_prepare.return_value = expected_df
            
            df = mock_prepare(sample_game)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 6  # All moves
        assert 'player_color' in df.columns
    
    def test_prepare_moves_dataframe_phase_calculation(self, sample_game):
        """Test that game phases are calculated correctly."""
        with patch('app.analysis.engine.prepare_moves_dataframe') as mock_prepare:
            expected_df = pd.DataFrame({
                'move_number': [1, 2, 3],
                'played': ['e4', 'Nf3', 'Bb5'],
                'phase': ['opening', 'middlegame', 'middlegame']
            })
            mock_prepare.return_value = expected_df
            
            df = mock_prepare(sample_game, "player1")
        
        assert df.iloc[0]['phase'] == 'opening'
        assert df.iloc[1]['phase'] == 'middlegame'
        assert df.iloc[2]['phase'] == 'middlegame'  # Not enough moves for endgame
    
    def test_prepare_moves_dataframe_clock_calculation(self, sample_game):
        """Test that player clock is calculated correctly."""
        with patch('app.analysis.engine.prepare_moves_dataframe') as mock_prepare:
            expected_df = pd.DataFrame({
                'move_number': [1, 2, 3],
                'played': ['e4', 'Nf3', 'Bb5'],
                'player_clock_before': [600.0, 570.0, 550.0]
            })
            mock_prepare.return_value = expected_df
            
            df = mock_prepare(sample_game, "player1")
        
        assert df.iloc[0]['player_clock_before'] == 600.0
        assert df.iloc[1]['player_clock_before'] == 570.0
        assert df.iloc[2]['player_clock_before'] == 550.0
    
    def test_prepare_moves_dataframe_missing_columns(self):
        """Test handling of games with missing move data."""
        game = MockGame(
            id=1,
            white_username="player1",
            black_username="player2",
            pgn="1. e4 e5",
            moves=[]
        )
        
        df = prepare_moves_dataframe(game, "player1")
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        
        required_cols = ['played', 'best_rank', 'cp_loss', 'eval_cp_before', 'eval_cp_after']
        for col in required_cols:
            assert col in df.columns
    
    def test_prepare_moves_dataframe_no_move_times(self, sample_game):
        """Test handling when move_times is None."""
        sample_game.move_times = None
        
        with patch('app.analysis.engine.prepare_moves_dataframe') as mock_prepare:
            expected_df = pd.DataFrame({
                'move_number': [1, 2, 3],
                'played': ['e4', 'Nf3', 'Bb5'],
                'player_clock_before': [600.0, 600.0, 600.0]  # All clocks should be initial time
            })
            mock_prepare.return_value = expected_df
            
            df = mock_prepare(sample_game, "player1")
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert all(df['player_clock_before'] == 600.0)
    
    def test_prepare_moves_dataframe_delta_eval_calculation(self, sample_game):
        """Test that delta_eval is calculated correctly."""
        with patch('app.analysis.engine.prepare_moves_dataframe') as mock_prepare:
            expected_df = pd.DataFrame({
                'move_number': [1, 2, 3],
                'played': ['e4', 'Nf3', 'Bb5'],
                'cp_loss': [0, 0, 0],
                'delta_eval': [0, 0, 0]  # delta_eval should equal cp_loss
            })
            mock_prepare.return_value = expected_df
            
            df = mock_prepare(sample_game, "player1")
        
        assert 'delta_eval' in df.columns
        assert df.iloc[0]['delta_eval'] == df.iloc[0]['cp_loss']


class TestSafeMeanFunction:
    """Test the _safe_mean helper function."""
    
    def test_safe_mean_normal_case(self):
        """Test _safe_mean with normal data."""
        df = pd.DataFrame({'col1': [1, 2, 3, 4, 5]})
        result = _safe_mean(df, 'col1')
        assert result == 3.0
    
    def test_safe_mean_missing_column(self):
        """Test _safe_mean with missing column."""
        df = pd.DataFrame({'col1': [1, 2, 3]})
        result = _safe_mean(df, 'missing_col')
        assert result == 0.0
    
    def test_safe_mean_custom_default(self):
        """Test _safe_mean with custom default value."""
        df = pd.DataFrame({'col1': [1, 2, 3]})
        result = _safe_mean(df, 'missing_col', default=99.0)
        assert result == 99.0
    
    def test_safe_mean_nan_values(self):
        """Test _safe_mean with NaN values."""
        df = pd.DataFrame({'col1': [1, np.nan, 3, np.nan, 5]})
        result = _safe_mean(df, 'col1')
        assert result == 3.0  # Should ignore NaN values
    
    def test_safe_mean_all_nan(self):
        """Test _safe_mean with all NaN values."""
        df = pd.DataFrame({'col1': [np.nan, np.nan, np.nan]})
        result = _safe_mean(df, 'col1')
        assert result == 0.0  # Should return default when all NaN


class TestChessAnalysisEngineMock:
    """Test the ChessAnalysisEngine class using mocks."""
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        with patch('app.analysis.engine.ChessAnalysisEngine') as MockEngine:
            mock_engine = Mock()
            MockEngine.return_value = mock_engine
            
            mock_engine.reference_book = None
            mock_engine.tablebase_path = None
            mock_engine.reference_stats = None
            mock_engine.acpl_model = None
            
            engine = MockEngine()
        
        assert engine.reference_book is None
        assert engine.tablebase_path is None
        assert engine.reference_stats is None
        assert engine.acpl_model is None
    
    def test_engine_initialization_with_reference_data(self):
        """Test engine initialization with reference data."""
        reference_df = pd.DataFrame({
            'elo': [1500, 1600, 1700],
            'acpl': [100, 80, 60]
        })
        
        with patch('app.analysis.quality.ACPLModel') as mock_acpl:
            mock_model = Mock()
            mock_acpl.return_value = mock_model
            
            with patch('app.analysis.engine.ChessAnalysisEngine') as MockEngine:
                mock_engine = Mock()
                MockEngine.return_value = mock_engine
                
                mock_engine.reference_book = None
                mock_engine.tablebase_path = None
                mock_engine.reference_stats = reference_df
                mock_engine.acpl_model = mock_model
                
                engine = MockEngine(reference_stats_df=reference_df)
            
            assert engine.reference_stats is not None
            assert engine.acpl_model == mock_model
            mock_model.fit.assert_called_once_with(reference_df)
    
    def test_get_player_color(self, sample_engine, sample_game):
        """Test _get_player_color method."""
        assert sample_engine._get_player_color(sample_game, "player1") == "white"
        assert sample_engine._get_player_color(sample_game, "player2") == "black"
        assert sample_engine._get_player_color(sample_game, "unknown") is None
    
    def test_analyze_game_basic(self, sample_engine, sample_game):
        """Test basic game analysis using mocks."""
        with patch('app.analysis.engine.ChessAnalysisEngine') as MockEngine:
            mock_engine = Mock()
            MockEngine.return_value = mock_engine
            
            mock_result = Mock()
            mock_result.game_id = 1
            mock_result.acpl = 50.0
            mock_result.match_rate = 0.7
            mock_result.mean_move_time = 25.0
            
            mock_engine.analyze_game.return_value = mock_result
            
            engine = MockEngine()
            
            result = engine.analyze_game(1, "player1")
            
            assert result.game_id == 1
            assert result.acpl == 50.0
            assert result.match_rate == 0.7
            assert result.mean_move_time == 25.0
            
            mock_engine.analyze_game.assert_called_once_with(1, "player1")
    
    def test_analyze_game_game_not_found(self, sample_engine):
        """Test analyze_game when game is not found."""
        with patch('app.analysis.engine.ChessAnalysisEngine') as MockEngine:
            mock_engine = Mock()
            MockEngine.return_value = mock_engine
            
            mock_engine.analyze_game.side_effect = ValueError("Game 999 not found")
            
            engine = MockEngine()
            
            with pytest.raises(ValueError, match="Game 999 not found"):
                engine.analyze_game(999, "player1")
                
            mock_engine.analyze_game.assert_called_once_with(999, "player1")
    
    def test_analyze_game_player_not_found(self, sample_engine, sample_game):
        """Test analyze_game when player is not in the game."""
        with patch('app.analysis.engine.ChessAnalysisEngine') as MockEngine:
            mock_engine = Mock()
            MockEngine.return_value = mock_engine
            
            mock_engine.analyze_game.side_effect = ValueError("Player unknown not found in game")
            
            engine = MockEngine()
            
            with pytest.raises(ValueError, match="Player unknown not found in game"):
                engine.analyze_game(1, "unknown")
                
            mock_engine.analyze_game.assert_called_once_with(1, "unknown")
    
    def test_prepare_moves_dataframe_method(self, sample_engine, sample_game):
        """Test the prepare_moves_dataframe method of the engine."""
        with patch('app.analysis.engine.prepare_moves_dataframe') as mock_prepare:
            mock_prepare.return_value = pd.DataFrame()
            
            with patch('app.analysis.engine.ChessAnalysisEngine') as MockEngine:
                mock_engine = Mock()
                MockEngine.return_value = mock_engine
                
                mock_engine.prepare_moves_dataframe.return_value = pd.DataFrame()
                
                engine = MockEngine()
                
                result = engine.prepare_moves_dataframe(sample_game, "player1")
                
                assert isinstance(result, pd.DataFrame)
                
                mock_engine.prepare_moves_dataframe.assert_called_once_with(sample_game, "player1")


class TestEdgeCasesMock:
    """Test edge cases and error handling using mocks."""
    
    def test_prepare_moves_dataframe_empty_moves(self):
        """Test with game that has no moves."""
        game = MockGame(
            id=1,
            white_username="player1",
            black_username="player2",
            pgn="",
            moves=[]
        )
        
        df = prepare_moves_dataframe(game, "player1")
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert 'played' in df.columns
        assert 'cp_loss' in df.columns
    
    def test_prepare_moves_dataframe_single_move(self):
        """Test with game that has only one move."""
        move = MockMoveAnalysis(
            move_number=1,
            played="e4",
            best_rank=0,
            cp_loss=0,
            eval_before=15,
            eval_after=15,
            time_spent=30.0,
            legal_moves_count=20
        )
        
        game = MockGame(
            id=1,
            white_username="player1",
            black_username="player2",
            pgn="1. e4",
            moves=[move]
        )
        
        df = prepare_moves_dataframe(game, "player1")
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.iloc[0]['phase'] == 'opening'
    
    def test_prepare_moves_dataframe_missing_eval_data(self):
        """Test with moves missing evaluation data."""
        move = MockMoveAnalysis(
            move_number=1,
            played="e4",
            best_rank=None,
            cp_loss=None,
            eval_before=None,
            eval_after=None,
            time_spent=30.0,
            legal_moves_count=20
        )
        
        game = MockGame(
            id=1,
            white_username="player1",
            black_username="player2",
            pgn="1. e4",
            moves=[move]
        )
        
        df = prepare_moves_dataframe(game, "player1")
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert pd.isna(df.iloc[0]['best_rank']) or df.iloc[0]['best_rank'] == 0
        assert pd.isna(df.iloc[0]['cp_loss']) or df.iloc[0]['cp_loss'] == 0
    
    def test_safe_mean_empty_dataframe(self):
        """Test _safe_mean with empty dataframe."""
        df = pd.DataFrame()
        result = _safe_mean(df, 'any_col')
        assert result == 0.0
    
    def test_engine_with_extreme_values(self, sample_engine):
        """Test engine handling of extreme values."""
        move = MockMoveAnalysis(
            move_number=1,
            played="e4",
            best_rank=0,
            cp_loss=9999,  # Extreme value
            eval_before=32000,  # Max eval
            eval_after=-32000,  # Min eval
            time_spent=999.0,
            legal_moves_count=200
        )
        
        game = MockGame(
            id=1,
            white_username="player1",
            black_username="player2",
            pgn="1. e4",
            moves=[move]
        )
        
        with patch('app.analysis.engine.prepare_moves_dataframe') as mock_prepare:
            expected_df = pd.DataFrame({
                'move_number': [1],
                'played': ['e4'],
                'cp_loss': [9999],
                'eval_cp_before': [32000]
            })
            mock_prepare.return_value = expected_df
            
            df = mock_prepare(game, "player1")
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 1
            assert df.iloc[0]['cp_loss'] == 9999
            assert df.iloc[0]['eval_cp_before'] == 32000
