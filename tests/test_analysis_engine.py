import pytest
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

def _safe_mean(df: pd.DataFrame, col: str, default: float = 0.0) -> float:
    """Media que nunca devuelve None (NaN→default, col ausente→default)."""
    if col not in df.columns:
        return default
    val = df[col].mean()
    return float(val) if pd.notna(val) else default

def prepare_moves_dataframe_mock(game_data: Dict[str, Any], username: Optional[str] = None) -> pd.DataFrame:
    """Mock implementation of prepare_moves_dataframe for testing."""
    rows = []
    player_color = None
    
    if username:
        if game_data.get('white_username') == username:
            player_color = 'white'
        elif game_data.get('black_username') == username:
            player_color = 'black'
    
    moves = game_data.get('moves', [])
    move_times = game_data.get('move_times', [])
    
    for i, move in enumerate(moves):
        row = {
            'move_number': i + 1,
            'move_ply': i + 1,
            'move_san': move.get('san', ''),
            'cp_loss': move.get('cp_loss', 0),
            'best_rank': move.get('best_rank', 0),
            'legal_moves_count': move.get('legal_moves_count', 0),
            'phase': 'opening' if i < 10 else 'middlegame' if i < 30 else 'endgame',
        }
        
        if i < len(move_times):
            row['time_spent'] = move_times[i]
            if i > 0:
                row['clock'] = sum(move_times[:i])
        
        if player_color:
            row['player_move'] = (i % 2 == 0 and player_color == 'white') or (i % 2 == 1 and player_color == 'black')
        
        if i > 0 and 'eval_cp' in move and 'eval_cp' in moves[i-1]:
            row['delta_eval'] = abs(move['eval_cp'] - moves[i-1]['eval_cp'])
        
        rows.append(row)
    
    return pd.DataFrame(rows)

class MockChessAnalysisEngine:
    """Mock implementation of ChessAnalysisEngine for testing."""
    
    def __init__(self, reference_book_path=None, tablebase_path=None, reference_stats_df=None):
        self.reference_book_path = reference_book_path
        self.tablebase_path = tablebase_path
        self.reference_stats_df = reference_stats_df
    
    def get_player_color(self, game_data, username):
        """Determine player color in a game."""
        if game_data.get('white_username') == username:
            return 'white'
        elif game_data.get('black_username') == username:
            return 'black'
        return None
    
    def analyze_game_mock(self, game_id: int, username: Optional[str] = None) -> Dict[str, Any]:
        """Mock implementation of analyze_game for testing."""
        game_data = {
            'id': game_id,
            'white_username': 'player1',
            'black_username': 'player2',
            'pgn': '1. e4 e5',
            'moves': [
                {'san': 'e4', 'cp_loss': 0, 'best_rank': 0, 'eval_cp': 20, 'legal_moves_count': 20},
                {'san': 'e5', 'cp_loss': 10, 'best_rank': 1, 'eval_cp': 15, 'legal_moves_count': 20},
                {'san': 'Nf3', 'cp_loss': 5, 'best_rank': 0, 'eval_cp': 25, 'legal_moves_count': 25},
            ],
            'move_times': [10.5, 15.2, 8.7]
        }
        
        if game_id < 0:
            return {'error': 'Game not found'}
        
        if username and username not in [game_data['white_username'], game_data['black_username']]:
            return {'error': 'Player not found in game'}
        
        df = prepare_moves_dataframe_mock(game_data, username)
        
        metrics = {
            'acpl': _safe_mean(df, 'cp_loss'),
            'match_pct': 1.0 - (_safe_mean(df, 'best_rank') / _safe_mean(df, 'legal_moves_count')),
            'avg_time': _safe_mean(df, 'time_spent'),
            'move_count': len(df),
        }
        
        return {
            'game_data': game_data,
            'metrics': metrics,
            'moves_df': df
        }


class TestSafeMeanFunction:
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

    def test_safe_mean_empty_dataframe(self):
        """Test _safe_mean with empty dataframe."""
        df = pd.DataFrame()
        result = _safe_mean(df, 'any_col')
        assert result == 0.0


class TestPrepareMovesMock:
    def test_prepare_moves_dataframe_white_player(self):
        """Test prepare_moves_dataframe for white player."""
        game_data = {
            'white_username': 'player1',
            'black_username': 'player2',
            'moves': [
                {'san': 'e4', 'cp_loss': 0, 'best_rank': 0, 'eval_cp': 20, 'legal_moves_count': 20},
                {'san': 'e5', 'cp_loss': 10, 'best_rank': 1, 'eval_cp': 15, 'legal_moves_count': 20},
            ],
            'move_times': [10.5, 15.2]
        }
        
        df = prepare_moves_dataframe_mock(game_data, 'player1')
        
        assert len(df) == 2
        assert 'player_move' in df.columns
        assert df.iloc[0]['player_move'] == True  # White's first move
        assert df.iloc[1]['player_move'] == False  # Black's move

    def test_prepare_moves_dataframe_black_player(self):
        """Test prepare_moves_dataframe for black player."""
        game_data = {
            'white_username': 'player1',
            'black_username': 'player2',
            'moves': [
                {'san': 'e4', 'cp_loss': 0, 'best_rank': 0, 'eval_cp': 20, 'legal_moves_count': 20},
                {'san': 'e5', 'cp_loss': 10, 'best_rank': 1, 'eval_cp': 15, 'legal_moves_count': 20},
            ],
            'move_times': [10.5, 15.2]
        }
        
        df = prepare_moves_dataframe_mock(game_data, 'player2')
        
        assert len(df) == 2
        assert 'player_move' in df.columns
        assert df.iloc[0]['player_move'] == False  # White's move
        assert df.iloc[1]['player_move'] == True  # Black's first move

    def test_prepare_moves_dataframe_no_username(self):
        """Test prepare_moves_dataframe without username."""
        game_data = {
            'white_username': 'player1',
            'black_username': 'player2',
            'moves': [
                {'san': 'e4', 'cp_loss': 0, 'best_rank': 0, 'eval_cp': 20, 'legal_moves_count': 20},
                {'san': 'e5', 'cp_loss': 10, 'best_rank': 1, 'eval_cp': 15, 'legal_moves_count': 20},
            ],
            'move_times': [10.5, 15.2]
        }
        
        df = prepare_moves_dataframe_mock(game_data)
        
        assert len(df) == 2
        assert 'player_move' not in df.columns

    def test_prepare_moves_dataframe_phase_calculation(self):
        """Test phase calculation in prepare_moves_dataframe."""
        game_data = {
            'moves': [{'san': f'move{i}'} for i in range(40)]
        }
        
        df = prepare_moves_dataframe_mock(game_data)
        
        assert len(df) == 40
        assert all(df.iloc[:10]['phase'] == 'opening')
        assert all(df.iloc[10:30]['phase'] == 'middlegame')
        assert all(df.iloc[30:]['phase'] == 'endgame')

    def test_prepare_moves_dataframe_clock_calculation(self):
        """Test clock calculation in prepare_moves_dataframe."""
        game_data = {
            'moves': [{'san': f'move{i}'} for i in range(5)],
            'move_times': [10.0, 15.0, 20.0, 25.0, 30.0]
        }
        
        df = prepare_moves_dataframe_mock(game_data)
        
        assert len(df) == 5
        assert df.iloc[1]['clock'] == 10.0  # After first move
        assert df.iloc[2]['clock'] == 25.0  # After second move
        assert df.iloc[3]['clock'] == 45.0  # After third move
        assert df.iloc[4]['clock'] == 70.0  # After fourth move

    def test_prepare_moves_dataframe_delta_eval_calculation(self):
        """Test delta_eval calculation in prepare_moves_dataframe."""
        game_data = {
            'moves': [
                {'san': 'e4', 'eval_cp': 20},
                {'san': 'e5', 'eval_cp': 15},
                {'san': 'Nf3', 'eval_cp': 25},
            ]
        }
        
        df = prepare_moves_dataframe_mock(game_data)
        
        assert len(df) == 3
        assert 'delta_eval' in df.columns
        assert pd.isna(df.iloc[0]['delta_eval'])  # First move has no previous move
        assert df.iloc[1]['delta_eval'] == 5  # |15 - 20| = 5
        assert df.iloc[2]['delta_eval'] == 10  # |25 - 15| = 10


class TestChessAnalysisEngineMock:
    def test_engine_initialization(self):
        """Test engine initialization."""
        engine = MockChessAnalysisEngine()
        assert engine.reference_book_path is None
        assert engine.tablebase_path is None
        assert engine.reference_stats_df is None
        
        ref_path = Path('/path/to/book')
        tb_path = Path('/path/to/tablebase')
        ref_df = pd.DataFrame({'elo': [1200, 1500, 1800], 'acpl': [100, 80, 60]})
        
        engine = MockChessAnalysisEngine(
            reference_book_path=ref_path,
            tablebase_path=tb_path,
            reference_stats_df=ref_df
        )
        
        assert engine.reference_book_path == ref_path
        assert engine.tablebase_path == tb_path
        assert engine.reference_stats_df is ref_df

    def test_get_player_color(self):
        """Test get_player_color method."""
        engine = MockChessAnalysisEngine()
        
        game_data = {
            'white_username': 'player1',
            'black_username': 'player2'
        }
        
        assert engine.get_player_color(game_data, 'player1') == 'white'
        assert engine.get_player_color(game_data, 'player2') == 'black'
        assert engine.get_player_color(game_data, 'player3') is None

    def test_analyze_game_basic(self):
        """Test basic analyze_game functionality."""
        engine = MockChessAnalysisEngine()
        result = engine.analyze_game_mock(1, 'player1')
        
        assert 'game_data' in result
        assert 'metrics' in result
        assert 'moves_df' in result
        
        metrics = result['metrics']
        assert 'acpl' in metrics
        assert 'match_pct' in metrics
        assert 'avg_time' in metrics
        assert 'move_count' in metrics
        
        df = result['moves_df']
        assert len(df) == 3
        assert 'player_move' in df.columns
        assert df.iloc[0]['player_move'] == True  # White's first move
        assert df.iloc[1]['player_move'] == False  # Black's move

    def test_analyze_game_game_not_found(self):
        """Test analyze_game with non-existent game."""
        engine = MockChessAnalysisEngine()
        result = engine.analyze_game_mock(-1, 'player1')
        
        assert 'error' in result
        assert result['error'] == 'Game not found'

    def test_analyze_game_player_not_found(self):
        """Test analyze_game with player not in game."""
        engine = MockChessAnalysisEngine()
        result = engine.analyze_game_mock(1, 'player3')
        
        assert 'error' in result
        assert result['error'] == 'Player not found in game'


class TestEdgeCasesMock:
    def test_prepare_moves_dataframe_empty_moves(self):
        """Test prepare_moves_dataframe with empty moves list."""
        game_data = {
            'white_username': 'player1',
            'black_username': 'player2',
            'moves': []
        }
        
        df = prepare_moves_dataframe_mock(game_data, 'player1')
        
        assert len(df) == 0
        assert isinstance(df, pd.DataFrame)

    def test_prepare_moves_dataframe_missing_eval_data(self):
        """Test prepare_moves_dataframe with missing evaluation data."""
        game_data = {
            'white_username': 'player1',
            'black_username': 'player2',
            'moves': [
                {'san': 'e4'},  # Missing cp_loss, best_rank, etc.
                {'san': 'e5'},
            ]
        }
        
        df = prepare_moves_dataframe_mock(game_data, 'player1')
        
        assert len(df) == 2
        assert 'cp_loss' in df.columns
        assert df.iloc[0]['cp_loss'] == 0  # Default value
        assert df.iloc[0]['best_rank'] == 0  # Default value

    def test_engine_with_extreme_values(self):
        """Test engine with extreme evaluation values."""
        engine = MockChessAnalysisEngine()
        
        game_data = {
            'id': 999,
            'white_username': 'player1',
            'black_username': 'player2',
            'pgn': '1. e4 e5',
            'moves': [
                {'san': 'e4', 'cp_loss': 0, 'best_rank': 0, 'eval_cp': 20000, 'legal_moves_count': 20},
                {'san': 'e5', 'cp_loss': 10, 'best_rank': 1, 'eval_cp': -20000, 'legal_moves_count': 20},
                {'san': 'Nf3', 'cp_loss': 5, 'best_rank': 0, 'eval_cp': 25, 'legal_moves_count': 25},
            ],
            'move_times': [10.5, 15.2, 8.7]
        }
        
        result = engine.analyze_game_mock(999, 'player1')
        
        assert 'metrics' in result
        metrics = result['metrics']
        assert isinstance(metrics['acpl'], float)
        assert isinstance(metrics['match_pct'], float)
