"""
Stockfish engine implementation for analysis.
Replaces direct Stockfish usage with clean interface.
"""
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

import chess
import chess.engine
import chess.pgn
from io import StringIO

from ...core.config import StockfishConfig

logger = logging.getLogger(__name__)


@dataclass
class MoveAnalysis:
    """Analysis result for a single move."""
    move: str
    evaluation: float
    is_best: bool
    centipawn_loss: float
    wdl_loss: float
    match_rate: float


@dataclass
class EngineAnalysisResult:
    """Complete analysis result from engine."""
    moves: List[MoveAnalysis]
    avg_centipawn_loss: float
    avg_wdl_loss: float
    avg_match_rate: float
    total_moves: int


class StockfishEngine:
    """
    Clean interface to Stockfish engine.
    Handles all low-level engine communication.
    """

    def __init__(self, config: StockfishConfig):
        self.config = config
        self._engine: Optional[chess.engine.SimpleEngine] = None

    def __enter__(self):
        """Context manager entry."""
        self._engine = chess.engine.SimpleEngine.popen_uci(self.config.path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._engine:
            self._engine.quit()
            self._engine = None

    async def analyze_moves(self, pgn_data: str, move_times: Optional[List[float]] = None) -> EngineAnalysisResult:
        """
        Analyze moves from PGN data.
        Returns comprehensive analysis with metrics.
        """
        if not self._engine:
            raise RuntimeError("Engine not initialized. Use as context manager.")

        try:
            # Parse PGN
            game = chess.pgn.read_game(StringIO(pgn_data))
            if not game:
                raise ValueError("Invalid PGN data")

            # Extract moves
            moves = self._extract_moves_from_game(game)
            if not moves:
                raise ValueError("No moves found in PGN")

            # Analyze each position
            move_analyses = []
            board = game.board()

            for i, move in enumerate(moves):
                try:
                    # Analyze position before move
                    info = self._engine.analyse(board, chess.engine.Limit(depth=self.config.depth))
                    position_eval = self._extract_evaluation(info)

                    # Make the move
                    board.push(move)

                    # Analyze position after move
                    post_info = self._engine.analyse(board, chess.engine.Limit(depth=self.config.depth))
                    post_eval = self._extract_evaluation(post_info)

                    # Calculate best move and metrics
                    best_move_info = self._engine.analyse(
                        board.copy().pop(),  # Position before move
                        chess.engine.Limit(depth=self.config.depth),
                        multipv=1
                    )
                    best_move = best_move_info.get("pv", [chess.Move.null()])[0]

                    # Calculate metrics
                    move_analysis = self._calculate_move_metrics(
                        move=move,
                        best_move=best_move,
                        position_eval=position_eval,
                        post_eval=post_eval,
                        board=board
                    )

                    move_analyses.append(move_analysis)

                except Exception as e:
                    logger.warning(f"Failed to analyze move {i}: {e}")
                    continue

            # Calculate aggregate metrics
            if not move_analyses:
                raise ValueError("No moves could be analyzed")

            avg_centipawn_loss = sum(m.centipawn_loss for m in move_analyses) / len(move_analyses)
            avg_wdl_loss = sum(m.wdl_loss for m in move_analyses) / len(move_analyses)
            avg_match_rate = sum(m.match_rate for m in move_analyses) / len(move_analyses)

            return EngineAnalysisResult(
                moves=move_analyses,
                avg_centipawn_loss=avg_centipawn_loss,
                avg_wdl_loss=avg_wdl_loss,
                avg_match_rate=avg_match_rate,
                total_moves=len(move_analyses)
            )

        except Exception as e:
            logger.error(f"Engine analysis failed: {e}")
            raise

    def _extract_moves_from_game(self, game: chess.pgn.Game) -> List[chess.Move]:
        """Extract move sequence from PGN game."""
        moves = []
        node = game
        while node.variations:
            move = node.variations[0].move
            moves.append(move)
            node = node.variations[0]
        return moves

    def _extract_evaluation(self, info: Dict) -> float:
        """Extract numerical evaluation from engine info."""
        score = info.get("score")
        if not score:
            return 0.0

        # Handle mate scores
        if score.is_mate():
            mate_in = score.mate()
            return 1000.0 if mate_in > 0 else -1000.0

        # Handle centipawn scores
        cp_score = score.relative.score(mate_score=1000)
        return cp_score / 100.0 if cp_score else 0.0

    def _calculate_move_metrics(
        self,
        move: chess.Move,
        best_move: chess.Move,
        position_eval: float,
        post_eval: float,
        board: chess.Board
    ) -> MoveAnalysis:
        """Calculate comprehensive metrics for a move."""
        is_best = move == best_move

        # Centipawn loss calculation
        eval_change = abs(post_eval - position_eval)
        centipawn_loss = eval_change * 100 if not is_best else 0.0

        # WDL (Win-Draw-Loss) loss approximation
        # Convert centipawn difference to probability loss
        wdl_loss = min(eval_change * 10, 100.0) if not is_best else 0.0

        # Match rate (binary: 1 if best move, 0 otherwise)
        match_rate = 1.0 if is_best else 0.0

        return MoveAnalysis(
            move=str(move),
            evaluation=post_eval,
            is_best=is_best,
            centipawn_loss=centipawn_loss,
            wdl_loss=wdl_loss,
            match_rate=match_rate
        )

    def test_engine(self) -> bool:
        """Test if engine is working correctly."""
        try:
            with chess.engine.SimpleEngine.popen_uci(self.config.path) as engine:
                board = chess.Board()
                info = engine.analyse(board, chess.engine.Limit(depth=1))
                return "score" in info
        except Exception as e:
            logger.error(f"Engine test failed: {e}")
            return False