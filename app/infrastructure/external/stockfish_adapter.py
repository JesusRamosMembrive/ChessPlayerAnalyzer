"""
Adapter for StockfishEngine to match domain service interface.
Bridges infrastructure and domain layers.
"""
from typing import List, Optional

from ...core.config import StockfishConfig
from .stockfish_engine import StockfishEngine, EngineAnalysisResult


class StockfishEngineAdapter:
    """
    Adapts StockfishEngine to match the ChessEngine protocol expected by AnalysisService.
    """

    def __init__(self, config: StockfishConfig):
        self.config = config

    async def analyze_moves(self, pgn_data: str, move_times: Optional[List[float]] = None) -> EngineAnalysisResult:
        """
        Analyze moves from PGN data.
        Matches the ChessEngine protocol interface.
        """
        with StockfishEngine(self.config) as engine:
            return await engine.analyze_moves(pgn_data, move_times)