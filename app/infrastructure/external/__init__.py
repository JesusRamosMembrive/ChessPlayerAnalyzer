"""
External integrations - Chess.com API, Stockfish engine, etc.
"""
from .stockfish_engine import StockfishEngine, EngineAnalysisResult, MoveAnalysis

__all__ = [
    "StockfishEngine",
    "EngineAnalysisResult",
    "MoveAnalysis"
]