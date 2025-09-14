# app/analysis/engine_facade.py
"""
Main interface facade for the analysis engine.
Coordinates data provider, game analyzer, and player analyzer.
Created as part of Refactor Fase 2 - Estructural.
"""
from __future__ import annotations
import logging
from typing import Dict, Optional
from pathlib import Path

import pandas as pd
from sqlmodel import Session

# Core imports
from app.models import Game, GameAnalysisDetailed, PlayerAnalysisDetailed
from app.database import engine as db_engine

# Analysis components
from .data_provider import AnalysisDataProvider
from .game_analyzer import GameAnalyzer
from .player_analyzer import PlayerAnalyzer

# Utils
from app.utils_debugging.tracer import trace

logger = logging.getLogger(__name__)


class AnalysisEngineFacade:
    """
    Main facade for the chess analysis engine.
    Coordinates all analysis components and provides unified interface.
    """

    def __init__(self,
                 reference_book_path: Optional[Path] = None,
                 tablebase_path: Optional[Path] = None,
                 reference_stats: Optional[pd.DataFrame] = None):
        """
        Args:
            reference_book_path: Path to Polyglot opening book
            tablebase_path: Path to Syzygy tablebases
            reference_stats: DataFrame with reference statistics by ELO
        """
        self.reference_book_path = reference_book_path
        self.tablebase_path = tablebase_path
        self.reference_stats = reference_stats

    @trace
    def analyze_game(self, game_id: int, username: str) -> GameAnalysisDetailed:
        """
        Analyze a single game completely.

        Args:
            game_id: Game ID in database
            username: Username of player to analyze

        Returns:
            GameAnalysisDetailed with all metrics
        """
        logger.info(f"ENGINE_FACADE: Starting game analysis - game_id: {game_id}, username: {username}")

        with Session(db_engine) as session:
            # Initialize components
            data_provider = AnalysisDataProvider(session)
            game_analyzer = GameAnalyzer(
                data_provider=data_provider,
                reference_book_path=self.reference_book_path,
                tablebase_path=self.tablebase_path
            )

            # Analyze game
            result = game_analyzer.analyze_game(game_id, username)
            logger.info(f"ENGINE_FACADE: Game analysis completed - suspicion score: {result.overall_suspicion_score}")

            return result

    @trace
    def analyze_player(self, username: str) -> PlayerAnalysisDetailed:
        """
        Analyze all games of a player and generate longitudinal metrics.

        Args:
            username: Player username

        Returns:
            PlayerAnalysisDetailed with longitudinal metrics
        """
        logger.info(f"ENGINE_FACADE: Starting player analysis - username: {username}")

        with Session(db_engine) as session:
            # Initialize components
            data_provider = AnalysisDataProvider(session)
            player_analyzer = PlayerAnalyzer(
                data_provider=data_provider,
                reference_stats=self.reference_stats
            )

            # Analyze player
            result = player_analyzer.analyze_player(username)
            logger.info(f"ENGINE_FACADE: Player analysis completed - risk score: {result.risk_score}, total games: {result.games_analyzed}")

            return result

    @trace
    def batch_analyze_games(self, game_ids: list[int], username: str) -> list[GameAnalysisDetailed]:
        """
        Analyze multiple games for the same player.

        Args:
            game_ids: List of game IDs to analyze
            username: Username of player to analyze

        Returns:
            List of GameAnalysisDetailed results
        """
        logger.info(f"ENGINE_FACADE: Starting batch game analysis - {len(game_ids)} games for {username}")

        results = []
        with Session(db_engine) as session:
            # Initialize components once for efficiency
            data_provider = AnalysisDataProvider(session)
            game_analyzer = GameAnalyzer(
                data_provider=data_provider,
                reference_book_path=self.reference_book_path,
                tablebase_path=self.tablebase_path
            )

            # Analyze each game
            for game_id in game_ids:
                try:
                    result = game_analyzer.analyze_game(game_id, username)
                    results.append(result)
                    logger.info(f"ENGINE_FACADE: Completed game {game_id}")
                except Exception as e:
                    logger.error(f"ENGINE_FACADE: Failed to analyze game {game_id}: {e}")
                    continue

        logger.info(f"ENGINE_FACADE: Batch analysis completed - {len(results)}/{len(game_ids)} games successful")
        return results

    @trace
    def get_player_stats(self, username: str) -> Dict:
        """
        Get basic player statistics without full analysis.

        Args:
            username: Player username

        Returns:
            Dictionary with basic stats
        """
        with Session(db_engine) as session:
            data_provider = AnalysisDataProvider(session)

            # Get games with analysis
            games_df = data_provider.get_player_games_with_analysis(username)

            if games_df.empty:
                return {
                    "username": username,
                    "total_games": 0,
                    "avg_acpl": 0.0,
                    "avg_match_rate": 0.0,
                    "avg_suspicion_score": 0.0,
                    "estimated_rating": data_provider.estimate_player_elo(username)
                }

            return {
                "username": username,
                "total_games": len(games_df),
                "avg_acpl": data_provider.safe_mean(games_df, 'acpl', 50.0),
                "avg_match_rate": data_provider.safe_mean(games_df, 'match_rate', 0.5),
                "avg_suspicion_score": data_provider.safe_mean(games_df, 'overall_suspicion_score', 0.0),
                "estimated_rating": data_provider.estimate_player_elo(username),
                "analysis_period_start": games_df['created_at'].min() if not games_df.empty else None,
                "analysis_period_end": games_df['created_at'].max() if not games_df.empty else None
            }


# Convenience functions for backward compatibility
@trace
def analyze_game(game_id: int, username: str,
                reference_book_path: Optional[Path] = None,
                tablebase_path: Optional[Path] = None) -> GameAnalysisDetailed:
    """
    Backward compatibility function for single game analysis.
    """
    facade = AnalysisEngineFacade(
        reference_book_path=reference_book_path,
        tablebase_path=tablebase_path
    )
    return facade.analyze_game(game_id, username)


@trace
def analyze_player(username: str,
                  reference_stats: Optional[pd.DataFrame] = None) -> PlayerAnalysisDetailed:
    """
    Backward compatibility function for player analysis.
    """
    facade = AnalysisEngineFacade(reference_stats=reference_stats)
    return facade.analyze_player(username)