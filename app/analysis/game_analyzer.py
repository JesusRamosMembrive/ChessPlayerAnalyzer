# app/analysis/game_analyzer.py
"""
Single game analysis orchestration.
Extracted from engine.py as part of Refactor Fase 2 - Estructural.
"""
from __future__ import annotations
import logging
import io
import numpy as np
from typing import Dict, Optional
from pathlib import Path

import pandas as pd
import chess.pgn

# Core imports
from app.models import Game, GameAnalysisDetailed
from app.database import engine as db_engine
from sqlmodel import Session

# Analysis modules
from . import quality, timing, openings, endgame, anomaly
from .clustering import assign_cluster_from_values
from .bayesian import BayesianSuspicionModel
from .ml_classifier import MLSuspicionClassifier

# Data provider
from .data_provider import AnalysisDataProvider

# Utils
from app.utils_debugging.tracer import trace
from app.utils import clean_json_numbers

logger = logging.getLogger(__name__)


class GameAnalyzer:
    """
    Single game analysis orchestration.
    Handles the complete analysis workflow for individual games.
    """

    def __init__(self,
                 data_provider: AnalysisDataProvider,
                 reference_book_path: Optional[Path] = None,
                 tablebase_path: Optional[Path] = None):
        """
        Args:
            data_provider: Data access layer
            reference_book_path: Ruta al libro de aperturas Polyglot
            tablebase_path: Ruta a las tablebases Syzygy
        """
        self.data_provider = data_provider
        self.reference_book = reference_book_path
        self.tablebase_path = tablebase_path

    @trace
    def analyze_game(self, game_id: int, username: str) -> GameAnalysisDetailed:
        """
        Analiza una partida completa con todos los módulos.
        Extraído de engine.py líneas 238-372.

        Args:
            game_id: ID de la partida en la BD
            username: Username del jugador a analizar

        Returns:
            GameAnalysisDetailed con todas las métricas
        """
        logger.info(f"DEBUG GAME_ANALYZER: Starting analyze_game for game_id: {game_id}")

        with Session(db_engine) as session:
            # Cargar partida y movimientos
            game = session.get(Game, game_id)
            if not game:
                raise ValueError(f"Game {game_id} not found")

            logger.info(f"DEBUG GAME_ANALYZER: Loaded game - white: {game.white_username}, black: {game.black_username}")
            logger.info(f"DEBUG GAME_ANALYZER: Game metadata - eco_code: {game.eco_code}, opening: {game.opening_key}")

            player_color = self.data_provider.get_player_color(game, username)
            if not player_color:
                raise ValueError(f"Player {username} not found in game {game_id}")

            # Preparar DataFrames para análisis
            moves_df = self.data_provider.prepare_moves_dataframe(game, username)
            logger.info(f"DEBUG GAME_ANALYZER: Prepared moves DataFrame - shape: {moves_df.shape}")
            logger.info(f"DEBUG GAME_ANALYZER: Moves DataFrame columns: {list(moves_df.columns)}")
            logger.info(f"DEBUG GAME_ANALYZER: Sample moves data:\n{moves_df.head(3).to_string()}")

            # Parsear PGN para análisis que lo requieren
            pgn_game = chess.pgn.read_game(io.StringIO(game.pgn))
            logger.info("DEBUG GAME_ANALYZER: Parsed PGN game successfully")

            # 1. MÉTRICAS DE CALIDAD
            quality_features = self._analyze_quality(moves_df, game, player_color)
            logger.info("DEBUG GAME_ANALYZER: Starting quality analysis")
            logger.info(f"DEBUG GAME_ANALYZER: Quality features: {quality_features}")

            # 2. MÉTRICAS DE TIEMPO
            logger.info("DEBUG GAME_ANALYZER: Starting timing analysis")
            timing_features = timing.aggregate_time_features(moves_df)
            logger.info(f"DEBUG GAME_ANALYZER: Timing features: {timing_features}")

            # Obtener todas las partidas del jugador para experiencia y análisis de apertura
            games_df = self.data_provider.get_player_games_df(username)
            logger.info(f"DEBUG GAME_ANALYZER: Player games DataFrame shape: {games_df.shape}")

            # 3. MÉTRICAS DE APERTURA (si es aplicable)
            logger.info("DEBUG GAME_ANALYZER: Starting opening analysis")
            opening_features = self._analyze_opening(game, games_df, moves_df)
            logger.info(f"DEBUG GAME_ANALYZER: Opening features: {opening_features}")

            # 4. MÉTRICAS DE FINAL (si es aplicable)
            endgame_features = {}
            if self.data_provider.has_endgame(moves_df):
                logger.info("DEBUG GAME_ANALYZER: Starting endgame analysis")
                endgame_features = self._analyze_endgame(moves_df, pgn_game)
                logger.info(f"DEBUG GAME_ANALYZER: Endgame features: {endgame_features}")

            # 5. DETECCIÓN DE ANOMALÍAS
            logger.info("DEBUG GAME_ANALYZER: Starting anomaly detection")
            anomaly_features = self._analyze_anomalies(moves_df, game, username)
            logger.info(f"DEBUG GAME_ANALYZER: Anomaly features: {anomaly_features}")

            # 6. AGREGACIÓN FINAL
            all_features = self._aggregate_features(
                quality_features, timing_features, opening_features,
                endgame_features, anomaly_features
            )
            logger.info(f"DEBUG GAME_ANALYZER: Aggregated features: {list(all_features.keys())}")

            # 7. CÁLCULO DE SUSPICION SCORE
            rating = self.data_provider.estimate_player_elo(username, game)
            experience = len(games_df)
            suspicion_score = self._calculate_suspicion_score(all_features, rating, experience)
            logger.info(f"DEBUG GAME_ANALYZER: Calculated suspicion score: {suspicion_score}")

            # 8. CONSTRUIR RESULTADO
            return self._build_game_analysis_result(
                game_id, username, player_color, all_features, suspicion_score
            )

    def _analyze_quality(self, moves_df: pd.DataFrame, game: Game, player_color: str) -> Dict:
        """
        Análisis de calidad de juego.
        Extraído de engine.py líneas 639-672.
        """
        features = {}

        try:
            # ACPL (Average Centipawn Loss)
            features['acpl'] = quality.acpl(moves_df, player_color)

            # Match rate
            features['match_rate'] = moves_df['is_engine_best'].mean()

            # Weighted match rate
            features['weighted_match_rate'] = quality.weighted_match_rate(moves_df)

            # Phase quality
            features['phase_quality'] = quality.compute_phase_quality(moves_df)

            # Clutch accuracy
            if 'player_clock_before' in moves_df.columns:
                features['clutch_accuracy'] = quality.aggregate_clutch_accuracy(moves_df)

            # Tactical trends
            features['tactical_trends'] = quality.aggregate_tactical_trends(moves_df)

            # Blunders by phase
            features['blunders_by_phase'] = quality.aggregate_blunders_by_phase(moves_df)

            logger.info("DEBUG GAME_ANALYZER: Quality analysis completed successfully")

        except Exception as e:
            logger.error(f"DEBUG GAME_ANALYZER: Error in quality analysis: {e}")
            features['acpl'] = np.nan
            features['match_rate'] = 0.0

        return features

    def _analyze_opening(self, game: Game, games_df: pd.DataFrame, moves_df: pd.DataFrame) -> Dict:
        """Análisis de apertura."""
        try:
            opening_features = openings.aggregate_opening_features(
                opening_key=game.opening_key or "",
                eco_code=game.eco_code,
                moves_df=moves_df,
                games_df=games_df
            )
            return opening_features
        except Exception as e:
            logger.error(f"DEBUG GAME_ANALYZER: Error in opening analysis: {e}")
            return {
                "opening_entropy": 0.0,
                "novelty_depth": 0,
                "second_choice_rate": 0.0,
                "opening_breadth": 0,
                "opening_score": 50.0
            }

    def _analyze_endgame(self, moves_df: pd.DataFrame, pgn_game: chess.pgn.Game) -> Dict:
        """Análisis de final."""
        try:
            if self.tablebase_path:
                endgame_features = endgame.aggregate_endgame_efficiency(
                    moves_df, pgn_game, self.tablebase_path
                )
            else:
                # Análisis básico sin tablebases
                endgame_features = {
                    "conversion_efficiency": 0.5,
                    "tb_match_rate": 0.0,
                    "dtz_deviation": 0.0
                }
            return endgame_features
        except Exception as e:
            logger.error(f"DEBUG GAME_ANALYZER: Error in endgame analysis: {e}")
            return {
                "conversion_efficiency": 0.5,
                "tb_match_rate": 0.0,
                "dtz_deviation": 0.0
            }

    def _analyze_anomalies(self, moves_df: pd.DataFrame, game: Game, username: str) -> Dict:
        """Detección de anomalías."""
        try:
            anomaly_features = anomaly.detect_game_anomalies(
                moves_df, game, username
            )
            return anomaly_features
        except Exception as e:
            logger.error(f"DEBUG GAME_ANALYZER: Error in anomaly analysis: {e}")
            return {
                "anomaly_score": 0.0,
                "anomaly_flags": [],
                "pattern_consistency": 0.5
            }

    def _aggregate_features(self, quality_features: Dict, timing_features: Dict,
                          opening_features: Dict, endgame_features: Dict,
                          anomaly_features: Dict) -> Dict:
        """Agrega todas las características en un solo diccionario."""
        all_features = {}
        all_features.update(quality_features)
        all_features.update(timing_features)
        all_features.update(opening_features)
        all_features.update(endgame_features)
        all_features.update(anomaly_features)

        # Clean numeric values
        all_features = clean_json_numbers(all_features)

        return all_features

    def _calculate_suspicion_score(self, features: Dict, rating: int, experience: int) -> float:
        """
        Calcula score de sospecha usando modelos Bayesiano y ML.
        Extraído de engine.py líneas 674-702.
        """
        try:
            # Modelo Bayesiano
            bayesian_model = BayesianSuspicionModel()
            bayesian_score = bayesian_model.calculate_suspicion(features, rating, experience)

            # Modelo ML
            ml_model = MLSuspicionClassifier()
            ml_score = ml_model.predict_suspicion(features, rating, experience)

            # Combinar ambos modelos (peso 70% Bayesiano, 30% ML)
            final_score = 0.7 * bayesian_score + 0.3 * ml_score

            # Asignar cluster
            cluster = assign_cluster_from_values(
                acpl=features.get('acpl', 50),
                match_rate=features.get('match_rate', 0.5),
                anomaly_score=features.get('anomaly_score', 0.0)
            )

            logger.info(f"DEBUG GAME_ANALYZER: Bayesian: {bayesian_score:.3f}, ML: {ml_score:.3f}, "
                       f"Final: {final_score:.3f}, Cluster: {cluster}")

            return float(final_score)

        except Exception as e:
            logger.error(f"DEBUG GAME_ANALYZER: Error calculating suspicion score: {e}")
            return 0.5  # Neutral score

    def _build_game_analysis_result(self, game_id: int, username: str, player_color: str,
                                  features: Dict, suspicion_score: float) -> GameAnalysisDetailed:
        """Construye el resultado final del análisis."""
        return GameAnalysisDetailed(
            game_id=game_id,
            username=username,
            player_color=player_color,
            acpl=features.get('acpl', np.nan),
            match_rate=features.get('match_rate', 0.0),
            weighted_match_rate=features.get('weighted_match_rate', 0.0),
            overall_suspicion_score=suspicion_score,

            # Timing features
            mean_move_time=features.get('mean_move_time', 0.0),
            time_variance=features.get('time_variance', 0.0),
            time_complexity_corr=features.get('time_complexity_corr', 0.0),

            # Opening features
            opening_entropy=features.get('opening_entropy', 0.0),
            novelty_depth=features.get('novelty_depth', 0),
            opening_score=features.get('opening_score', 50.0),

            # Endgame features
            conversion_efficiency=features.get('conversion_efficiency', 0.5),
            tb_match_rate=features.get('tb_match_rate', 0.0),
            dtz_deviation=features.get('dtz_deviation', 0.0),

            # Quality features by phase
            phase_quality=features.get('phase_quality', {}),
            clutch_accuracy_diff=features.get('clutch_accuracy', {}).get('diff', 0.0) if isinstance(features.get('clutch_accuracy'), dict) else 0.0,

            # Anomaly features
            anomaly_score=features.get('anomaly_score', 0.0),
            pattern_consistency=features.get('pattern_consistency', 0.5),

            # Estimate rating
            estimated_rating=self.data_provider.estimate_player_elo(username),

            # All features as JSON
            all_features=features
        )