# app/analysis/player_analyzer.py
"""
Player longitudinal analysis orchestration.
Extracted from engine.py as part of Refactor Fase 2 - Estructural.
"""
from __future__ import annotations
import logging
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timezone

import pandas as pd

# Core imports
from app.models import Game, PlayerAnalysisDetailed
from app.database import engine as db_engine
from sqlmodel import Session

# Analysis modules
from . import timing, longitudinal
from .longitudinal import roi_per_game, compute_trends
from .openings import aggregate_player_opening_patterns
from .timing import aggregate_time_management, aggregate_time_complexity_corr

# Data provider
from .data_provider import AnalysisDataProvider

# Utils
from app.utils_debugging.tracer import trace
from app.utils import clean_json_numbers

logger = logging.getLogger(__name__)


class PlayerAnalyzer:
    """
    Player longitudinal analysis orchestration.
    Handles the complete analysis workflow for player performance over time.
    """

    def __init__(self,
                 data_provider: AnalysisDataProvider,
                 reference_stats: Optional[pd.DataFrame] = None):
        """
        Args:
            data_provider: Data access layer
            reference_stats: DataFrame con estadísticas de referencia por ELO
        """
        self.data_provider = data_provider
        self.reference_stats = reference_stats

    @trace
    def analyze_player(self, username: str) -> PlayerAnalysisDetailed:
        """
        Analiza todas las partidas de un jugador y genera métricas agregadas.
        Extraído de engine.py líneas 375-632.

        Args:
            username: Nombre del jugador

        Returns:
            PlayerAnalysisDetailed con métricas longitudinales
        """
        logger.info(f"DEBUG PLAYER_ANALYZER: Starting analyze_player for {username}")

        with Session(db_engine) as session:
            # ── 1. Recuperar todas las partidas + sus análisis ───────────────
            games_df = self.data_provider.get_player_games_with_analysis(username)

            if not games_df.empty:
                # Agregar columna ROI
                logger.info(f"DEBUG PLAYER_ANALYZER: Games DataFrame columns before ROI: {list(games_df.columns)}")
                games_df['roi'] = roi_per_game(games_df)
                logger.info(f"DEBUG PLAYER_ANALYZER: Added ROI column, shape: {games_df.shape}")

                # Preparar DataFrame para análisis de tendencias
                games_df_for_trends = self._prepare_trends_dataframe(games_df)

                # Calcular tendencias
                try:
                    logger.info(f"DEBUG PLAYER_ANALYZER: About to call compute_trends with DataFrame shape: {games_df_for_trends.shape}")
                    trend_feats = compute_trends(games_df_for_trends)
                    logger.info(f"DEBUG PLAYER_ANALYZER: compute_trends result: {trend_feats}")
                except Exception as e:
                    logger.error(f"DEBUG PLAYER_ANALYZER: compute_trends failed with error: {e}")
                    trend_feats = {}
            else:
                trend_feats = {}
                logger.info("DEBUG PLAYER_ANALYZER: Empty games_df, setting empty trend_feats")

            if games_df.empty:
                raise ValueError(f"No analyzed games found for {username}")

            # ── 2. DataFrames de movimientos de cada partida ─────────────
            moves_dfs = self._collect_moves_dataframes(games_df, username)
            logger.info(f"DEBUG PLAYER_ANALYZER: Collected {len(moves_dfs)} moves DataFrames")

            # ── 3. Análisis longitudinal completo ────────────────────────
            long_features = self._aggregate_longitudinal_features(games_df, moves_dfs)
            logger.info(f"DEBUG PLAYER_ANALYZER: Longitudinal features calculated")

            # ── 4. Cálculo de risk score ─────────────────────────────────
            risk_score, risk_factors = self._calculate_risk_score(games_df, long_features)
            logger.info(f"DEBUG PLAYER_ANALYZER: Risk score: {risk_score}")

            # ── 5. Construcción del resultado final ──────────────────────
            return self._build_player_analysis_result(
                username, games_df, long_features, trend_feats,
                risk_score, risk_factors
            )

    def _prepare_trends_dataframe(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Prepara el DataFrame para análisis de tendencias."""
        logger.info(f"DEBUG PLAYER_ANALYZER: Before column handling - columns: {list(games_df.columns)}")
        logger.info(f"DEBUG PLAYER_ANALYZER: 'date' in columns: {'date' in games_df.columns}")

        if 'date' in games_df.columns:
            logger.info("DEBUG PLAYER_ANALYZER: Dropping existing 'date' column and renaming 'created_at' to 'date'")
            games_df_for_trends = games_df.drop(columns=['date']).rename(columns={'created_at': 'date'})
        else:
            logger.info("DEBUG PLAYER_ANALYZER: No existing 'date' column, just renaming 'created_at' to 'date'")
            games_df_for_trends = games_df.rename(columns={'created_at': 'date'})

        logger.info(f"DEBUG PLAYER_ANALYZER: After column handling - columns: {list(games_df_for_trends.columns)}")

        # Verificar y eliminar columnas duplicadas
        if games_df_for_trends.columns.duplicated().any():
            logger.error(f"DEBUG PLAYER_ANALYZER: Found duplicate columns: {games_df_for_trends.columns[games_df_for_trends.columns.duplicated()].tolist()}")
            games_df_for_trends = games_df_for_trends.loc[:, ~games_df_for_trends.columns.duplicated()]
            logger.info(f"DEBUG PLAYER_ANALYZER: After removing duplicates - columns: {list(games_df_for_trends.columns)}")

        return games_df_for_trends

    def _collect_moves_dataframes(self, games_df: pd.DataFrame, username: str) -> List[pd.DataFrame]:
        """Recolecta DataFrames de movimientos para cada partida."""
        moves_dfs = []

        for gid in games_df["game_id"]:
            try:
                with Session(db_engine) as session:
                    game = session.get(Game, gid)
                    if game:
                        moves_df = self.data_provider.prepare_moves_dataframe(game, username)
                        moves_dfs.append(moves_df)
                    else:
                        logger.warning(f"DEBUG PLAYER_ANALYZER: Game {gid} not found")
            except Exception as e:
                logger.error(f"DEBUG PLAYER_ANALYZER: Error getting moves for game {gid}: {e}")
                continue

        logger.info(f"DEBUG PLAYER_ANALYZER: Collected {len(moves_dfs)} valid moves DataFrames from {len(games_df)} games")
        return moves_dfs

    def _aggregate_longitudinal_features(self, games_df: pd.DataFrame, moves_dfs: List[pd.DataFrame]) -> Dict:
        """
        Agrega características longitudinales del jugador.
        Extraído de engine.py líneas 450-630.
        """
        long_features = {}

        try:
            # 1. Características de apertura
            logger.info("DEBUG PLAYER_ANALYZER: Computing opening patterns")
            opening_feats = aggregate_player_opening_patterns(games_df, moves_dfs)
            long_features.update(opening_feats)
            logger.info(f"DEBUG PLAYER_ANALYZER: Opening features: {opening_feats}")

            # 2. Gestión temporal
            logger.info("DEBUG PLAYER_ANALYZER: Computing time management")
            time_mgmt = aggregate_time_management(moves_dfs)
            long_features.update(time_mgmt)
            logger.info(f"DEBUG PLAYER_ANALYZER: Time management features: {time_mgmt}")

            # 3. Correlación tiempo-complejidad
            logger.info("DEBUG PLAYER_ANALYZER: Computing time-complexity correlation")
            time_corr = aggregate_time_complexity_corr(games_df)
            long_features.update(time_corr)
            logger.info(f"DEBUG PLAYER_ANALYZER: Time correlation features: {time_corr}")

            # 4. Características longitudinales avanzadas
            logger.info("DEBUG PLAYER_ANALYZER: Computing advanced longitudinal features")
            longitudinal_feats = longitudinal.aggregate_longitudinal_features(
                games_df, self.reference_stats
            )
            long_features.update(longitudinal_feats)
            logger.info(f"DEBUG PLAYER_ANALYZER: Advanced longitudinal features computed")

            # 5. Estadísticas básicas
            self._add_basic_statistics(games_df, long_features)

            logger.info("DEBUG PLAYER_ANALYZER: All longitudinal features computed successfully")

        except Exception as e:
            logger.error(f"DEBUG PLAYER_ANALYZER: Error in longitudinal analysis: {e}")
            import traceback
            logger.error(f"DEBUG PLAYER_ANALYZER: Full traceback: {traceback.format_exc()}")

        return long_features

    def _add_basic_statistics(self, games_df: pd.DataFrame, features: Dict):
        """Agrega estadísticas básicas del jugador."""
        # Estadísticas básicas
        features['total_games'] = len(games_df)
        features['avg_acpl'] = self.data_provider.safe_mean(games_df, 'acpl', 50.0)
        features['avg_match_rate'] = self.data_provider.safe_mean(games_df, 'match_rate', 0.5)
        features['avg_suspicion_score'] = self.data_provider.safe_mean(games_df, 'overall_suspicion_score', 0.0)

        # Distribución temporal
        if 'created_at' in games_df.columns and not games_df.empty:
            date_range = (games_df['created_at'].max() - games_df['created_at'].min()).days
            features['analysis_period_days'] = date_range
            features['games_per_day'] = len(games_df) / max(date_range, 1)

    @trace
    def _calculate_risk_score(self, games_df: pd.DataFrame, long_features: Dict) -> tuple[float, Dict]:
        """
        Calcula score de riesgo basado en patrones longitudinales.
        Extraído de engine.py líneas 703-769.
        """
        risk_score = 0
        risk_factors = {}

        try:
            # Factor 1: ACPL demasiado bajo
            avg_acpl = self.data_provider.safe_mean(games_df, 'acpl', 50.0)
            logger.info(f"DEBUG PLAYER_ANALYZER: Average ACPL: {avg_acpl}")

            if avg_acpl < 15.0:  # ACPL excepcionalmente bajo
                risk_score += 30
                risk_factors['low_acpl'] = True
                logger.info("DEBUG PLAYER_ANALYZER: Risk factor added - low ACPL")

            # Factor 2: Match rate muy alto
            avg_match_rate = self.data_provider.safe_mean(games_df, 'match_rate', 0.5)
            logger.info(f"DEBUG PLAYER_ANALYZER: Average match rate: {avg_match_rate}")

            if avg_match_rate > 0.85:  # Match rate excepcionalmente alto
                risk_score += 25
                risk_factors['high_match_rate'] = True
                logger.info("DEBUG PLAYER_ANALYZER: Risk factor added - high match rate")

            # Factor 3: Patrones de mejora súbita
            step_features = long_features.get('step_match_pct_flag', False)
            if step_features:
                risk_score += 20
                risk_factors['sudden_improvement'] = True
                logger.info("DEBUG PLAYER_ANALYZER: Risk factor added - sudden improvement")

            # Factor 4: Racha larga de alto rendimiento
            longest_streak = long_features.get('longest_streak', 0)
            logger.info(f"DEBUG PLAYER_ANALYZER: Longest streak: {longest_streak}")

            if longest_streak >= 8:
                risk_score += 15
                risk_factors['long_streak'] = True
                logger.info("DEBUG PLAYER_ANALYZER: Risk factor added - long streak")

            # Factor 5: Timing anormal
            timing_corr = self.data_provider.safe_mean(games_df, 'time_complexity_corr', 0.0)
            logger.info(f"DEBUG PLAYER_ANALYZER: Time complexity correlation: {timing_corr}")

            if timing_corr < -0.3:  # Correlación negativa fuerte es sospechosa
                risk_score += 10
                risk_factors['abnormal_timing'] = True
                logger.info("DEBUG PLAYER_ANALYZER: Risk factor added - abnormal timing")

            # Normalizar risk score (0-100)
            risk_score = min(risk_score, 100)
            logger.info(f"DEBUG PLAYER_ANALYZER: Final risk score: {risk_score}")
            logger.info(f"DEBUG PLAYER_ANALYZER: Risk factors: {risk_factors}")

        except Exception as e:
            logger.error(f"DEBUG PLAYER_ANALYZER: Error calculating risk score: {e}")
            risk_score = 0

        return float(risk_score), risk_factors

    def _build_player_analysis_result(self, username: str, games_df: pd.DataFrame,
                                    long_features: Dict, trend_feats: Dict,
                                    risk_score: float, risk_factors: Dict) -> PlayerAnalysisDetailed:
        """Construye el resultado final del análisis del jugador."""
        # Clean numeric values
        all_features = {**long_features, **trend_feats}
        all_features = clean_json_numbers(all_features)

        return PlayerAnalysisDetailed(
            username=username,
            total_games=len(games_df),
            avg_acpl=self.data_provider.safe_mean(games_df, 'acpl', 50.0),
            avg_match_rate=self.data_provider.safe_mean(games_df, 'match_rate', 0.5),
            avg_suspicion_score=self.data_provider.safe_mean(games_df, 'overall_suspicion_score', 0.0),

            # Longitudinal features
            opening_entropy=long_features.get('mean_entropy', 0.0),
            novelty_depth=long_features.get('novelty_depth', 0),
            opening_breadth=long_features.get('opening_breadth', 0),

            # Time management
            mean_move_time=long_features.get('mean_move_time', 3.0),
            time_variance=long_features.get('time_variance', 1.0),
            time_complexity_corr=long_features.get('time_complexity_corr', 0.0),

            # Longitudinal patterns
            roi_mean=long_features.get('roi_mean', 0.0),
            longest_streak=long_features.get('longest_streak', 0),
            step_function_detected=long_features.get('step_match_pct_flag', False),

            # Trends
            trend_acpl=trend_feats.get('trend_acpl', 0.0),
            trend_match_rate=trend_feats.get('trend_match_rate', 0.0),
            roi_curve=trend_feats.get('roi_curve', []),

            # Risk assessment
            risk_score=risk_score,
            risk_factors=risk_factors,

            # Estimate rating
            estimated_rating=self.data_provider.estimate_player_elo(username),

            # Analysis metadata
            analyzed_at=datetime.now(timezone.utc),
            analysis_period_start=games_df['created_at'].min() if not games_df.empty else None,
            analysis_period_end=games_df['created_at'].max() if not games_df.empty else None,

            # All features as JSON
            all_features=all_features
        )