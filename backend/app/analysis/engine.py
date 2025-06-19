# app/analysis/engine.py
"""
Motor principal de análisis que orquesta todos los módulos.
"""
from __future__ import annotations
import logging
from typing import Dict, List, Optional
import pandas as pd
import chess.pgn
import io
from pathlib import Path

from app.models import Game, MoveAnalysis, GameAnalysisDetailed, PlayerAnalysisDetailed
from app.database import engine as db_engine
from sqlmodel import Session, select

# Importar módulos de análisis
from . import quality
from . import timing
from . import openings
from . import endgame
from . import longitudinal
from app import models
from app.database import engine

import numpy as np

logger = logging.getLogger(__name__)

def prepare_moves_dataframe(game: Game) -> pd.DataFrame:
    """Versión libre de 'self' para reutilizar desde Celery u otros sitios."""
    moves_data = []
    for m in game.moves:
        moves_data.append({
            "move_number": m.move_number,
            "played": m.played,
            "best": m.best,
            "best_rank": m.best_rank,
            "cp_loss": m.cp_loss,
            "eval_cp_before": m.eval_before,
            "eval_cp_after":  m.eval_after,
            "move_time": game.move_times[m.move_number - 1] if game.move_times else None,
            "legal_moves": m.legal_moves_count,
        })
    return pd.DataFrame(moves_data)



class ChessAnalysisEngine:
    """Motor principal que coordina todos los análisis."""

    def __init__(self,
                 reference_book_path: Optional[Path] = None,
                 tablebase_path: Optional[Path] = None,
                 reference_stats_df: Optional[pd.DataFrame] = None):
        """
        Args:
            reference_book_path: Ruta al libro de aperturas Polyglot
            tablebase_path: Ruta a las tablebases Syzygy
            reference_stats_df: DataFrame con estadísticas de referencia por ELO
        """
        self.reference_book = reference_book_path
        self.tablebase_path = tablebase_path
        self.reference_stats = reference_stats_df
        self.acpl_model = None

        # Entrenar modelo ACPL si hay datos de referencia
        if reference_stats_df is not None and 'elo' in reference_stats_df.columns:
            self.acpl_model = quality.ACPLModel()
            self.acpl_model.fit(reference_stats_df)

    def analyze_game(self, game_id: int) -> GameAnalysisDetailed:
        """
        Analiza una partida completa con todos los módulos.

        Args:
            game_id: ID de la partida en la BD

        Returns:
            GameAnalysisDetailed con todas las métricas
        """
        logger.info(f"Iniciando análisis detallado de partida {game_id}")

        with Session(db_engine) as session:
            # Cargar partida y movimientos
            game = session.get(Game, game_id)
            if not game:
                raise ValueError(f"Game {game_id} not found")

            # Preparar DataFrames para análisis
            moves_df = self.prepare_moves_dataframe(game)

            # Parsear PGN para análisis que lo requieren
            pgn_game = chess.pgn.read_game(io.StringIO(game.pgn))

            # 1. MÉTRICAS DE CALIDAD
            quality_features = self._analyze_quality(moves_df, game)

            # 2. MÉTRICAS DE TIEMPO
            timing_features = timing.aggregate_time_features(moves_df)

            # 3. MÉTRICAS DE APERTURA (si es aplicable)
            opening_features = {}
            if self.reference_book:
                # Necesitamos múltiples partidas del jugador para entropía
                games_df = self._get_player_games_df(game.white_username, session)
                opening_features = openings.aggregate_opening_features(
                    games_df, moves_df, [pgn_game],
                    elo=self._estimate_player_elo(game.white_username),
                    book_path=self.reference_book
                )

            # 4. MÉTRICAS DE FINAL (si hay tablebases)
            endgame_features = {}
            if self.tablebase_path and self._has_endgame(moves_df):
                endgame_features = endgame.aggregate_endgame_features(
                    pgn_game, moves_df, self.tablebase_path
                )

            # 5. COMBINAR TODAS LAS FEATURES
            all_features = {
                **quality_features,
                **timing_features,
                **opening_features,
                **endgame_features
            }

            # 6. CALCULAR FLAGS DE SOSPECHA
            suspicious_flags = self._calculate_suspicious_flags(all_features)

            # 7. CREAR REGISTRO DE ANÁLISIS
            analysis = GameAnalysisDetailed(
                game_id=game_id,
                # Quality
                acpl=all_features.get('acpl', 0),
                match_rate=all_features.get('match_rate', 0),
                weighted_match_rate=all_features.get('match_weighted', 0),
                ipr=all_features.get('ipr', 0),
                ipr_z_score=all_features.get('ipr_z', 0),
                # Timing
                mean_move_time=all_features.get('mean_time', 0),
                time_variance=all_features.get('std_time', 0),
                time_complexity_corr=all_features.get('time_complexity_corr', 0),
                lag_spike_count=all_features.get('lag_spike_count', 0),
                uniformity_score=all_features.get('uniformity_score', 0),
                # Opening
                opening_entropy=all_features.get('H_opening', 0),
                novelty_depth=all_features.get('mean_tn_depth', 0),
                second_choice_rate=all_features.get('second_choice_pct', 0),
                # Endgame
                tb_match_rate=all_features.get('tb_match_pct'),
                conversion_efficiency=all_features.get('conversion_moves'),
                # Flags
                **suspicious_flags
            )

            session.add(analysis)
            session.commit()

            logger.info(f"Análisis detallado completado para partida {game_id}")
            return analysis

    def analyze_player(self, username: str) -> PlayerAnalysisDetailed:
        """
        Analiza todas las partidas de un jugador y genera métricas agregadas.

        Args:
            username: Nombre del jugador

        Returns:
            PlayerAnalysisDetailed con métricas longitudinales
        """
        logger.info(f"Iniciando análisis detallado de jugador {username}")

        with Session(db_engine) as session:
            # Obtener todas las partidas analizadas del jugador
            games_df = self._get_player_games_with_analysis(username, session)

            if games_df.empty:
                raise ValueError(f"No analyzed games found for {username}")

            # Análisis longitudinal
            long_features = longitudinal.aggregate_longitudinal_features(
                games_df, self.reference_stats
            )

            # Calcular risk score
            risk_score, risk_factors = self._calculate_risk_score(
                games_df, long_features
            )

            # Crear registro
            analysis = PlayerAnalysisDetailed(
                username=username,
                avg_acpl=games_df['acpl'].mean(),
                avg_match_rate=games_df['match_rate'].mean(),
                roi_mean=long_features.get('roi_mean', 0),
                roi_max=long_features.get('roi_max', 0),
                step_function_detected=long_features.get('step_acpl_flag', False),
                peer_delta_acpl=long_features.get('peer_acpl_delta', 0),
                longest_streak=long_features.get('longest_roi_streak', 0),
                risk_score=risk_score,
                risk_factors=risk_factors
            )

            session.add(analysis)
            session.commit()

            logger.info(f"Análisis detallado completado para jugador {username}")
            return analysis

    def prepare_moves_dataframe(self, game: Game) -> pd.DataFrame:
        """Convierte los movimientos de la BD a DataFrame para análisis."""
        return prepare_moves_dataframe(game)  # Llamada libre para reutilizar


    def _analyze_quality(self, moves_df: pd.DataFrame, game: Game) -> Dict:
        """Ejecuta análisis de calidad."""
        features = {}

        # ACPL básico
        features['acpl'] = quality.acpl(moves_df)

        # Match rate
        features['match_rate'] = moves_df['is_engine_best'].mean()

        # Weighted match rate
        features['match_weighted'] = quality.complexity_weighted_match(moves_df)

        # IPR
        features['ipr'] = quality.intrinsic_performance_rating(
            features['match_rate'], features['acpl']
        )

        # Z-scores si tenemos modelo
        if self.acpl_model:
            elo = self._estimate_player_elo(game.white_username)
            features['acpl_z'] = self.acpl_model.z_score(elo, features['acpl'])
            features['ipr_z'] = quality.ipr_z_score(features['ipr'], elo)

        # Detección de rachas
        bursts = quality.precision_bursts(moves_df)
        features['precision_burst_count'] = len(bursts)

        return features

    def _calculate_suspicious_flags(self, features: Dict) -> Dict:
        """Calcula flags de comportamiento sospechoso."""
        return {
            'suspicious_quality': (
                    features.get('acpl', 100) < 20 and
                    features.get('match_rate', 0) > 0.70
            ),
            'suspicious_timing': (
                    features.get('time_complexity_corr', 1) < 0.1 or
                    features.get('lag_spike_count', 0) > 2
            ),
            'suspicious_opening': (
                    features.get('H_opening', 10) < 1.0 and
                    features.get('second_choice_pct', 0) > 0.80
            )
        }

    def _calculate_risk_score(self, games_df: pd.DataFrame,
                              long_features: Dict) -> tuple[float, Dict]:
        """
        Calcula un score de riesgo 0-100 basado en múltiples factores.
        """
        risk_factors = {}
        risk_score = 0

        # Factor 1: ACPL demasiado bajo
        if games_df['acpl'].mean() < 25:
            risk_factors['low_acpl'] = True
            risk_score += 20

        # Factor 2: ROI consistentemente alto
        if long_features.get('roi_mean', 0) > 2.0:
            risk_factors['high_roi'] = True
            risk_score += 25

        # Factor 3: Step function detectado
        if long_features.get('step_acpl_flag', False):
            risk_factors['step_function'] = True
            risk_score += 20

        # Factor 4: Streaks largos
        if long_features.get('longest_roi_streak', 0) >= 8:
            risk_factors['long_streak'] = True
            risk_score += 15

        # Factor 5: Timing anormal
        if games_df['time_complexity_corr'].mean() < 0.1:
            risk_factors['abnormal_timing'] = True
            risk_score += 20

        return min(risk_score, 100), risk_factors

    def _get_player_games_df(self, username: str, session: Session) -> pd.DataFrame:
        """Obtiene DataFrame con todas las partidas del jugador."""
        stmt = select(Game).where(
            (Game.white_username == username) |
            (Game.black_username == username)
        )
        games = session.exec(stmt).all()

        return pd.DataFrame([{
            'game_id': g.id,
            'eco': g.eco_code or 'A00',
            'opening_key': g.opening_key
        } for g in games])

    # --------------------------------------------------------------------------- #
    # 1.  Partidas + análisis detallado                                           #
    # --------------------------------------------------------------------------- #
    def _get_player_games_with_analysis(self, username: str,
                                        session: Session) -> pd.DataFrame:
        """
        Devuelve un DataFrame que une Game ←→ GameAnalysisDetailed
        para todas las partidas en las que `username` jugó con blancas o negras.

        Columnas devueltas:
            game_id, created_at, white_username, black_username,
            result, acpl, match_rate, overall_suspicion_score, analyzed_at
        """
        stmt = (
            select(models.Game, models.GameAnalysisDetailed)
            .join(models.GameAnalysisDetailed,
                  models.Game.id == models.GameAnalysisDetailed.game_id)
            .where(
                (models.Game.white_username == username) |
                (models.Game.black_username == username)
            )
        )
        rows = session.exec(stmt).all()

        # -- a DataFrame --------------------------------------------------------
        data = []
        for game, detail in rows:
            data.append({
                "game_id": game.id,
                "created_at": game.created_at,
                "white": game.white_username,
                "black": game.black_username,
                "result": game.result,  # asumiendo columna `result`
                # métricas de GameAnalysisDetailed
                "acpl": detail.acpl,
                "match_rate": detail.match_rate,
                "suspicion": detail.overall_suspicion_score,
                "analyzed_at": detail.analyzed_at,
            })
        return pd.DataFrame(data)

    # --------------------------------------------------------------------------- #
    # 2.  Estimación de ELO por media robusta                                     #
    # --------------------------------------------------------------------------- #
    def _estimate_player_elo(self, username: str) -> int:
        """
        Estima el ELO del jugador como la mediana de sus ratings
        (blanca + negra) en las partidas guardadas.

        - Usa la BD directamente → no hace peticiones externas.
        - Si no hay datos, devuelve 1800 por defecto.
        """
        with Session(engine) as s:
            stmt = select(models.Game.white_elo, models.Game.black_elo).where(
                (models.Game.white_username == username) |
                (models.Game.black_username == username)
            )
            elos = []
            for white_elo, black_elo in s.exec(stmt):
                if white_elo and white_elo > 0:
                    elos.append(white_elo)
                if black_elo and black_elo > 0:
                    elos.append(black_elo)

        if not elos:
            return 1800  # sin datos ⇒ default

        # mediana robusta contra outliers
        return int(np.median(elos))

    # --------------------------------------------------------------------------- #
    # 3.  Detección de final real                                                #
    # --------------------------------------------------------------------------- #
    def _has_endgame(self, moves_df: pd.DataFrame) -> bool:
        """
        Considera que hay final si:
            • Se alcanzó una posición con ≤ 7 piezas   *o*
            • Hay tabla Syzygy disponible (ECO = 'E*') *o*
            • La partida superó 60 jugadas y no quedan damas

        Necesita las SAN de cada jugada en `moves_df.played`.
        """
        board = chess.Board()
        for san in moves_df.played:
            move = board.parse_san(san)
            board.push(move)

            # condición 1: pocas piezas (reloj Syzygy)
            if len(board.piece_map()) <= 7:
                return True

        # condición 2: jugadas largas sin damas
        if board.fullmove_number > 60 and not board.pieces(chess.QUEEN, chess.WHITE | chess.BLACK):
            return True

        return False