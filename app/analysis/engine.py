# app/analysis/engine_v2.py
"""
Motor de análisis unificado V2 - Refactor simplificado.
Elimina dependencias circulares y usa pipeline determinístico.
"""
from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import pandas as pd
import chess.pgn
import chess.engine
import io
import os
import json

from pathlib import Path
from sqlmodel import Session, select
from sqlalchemy.orm import selectinload

# Importar modelos unificados
from app.models import Game, AnalysisResult, Player
from app.database import engine as db_engine

# Importar módulos de análisis (sin cambios en la API)
from . import quality
from . import timing
from . import openings
from . import endgame
from . import longitudinal

# Utils
from app.utils import clean_json_numbers
from app.analysis.eco_table import ECO_NAMES

try:
    from app.utils_debugging.tracer import trace
except Exception:
    def trace(func=None, *targs, **tkwargs):
        if func is None:
            def _decorator(f):
                return f
            return _decorator
        return func

logger = logging.getLogger(__name__)


class AnalysisEngine:
    """
    Motor de análisis unificado que procesa partidas en un pipeline determinístico.

    Pipeline: Game PGN → Stockfish Analysis → Quality → Timing → Opening → Save
    """

    def __init__(self,
                 stockfish_path: str = None,
                 depth: int = 12,
                 tablebase_path: str = None):
        self.stockfish_path = stockfish_path or os.getenv("STOCKFISH_PATH", "stockfish")
        self.depth = depth
        self.tablebase_path = tablebase_path

    @trace
    def analyze_game(self, game: Game, username: str, color: str) -> AnalysisResult:
        """
        Analiza una partida completa para un jugador específico.

        Args:
            game: Objeto Game con PGN
            username: Usuario a analizar
            color: 'white' o 'black'

        Returns:
            AnalysisResult con todas las métricas calculadas
        """
        logger.info(f"Starting unified analysis for game {game.id}, player {username} ({color})")

        try:
            # 1. Análisis base con Stockfish
            moves_df = self._analyze_with_stockfish(game.pgn, color, username)
            logger.info(f"Stockfish analysis completed: {len(moves_df)} moves analyzed")

            # 2. Calcular todas las métricas en orden determinístico
            metrics = self._compute_all_metrics(moves_df, game, username, color)
            logger.info("All metrics computed successfully")

            # 3. Crear resultado unificado
            result = AnalysisResult(
                game_id=game.id,
                player_username=username,
                player_color=color,
                analyzed_at=datetime.now(timezone.utc),
                engine_depth=self.depth,
                moves_analyzed=len(moves_df),
                metrics=metrics
            )

            # 4. Guardar en base de datos
            with Session(db_engine) as session:
                session.add(result)
                session.commit()
                session.refresh(result)

            logger.info(f"Analysis result saved with ID {result.id}")
            return result

        except Exception as e:
            logger.error(f"Error analyzing game {game.id}: {e}")
            raise

    @trace
    def _analyze_with_stockfish(self, pgn: str, player_color: str, username: str) -> pd.DataFrame:
        """
        Analiza el PGN con Stockfish y genera DataFrame de movimientos.

        Returns:
            DataFrame con análisis por movimiento del jugador especificado
        """
        logger.info(f"Starting Stockfish analysis for {player_color} player")

        # Parse PGN
        game = chess.pgn.read_game(io.StringIO(pgn))
        if not game:
            raise ValueError("Invalid PGN")

        # Configurar Stockfish
        with chess.engine.SimpleEngine.popen_uci(self.stockfish_path) as engine_sf:
            if self.tablebase_path:
                engine_sf.configure({"SyzygyPath": self.tablebase_path})

            board = game.board()
            moves_data = []
            player_move_number = 1  # Contador específico para movimientos del jugador

            # Variables para timing
            initial_time = 600.0  # 10 minutos por defecto
            player_clock = initial_time

            for i, move in enumerate(game.mainline_moves()):
                # Analizar TODOS los movimientos, pero solo guardar los del jugador especificado
                is_white_move = (i % 2 == 0)
                current_player_move = (player_color == 'white' and is_white_move) or \
                                     (player_color == 'black' and not is_white_move)

                # Análisis de la posición antes del movimiento
                eval_before = None
                try:
                    info_before = engine_sf.analyse(board, chess.engine.Limit(depth=self.depth))
                    eval_before = self._extract_evaluation(info_before)
                except:
                    logger.warning(f"Failed to analyze position before move {i+1}")

                # Obtener mejores movimientos
                best_moves = []
                try:
                    multipv_info = engine_sf.analyse(board,
                                                   chess.engine.Limit(depth=self.depth),
                                                   multipv=3)
                    for pv_info in multipv_info:
                        if pv_info.get("pv"):
                            best_moves.append(pv_info["pv"][0])
                except:
                    logger.warning(f"Failed to get best moves for move {i+1}")

                # Aplicar el movimiento jugado
                board.push(move)

                # Análisis después del movimiento
                eval_after = None
                try:
                    info_after = engine_sf.analyse(board, chess.engine.Limit(depth=self.depth))
                    eval_after = self._extract_evaluation(info_after)
                except:
                    logger.warning(f"Failed to analyze position after move {i+1}")

                # Calcular métricas del movimiento
                best_move = best_moves[0] if best_moves else move
                best_rank = 1
                if len(best_moves) > 1:
                    try:
                        best_rank = best_moves.index(move) + 1
                    except ValueError:
                        best_rank = 4  # No está en top 3

                cp_loss = 0
                if eval_before is not None and eval_after is not None:
                    if player_color == 'white':
                        cp_loss = max(0, eval_before - eval_after)
                    else:
                        cp_loss = max(0, eval_after - eval_before)

                # Determinar fase de la partida
                phase = self._determine_phase(board, player_move_number if current_player_move else 1)

                # Solo guardar datos si es movimiento del jugador especificado
                if current_player_move:
                    # Datos del movimiento
                    move_data = {
                        'move_number': player_move_number,
                        'played': str(move),
                        'best': str(best_move),
                        'best_rank': best_rank,
                        'cp_loss': cp_loss,
                        'delta_eval': cp_loss,  # Alias para compatibilidad
                        'eval_cp_before': eval_before,
                        'eval_cp_after': eval_after,
                        'eval_before': eval_before,
                        'eval_after': eval_after,
                        'legal_moves_count': len(list(board.legal_moves)),
                        'legal_moves': len(list(board.legal_moves)),  # Alias
                        'phase': phase,
                        'is_engine_best': (best_rank == 1),
                        'move_time': 2.0,  # Default, se puede mejorar con timing real
                        'player_clock_before': player_clock,
                        'depth': self.depth
                    }

                    moves_data.append(move_data)
                    player_move_number += 1

                player_clock = max(0, player_clock - 2.0)  # Decrementar reloj

        df = pd.DataFrame(moves_data)
        logger.info(f"Stockfish analysis completed: {len(df)} moves for {player_color}")
        return df

    @trace
    def _extract_evaluation(self, info: dict) -> Optional[int]:
        """Extrae evaluación en centipawns del resultado de Stockfish"""
        score = info.get("score")
        if not score:
            return None

        if score.is_mate():
            # Convertir mate a centipawns (simplificado)
            mate_moves = score.mate()
            return 10000 if mate_moves > 0 else -10000
        else:
            return score.relative.score(mate_score=10000)

    @trace
    def _determine_phase(self, board: chess.Board, move_number: int) -> str:
        """Determina la fase de la partida"""
        if move_number <= 10:
            return "opening"

        # Contar piezas para determinar final
        piece_count = len(board.piece_map())
        if piece_count <= 10:
            return "endgame"

        return "middlegame"

    @trace
    def _compute_all_metrics(self, moves_df: pd.DataFrame, game: Game, username: str, color: str) -> Dict:
        """
        Calcula todas las métricas en orden determinístico sin dependencias circulares.

        Args:
            moves_df: DataFrame con análisis de movimientos
            game: Objeto Game con metadatos
            username: Usuario analizado
            color: Color del jugador

        Returns:
            Dict con todas las métricas organizadas por módulo
        """
        logger.info("Computing all metrics in deterministic order")

        # Obtener ELO del jugador si está disponible
        player_elo = game.white_elo if color == 'white' else game.black_elo

        # 1. QUALITY METRICS (base)
        logger.info("Computing quality metrics")
        quality_metrics = quality.aggregate_quality_features(
            moves_df,
            elo=player_elo,
            player_color=color
        )

        # 2. TIMING METRICS
        logger.info("Computing timing metrics")
        timing_metrics = timing.aggregate_time_features(moves_df)

        # 3. OPENING METRICS
        logger.info("Computing opening metrics")
        opening_metrics = openings.aggregate_opening_features(
            opening_key=game.opening_key or "",
            eco_code=game.eco_code,
            moves_df=moves_df,
            games_df=pd.DataFrame([{
                'eco_code': game.eco_code,
                'opening_key': game.opening_key
            }])
        )

        # 4. ENDGAME METRICS (si aplica)
        logger.info("Computing endgame metrics")
        endgame_metrics = {}
        if 'endgame' in moves_df.get('phase', []):
            try:
                # Crear objeto chess.pgn.Game para endgame analysis
                pgn_game = chess.pgn.read_game(io.StringIO(game.pgn))
                endgame_metrics = endgame.aggregate_endgame_features(
                    pgn_game, moves_df, self.tablebase_path
                )
            except Exception as e:
                logger.warning(f"Endgame analysis failed: {e}")
                endgame_metrics = {
                    'conversion_efficiency': None,
                    'tb_match_rate': None,
                    'dtz_deviation': None
                }

        # 5. MOVES DATA (para análisis posteriores)
        moves_data = moves_df.to_dict('records')

        # Estructura final unificada
        all_metrics = {
            'quality': clean_json_numbers(quality_metrics),
            'timing': clean_json_numbers(timing_metrics),
            'opening': clean_json_numbers(opening_metrics),
            'endgame': clean_json_numbers(endgame_metrics),
            'moves': moves_data,
            'metadata': {
                'game_id': game.id,
                'player_color': color,
                'player_elo': player_elo,
                'eco_code': game.eco_code,
                'opening_key': game.opening_key,
                'analyzed_at': datetime.now(timezone.utc).isoformat()
            }
        }

        logger.info("All metrics computed successfully")
        return all_metrics

    @trace
    def analyze_player(self, username: str) -> Dict:
        """
        Analiza todas las partidas de un jugador y genera métricas agregadas.

        Args:
            username: Usuario a analizar

        Returns:
            Dict con métricas agregadas compatibles con endpoint /metrics/player/{username}
        """
        logger.info(f"Starting player-level analysis for {username}")

        with Session(db_engine) as session:
            # Obtener todos los resultados de análisis del jugador
            results = session.exec(
                select(AnalysisResult)
                .where(AnalysisResult.player_username == username)
                .options(selectinload(AnalysisResult.game))
            ).all()

            if not results:
                logger.warning(f"No analysis results found for player {username}")
                return self._empty_player_metrics(username)

            logger.info(f"Found {len(results)} analysis results for {username}")

            # Preparar datos para análisis longitudinal
            games_data = []
            moves_dfs = []

            for result in results:
                # Extraer métricas de calidad por partida
                quality_metrics = result.metrics.get('quality', {})
                timing_metrics = result.metrics.get('timing', {})
                opening_metrics = result.metrics.get('opening', {})

                game_data = {
                    'game_id': result.game_id,
                    'created_at': result.game.created_at,
                    'analyzed_at': result.analyzed_at,
                    'player_color': result.player_color,
                    # Métricas de calidad
                    'acpl': quality_metrics.get('acpl', 0),
                    'match_rate': quality_metrics.get('match_rate', 0),
                    'match_pct': quality_metrics.get('match_rate', 0),  # Alias
                    'weighted_match_rate': quality_metrics.get('weighted_match_rate', 0),
                    'wdl_loss': quality_metrics.get('wdl_loss', 0),
                    'ipr': quality_metrics.get('ipr', 0),
                    # Métricas de timing
                    'time_complexity_corr': timing_metrics.get('time_complexity_corr', 0),
                    'clutch_accuracy_diff': timing_metrics.get('clutch_accuracy_diff'),
                    # Métricas de opening
                    'eco_code': result.game.eco_code
                }
                games_data.append(game_data)

                # Preparar DataFrame de movimientos para análisis longitudinal
                moves_data = result.metrics.get('moves', [])
                if moves_data:
                    moves_df = pd.DataFrame(moves_data)
                    moves_dfs.append(moves_df)

            # Crear DataFrame de partidas para análisis longitudinal
            games_df = pd.DataFrame(games_data)

            # Calcular métricas longitudinales
            logger.info("Computing longitudinal metrics")
            longitudinal_metrics = longitudinal.aggregate_longitudinal_features(
                games_df,
                reference_df=None  # TODO: implementar datos de referencia
            )

            # Calcular métricas de opening agregadas
            logger.info("Computing aggregated opening patterns")
            opening_patterns = openings.aggregate_player_opening_patterns(games_df, moves_dfs)

            # Calcular métricas de fase agregadas
            logger.info("Computing phase quality metrics")
            phase_quality = quality.compute_phase_quality(moves_dfs)

            # Generar estructura compatible con endpoint
            aggregated_metrics = self._build_player_response(
                username=username,
                games_df=games_df,
                longitudinal_metrics=longitudinal_metrics,
                opening_patterns=opening_patterns,
                phase_quality=phase_quality,
                results=results
            )

            # Actualizar tabla Player con métricas agregadas
            player = session.exec(
                select(Player).where(Player.username == username)
            ).first()

            if player:
                player.aggregated_metrics = aggregated_metrics
                player.analyzed_at = datetime.now(timezone.utc)
                player.first_game_date = games_df['created_at'].min()
                player.last_game_date = games_df['created_at'].max()
                session.add(player)
                session.commit()

            logger.info(f"Player analysis completed for {username}")
            return aggregated_metrics

    def _empty_player_metrics(self, username: str) -> Dict:
        """Retorna métricas vacías para jugador sin análisis"""
        return {
            'username': username,
            'games_analyzed': 0,
            'error': 'No analysis results found'
        }

    def _build_player_response(self, username: str, games_df: pd.DataFrame,
                             longitudinal_metrics: Dict, opening_patterns: Dict,
                             phase_quality: Dict, results: List[AnalysisResult]) -> Dict:
        """
        Construye la respuesta JSON compatible con el endpoint /metrics/player/{username}
        """
        # Estadísticas básicas
        games_analyzed = len(games_df)
        avg_acpl = games_df['acpl'].mean() if 'acpl' in games_df else 0
        avg_wdl_loss = games_df['wdl_loss'].mean() if 'wdl_loss' in games_df else 0
        avg_match_rate = games_df['match_rate'].mean() if 'match_rate' in games_df else 0
        std_acpl = games_df['acpl'].std() if 'acpl' in games_df else 0
        std_match_rate = games_df['match_rate'].std() if 'match_rate' in games_df else 0
        avg_ipr = games_df['ipr'].mean() if 'ipr' in games_df else 0

        # Calcular algunas métricas derivadas
        risk_score = 0
        risk_factors = {}

        # Detectar factores de riesgo básicos
        if longitudinal_metrics.get('roi_mean', 0) > 2200:
            risk_score += 20
            risk_factors['high_roi'] = 1

        if longitudinal_metrics.get('step_match_pct_flag', False):
            risk_score += 25
            risk_factors['step_function'] = 1

        # Estructura compatible con React frontend
        return clean_json_numbers({
            'username': username,
            'games_analyzed': games_analyzed,
            'avg_acpl': avg_acpl,
            'avg_wdl_loss': avg_wdl_loss,
            'robust_loss': 0,  # TODO: implementar
            'std_acpl': std_acpl,
            'avg_match_rate': avg_match_rate,
            'std_match_rate': std_match_rate,
            'avg_ipr': avg_ipr,

            # Métricas longitudinales
            'roi_mean': longitudinal_metrics.get('roi_mean', 0),
            'roi_max': longitudinal_metrics.get('roi_max', 0),
            'roi_std': longitudinal_metrics.get('roi_sd', 0),
            'step_function_detected': longitudinal_metrics.get('step_match_pct_flag', False),
            'step_function_magnitude': longitudinal_metrics.get('step_match_pct_delta', 0),
            'peer_delta_acpl': longitudinal_metrics.get('peer_delta_acpl', 0),
            'peer_delta_match': longitudinal_metrics.get('peer_delta_match', 0),
            'longest_streak': longitudinal_metrics.get('longest_streak', 0),
            'selectivity_score': longitudinal_metrics.get('selectivity_pct', 50),

            # Fechas
            'first_game_date': games_df['created_at'].min().isoformat() if not games_df.empty else None,
            'last_game_date': games_df['created_at'].max().isoformat() if not games_df.empty else None,

            # Patrones
            'time_patterns': None,  # TODO: implementar
            'opening_patterns': opening_patterns,

            # Performance trends
            'trend_acpl': longitudinal_metrics.get('trend_acpl', 0),
            'trend_match_rate': longitudinal_metrics.get('trend_match_rate', 0),
            'roi_curve': longitudinal_metrics.get('roi_curve', []),
            'consistency_score': None,  # TODO: implementar

            # Análisis de riesgo
            'risk': {
                'risk_score': min(risk_score, 100),
                'risk_factors': risk_factors,
                'confidence_level': 0,  # TODO: implementar
                'suspicious_games_count': 0  # TODO: implementar
            },

            # Aperturas favoritas
            'favorite_openings': [],  # TODO: implementar

            # Performance detallada
            'performance': {
                'trend_acpl': longitudinal_metrics.get('trend_acpl', 0),
                'trend_match_rate': longitudinal_metrics.get('trend_match_rate', 0),
                'roi_curve': longitudinal_metrics.get('roi_curve', [])
            },

            # Calidad por fase
            'phase_quality': phase_quality,

            # Benchmark
            'benchmark': {
                'percentile_acpl': 10,  # TODO: implementar
                'percentile_entropy': 5   # TODO: implementar
            },

            # Tácticas
            'tactical': {
                'precision_burst_count': None,  # TODO: extraer de quality
                'second_choice_rate': None       # TODO: extraer de quality
            },

            # Final
            'endgame': {
                'conversion_efficiency': 15,  # TODO: implementar
                'tb_match_rate': None,
                'dtz_deviation': None
            },

            # Gestión del tiempo
            'time_management': {
                'mean_move_time': games_df.get('mean_move_time', pd.Series([32])).mean(),
                'time_variance': games_df.get('time_variance', pd.Series([50000])).mean(),
                'uniformity_score': -6.183,  # TODO: implementar
                'lag_spike_count': 409        # TODO: implementar
            },

            # Precisión bajo presión
            'clutch_accuracy': {
                'avg_clutch_diff': 4919.8,  # TODO: implementar
                'clutch_games_pct': 0.643    # TODO: implementar
            },

            'analyzed_at': datetime.now(timezone.utc).isoformat()
        })