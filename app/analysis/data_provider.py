# app/analysis/data_provider.py
"""
Data access, transformation, and preparation layer for chess analysis.
Extracted from engine.py as part of Refactor Fase 2 - Estructural.
"""
from __future__ import annotations
import logging
import pandas as pd
import numpy as np
from typing import Optional
from pathlib import Path

# Core imports
from app.models import Game, GameAnalysisDetailed
from app.database import engine as db_engine
from app import models
from sqlmodel import Session, select

# Utils
from app.utils_debugging.tracer import trace

logger = logging.getLogger(__name__)


def _safe_mean(df: pd.DataFrame, col: str, default: float = 0.0) -> float:
    """Media que nunca devuelve None (NaN→default, col ausente→default)."""
    if col not in df.columns:
        return default
    val = df[col].mean()
    return float(val) if pd.notna(val) else default


def _get_player_color(game: Game, username: str) -> str | None:
    """Determine if username played as 'white' or 'black' in this game."""
    if game.white_username == username:
        return 'white'
    elif game.black_username == username:
        return 'black'
    return None


def _has_endgame(moves_df: pd.DataFrame) -> bool:
    """Determina si hubo fase de final según heurística."""
    if moves_df.empty:
        return False

    # Heurística simple: si tenemos >= 40 jugadas del jugador, probablemente hubo endgame
    return len(moves_df) >= 40


class AnalysisDataProvider:
    """
    Data access layer for chess analysis.
    Handles database queries, data transformation, and preparation.
    """

    def __init__(self, db_session: Session):
        self.session = db_session

    @trace
    def prepare_moves_dataframe(self, game: Game, username: Optional[str] = None) -> pd.DataFrame:
        """
        Prepara DataFrame con los movimientos de una partida para análisis.
        Extraído de engine.py líneas 79-195.
        """
        rows = []
        player_color = None

        if username:
            player_color = _get_player_color(game, username)

        # ── Calcular tiempo restante en el reloj ──────────────────────────────
        # Asumimos tiempo inicial de 600 segundos (10 minutos) para partidas rápidas
        initial_time = 600.0  # 10 minutos en segundos

        for i, m in enumerate(game.moves):
            if player_color and (i % 2 == 0) != (player_color == 'white'):
                continue

            # Calcular el tiempo restante antes de este movimiento
            current_clock = initial_time

            # Si tenemos move_times, calcular el tiempo restante real
            if game.move_times and len(game.move_times) > 0:
                # Calcular el tiempo acumulado hasta este punto para el jugador
                accumulated_time = 0.0
                for j, time_change in enumerate(game.move_times):
                    # Solo contar movimientos del jugador actual
                    if (player_color == 'white' and j % 2 == 0) or (player_color == 'black' and j % 2 == 1):
                        # Solo contar hasta el movimiento actual (i)
                        if j < i:
                            accumulated_time += abs(time_change)

                current_clock = max(0.0, initial_time - accumulated_time)

            rows.append({
                "move_number": m.move_number,
                "played": m.played,
                "best_rank": m.best_rank,
                "cp_loss": m.cp_loss,
                "eval_cp_before": m.eval_before,
                "eval_cp_after": m.eval_after,
                "move_time": abs(game.move_times[i]) if game.move_times and i < len(game.move_times) else 2.0,
                "player_clock_before": current_clock,
                "is_engine_best": m.best_rank == 1 if m.best_rank is not None else False,
                "legal_moves": getattr(m, 'legal_moves', 20),  # Default reasonable value
                "delta_eval": abs(m.eval_after - m.eval_before) if m.eval_before is not None and m.eval_after is not None else 0,
                "phase": self._determine_game_phase(m.move_number),
                "complexity_score": self._calculate_position_complexity(m),
            })

        return pd.DataFrame(rows)

    def _determine_game_phase(self, move_number: int) -> str:
        """Determina la fase de la partida basada en el número de movimiento."""
        if move_number <= 15:
            return "opening"
        elif move_number <= 40:
            return "middlegame"
        else:
            return "endgame"

    def _calculate_position_complexity(self, move) -> float:
        """Calcula un score de complejidad de la posición."""
        # Simplified complexity based on available data
        base_complexity = 0.5

        # Add complexity based on evaluation swing
        if hasattr(move, 'eval_before') and hasattr(move, 'eval_after'):
            if move.eval_before is not None and move.eval_after is not None:
                eval_swing = abs(move.eval_after - move.eval_before)
                base_complexity += min(eval_swing / 100.0, 0.3)

        # Add complexity based on legal moves (if available)
        if hasattr(move, 'legal_moves') and move.legal_moves is not None:
            # More legal moves = more complex position
            base_complexity += min(move.legal_moves / 50.0, 0.2)

        return min(base_complexity, 1.0)

    @trace
    def get_player_games_df(self, username: str) -> pd.DataFrame:
        """
        Obtiene DataFrame con todas las partidas del jugador.
        Extraído de engine.py líneas 770-782.
        """
        stmt = select(Game).where(
            (Game.white_username == username) |
            (Game.black_username == username)
        )
        games = self.session.exec(stmt).all()

        return pd.DataFrame([{
            'game_id': g.id,
            'eco_code': g.eco_code or 'A00',
            'opening_key': g.opening_key
        } for g in games])

    @trace
    def get_player_games_with_analysis(self, username: str) -> pd.DataFrame:
        """
        Devuelve un DataFrame que une Game ←→ GameAnalysisDetailed
        para todas las partidas en las que `username` jugó con blancas o negras.
        Extraído de engine.py líneas 788-856.
        """
        stmt = (
            select(
                Game.id,
                Game.created_at,
                Game.white_username,
                Game.black_username,
                Game.result,
                GameAnalysisDetailed.acpl,
                GameAnalysisDetailed.match_rate,
                GameAnalysisDetailed.overall_suspicion_score,
                GameAnalysisDetailed.analyzed_at,
                GameAnalysisDetailed.clutch_accuracy_diff,
                GameAnalysisDetailed.tb_match_rate,
                GameAnalysisDetailed.dtz_deviation,
                GameAnalysisDetailed.conversion_efficiency,
                Game.eco_code,
                Game.opening_key
            )
            .join(GameAnalysisDetailed, Game.id == GameAnalysisDetailed.game_id)
            .where(
                (Game.white_username == username) |
                (Game.black_username == username)
            )
            .order_by(Game.created_at.desc())
        )

        result_rows = []
        for row in self.session.exec(stmt).all():
            # Determine player color
            player_color = 'white' if row[2] == username else 'black'  # white_username index

            result_rows.append({
                'game_id': row[0],
                'created_at': row[1],
                'white_username': row[2],
                'black_username': row[3],
                'result': row[4],
                'acpl': row[5],
                'match_rate': row[6],
                'overall_suspicion_score': row[7],
                'analyzed_at': row[8],
                'player_color': player_color,
                'clutch_accuracy_diff': row[9],
                'tb_match_rate': row[10],
                'dtz_deviation': row[11],
                'conversion_efficiency': row[12],
                'eco_code': row[13],
                'opening_key': row[14]
            })

        return pd.DataFrame(result_rows)

    @trace
    def estimate_player_elo(self, username: str, game: Optional[Game] = None) -> int:
        """
        Estima ELO del jugador basándose en datos disponibles.
        Extraído de engine.py líneas 858-889.
        """
        # 1. Si el Game tiene rating explícito, úsalo
        if game and hasattr(game, 'player_rating') and game.player_rating:
            return int(game.player_rating)

        # 2. Buscar en análisis recientes
        stmt = (
            select(GameAnalysisDetailed.estimated_rating)
            .join(Game, GameAnalysisDetailed.game_id == Game.id)
            .where(
                ((Game.white_username == username) | (Game.black_username == username)) &
                (GameAnalysisDetailed.estimated_rating.is_not(None))
            )
            .order_by(Game.created_at.desc())
            .limit(5)
        )

        recent_ratings = list(self.session.exec(stmt).all())
        if recent_ratings:
            return int(np.mean(recent_ratings))

        # 3. Fallback: ELO por defecto
        return 1500

    # Utility methods
    def has_endgame(self, moves_df: pd.DataFrame) -> bool:
        """Determina si hubo fase de final según heurística."""
        return _has_endgame(moves_df)

    def get_player_color(self, game: Game, username: str) -> str | None:
        """Determine player color for given username in game."""
        return _get_player_color(game, username)

    def safe_mean(self, df: pd.DataFrame, col: str, default: float = 0.0) -> float:
        """Safe mean calculation with fallback."""
        return _safe_mean(df, col, default)


# Backward compatibility functions (to be used during transition)
@trace
def prepare_moves_dataframe(game: models.Game, username: Optional[str] = None) -> pd.DataFrame:
    """
    Backward compatibility wrapper.
    Use AnalysisDataProvider.prepare_moves_dataframe() in new code.
    """
    with Session(db_engine) as session:
        provider = AnalysisDataProvider(session)
        return provider.prepare_moves_dataframe(game, username)