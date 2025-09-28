# app/models.py - BULLDOZER TOTAL
"""
BULLDOZER TOTAL: Modelo ultra-simplificado.

FILOSOFÍA: Una sola tabla para análisis. Sin foreign keys complejas,
sin relaciones que generen dependencias circulares, sin over-engineering.

TODO en una tabla: pgn, usuario, análisis, métricas.
Máxima simplificidad, fácil de entender, fácil de mantener.
"""
from datetime import timezone, datetime
from typing import Dict, Optional

from sqlalchemy import Column, JSON, Text
from sqlmodel import Field, SQLModel


class GameAnalysis(SQLModel, table=True):
    """
    TABLA ÚNICA BULLDOZER: Todo el análisis de chess en una sola tabla.

    Principios:
    - Sin foreign keys → Sin dependencias complejas
    - Sin relationships → Sin circular imports
    - Todo el contexto en un solo registro
    - JSON para métricas → Flexibilidad total
    - Índices simples en username → Performance básica
    """
    __tablename__ = "game_analysis_bulldozer"

    # Primary key simple
    id: Optional[int] = Field(default=None, primary_key=True)

    # ═══ DATOS DEL JUEGO ═══
    pgn: str = Field(description="PGN completo de la partida")
    white_username: Optional[str] = Field(default=None, index=True)
    black_username: Optional[str] = Field(default=None, index=True)
    white_elo: Optional[int] = None
    black_elo: Optional[int] = None
    time_control: Optional[str] = None
    game_date: Optional[str] = None
    termination: Optional[str] = None

    # ═══ ANÁLISIS CONTEXT ═══
    analyzed_username: str = Field(index=True, description="Usuario que se analizó (white o black)")
    analyzed_color: str = Field(description="'white' o 'black'")
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    engine_depth: int = Field(default=12)
    moves_analyzed: int = Field(default=0)

    # ═══ RESULTADOS COMPLETOS ═══
    # TODO en JSON - sin tablas relacionadas, sin complejidad
    analysis: Dict = Field(
        sa_column=Column(JSON),
        default_factory=dict,
        description="TODO el análisis: quality, timing, moves, opening, endgame, etc."
    )

    # ═══ ERROR HANDLING ═══
    error_message: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True),
        description="Error durante análisis, si lo hubo"
    )


class PlayerProgress(SQLModel, table=True):
    """
    Tabla simple para tracking de progreso de análisis.
    Separada de GameAnalysis para evitar updates concurrentes.
    """
    __tablename__ = "player_progress_bulldozer"

    username: str = Field(primary_key=True)

    # Estado simple
    status: str = Field(default="pending")  # "pending", "ready", "error"
    progress: int = Field(default=0)  # 0-100
    total_games: int = Field(default=0)
    done_games: int = Field(default=0)

    # Timestamps básicos
    requested_at: Optional[datetime] = Field(default=None)
    finished_at: Optional[datetime] = Field(default=None)

    # Celery task tracking simple
    last_task_id: Optional[str] = Field(default=None)

    # Error handling simple
    error_message: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True)
    )



# ════════════════════════════════════════════════════════════════════════════════
# BULLDOZER TOTAL: Estructura del JSON analysis
# ════════════════════════════════════════════════════════════════════════════════

"""
Estructura del campo GameAnalysis.analysis (TODO en un JSON):

{
    # MÉTRICAS DE CALIDAD DE JUEGO
    "quality": {
        "acpl": float or None,
        "match_rate": float or None,
        "blunders": int,
        "mistakes": int,
        "inaccuracies": int,
        "opening_acpl": float or None,
        "middlegame_acpl": float or None,
        "endgame_acpl": float or None
    },

    # MÉTRICAS DE TIMING
    "timing": {
        "mean_move_time": float or None,
        "time_variance": float or None,
        "moves": [float or None, ...]  # tiempo por movimiento
    },

    # ANÁLISIS MOVIMIENTO POR MOVIMIENTO (opcional)
    "moves": [
        {
            "move_number": int,
            "played": str,
            "best": str,
            "cp_loss": int or None,
            "time_spent": float or None
        },
        ...
    ]
}

PRINCIPIO BULLDOZER: Si no hay dato, None. Si hay problema matemático, error.
NO sanitization, NO valores fake, NO complejidad.
"""