# app/models_v2.py
"""
Modelos simplificados para el refactor unificado.
Elimina dependencias circulares y reduce de 4 tablas principales a 2.
"""
import enum
from datetime import timezone, datetime
from typing import Dict, List, Optional

from sqlalchemy import Column, ForeignKey, JSON, String, Text
from sqlmodel import Field, Relationship, SQLModel
from sqlalchemy import Enum as SQLEnum


# ============================================================
# MODELOS SIMPLIFICADOS V2
# ============================================================

class Game(SQLModel, table=True):
    """
    Tabla simplificada para partidas - solo metadatos básicos.
    Todo el análisis se almacena en AnalysisResult.
    """
    __tablename__ = "game_v2"

    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    pgn: str = Field(description="PGN completo de la partida")

    # Metadatos básicos extraídos del PGN
    white_username: Optional[str] = Field(default=None, index=True)
    black_username: Optional[str] = Field(default=None, index=True)
    white_elo: Optional[int] = None
    black_elo: Optional[int] = None
    time_control: Optional[str] = None
    termination: Optional[str] = None
    eco_code: Optional[str] = None
    opening_key: Optional[str] = None

    # Metadatos de análisis
    move_times: Optional[List[int]] = Field(sa_column=Column(JSON))

    # Relación con resultados de análisis
    analysis_results: List["AnalysisResult"] = Relationship(back_populates="game")


class AnalysisResult(SQLModel, table=True):
    """
    Tabla unificada que almacena TODOS los resultados de análisis para un jugador en una partida.
    Elimina la necesidad de MoveAnalysis, GameAnalysisDetailed, PlayerAnalysisDetailed separadas.
    """
    __tablename__ = "analysis_result_v2"

    id: Optional[int] = Field(default=None, primary_key=True)

    # Referencias
    game_id: int = Field(
        sa_column=Column(
            ForeignKey("game_v2.id", ondelete="CASCADE"),
            nullable=False,
            index=True
        )
    )
    player_username: str = Field(index=True, description="Usuario analizado")
    player_color: str = Field(description="'white' o 'black'")

    # Metadatos de análisis
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    engine_depth: int = Field(default=12, description="Profundidad de análisis Stockfish")
    moves_analyzed: int = Field(default=0, description="Número de movimientos analizados")

    # TODAS las métricas en una estructura JSON unificada
    metrics: Dict = Field(
        sa_column=Column(JSON),
        default_factory=dict,
        description="Estructura JSON con todas las métricas: quality, timing, opening, etc."
    )

    # Relaciones
    game: Game = Relationship(back_populates="analysis_results")


class PlayerStatus(str, enum.Enum):
    """Estado del procesamiento de un jugador - sin cambios"""
    not_analyzed = "not_analyzed"
    pending = "pending"
    ready = "ready"
    error = "error"


class Player(SQLModel, table=True):
    """
    Tabla de jugadores - mantiene compatibilidad con endpoints existentes.
    Se relaciona con AnalysisResult para obtener análisis agregados.
    """
    __tablename__ = "player_v2"

    username: str = Field(primary_key=True)

    # Estado de proceso (para endpoints)
    status: PlayerStatus = Field(
        default=PlayerStatus.pending,
        sa_column=Column(
            SQLEnum(PlayerStatus, name="player_status_v2", native_enum=True),
            nullable=False,
        ),
    )
    requested_at: datetime | None = Field(default=None)
    finished_at: datetime | None = Field(default=None)
    progress: int = Field(default=0)
    total_games: int = Field(default=0)
    done_games: int = Field(default=0)

    error: str | None = Field(
        default=None,
        sa_column=Column(Text, nullable=True),
    )

    last_task_id: str | None = Field(
        default=None,
        sa_column=Column(String, nullable=True),
        description="Celery task id en ejecución",
    )

    # Métricas agregadas del jugador (calculadas desde AnalysisResult)
    aggregated_metrics: Optional[Dict] = Field(
        sa_column=Column(JSON),
        default_factory=dict,
        description="Métricas agregadas a nivel jugador para el endpoint /metrics/player/{username}"
    )

    # Fechas de análisis
    first_game_date: datetime | None = None
    last_game_date: datetime | None = None
    analyzed_at: datetime | None = None


# ============================================================
# TABLAS DE REFERENCIA (sin cambios)
# ============================================================

class ReferenceStats(SQLModel, table=True):
    """Estadísticas de referencia por ELO para calibración - sin cambios."""
    __tablename__ = "reference_stats_v2"

    id: Optional[int] = Field(default=None, primary_key=True)
    elo_range_min: int
    elo_range_max: int

    # Estadísticas esperadas
    expected_acpl: float
    expected_match_rate: float
    expected_time_variance: float
    expected_opening_entropy: float

    # Desviaciones estándar
    std_acpl: float
    std_match_rate: float
    std_time_variance: float
    std_opening_entropy: float

    # Metadatos
    sample_size: int
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ============================================================
# ESTRUCTURA ESPERADA DE metrics JSON
# ============================================================

"""
Estructura del campo AnalysisResult.metrics:

{
    "quality": {
        "acpl": float,
        "wdl_loss": float,
        "match_rate": float,
        "weighted_match_rate": float,
        "ipr": float,
        "ipr_z_score": float,
        "precision_burst_count": int,
        "opening_acpl": float,
        "middlegame_acpl": float,
        "endgame_acpl": float,
        "opening_blunder_rate": float,
        "middlegame_blunder_rate": float,
        "endgame_blunder_rate": float,
        "blunder_rate": float,
        "second_choice_rate": float,
        ...
    },
    "timing": {
        "mean_move_time": float,
        "time_variance": float,
        "time_complexity_corr": float,
        "lag_spike_count": int,
        "uniformity_score": float,
        "clutch_accuracy_diff": float,
        ...
    },
    "opening": {
        "opening_entropy": float,
        "novelty_depth": int,
        "second_choice_rate": float,
        "opening_breadth": int,
        ...
    },
    "moves": [
        {
            "move_number": int,
            "played": str,
            "best": str,
            "best_rank": int,
            "cp_loss": int,
            "eval_before": int,
            "eval_after": int,
            "legal_moves_count": int,
            "time_spent": float,
            "phase": str,
            ...
        }
    ]
}

Estructura del campo Player.aggregated_metrics (para endpoint /metrics/player/{username}):

{
    "username": str,
    "games_analyzed": int,
    "avg_acpl": float,
    "avg_wdl_loss": float,
    "robust_loss": float,
    "std_acpl": float,
    "avg_match_rate": float,
    "std_match_rate": float,
    "avg_ipr": float,
    "roi_mean": float,
    "roi_max": float,
    "roi_std": float,
    "step_function_detected": bool,
    "step_function_magnitude": float,
    "peer_delta_acpl": float,
    "peer_delta_match": float,
    "longest_streak": int,
    "first_game_date": datetime,
    "last_game_date": datetime,
    "selectivity_score": float,
    "time_patterns": dict,
    "opening_patterns": {
        "mean_entropy": float,
        "novelty_depth": float,
        "opening_breadth": int,
        "second_choice_rate": float
    },
    "trend_acpl": float,
    "trend_match_rate": float,
    "roi_curve": list,
    "consistency_score": float,
    "risk": {
        "risk_score": int,
        "risk_factors": dict,
        "confidence_level": int,
        "suspicious_games_count": int
    },
    "favorite_openings": list,
    "performance": {
        "trend_acpl": float,
        "trend_match_rate": float,
        "roi_curve": list
    },
    "phase_quality": {
        "opening_acpl": float,
        "middlegame_acpl": float,
        "endgame_acpl": float,
        "opening_blunder_rate": float,
        "middlegame_blunder_rate": float,
        "endgame_blunder_rate": float,
        "blunder_rate": float
    },
    "benchmark": {
        "percentile_acpl": int,
        "percentile_entropy": int
    },
    "tactical": {
        "precision_burst_count": int,
        "second_choice_rate": float
    },
    "endgame": {
        "conversion_efficiency": int,
        "tb_match_rate": float,
        "dtz_deviation": float
    },
    "time_management": {
        "mean_move_time": float,
        "time_variance": float,
        "uniformity_score": float,
        "lag_spike_count": int
    },
    "clutch_accuracy": {
        "avg_clutch_diff": float,
        "clutch_games_pct": float
    },
    "analyzed_at": datetime
}
"""