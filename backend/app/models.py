# app/models.py
"""
Modelos actualizados para soportar análisis detallado.
Mantiene compatibilidad con el sistema actual.
"""
from datetime import datetime, UTC, timezone
from typing import Optional, List
from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy import Column, JSON
import sqlalchemy as sa


# ============================================================
# MODELOS EXISTENTES (sin cambios para compatibilidad)
# ============================================================

class Game(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    pgn: str

    moves: List["MoveAnalysis"] = Relationship(back_populates="game")
    metrics: Optional["GameMetrics"] = Relationship(back_populates="game")
    detailed_analysis: Optional["GameAnalysisDetailed"] = Relationship(back_populates="game")

    move_times: Optional[List[int]] = Field(sa_column=Column(JSON))
    eco_code: Optional[str] = None
    opening_key: Optional[str] = None
    white_username: Optional[str] = None
    black_username: Optional[str] = None

    # Nuevos campos para análisis mejorado
    white_elo: Optional[int] = None
    black_elo: Optional[int] = None
    time_control: Optional[str] = None
    termination: Optional[str] = None


class MoveAnalysis(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    game_id: int = Field(foreign_key="game.id")

    move_number: int
    played: str
    best: str
    best_rank: int
    cp_loss: int

    # Nuevos campos para análisis detallado
    eval_before: Optional[int] = None  # Evaluación antes del movimiento
    eval_after: Optional[int] = None  # Evaluación después
    legal_moves_count: Optional[int] = None  # Número de movimientos legales
    time_spent: Optional[float] = None  # Tiempo usado en segundos

    game: Game = Relationship(back_populates="moves")


class GameMetrics(SQLModel, table=True):
    """Métricas básicas (compatibilidad)"""
    id: Optional[int] = Field(default=None, primary_key=True)
    game_id: int = Field(foreign_key="game.id", unique=True)

    sigma_total: Optional[float] = None
    constant_time: bool = False
    pause_spike: bool = False

    pct_top1: float
    pct_top3: float
    acl: float
    suspicious: bool = False
    computed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    game: Game = Relationship(back_populates="metrics")


class PlayerMetrics(SQLModel, table=True):
    """Métricas básicas del jugador (compatibilidad)"""
    username: str = Field(primary_key=True)
    game_count: int
    opening_entropy: float
    most_played: Optional[str] = None
    low_entropy: bool = False
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Player(SQLModel, table=True):
    username: str = Field(primary_key=True)
    status: str = Field(
        sa_column=sa.Column(
            sa.Enum("not_analyzed", "pending", "ready", "error",
                    name="player_status"),
            nullable=False,
            default="not_analyzed",
        )
    )
    progress: int = 0
    total_games: Optional[int] = None
    done_games: Optional[int] = None
    requested_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error: Optional[str] = None
    last_task_id: str | None = Field(default=None, description="Celery id de la tarea en curso")


    # Relación con análisis detallado
    detailed_analysis: Optional["PlayerAnalysisDetailed"] = Relationship(back_populates="player")


# ============================================================
# NUEVOS MODELOS PARA ANÁLISIS DETALLADO
# ============================================================

class GameAnalysisDetailed(SQLModel, table=True):
    """Análisis detallado por partida usando los nuevos módulos."""
    __tablename__ = "game_analysis_detailed"

    game_id: int = Field(foreign_key="game.id", primary_key=True)

    # === QUALITY METRICS ===
    acpl: float = Field(description="Average Centipawn Loss")
    match_rate: float = Field(description="% de coincidencia con motor")
    weighted_match_rate: float = Field(description="Match rate ponderado por complejidad")
    ipr: float = Field(description="Intrinsic Performance Rating")
    ipr_z_score: float = Field(default=0, description="Z-score del IPR")
    precision_burst_count: int = Field(default=0, description="Número de rachas de precisión")

    # === TIMING METRICS ===
    mean_move_time: float
    time_variance: float
    time_complexity_corr: float = Field(description="Correlación tiempo-complejidad")
    lag_spike_count: int = Field(default=0)
    uniformity_score: float = Field(description="Score de uniformidad temporal")
    clutch_accuracy_diff: Optional[float] = Field(default=None, description="Diferencia de precisión bajo presión")

    # === OPENING METRICS ===
    opening_entropy: float = Field(default=0, description="Entropía del repertorio")
    novelty_depth: Optional[int] = Field(default=None, description="Profundidad de novedad")
    second_choice_rate: float = Field(default=0, description="% de segundas mejores jugadas")
    opening_breadth: int = Field(default=0, description="Variedad de aperturas")

    # === ENDGAME METRICS ===
    tb_match_rate: Optional[float] = Field(default=None, description="% coincidencia con tablebases")
    dtz_deviation: Optional[float] = Field(default=None, description="Desviación DTZ promedio")
    conversion_efficiency: Optional[int] = Field(default=None, description="Movimientos para convertir ventaja")

    # === FLAGS & SCORES ===
    suspicious_quality: bool = Field(default=False)
    suspicious_timing: bool = Field(default=False)
    suspicious_opening: bool = Field(default=False)
    overall_suspicion_score: float = Field(default=0, description="Score agregado 0-100")

    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Relaciones
    game: Game = Relationship(back_populates="detailed_analysis")


class PlayerAnalysisDetailed(SQLModel, table=True):
    """Análisis agregado detallado por jugador."""
    __tablename__ = "player_analysis_detailed"

    username: str = Field(foreign_key="player.username", primary_key=True)

    # === AGGREGATED METRICS ===
    games_analyzed: int = Field(description="Número de partidas analizadas")
    avg_acpl: float
    avg_match_rate: float
    avg_ipr: float
    std_acpl: float = Field(description="Desviación estándar del ACPL")
    std_match_rate: float

    # === LONGITUDINAL METRICS ===
    roi_mean: float = Field(description="ROI promedio")
    roi_max: float = Field(description="ROI máximo")
    roi_std: float = Field(description="Desviación estándar ROI")
    step_function_detected: bool = Field(default=False)
    step_function_magnitude: Optional[float] = Field(default=None)
    peer_delta_acpl: float = Field(default=0, description="Diferencia vs peers")
    peer_delta_match: float = Field(default=0)
    longest_streak: int = Field(default=0, description="Racha más larga de alto rendimiento")
    selectivity_score: float = Field(default=0, description="Score de selectividad")

    # === PATTERN ANALYSIS ===
    time_patterns: dict = Field(default_factory=dict, sa_column=Column(JSON))
    opening_patterns: dict = Field(default_factory=dict, sa_column=Column(JSON))
    suspicious_games_ids: List[int] = Field(default_factory=list, sa_column=Column(JSON))

    # === RISK ASSESSMENT ===
    risk_score: float = Field(default=0, description="Score de riesgo 0-100")
    risk_factors: dict = Field(default_factory=dict, sa_column=Column(JSON))
    confidence_level: float = Field(default=0, description="Confianza en el análisis 0-1")

    # === TIMESTAMPS ===
    first_game_date: Optional[datetime] = None
    last_game_date: Optional[datetime] = None
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Relaciones
    player: Player = Relationship(back_populates="detailed_analysis")


# ============================================================
# TABLAS DE REFERENCIA (para calibración)
# ============================================================

class ReferenceStats(SQLModel, table=True):
    """Estadísticas de referencia por ELO para calibración."""
    __tablename__ = "reference_stats"

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
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))