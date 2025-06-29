# app/models.py
"""
Modelos actualizados para soportar análisis detallado.
Mantiene compatibilidad con el sistema actual.
"""
import enum
from datetime import timezone
from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy import Column, ForeignKey, JSON, String, Text
from sqlmodel import Field, Relationship, SQLModel
from sqlalchemy import Enum as SQLEnum


# ============================================================
# MODELOS EXISTENTES (sin cambios para compatibilidad)
# ============================================================

class Game(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    pgn: str

    moves: List["MoveAnalysis"] = Relationship(back_populates="game")
    detailed_analysis: Optional["GameAnalysisDetailed"] = Relationship(back_populates="game")

    move_times: Optional[List[int]] = Field(sa_column=Column(JSON))
    eco_code: Optional[str] = None
    opening_key: Optional[str] = None

    white_username: Optional[str] = Field(default=None, index=True)
    black_username: Optional[str] = Field(default=None, index=True)
    # Nuevos campos para análisis mejorado
    white_elo: Optional[int] = None
    black_elo: Optional[int] = None
    time_control: Optional[str] = None
    termination: Optional[str] = None



class MoveAnalysis(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    game_id: int = Field(foreign_key="game.id", index=True)

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

    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Relaciones
    game: Game = Relationship(back_populates="detailed_analysis")


class PlayerStatus(str, enum.Enum):
    pending = "pending"
    ready   = "ready"
    error   = "error"

class Player(SQLModel, table=True):
    """
    Jugador individual.

    Se declara la relación `analysis` con ON-DELETE CASCADE para que al borrar
    el jugador se elimine automáticamente su análisis detallado y no sea
    necesario borrarlo a mano.
    """
    __tablename__ = "player"

    username: str = Field(primary_key=True)

    # ── Estado de proceso ──────────────────────────────────────────────
    status: PlayerStatus = Field(
        default=PlayerStatus.pending,
        sa_column=Column(
            SQLEnum(PlayerStatus, name="player_status", native_enum=True),
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
        description="Celery task id en ejecución (process_player_enhanced)",
    )

    # relación 1-a-1 con PlayerAnalysisDetailed
    analysis: Optional["PlayerAnalysisDetailed"] = Relationship(
        back_populates="player",
        sa_relationship_kwargs={
            "cascade": "all, delete-orphan",   # borra el hijo al borrar el padre
            "passive_deletes": True,           # evita DetachedInstanceError
        },
    )


class PlayerAnalysisDetailed(SQLModel, table=True):
    """
    Resumen estadístico y de riesgo de un jugador tras analizar todas sus
    partidas.
    """
    __tablename__ = "player_analysis_detailed"

    # PK y FK al mismo tiempo, con borrado en cascada
    username: str = Field(
        sa_column=Column(
            String,
            ForeignKey("player.username", ondelete="CASCADE"),
            primary_key=True,
        )
    )

    # ── Métricas globales de calidad ───────────────────────────────────
    games_analyzed: int = Field(nullable=False, ge=0)
    avg_acpl: float = Field(nullable=False, ge=0)
    avg_match_rate: float = Field(nullable=False, ge=0)
    avg_ipr: float = Field(nullable=False, ge=0)
    std_acpl: float | None = None
    std_match_rate: float | None = None

    # ── Métricas longitudinales ───────────────────────────────────────
    roi_mean: float | None = None
    roi_max: float | None = None
    roi_std: float | None = None
    step_function_detected: bool = Field(default=False)
    step_function_magnitude: float | None = None
    peer_delta_acpl: float = Field(default=0)
    peer_delta_match: float = Field(default=0)
    longest_streak: int = Field(default=0, ge=0)
    selectivity_score: float = Field(default=0)

    # ── Patrones y banderas ───────────────────────────────────────────
    time_patterns: Dict | None = Field(sa_column=Column(JSON, default=dict))
    opening_patterns: Dict | None = Field(sa_column=Column(JSON, default=dict))
    suspicious_games_ids: List[int] = Field(
        sa_column=Column(JSON, default=list)
    )

    performance: Dict | None = Field(sa_column=Column(JSON, default=dict))
    phase_quality: Dict | None = Field(sa_column=Column(JSON, default=dict))
    benchmark: Dict | None = Field(sa_column=Column(JSON, default=dict))

    # ── Evaluación final ──────────────────────────────────────────────
    risk_score: int = Field(default=0)
    risk_factors: Dict = Field(sa_column=Column(JSON, default=dict))
    confidence_level: int = Field(default=0)

    # ── Información temporal ─────────────────────────────────────────
    first_game_date: datetime | None = None
    last_game_date: datetime | None = None
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)

    time_management: Dict | None = Field(sa_column=Column(JSON, default=dict))
    clutch_accuracy: Dict | None = Field(sa_column=Column(JSON, default=dict))
    tactical: Dict | None = Field(sa_column=Column(JSON, default=dict))
    endgame: Dict | None = Field(sa_column=Column(JSON, default=dict))

    time_complexity: Dict | None = Field(sa_column=Column(JSON, default=dict))
    # back-ref al jugador
    player: Optional[Player] = Relationship(back_populates="analysis")


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
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
