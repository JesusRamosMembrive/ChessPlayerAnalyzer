"""
schemas.py – Pydantic response models exposed by the public API.
Refactored to make optional fields truly optional and match
PlayerAnalysisDetailed without triggering ResponseValidationError.
"""
from datetime import datetime
from typing import Dict, Optional
from pydantic import BaseModel, Field

# ── Public API response models ────────────────────────────────────────────────
class TimeManagementOut(BaseModel):
    mean_move_time: float          # segundos
    time_variance: float           # varianza en seg²
    uniformity_score: float        # 0-1 (1 = tiempos clavados)
    lag_spike_count: int           # nº de spikes >5× promedio

class ClutchAccuracyOut(BaseModel):
    avg_clutch_diff: Optional[float] = None      # media de |Δeval| en últimos 5 movimientos
    clutch_games_pct: Optional[float] = None     # % partidas con clutch_diff < 100 cp

# ── Nested models ──────────────────────────────────────────────────────
class TimePatternsOut(BaseModel):
    mean_move_time: float
    time_variance: float
    uniformity_score: float
    clutch_accuracy_diff: Optional[float] = Field(
        None, description="Accuracy drop in critical positions"
    )

class TacticalTrendOut(BaseModel):
    """Measures of tactical bursts and second-line choices."""

    precision_burst_count: Optional[int] = None   # rachas ≥3 jugadas casi perfectas
    second_choice_rate: Optional[float] = None    # ya incluido en opening_patterns, duplicamos aquí para UI cómoda


class EndgameEfficiencyOut(BaseModel):
    """Quality of play in endings and TB positions."""

    conversion_efficiency: Optional[int] = Field(
        None, ge=0, description="Number of moves to convert advantage to victory"
    )
    tb_match_rate: Optional[float] = Field(
        None, ge=0, le=1, description="Fraction of moves matching tablebase line"
    )
    dtz_deviation: Optional[float] = None         # average DTZ deviation


class PerformanceTrendsOut(BaseModel):
    """Long‑term evolution of playing strength (aggregated per month)."""

    trend_acpl: Optional[float] = Field(
        None, description="Slope of ACPL over time (cp per 100 games; positive = getting worse)"
    )
    trend_match_rate: Optional[float] = Field(
        None, description="Slope of match‑rate over time (Δ per 100 games)"
    )
    roi_curve: Optional[list[float]] = Field(
        None, description="Monthly ROI means oldest→newest (max 24)"
    )


class OpeningPatternsOut(BaseModel):
    mean_entropy: float
    novelty_depth: float  # can be fractional once averaged
    opening_breadth: int
    second_choice_rate: Optional[float] = Field(
        None, description="Proportion of 2nd/3rd engine line when >50 cp diff"
    )

class OpeningSummaryOut(BaseModel):
    eco_code: str = Field(..., min_length=3, max_length=3)
    name: str
    count: int


class PhaseQualityOut(BaseModel):
    """Quality metrics split by game phase."""
    opening_acpl: Optional[float] = None
    middlegame_acpl: Optional[float] = None
    endgame_acpl: Optional[float] = None

    opening_blunder_rate: Optional[float] = None
    middlegame_blunder_rate: Optional[float] = None
    endgame_blunder_rate: Optional[float] = None

    blunder_rate: Optional[float] = None  # global, ya existía

class BenchmarkOut(BaseModel):
    """Position of the player relative to a cohort of similar Elo."""

    percentile_acpl: Optional[int] = Field(
        None, ge=0, le=100, description="Percentile of average ACPL vs peers"
    )
    percentile_entropy: Optional[int] = Field(
        None, ge=0, le=100, description="Percentile of opening variety"
    )

class RiskAssessmentOut(BaseModel):
    risk_score: int
    risk_factors: Dict[str, float]
    confidence_level: int
    suspicious_games_count: int


# ── Main public schema ─────────────────────────────────────────────────
class PlayerMetricsOut(BaseModel):
    # Identification
    username: str

    # Quality metrics
    games_analyzed: int
    avg_acpl: float
    std_acpl: float
    avg_match_rate: float
    std_match_rate: float
    avg_ipr: float

    # Longitudinal metrics
    roi_mean: Optional[float] = None
    roi_max: Optional[float] = None
    roi_std: Optional[float] = None
    step_function_detected: bool
    step_function_magnitude: Optional[float] = None

    # Comparative & volume
    peer_delta_acpl: float
    peer_delta_match: float
    longest_streak: int
    first_game_date: Optional[datetime] = None
    last_game_date: Optional[datetime] = None

    # Selectivity & patterns
    selectivity_score: float
    time_patterns: Optional[TimePatternsOut] = None
    opening_patterns: Optional[OpeningPatternsOut] = None

    trend_acpl: float | None = None
    trend_match_rate: float | None = None
    roi_curve: list[float] | None = None
    consistency_score: float | None = None
    
    # Risk evaluation (optional because legacy data may not have it yet)
    risk: Optional[RiskAssessmentOut] = None

    # Opening repertoire summary
    favorite_openings: Optional[list[OpeningSummaryOut]] = None
    performance: Optional[PerformanceTrendsOut] = None
    phase_quality: Optional[PhaseQualityOut] = None
    benchmark: Optional[BenchmarkOut] = None
    tactical: Optional[TacticalTrendOut] = None
    endgame: Optional[EndgameEfficiencyOut] = None
    time_management: Optional[TimeManagementOut] = None
    clutch_accuracy: Optional[ClutchAccuracyOut] = None

    # Metadata
    analyzed_at: datetime

    class Config:
        orm_mode = True  # allow returning SQLModel instances directly


__all__ = [
    "PlayerMetricsOut",
    "TimePatternsOut",
    "OpeningPatternsOut",
    "RiskAssessmentOut",
]
