"""
schemas.py – Pydantic response models exposed by the public API.
Refactored to make optional fields truly optional and match
PlayerAnalysisDetailed without triggering ResponseValidationError.
"""
from datetime import datetime
from typing import Dict, Optional, List, Literal
from pydantic import BaseModel, Field

# ── Public API response models ────────────────────────────────────────────────
class TimeManagementOut(BaseModel):
    mean_move_time: float          # segundos
    time_variance: float           # varianza en seg²
    uniformity_score: float        # 0-1 (1 = tiempos clavados)
    lag_spike_count: int           # nº de spikes >5× promedio

    class Config:
        schema_extra = {
            "example": {
                "mean_move_time": 12.5,
                "time_variance": 30.2,
                "uniformity_score": 0.87,
                "lag_spike_count": 3
            }
        }

class ClutchAccuracyOut(BaseModel):
    avg_clutch_diff: Optional[float] = None      # media de |Δeval| en últimos 5 movimientos
    clutch_games_pct: Optional[float] = None     # % partidas con clutch_diff < 100 cp

    class Config:
        schema_extra = {
            "example": {
                "avg_clutch_diff": 55.2,
                "clutch_games_pct": 0.42
            }
        }

# ── Nested models ──────────────────────────────────────────────────────
class TimePatternsOut(BaseModel):
    mean_move_time: float
    time_variance: float
    uniformity_score: float
    clutch_accuracy_diff: Optional[float] = Field(
        None, description="Accuracy drop in critical positions"
    )

    class Config:
        schema_extra = {
            "example": {
                "mean_move_time": 12.5,
                "time_variance": 28.1,
                "uniformity_score": 0.9,
                "clutch_accuracy_diff": -15.4
            }
        }

class TacticalTrendOut(BaseModel):
    """Measures of tactical bursts and second-line choices."""

    precision_burst_count: Optional[int] = None   # rachas ≥3 jugadas casi perfectas
    second_choice_rate: Optional[float] = None    # ya incluido en opening_patterns, duplicamos aquí para UI cómoda

    class Config:
        schema_extra = {
            "example": {
                "precision_burst_count": 7,
                "second_choice_rate": 0.18
            }
        }


class EndgameEfficiencyOut(BaseModel):
    """Quality of play in endings and TB positions."""

    conversion_efficiency: Optional[int] = Field(
        None, ge=0, description="Number of moves to convert advantage to victory"
    )
    tb_match_rate: Optional[float] = Field(
        None, ge=0, le=1, description="Fraction of moves matching tablebase line"
    )
    dtz_deviation: Optional[float] = None         # average DTZ deviation

    class Config:
        schema_extra = {
            "example": {
                "conversion_efficiency": 32,
                "tb_match_rate": 0.76,
                "dtz_deviation": 1.4
            }
        }


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

    class Config:
        schema_extra = {
            "example": {
                "trend_acpl": -1.3,
                "trend_match_rate": 0.4,
                "roi_curve": [0.02, 0.01, -0.01, 0.03]
            }
        }


class OpeningPatternsOut(BaseModel):
    mean_entropy: float
    novelty_depth: float  # can be fractional once averaged
    opening_breadth: int
    second_choice_rate: Optional[float] = Field(
        None, description="Proportion of 2nd/3rd engine line when >50 cp diff"
    )

    class Config:
        schema_extra = {
            "example": {
                "mean_entropy": 2.1,
                "novelty_depth": 10.5,
                "opening_breadth": 42,
                "second_choice_rate": 0.21
            }
        }

class OpeningSummaryOut(BaseModel):
    eco_code: str = Field(..., min_length=3, max_length=3)
    name: str
    count: int

    class Config:
        schema_extra = {
            "example": {
                "eco_code": "B22",
                "name": "Sicilian Alapin",
                "count": 15
            }
        }


class PhaseQualityOut(BaseModel):
    """Quality metrics split by game phase."""
    opening_acpl: Optional[float] = None
    middlegame_acpl: Optional[float] = None
    endgame_acpl: Optional[float] = None

    opening_blunder_rate: Optional[float] = None
    middlegame_blunder_rate: Optional[float] = None
    endgame_blunder_rate: Optional[float] = None

    blunder_rate: Optional[float] = None  # global, ya existía

    class Config:
        schema_extra = {
            "example": {
                "opening_acpl": 18.2,
                "middlegame_acpl": 24.5,
                "endgame_acpl": 15.1,
                "opening_blunder_rate": 0.05,
                "middlegame_blunder_rate": 0.07,
                "endgame_blunder_rate": 0.03,
                "blunder_rate": 0.05
            }
        }

class BenchmarkOut(BaseModel):
    """Position of the player relative to a cohort of similar Elo."""

    percentile_acpl: Optional[int] = Field(
        None, ge=0, le=100, description="Percentile of average ACPL vs peers"
    )
    percentile_entropy: Optional[int] = Field(
        None, ge=0, le=100, description="Percentile of opening variety"
    )

    class Config:
        schema_extra = {
            "example": {
                "percentile_acpl": 85,
                "percentile_entropy": 60
            }
        }

class RiskAssessmentOut(BaseModel):
    risk_score: int
    risk_factors: Dict[str, float]
    confidence_level: int
    suspicious_games_count: int

    class Config:
        schema_extra = {
            "example": {
                "risk_score": 68,
                "risk_factors": {"acpl_spike": 0.7, "time_uniformity": 0.3},
                "confidence_level": 90,
                "suspicious_games_count": 12
            }
        }


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
        schema_extra = {
            "example": {
                "username": "MagnusCarlsen",
                "games_analyzed": 350,
                "avg_acpl": 18.4,
                "std_acpl": 6.2,
                "avg_match_rate": 0.78,
                "std_match_rate": 0.05,
                "avg_ipr": 65.2,
                "roi_mean": 0.02,
                "roi_max": 0.08,
                "roi_std": 0.01,
                "step_function_detected": False,
                "step_function_magnitude": None,
                "peer_delta_acpl": -3.5,
                "peer_delta_match": 0.02,
                "longest_streak": 45,
                "first_game_date": "2020-01-05T00:00:00Z",
                "last_game_date": "2020-12-30T00:00:00Z",
                "selectivity_score": 0.24,
                "time_patterns": {
                    "mean_move_time": 12.5,
                    "time_variance": 30.2,
                    "uniformity_score": 0.87,
                    "clutch_accuracy_diff": -10.0
                },
                "opening_patterns": {
                    "mean_entropy": 2.1,
                    "novelty_depth": 9.5,
                    "opening_breadth": 40,
                    "second_choice_rate": 0.19
                },
                "trend_acpl": -1.4,
                "trend_match_rate": 0.03,
                "roi_curve": [0.01, -0.02, 0.03],
                "consistency_score": 0.88,
                "risk": {
                    "risk_score": 68,
                    "risk_factors": {"acpl_spike": 0.7},
                    "confidence_level": 90,
                    "suspicious_games_count": 12
                },
                "favorite_openings": [
                    {"eco_code": "C65", "name": "Ruy Lopez, Berlin Defense", "count": 40}
                ],
                "performance": {
                    "trend_acpl": -1.4,
                    "trend_match_rate": 0.03,
                    "roi_curve": [0.01, 0.02, -0.01]
                },
                "phase_quality": {
                    "opening_acpl": 15.0,
                    "middlegame_acpl": 19.0,
                    "endgame_acpl": 12.0
                },
                "benchmark": {
                    "percentile_acpl": 95,
                    "percentile_entropy": 80
                },
                "tactical": {
                    "precision_burst_count": 7,
                    "second_choice_rate": 0.18
                },
                "endgame": {
                    "conversion_efficiency": 32,
                    "tb_match_rate": 0.76,
                    "dtz_deviation": 1.4
                },
                "time_management": {
                    "mean_move_time": 12.5,
                    "time_variance": 30.2,
                    "uniformity_score": 0.87,
                    "lag_spike_count": 3
                },
                "clutch_accuracy": {
                    "avg_clutch_diff": 55.2,
                    "clutch_games_pct": 0.42
                },
                "analyzed_at": "2020-12-31T00:00:00Z"
            }
        }


# ── New request / response models for API validation ─────────────────────────
class AnalyzeGameIn(BaseModel):
    """Request payload for /analyze endpoint."""

    pgn: str = Field(..., description="PGN text of the game to analyze")
    move_times: Optional[List[int]] = Field(
        None, description="Optional list of move times in milliseconds"
    )

    class Config:
        schema_extra = {
            "example": {
                "pgn": "[Event \"Friendly\"]\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6",
                "move_times": [1234, 2100, 1800, 1500]
            }
        }


class TaskQueuedOut(BaseModel):
    """Standard response when a background task has been queued."""

    game_id: Optional[int] = Field(None, description="Identifier of the game, if applicable")
    task_id: str = Field(..., description="Celery task identifier")
    status: Literal["queued"]

    class Config:
        schema_extra = {
            "example": {
                "game_id": 42,
                "task_id": "9d0c1d6a-865b-4c0f-a9d0-1c2b3a4d5e6f",
                "status": "queued"
            }
        }


class MoveOut(BaseModel):
    move_number: int
    played: str
    best: Optional[str] = None
    best_rank: Optional[int] = None
    cp_loss: Optional[int] = None

    class Config:
        schema_extra = {
            "example": {
                "move_number": 1,
                "played": "e4",
                "best": "e4",
                "best_rank": 1,
                "cp_loss": 0
            }
        }


class GameOut(BaseModel):
    id: int
    created_at: datetime
    pgn: str
    white_username: Optional[str] = None
    black_username: Optional[str] = None
    eco_code: Optional[str] = None
    opening_key: Optional[str] = None
    moves: List[MoveOut] | None = None

    class Config:
        orm_mode = True
        schema_extra = {
            "example": {
                "id": 42,
                "created_at": "2024-01-01T12:00:00Z",
                "pgn": "[Event \"Friendly\"]\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6",
                "white_username": "WhitePlayer",
                "black_username": "BlackPlayer",
                "eco_code": "C50",
                "opening_key": "taylor_opening",
                "moves": [
                    {
                        "move_number": 1,
                        "played": "e4",
                        "best": "e4",
                        "best_rank": 1,
                        "cp_loss": 0
                    }
                ]
            }
        }


# ── Nuevos modelos para documentación de endpoints analysis ──────────────


class GameMetricsOut(BaseModel):
    game_id: int
    acpl: float
    match_rate: float
    weighted_match_rate: float
    ipr: float
    ipr_z_score: float
    precision_burst_count: int
    mean_move_time: float
    time_variance: float
    time_complexity_corr: float
    suspicious_quality: float
    suspicious_timing: float
    suspicious_opening: float
    overall_suspicion_score: float
    analyzed_at: datetime

    class Config:
        orm_mode = True
        schema_extra = {
            "example": {
                "game_id": 42,
                "acpl": 18.2,
                "match_rate": 0.78,
                "weighted_match_rate": 0.81,
                "ipr": 65.3,
                "ipr_z_score": 1.2,
                "precision_burst_count": 3,
                "mean_move_time": 12.5,
                "time_variance": 30.2,
                "time_complexity_corr": 0.45,
                "suspicious_quality": 0.1,
                "suspicious_timing": 0.2,
                "suspicious_opening": 0.05,
                "overall_suspicion_score": 0.12,
                "analyzed_at": "2024-06-28T15:00:00Z"
            }
        }


class PlayerMetricsSummaryOut(BaseModel):
    username: str
    games_analyzed: int
    avg_acpl: float
    std_acpl: float
    avg_match_rate: float
    std_match_rate: float
    avg_ipr: float
    roi_mean: float | None = None
    roi_max: float | None = None
    roi_std: float | None = None
    step_function_detected: bool
    step_function_magnitude: float | None = None
    peer_delta_acpl: float
    peer_delta_match: float
    longest_streak: int
    selectivity_score: float

    class Config:
        orm_mode = True
        schema_extra = {
            "example": {
                "username": "MagnusCarlsen",
                "games_analyzed": 350,
                "avg_acpl": 18.4,
                "std_acpl": 6.2,
                "avg_match_rate": 0.78,
                "std_match_rate": 0.05,
                "avg_ipr": 65.2,
                "roi_mean": 0.02,
                "roi_max": 0.08,
                "roi_std": 0.01,
                "step_function_detected": False,
                "step_function_magnitude": None,
                "peer_delta_acpl": -3.5,
                "peer_delta_match": 0.02,
                "longest_streak": 45,
                "selectivity_score": 0.24
            }
        }


class GameAnalysisStatusOut(BaseModel):
    game_id: int
    task_id: str
    status: str
    result: Optional[dict] | None = None

    class Config:
        schema_extra = {
            "example": {
                "game_id": 42,
                "task_id": "9d0c1d6a-865b-4c0f-a9d0-1c2b3a4d5e6f",
                "status": "STARTED",
                "result": None
            }
        }


class GameAnalysisCancelOut(BaseModel):
    status: str
    game_id: int
    task_id: str

    class Config:
        schema_extra = {
            "example": {
                "status": "cancelled",
                "game_id": 42,
                "task_id": "9d0c1d6a-865b-4c0f-a9d0-1c2b3a4d5e6f"
            }
        }


# ── Modelos para endpoints restantes ──────────────────────────────────────


class HealthOut(BaseModel):
    status: str
    version: str
    timestamp: datetime
    database: str
    services: dict[str, bool]

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "version": "v1",
                "timestamp": "2024-06-28T15:30:00Z",
                "database": "connected",
                "services": {"database": True, "celery": True, "redis": True}
            }
        }


class PlayerStatusOut(BaseModel):
    username: str
    status: str
    progress: int
    total_games: Optional[int] = None
    done_games: Optional[int] = None
    requested_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error: Optional[str] = None
    last_task_id: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "username": "MagnusCarlsen",
                "status": "ready",
                "progress": 100,
                "total_games": 350,
                "done_games": 350,
                "requested_at": "2024-06-20T10:00:00Z",
                "finished_at": "2024-06-21T12:00:00Z",
                "last_task_id": "abcd-efgh",
            }
        }


class PlayerAnalyzeOut(BaseModel):
    status: str  # queued | already_processing
    username: str
    task_id: str
    progress: Optional[int] = None

    class Config:
        schema_extra = {
            "example": {
                "status": "queued",
                "username": "MagnusCarlsen",
                "task_id": "abcd-efgh",
                "progress": None
            }
        }


class PlayerDeleteOut(BaseModel):
    status: str
    username: str

    class Config:
        schema_extra = {
            "example": {
                "status": "deleted",
                "username": "MagnusCarlsen"
            }
        }


class PlayerListItemOut(BaseModel):
    username: str
    status: str
    progress: int
    total_games: Optional[int] = None
    done_games: Optional[int] = None
    requested_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    class Config:
        schema_extra = {
            "example": {
                "username": "Hikaru",
                "status": "pending",
                "progress": 45,
                "total_games": 200,
                "done_games": 90,
                "requested_at": "2024-06-27T09:00:00Z",
                "finished_at": None
            }
        }


class TaskStatusOut(BaseModel):
    task_id: str
    state: str
    status: str
    progress: Optional[int] = None

    class Config:
        schema_extra = {
            "example": {
                "task_id": "abcd-1234",
                "state": "STARTED",
                "status": "Task is in progress",
                "progress": 30
            }
        }


class TaskCancelOut(BaseModel):
    task_id: str
    status: str
    message: str

    class Config:
        schema_extra = {
            "example": {
                "task_id": "abcd-1234",
                "status": "cancelled",
                "message": "Task has been cancelled successfully"
            }
        }


class TaskResultOut(BaseModel):
    task_id: str
    status: str
    state: str
    result: Optional[dict] | None = None
    error: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "task_id": "abcd-1234",
                "status": "success",
                "state": "SUCCESS",
                "result": {"foo": "bar"}
            }
        }


__all__ = [
    "PlayerMetricsOut",
    "TimePatternsOut",
    "OpeningPatternsOut",
    "RiskAssessmentOut",
    "AnalyzeGameIn",
    "TaskQueuedOut",
    "MoveOut",
    "GameOut",
    "GameMetricsOut",
    "PlayerMetricsSummaryOut",
    "GameAnalysisStatusOut",
    "GameAnalysisCancelOut",
    "HealthOut",
    "PlayerStatusOut",
    "PlayerAnalyzeOut",
    "PlayerDeleteOut",
    "PlayerListItemOut",
    "TaskStatusOut",
    "TaskCancelOut",
    "TaskResultOut",
]
