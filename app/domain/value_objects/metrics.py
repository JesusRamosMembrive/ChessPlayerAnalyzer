"""
Value Objects para métricas de análisis de ajedrez.
Estos objetos reemplazan los campos JSON no tipados del modelo actual.
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass(frozen=True)
class QualityMetrics:
    """Métricas de calidad de juego."""
    avg_acpl: float
    avg_wdl_loss: float
    robust_loss: float
    avg_match_rate: float
    avg_ipr: float
    std_acpl: Optional[float] = None
    std_match_rate: Optional[float] = None

    def __post_init__(self):
        """Validaciones."""
        if self.avg_acpl < 0:
            raise ValueError("avg_acpl debe ser >= 0")
        if not 0 <= self.avg_match_rate <= 1:
            raise ValueError("avg_match_rate debe estar entre 0 y 1")


@dataclass(frozen=True)
class TimingMetrics:
    """Métricas de gestión del tiempo."""
    mean_move_time: float
    time_variance: float
    uniformity_score: float
    lag_spike_count: int

    def __post_init__(self):
        """Validaciones."""
        if self.mean_move_time < 0:
            raise ValueError("mean_move_time debe ser >= 0")
        if self.lag_spike_count < 0:
            raise ValueError("lag_spike_count debe ser >= 0")


@dataclass(frozen=True)
class OpeningMetrics:
    """Métricas de repertorio de aperturas."""
    mean_entropy: float
    novelty_depth: Optional[float]
    opening_breadth: int
    second_choice_rate: float

    def __post_init__(self):
        """Validaciones."""
        if self.opening_breadth < 0:
            raise ValueError("opening_breadth debe ser >= 0")
        if not 0 <= self.second_choice_rate <= 1:
            raise ValueError("second_choice_rate debe estar entre 0 y 1")


@dataclass(frozen=True)
class EndgameMetrics:
    """Métricas de finales."""
    conversion_efficiency: Optional[int]
    tb_match_rate: Optional[float]
    dtz_deviation: Optional[float]

    def __post_init__(self):
        """Validaciones."""
        if self.tb_match_rate is not None and not 0 <= self.tb_match_rate <= 1:
            raise ValueError("tb_match_rate debe estar entre 0 y 1")


@dataclass(frozen=True)
class RiskAssessment:
    """Evaluación de riesgo de trampas."""
    risk_score: int
    risk_factors: Dict[str, bool]
    confidence_level: int

    def __post_init__(self):
        """Validaciones."""
        if not 0 <= self.risk_score <= 100:
            raise ValueError("risk_score debe estar entre 0 y 100")
        if not 0 <= self.confidence_level <= 100:
            raise ValueError("confidence_level debe estar entre 0 y 100")


@dataclass(frozen=True)
class PerformanceMetrics:
    """Métricas de rendimiento longitudinal."""
    trend_acpl: Optional[float]
    trend_match_rate: Optional[float]
    roi_curve: List[float]
    roi_mean: Optional[float] = None
    roi_max: Optional[float] = None
    roi_std: Optional[float] = None
    step_function_detected: bool = False
    step_function_magnitude: Optional[float] = None
    peer_delta_acpl: float = 0.0
    peer_delta_match: float = 0.0
    longest_streak: int = 0
    selectivity_score: float = 0.0


@dataclass(frozen=True)
class PhaseQuality:
    """Calidad de juego por fase."""
    opening_acpl: float
    middlegame_acpl: float
    endgame_acpl: float
    opening_blunder_rate: float
    middlegame_blunder_rate: float
    endgame_blunder_rate: float
    blunder_rate: float

    def __post_init__(self):
        """Validaciones."""
        for rate in [self.opening_blunder_rate, self.middlegame_blunder_rate,
                    self.endgame_blunder_rate, self.blunder_rate]:
            if not 0 <= rate <= 1:
                raise ValueError(f"blunder_rate debe estar entre 0 y 1, got {rate}")


@dataclass(frozen=True)
class ClutchAccuracy:
    """Precisión bajo presión."""
    avg_clutch_diff: Optional[float]
    clutch_games_pct: float

    def __post_init__(self):
        """Validaciones."""
        if not 0 <= self.clutch_games_pct <= 1:
            raise ValueError("clutch_games_pct debe estar entre 0 y 1")


@dataclass(frozen=True)
class TimeComplexity:
    """Correlación tiempo-complejidad."""
    time_complexity_corr: Optional[float]

    def __post_init__(self):
        """Validaciones."""
        if (self.time_complexity_corr is not None and
            not -1 <= self.time_complexity_corr <= 1):
            raise ValueError("time_complexity_corr debe estar entre -1 y 1")


@dataclass(frozen=True)
class Benchmark:
    """Métricas de benchmark comparativo."""
    percentile_acpl: int
    percentile_entropy: int

    def __post_init__(self):
        """Validaciones."""
        for percentile in [self.percentile_acpl, self.percentile_entropy]:
            if not 0 <= percentile <= 100:
                raise ValueError(f"percentile debe estar entre 0 y 100, got {percentile}")


@dataclass(frozen=True)
class Segments:
    """Segmento de análisis temporal."""
    start: int
    end: int
    mean_acpl: float
    mean_time: Optional[float] = None

    def __post_init__(self):
        """Validaciones."""
        if self.start < 0 or self.end < 0:
            raise ValueError("start y end deben ser >= 0")
        if self.start >= self.end:
            raise ValueError("start debe ser < end")


# Funciones de conversión para migrar desde JSON
def quality_metrics_from_dict(data: Dict[str, Any]) -> QualityMetrics:
    """Convierte dict JSON a QualityMetrics."""
    return QualityMetrics(
        avg_acpl=data.get("avg_acpl", 0.0),
        avg_wdl_loss=data.get("avg_wdl_loss", 0.0),
        robust_loss=data.get("robust_loss", 0.0),
        avg_match_rate=data.get("avg_match_rate", 0.0),
        avg_ipr=data.get("avg_ipr", 0.0),
        std_acpl=data.get("std_acpl"),
        std_match_rate=data.get("std_match_rate")
    )


def timing_metrics_from_dict(data: Dict[str, Any]) -> TimingMetrics:
    """Convierte dict JSON a TimingMetrics."""
    return TimingMetrics(
        mean_move_time=data.get("mean_move_time", 0.0),
        time_variance=data.get("time_variance", 0.0),
        uniformity_score=data.get("uniformity_score", 0.0),
        lag_spike_count=data.get("lag_spike_count", 0)
    )


def opening_metrics_from_dict(data: Dict[str, Any]) -> OpeningMetrics:
    """Convierte dict JSON a OpeningMetrics."""
    return OpeningMetrics(
        mean_entropy=data.get("mean_entropy", 0.0),
        novelty_depth=data.get("novelty_depth"),
        opening_breadth=data.get("opening_breadth", 0),
        second_choice_rate=data.get("second_choice_rate", 0.0)
    )