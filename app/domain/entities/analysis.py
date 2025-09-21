"""
Entidades de análisis del dominio.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from ..value_objects.metrics import (
    QualityMetrics,
    TimingMetrics,
    OpeningMetrics,
    EndgameMetrics,
    RiskAssessment,
    PerformanceMetrics,
    PhaseQuality,
    ClutchAccuracy,
    TimeComplexity,
    Benchmark,
    Segments
)


@dataclass
class GameAnalysis:
    """Análisis de una partida."""
    game_id: int
    quality_metrics: QualityMetrics
    timing_metrics: TimingMetrics
    opening_metrics: OpeningMetrics
    endgame_metrics: EndgameMetrics
    suspicious_quality: bool = False
    suspicious_timing: bool = False
    suspicious_opening: bool = False
    overall_suspicion_score: float = 0.0
    analyzed_at: Optional[datetime] = None

    def is_suspicious(self) -> bool:
        """Verifica si la partida es sospechosa."""
        return (self.suspicious_quality or
                self.suspicious_timing or
                self.suspicious_opening or
                self.overall_suspicion_score > 70)


@dataclass
class PlayerAnalysis:
    """Análisis completo de un jugador."""
    username: str
    games_analyzed: int
    quality_metrics: QualityMetrics
    timing_metrics: TimingMetrics
    opening_metrics: OpeningMetrics
    endgame_metrics: EndgameMetrics
    performance_metrics: PerformanceMetrics
    phase_quality: PhaseQuality
    clutch_accuracy: ClutchAccuracy
    time_complexity: TimeComplexity
    benchmark: Benchmark
    risk_assessment: RiskAssessment

    # Metadatos temporales
    first_game_date: Optional[datetime] = None
    last_game_date: Optional[datetime] = None
    analyzed_at: Optional[datetime] = None

    # Datos adicionales
    suspicious_games_ids: List[int] = None
    segments: List[Segments] = None
    change_points: List[int] = None
    cluster_id: int = -1
    cluster_distance: float = 0.0

    def __post_init__(self):
        """Inicialización post-creación."""
        if self.suspicious_games_ids is None:
            self.suspicious_games_ids = []
        if self.segments is None:
            self.segments = []
        if self.change_points is None:
            self.change_points = []

    def is_high_risk(self) -> bool:
        """Verifica si el jugador tiene alto riesgo."""
        return self.risk_assessment.risk_score >= 80

    def get_suspicious_games_count(self) -> int:
        """Obtiene el número de partidas sospechosas."""
        return len(self.suspicious_games_ids)

    def add_suspicious_game(self, game_id: int) -> None:
        """Añade una partida sospechosa."""
        if game_id not in self.suspicious_games_ids:
            self.suspicious_games_ids.append(game_id)

    def get_analysis_summary(self) -> dict:
        """Obtiene resumen del análisis para la UI."""
        return {
            "username": self.username,
            "games_analyzed": self.games_analyzed,
            "avg_acpl": self.quality_metrics.avg_acpl,
            "avg_wdl_loss": self.quality_metrics.avg_wdl_loss,
            "robust_loss": self.quality_metrics.robust_loss,
            "avg_match_rate": self.quality_metrics.avg_match_rate,
            "avg_ipr": self.quality_metrics.avg_ipr,
            "std_acpl": self.quality_metrics.std_acpl,
            "std_match_rate": self.quality_metrics.std_match_rate,
            "roi_mean": self.performance_metrics.roi_mean,
            "roi_max": self.performance_metrics.roi_max,
            "roi_std": self.performance_metrics.roi_std,
            "step_function_detected": self.performance_metrics.step_function_detected,
            "step_function_magnitude": self.performance_metrics.step_function_magnitude,
            "peer_delta_acpl": self.performance_metrics.peer_delta_acpl,
            "peer_delta_match": self.performance_metrics.peer_delta_match,
            "longest_streak": self.performance_metrics.longest_streak,
            "selectivity_score": self.performance_metrics.selectivity_score,
            "time_patterns": None,  # TODO: Implementar
            "opening_patterns": {
                "mean_entropy": self.opening_metrics.mean_entropy,
                "novelty_depth": self.opening_metrics.novelty_depth,
                "opening_breadth": self.opening_metrics.opening_breadth,
                "second_choice_rate": self.opening_metrics.second_choice_rate
            },
            "suspicious_games_ids": self.suspicious_games_ids,
            "performance": {
                "trend_acpl": self.performance_metrics.trend_acpl,
                "trend_match_rate": self.performance_metrics.trend_match_rate,
                "roi_curve": self.performance_metrics.roi_curve
            },
            "phase_quality": {
                "opening_acpl": self.phase_quality.opening_acpl,
                "middlegame_acpl": self.phase_quality.middlegame_acpl,
                "endgame_acpl": self.phase_quality.endgame_acpl,
                "opening_blunder_rate": self.phase_quality.opening_blunder_rate,
                "middlegame_blunder_rate": self.phase_quality.middlegame_blunder_rate,
                "endgame_blunder_rate": self.phase_quality.endgame_blunder_rate,
                "blunder_rate": self.phase_quality.blunder_rate
            },
            "benchmark": {
                "percentile_acpl": self.benchmark.percentile_acpl,
                "percentile_entropy": self.benchmark.percentile_entropy
            },
            "risk_score": self.risk_assessment.risk_score,
            "risk_factors": self.risk_assessment.risk_factors,
            "confidence_level": self.risk_assessment.confidence_level,
            "first_game_date": self.first_game_date.isoformat() if self.first_game_date else None,
            "last_game_date": self.last_game_date.isoformat() if self.last_game_date else None,
            "analyzed_at": self.analyzed_at.isoformat() if self.analyzed_at else None,
            "time_management": {
                "mean_move_time": self.timing_metrics.mean_move_time,
                "time_variance": self.timing_metrics.time_variance,
                "uniformity_score": self.timing_metrics.uniformity_score,
                "lag_spike_count": self.timing_metrics.lag_spike_count
            },
            "clutch_accuracy": {
                "avg_clutch_diff": self.clutch_accuracy.avg_clutch_diff,
                "clutch_games_pct": self.clutch_accuracy.clutch_games_pct
            },
            "tactical": {},  # TODO: Implementar
            "endgame": {
                "conversion_efficiency": self.endgame_metrics.conversion_efficiency,
                "tb_match_rate": self.endgame_metrics.tb_match_rate,
                "dtz_deviation": self.endgame_metrics.dtz_deviation
            },
            "time_complexity": {
                "time_complexity_corr": self.time_complexity.time_complexity_corr
            },
            "segments": [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "mean_acpl": seg.mean_acpl,
                    "mean_time": seg.mean_time
                }
                for seg in self.segments
            ],
            "change_points": self.change_points,
            "cluster_id": self.cluster_id,
            "cluster_distance": self.cluster_distance
        }