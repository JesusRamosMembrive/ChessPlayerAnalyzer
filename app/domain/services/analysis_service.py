"""
Servicio de dominio para análisis de ajedrez.
Contiene la lógica pura sin dependencias de infraestructura.
"""
from typing import List, Optional, Protocol
from datetime import datetime

from ..entities.game import Game, MoveData
from ..entities.analysis import GameAnalysis, PlayerAnalysis
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


class ChessEngine(Protocol):
    """Protocol for chess engine implementations."""

    async def analyze_moves(self, pgn_data: str, move_times: Optional[List[float]] = None):
        """Analyze moves and return engine results."""
        ...


class AnalysisService:
    """
    Servicio de dominio para análisis de partidas y jugadores.
    Usa inyección de dependencias para engine de ajedrez.
    """

    def __init__(self, chess_engine: ChessEngine):
        self._engine = chess_engine

    async def analyze_game(self, game: Game, moves_data: Optional[List[MoveData]] = None) -> GameAnalysis:
        """
        Analiza una partida individual usando engine de ajedrez.

        Args:
            game: Partida a analizar
            moves_data: Datos de movimientos (opcional, se extraen del PGN si no se proveen)

        Returns:
            GameAnalysis con métricas calculadas
        """
        if not game.pgn_data:
            raise ValueError("Game must have PGN data for analysis")

        # Si no hay moves_data, analizar con engine
        if not moves_data:
            engine_result = await self._engine.analyze_moves(game.pgn_data, game.move_times)
            moves_data = self._convert_engine_result_to_moves_data(engine_result)

        if not moves_data:
            raise ValueError("No moves data available after analysis")

        # Calcular métricas de calidad
        quality_metrics = self._calculate_quality_metrics(moves_data)

        # Calcular métricas de timing
        timing_metrics = self._calculate_timing_metrics(moves_data, game.move_times)

        # Calcular métricas de apertura
        opening_metrics = self._calculate_opening_metrics(moves_data, getattr(game, 'eco_code', None))

        # Calcular métricas de final
        endgame_metrics = self._calculate_endgame_metrics(moves_data)

        # Detectar patrones sospechosos
        suspicious_quality = self._detect_suspicious_quality(quality_metrics)
        suspicious_timing = self._detect_suspicious_timing(timing_metrics)
        suspicious_opening = self._detect_suspicious_opening(opening_metrics)

        # Calcular score de sospecha general
        overall_suspicion_score = self._calculate_suspicion_score(
            quality_metrics, timing_metrics, opening_metrics
        )

        return GameAnalysis(
            game_id=game.id,
            quality_metrics=quality_metrics,
            timing_metrics=timing_metrics,
            opening_metrics=opening_metrics,
            endgame_metrics=endgame_metrics,
            suspicious_quality=suspicious_quality,
            suspicious_timing=suspicious_timing,
            suspicious_opening=suspicious_opening,
            overall_suspicion_score=overall_suspicion_score,
            analyzed_at=datetime.utcnow()
        )

    def analyze_player(self, username: str, game_analyses: List[GameAnalysis],
                      games: List[Game]) -> PlayerAnalysis:
        """
        Analiza un jugador basado en sus análisis de partidas.

        Args:
            username: Nombre del jugador
            game_analyses: Lista de análisis de partidas
            games: Lista de partidas del jugador

        Returns:
            PlayerAnalysis con métricas agregadas
        """
        if not game_analyses:
            raise ValueError("No game analyses provided for player analysis")

        # Métricas agregadas de calidad
        quality_metrics = self._aggregate_quality_metrics(game_analyses)

        # Métricas agregadas de timing
        timing_metrics = self._aggregate_timing_metrics(game_analyses)

        # Métricas agregadas de apertura
        opening_metrics = self._aggregate_opening_metrics(game_analyses)

        # Métricas agregadas de final
        endgame_metrics = self._aggregate_endgame_metrics(game_analyses)

        # Análisis longitudinal (performance a lo largo del tiempo)
        performance_metrics = self._calculate_performance_metrics(game_analyses, games)

        # Análisis por fases del juego
        phase_quality = self._calculate_phase_quality(game_analyses)

        # Análisis de precisión bajo presión
        clutch_accuracy = self._calculate_clutch_accuracy(game_analyses, games)

        # Correlación tiempo-complejidad
        time_complexity = self._calculate_time_complexity(game_analyses)

        # Benchmark comparativo
        benchmark = self._calculate_benchmark(quality_metrics, opening_metrics)

        # Evaluación de riesgo
        risk_assessment = self._calculate_risk_assessment(
            quality_metrics, timing_metrics, opening_metrics, performance_metrics
        )

        # Detectar partidas sospechosas
        suspicious_games_ids = [
            analysis.game_id for analysis in game_analyses
            if analysis.is_suspicious()
        ]

        # Calcular segmentos temporales
        segments = self._calculate_segments(game_analyses, games)

        # Detectar puntos de cambio en rendimiento
        change_points = self._detect_change_points(game_analyses)

        return PlayerAnalysis(
            username=username,
            games_analyzed=len(game_analyses),
            quality_metrics=quality_metrics,
            timing_metrics=timing_metrics,
            opening_metrics=opening_metrics,
            endgame_metrics=endgame_metrics,
            performance_metrics=performance_metrics,
            phase_quality=phase_quality,
            clutch_accuracy=clutch_accuracy,
            time_complexity=time_complexity,
            benchmark=benchmark,
            risk_assessment=risk_assessment,
            first_game_date=min(g.created_at for g in games if g.created_at),
            last_game_date=max(g.created_at for g in games if g.created_at),
            analyzed_at=datetime.utcnow(),
            suspicious_games_ids=suspicious_games_ids,
            segments=segments,
            change_points=change_points
        )

    # ===== MÉTODOS PRIVADOS PARA CÁLCULOS =====

    def _calculate_quality_metrics(self, moves_data: List[MoveData]) -> QualityMetrics:
        """Calcula métricas de calidad de movimientos."""
        if not moves_data:
            return QualityMetrics(0.0, 0.0, 0.0, 0.0, 0.0)

        # ACPL (Average Centipawn Loss)
        cp_losses = [move.cp_loss for move in moves_data if move.cp_loss < 500]  # Filter outliers
        avg_acpl = sum(cp_losses) / len(cp_losses) if cp_losses else 0.0

        # Match rate (coincidencia con mejor jugada)
        matches = sum(1 for move in moves_data if move.played == move.best)
        match_rate = matches / len(moves_data) if moves_data else 0.0

        # WDL loss - aproximación basada en CP loss
        avg_wdl_loss = min(avg_acpl / 100.0, 1.0)  # Conversión aproximada

        # Robust loss (filtrar outliers extremos)
        filtered_losses = [cp for cp in cp_losses if cp < 200]
        robust_loss = sum(filtered_losses) / len(filtered_losses) if filtered_losses else 0.0

        # IPR (Intrinsic Performance Rating) - aproximación
        avg_ipr = max(800, 2800 - (avg_acpl * 20))  # Fórmula aproximada

        return QualityMetrics(
            avg_acpl=avg_acpl,
            avg_wdl_loss=avg_wdl_loss,
            robust_loss=robust_loss,
            avg_match_rate=match_rate,
            avg_ipr=avg_ipr
        )

    def _calculate_timing_metrics(self, moves_data: List[MoveData],
                                 move_times: Optional[List[int]]) -> TimingMetrics:
        """Calcula métricas de gestión del tiempo."""
        if not move_times:
            return TimingMetrics(0.0, 0.0, 0.0, 0)

        # Tiempo medio por movimiento (convertir de milisegundos a segundos)
        times_seconds = [t / 1000.0 for t in move_times if t > 0]
        mean_time = sum(times_seconds) / len(times_seconds) if times_seconds else 0.0

        # Varianza del tiempo
        if len(times_seconds) > 1:
            variance = sum((t - mean_time) ** 2 for t in times_seconds) / (len(times_seconds) - 1)
        else:
            variance = 0.0

        # Score de uniformidad (inverso del coeficiente de variación)
        cv = (variance ** 0.5) / mean_time if mean_time > 0 else 0
        uniformity_score = max(0, 1 - cv)

        # Contar spikes de lag (tiempos > 3 desviaciones estándar)
        if variance > 0:
            std_dev = variance ** 0.5
            threshold = mean_time + (3 * std_dev)
            lag_spike_count = sum(1 for t in times_seconds if t > threshold)
        else:
            lag_spike_count = 0

        return TimingMetrics(
            mean_move_time=mean_time,
            time_variance=variance,
            uniformity_score=uniformity_score,
            lag_spike_count=lag_spike_count
        )

    def _calculate_opening_metrics(self, moves_data: List[MoveData],
                                  eco_code: Optional[str]) -> OpeningMetrics:
        """Calcula métricas de repertorio de aperturas."""
        opening_moves = moves_data[:10]  # Primeros 10 movimientos

        # Entropía simple basada en variedad de jugadas
        unique_moves = set(move.played for move in opening_moves)
        entropy = len(unique_moves) / len(opening_moves) if opening_moves else 0.0

        # Profundidad de novedad (movimientos hasta primera desviación de libro)
        novelty_depth = None
        for i, move in enumerate(opening_moves):
            if move.played != move.best:  # Simplificación
                novelty_depth = float(i + 1)
                break

        # Amplitud del repertorio
        opening_breadth = len(unique_moves)

        # Tasa de segunda mejor jugada
        second_choice_count = sum(1 for move in opening_moves
                                if move.best_rank == 2) if hasattr(opening_moves[0], 'best_rank') else 0
        second_choice_rate = second_choice_count / len(opening_moves) if opening_moves else 0.0

        return OpeningMetrics(
            mean_entropy=entropy,
            novelty_depth=novelty_depth,
            opening_breadth=opening_breadth,
            second_choice_rate=second_choice_rate
        )

    def _calculate_endgame_metrics(self, moves_data: List[MoveData]) -> EndgameMetrics:
        """Calcula métricas de finales."""
        # Para este servicio básico, solo implementamos métricas simples
        # Las métricas avanzadas como tablebase se calcularían en servicios especializados

        endgame_moves = moves_data[-15:] if len(moves_data) > 15 else moves_data

        # Eficiencia de conversión (movimientos para ganar material)
        conversion_efficiency = len(endgame_moves) if endgame_moves else None

        return EndgameMetrics(
            conversion_efficiency=conversion_efficiency,
            tb_match_rate=None,  # Requiere acceso a tablebases
            dtz_deviation=None   # Requiere acceso a tablebases
        )

    def _detect_suspicious_quality(self, quality_metrics: QualityMetrics) -> bool:
        """Detecta patrones sospechosos en calidad."""
        return (quality_metrics.avg_acpl < 5.0 and
                quality_metrics.avg_match_rate > 0.95)

    def _detect_suspicious_timing(self, timing_metrics: TimingMetrics) -> bool:
        """Detecta patrones sospechosos en timing."""
        return (timing_metrics.uniformity_score > 0.95 and
                timing_metrics.lag_spike_count == 0)

    def _detect_suspicious_opening(self, opening_metrics: OpeningMetrics) -> bool:
        """Detecta patrones sospechosos en aperturas."""
        return opening_metrics.mean_entropy < 0.1

    def _calculate_suspicion_score(self, quality: QualityMetrics,
                                  timing: TimingMetrics,
                                  opening: OpeningMetrics) -> float:
        """Calcula score general de sospecha (0-100)."""
        score = 0.0

        # Factores de calidad
        if quality.avg_acpl < 10.0:
            score += 30
        if quality.avg_match_rate > 0.90:
            score += 25

        # Factores de timing
        if timing.uniformity_score > 0.90:
            score += 20
        if timing.lag_spike_count == 0:
            score += 15

        # Factores de apertura
        if opening.mean_entropy < 0.2:
            score += 10

        return min(100.0, score)

    def _aggregate_quality_metrics(self, analyses: List[GameAnalysis]) -> QualityMetrics:
        """Agrega métricas de calidad de múltiples partidas."""
        acpls = [a.quality_metrics.avg_acpl for a in analyses]
        match_rates = [a.quality_metrics.avg_match_rate for a in analyses]

        return QualityMetrics(
            avg_acpl=sum(acpls) / len(acpls),
            avg_wdl_loss=sum(a.quality_metrics.avg_wdl_loss for a in analyses) / len(analyses),
            robust_loss=sum(a.quality_metrics.robust_loss for a in analyses) / len(analyses),
            avg_match_rate=sum(match_rates) / len(match_rates),
            avg_ipr=sum(a.quality_metrics.avg_ipr for a in analyses) / len(analyses),
            std_acpl=self._calculate_std(acpls),
            std_match_rate=self._calculate_std(match_rates)
        )

    def _aggregate_timing_metrics(self, analyses: List[GameAnalysis]) -> TimingMetrics:
        """Agrega métricas de timing de múltiples partidas."""
        times = [a.timing_metrics.mean_move_time for a in analyses]
        variances = [a.timing_metrics.time_variance for a in analyses]

        return TimingMetrics(
            mean_move_time=sum(times) / len(times),
            time_variance=sum(variances) / len(variances),
            uniformity_score=sum(a.timing_metrics.uniformity_score for a in analyses) / len(analyses),
            lag_spike_count=sum(a.timing_metrics.lag_spike_count for a in analyses)
        )

    def _aggregate_opening_metrics(self, analyses: List[GameAnalysis]) -> OpeningMetrics:
        """Agrega métricas de apertura de múltiples partidas."""
        entropies = [a.opening_metrics.mean_entropy for a in analyses]
        breadths = [a.opening_metrics.opening_breadth for a in analyses]

        return OpeningMetrics(
            mean_entropy=sum(entropies) / len(entropies),
            novelty_depth=None,  # Requiere análisis más avanzado
            opening_breadth=max(breadths) if breadths else 0,
            second_choice_rate=sum(a.opening_metrics.second_choice_rate for a in analyses) / len(analyses)
        )

    def _aggregate_endgame_metrics(self, analyses: List[GameAnalysis]) -> EndgameMetrics:
        """Agrega métricas de finales de múltiples partidas."""
        efficiencies = [a.endgame_metrics.conversion_efficiency
                       for a in analyses
                       if a.endgame_metrics.conversion_efficiency is not None]

        return EndgameMetrics(
            conversion_efficiency=int(sum(efficiencies) / len(efficiencies)) if efficiencies else None,
            tb_match_rate=None,
            dtz_deviation=None
        )

    def _calculate_performance_metrics(self, analyses: List[GameAnalysis],
                                     games: List[Game]) -> PerformanceMetrics:
        """Calcula métricas de rendimiento longitudinal."""
        acpls = [a.quality_metrics.avg_acpl for a in analyses]

        # Tendencia simple (pendiente)
        if len(acpls) > 1:
            trend_acpl = (acpls[-1] - acpls[0]) / len(acpls)
        else:
            trend_acpl = 0.0

        return PerformanceMetrics(
            trend_acpl=trend_acpl,
            trend_match_rate=None,
            roi_curve=acpls,  # Simplificado
            roi_mean=sum(acpls) / len(acpls) if acpls else None
        )

    def _calculate_phase_quality(self, analyses: List[GameAnalysis]) -> PhaseQuality:
        """Calcula calidad por fases del juego."""
        # Simplificación - en implementación real se analizarían las fases específicas
        avg_acpl = sum(a.quality_metrics.avg_acpl for a in analyses) / len(analyses)

        return PhaseQuality(
            opening_acpl=avg_acpl * 0.8,  # Suposición de mejor juego en apertura
            middlegame_acpl=avg_acpl * 1.2,  # Peor en medio juego
            endgame_acpl=avg_acpl * 0.9,  # Mejor en final
            opening_blunder_rate=0.01,  # Simplificado
            middlegame_blunder_rate=0.02,
            endgame_blunder_rate=0.015,
            blunder_rate=0.015
        )

    def _calculate_clutch_accuracy(self, analyses: List[GameAnalysis],
                                  games: List[Game]) -> ClutchAccuracy:
        """Calcula precisión bajo presión."""
        return ClutchAccuracy(
            avg_clutch_diff=None,  # Requiere análisis de presión temporal
            clutch_games_pct=0.5   # Simplificado
        )

    def _calculate_time_complexity(self, analyses: List[GameAnalysis]) -> TimeComplexity:
        """Calcula correlación tiempo-complejidad."""
        return TimeComplexity(
            time_complexity_corr=None  # Requiere análisis más avanzado
        )

    def _calculate_benchmark(self, quality: QualityMetrics,
                           opening: OpeningMetrics) -> Benchmark:
        """Calcula métricas de benchmark comparativo."""
        # Conversión aproximada a percentiles
        acpl_percentile = max(0, min(100, int((50 - quality.avg_acpl) * 2)))
        entropy_percentile = max(0, min(100, int(opening.mean_entropy * 100)))

        return Benchmark(
            percentile_acpl=acpl_percentile,
            percentile_entropy=entropy_percentile
        )

    def _calculate_risk_assessment(self, quality: QualityMetrics,
                                  timing: TimingMetrics,
                                  opening: OpeningMetrics,
                                  performance: PerformanceMetrics) -> RiskAssessment:
        """Calcula evaluación de riesgo."""
        risk_score = 0
        risk_factors = {}

        # Factor: ACPL muy bajo
        if quality.avg_acpl < 10.0:
            risk_score += 25
            risk_factors["low_acpl"] = True
        else:
            risk_factors["low_acpl"] = False

        # Factor: Match rate muy alto
        if quality.avg_match_rate > 0.85:
            risk_score += 20
            risk_factors["high_match_rate"] = True
        else:
            risk_factors["high_match_rate"] = False

        # Factor: Timing muy uniforme
        if timing.uniformity_score > 0.90:
            risk_score += 15
            risk_factors["uniform_timing"] = True
        else:
            risk_factors["uniform_timing"] = False

        # Factor: IPR muy alto para el nivel
        if quality.avg_ipr > 2500:
            risk_score += 20
            risk_factors["high_ipr"] = True
        else:
            risk_factors["high_ipr"] = False

        # Nivel de confianza basado en cantidad de datos
        confidence_level = min(100, len(performance.roi_curve) * 5)

        return RiskAssessment(
            risk_score=min(100, risk_score),
            risk_factors=risk_factors,
            confidence_level=confidence_level
        )

    def _calculate_segments(self, analyses: List[GameAnalysis],
                           games: List[Game]) -> List[Segments]:
        """Calcula segmentos temporales de análisis."""
        # Simplificación - un solo segmento
        acpls = [a.quality_metrics.avg_acpl for a in analyses]
        avg_acpl = sum(acpls) / len(acpls) if acpls else 0.0

        return [Segments(
            start=0,
            end=len(analyses),
            mean_acpl=avg_acpl,
            mean_time=None
        )]

    def _detect_change_points(self, analyses: List[GameAnalysis]) -> List[int]:
        """Detecta puntos de cambio en el rendimiento."""
        # Implementación simplificada
        return []

    def _calculate_std(self, values: List[float]) -> Optional[float]:
        """Calcula desviación estándar."""
        if len(values) < 2:
            return None

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5

    def _convert_engine_result_to_moves_data(self, engine_result) -> List[MoveData]:
        """Convierte resultado del engine a formato MoveData."""
        moves_data = []

        for move_analysis in engine_result.moves:
            move_data = MoveData(
                move=move_analysis.move,
                evaluation=move_analysis.evaluation,
                centipawn_loss=move_analysis.centipawn_loss,
                wdl_loss=move_analysis.wdl_loss,
                is_best=move_analysis.is_best,
                match_rate=move_analysis.match_rate
            )
            moves_data.append(move_data)

        return moves_data