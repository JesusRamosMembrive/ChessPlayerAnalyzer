"""
Mappers entre modelos SQLModel y entidades de dominio.
Bridge layer para mantener separación entre infrastructure y domain.
"""
from typing import Optional, List
from datetime import datetime

# Domain imports
from app.domain.entities.player import Player as DomainPlayer, PlayerStatus
from app.domain.entities.game import Game as DomainGame, MoveData
from app.domain.entities.analysis import PlayerAnalysis as DomainPlayerAnalysis, GameAnalysis as DomainGameAnalysis
from app.domain.value_objects.metrics import (
    QualityMetrics, TimingMetrics, OpeningMetrics, EndgameMetrics,
    RiskAssessment, PerformanceMetrics, PhaseQuality, ClutchAccuracy,
    TimeComplexity, Benchmark, Segments
)

# SQLModel imports (current models)
from app.models import Player as SQLPlayer, Game as SQLGame
from app.models import PlayerAnalysisDetailed as SQLPlayerAnalysis
from app.models import GameAnalysisDetailed as SQLGameAnalysis, MoveAnalysis as SQLMoveAnalysis


# ===== PLAYER MAPPERS =====

def player_to_domain(sql_player: SQLPlayer) -> DomainPlayer:
    """Convierte SQLPlayer a DomainPlayer."""
    return DomainPlayer(
        username=sql_player.username,
        status=PlayerStatus(sql_player.status.value),
        requested_at=sql_player.requested_at,
        finished_at=sql_player.finished_at,
        progress=sql_player.progress,
        total_games=sql_player.total_games,
        done_games=sql_player.done_games,
        error=sql_player.error,
        last_task_id=sql_player.last_task_id
    )


def domain_to_player(domain_player: DomainPlayer) -> SQLPlayer:
    """Convierte DomainPlayer a SQLPlayer."""
    return SQLPlayer(
        username=domain_player.username,
        status=domain_player.status,
        requested_at=domain_player.requested_at,
        finished_at=domain_player.finished_at,
        progress=domain_player.progress,
        total_games=domain_player.total_games,
        done_games=domain_player.done_games,
        error=domain_player.error,
        last_task_id=domain_player.last_task_id
    )


# ===== GAME MAPPERS =====

def game_to_domain(sql_game: SQLGame, include_moves: bool = False) -> DomainGame:
    """Convierte SQLGame a DomainGame."""
    moves_data = None
    if include_moves and sql_game.moves:
        moves_data = [
            MoveData(
                move_number=move.move_number,
                played=move.played,
                best=move.best,
                cp_loss=move.cp_loss,
                eval_before=move.eval_before,
                eval_after=move.eval_after,
                time_spent=move.time_spent,
                best_rank=getattr(move, "best_rank", None),
                match_rate=None
            )
            for move in sql_game.moves
        ]

    return DomainGame(
        id=sql_game.id,
        pgn=sql_game.pgn,
        white_username=sql_game.white_username,
        black_username=sql_game.black_username,
        white_elo=sql_game.white_elo,
        black_elo=sql_game.black_elo,
        time_control=sql_game.time_control,
        termination=sql_game.termination,
        eco_code=sql_game.eco_code,
        opening_key=sql_game.opening_key,
        move_times=sql_game.move_times,
        moves=moves_data,
        created_at=sql_game.created_at
    )


def domain_to_game(domain_game: DomainGame) -> SQLGame:
    """Convierte DomainGame a SQLGame."""
    return SQLGame(
        id=domain_game.id,
        pgn=domain_game.pgn,
        white_username=domain_game.white_username,
        black_username=domain_game.black_username,
        white_elo=domain_game.white_elo,
        black_elo=domain_game.black_elo,
        time_control=domain_game.time_control,
        termination=domain_game.termination,
        eco_code=domain_game.eco_code,
        opening_key=domain_game.opening_key,
        move_times=domain_game.move_times,
        created_at=domain_game.created_at or datetime.utcnow()
    )


# ===== ANALYSIS MAPPERS =====

def analysis_to_domain(sql_analysis: SQLPlayerAnalysis) -> DomainPlayerAnalysis:
    """Convierte SQLPlayerAnalysis a DomainPlayerAnalysis."""

    # Convertir métricas básicas
    quality_metrics = QualityMetrics(
        avg_acpl=sql_analysis.avg_acpl,
        avg_wdl_loss=sql_analysis.avg_wdl_loss,
        robust_loss=sql_analysis.robust_loss,
        avg_match_rate=sql_analysis.avg_match_rate,
        avg_ipr=sql_analysis.avg_ipr,
        std_acpl=sql_analysis.std_acpl,
        std_match_rate=sql_analysis.std_match_rate
    )

    # Extraer timing metrics del JSON
    time_mgmt = sql_analysis.time_management or {}
    timing_metrics = TimingMetrics(
        mean_move_time=time_mgmt.get("mean_move_time", 0.0),
        time_variance=time_mgmt.get("time_variance", 0.0),
        uniformity_score=time_mgmt.get("uniformity_score", 0.0),
        lag_spike_count=time_mgmt.get("lag_spike_count", 0)
    )

    # Extraer opening metrics del JSON
    opening_patterns = sql_analysis.opening_patterns or {}
    opening_metrics = OpeningMetrics(
        mean_entropy=opening_patterns.get("mean_entropy", 0.0),
        novelty_depth=opening_patterns.get("novelty_depth"),
        opening_breadth=opening_patterns.get("opening_breadth", 0),
        second_choice_rate=opening_patterns.get("second_choice_rate", 0.0)
    )

    # Extraer endgame metrics del JSON
    endgame_data = sql_analysis.endgame or {}
    endgame_metrics = EndgameMetrics(
        conversion_efficiency=endgame_data.get("conversion_efficiency"),
        tb_match_rate=endgame_data.get("tb_match_rate"),
        dtz_deviation=endgame_data.get("dtz_deviation")
    )

    # Performance metrics del JSON
    performance_data = sql_analysis.performance or {}
    performance_metrics = PerformanceMetrics(
        trend_acpl=performance_data.get("trend_acpl"),
        trend_match_rate=performance_data.get("trend_match_rate"),
        roi_curve=performance_data.get("roi_curve", []),
        roi_mean=sql_analysis.roi_mean,
        roi_max=sql_analysis.roi_max,
        roi_std=sql_analysis.roi_std,
        step_function_detected=sql_analysis.step_function_detected,
        step_function_magnitude=sql_analysis.step_function_magnitude,
        peer_delta_acpl=sql_analysis.peer_delta_acpl,
        peer_delta_match=sql_analysis.peer_delta_match,
        longest_streak=sql_analysis.longest_streak,
        selectivity_score=sql_analysis.selectivity_score
    )

    # Phase quality del JSON
    phase_data = sql_analysis.phase_quality or {}
    phase_quality = PhaseQuality(
        opening_acpl=phase_data.get("opening_acpl", 0.0),
        middlegame_acpl=phase_data.get("middlegame_acpl", 0.0),
        endgame_acpl=phase_data.get("endgame_acpl", 0.0),
        opening_blunder_rate=phase_data.get("opening_blunder_rate", 0.0),
        middlegame_blunder_rate=phase_data.get("middlegame_blunder_rate", 0.0),
        endgame_blunder_rate=phase_data.get("endgame_blunder_rate", 0.0),
        blunder_rate=phase_data.get("blunder_rate", 0.0)
    )

    # Clutch accuracy del JSON
    clutch_data = sql_analysis.clutch_accuracy or {}
    clutch_accuracy = ClutchAccuracy(
        avg_clutch_diff=clutch_data.get("avg_clutch_diff"),
        clutch_games_pct=clutch_data.get("clutch_games_pct", 0.0)
    )

    # Time complexity del JSON
    time_complex_data = sql_analysis.time_complexity or {}
    time_complexity = TimeComplexity(
        time_complexity_corr=time_complex_data.get("time_complexity_corr")
    )

    # Benchmark del JSON
    benchmark_data = sql_analysis.benchmark or {}
    benchmark = Benchmark(
        percentile_acpl=benchmark_data.get("percentile_acpl", 0),
        percentile_entropy=benchmark_data.get("percentile_entropy", 0)
    )

    # Risk assessment
    risk_assessment = RiskAssessment(
        risk_score=sql_analysis.risk_score,
        risk_factors=sql_analysis.risk_factors or {},
        confidence_level=sql_analysis.confidence_level
    )

    # Crear segments (simplificado)
    segments = []
    # TODO: Implementar parsing de segments desde JSON si es necesario

    return DomainPlayerAnalysis(
        username=sql_analysis.username,
        games_analyzed=sql_analysis.games_analyzed,
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
        first_game_date=sql_analysis.first_game_date,
        last_game_date=sql_analysis.last_game_date,
        analyzed_at=sql_analysis.analyzed_at,
        suspicious_games_ids=sql_analysis.suspicious_games_ids or [],
        segments=segments,
        change_points=[],  # TODO: Implementar si es necesario
        cluster_id=-1,
        cluster_distance=0.0
    )


def domain_to_analysis(domain_analysis: DomainPlayerAnalysis) -> SQLPlayerAnalysis:
    """Convierte DomainPlayerAnalysis a SQLPlayerAnalysis."""

    # Convertir time management a JSON
    time_management = {
        "mean_move_time": domain_analysis.timing_metrics.mean_move_time,
        "time_variance": domain_analysis.timing_metrics.time_variance,
        "uniformity_score": domain_analysis.timing_metrics.uniformity_score,
        "lag_spike_count": domain_analysis.timing_metrics.lag_spike_count
    }

    # Convertir opening patterns a JSON
    opening_patterns = {
        "mean_entropy": domain_analysis.opening_metrics.mean_entropy,
        "novelty_depth": domain_analysis.opening_metrics.novelty_depth,
        "opening_breadth": domain_analysis.opening_metrics.opening_breadth,
        "second_choice_rate": domain_analysis.opening_metrics.second_choice_rate
    }

    # Convertir performance a JSON
    performance = {
        "trend_acpl": domain_analysis.performance_metrics.trend_acpl,
        "trend_match_rate": domain_analysis.performance_metrics.trend_match_rate,
        "roi_curve": domain_analysis.performance_metrics.roi_curve
    }

    # Convertir phase quality a JSON
    phase_quality = {
        "opening_acpl": domain_analysis.phase_quality.opening_acpl,
        "middlegame_acpl": domain_analysis.phase_quality.middlegame_acpl,
        "endgame_acpl": domain_analysis.phase_quality.endgame_acpl,
        "opening_blunder_rate": domain_analysis.phase_quality.opening_blunder_rate,
        "middlegame_blunder_rate": domain_analysis.phase_quality.middlegame_blunder_rate,
        "endgame_blunder_rate": domain_analysis.phase_quality.endgame_blunder_rate,
        "blunder_rate": domain_analysis.phase_quality.blunder_rate
    }

    # Convertir clutch accuracy a JSON
    clutch_accuracy = {
        "avg_clutch_diff": domain_analysis.clutch_accuracy.avg_clutch_diff,
        "clutch_games_pct": domain_analysis.clutch_accuracy.clutch_games_pct
    }

    # Convertir endgame a JSON
    endgame = {
        "conversion_efficiency": domain_analysis.endgame_metrics.conversion_efficiency,
        "tb_match_rate": domain_analysis.endgame_metrics.tb_match_rate,
        "dtz_deviation": domain_analysis.endgame_metrics.dtz_deviation
    }

    # Convertir time complexity a JSON
    time_complexity = {
        "time_complexity_corr": domain_analysis.time_complexity.time_complexity_corr
    }

    # Convertir benchmark a JSON
    benchmark = {
        "percentile_acpl": domain_analysis.benchmark.percentile_acpl,
        "percentile_entropy": domain_analysis.benchmark.percentile_entropy
    }

    return SQLPlayerAnalysis(
        username=domain_analysis.username,
        games_analyzed=domain_analysis.games_analyzed,
        avg_acpl=domain_analysis.quality_metrics.avg_acpl,
        avg_wdl_loss=domain_analysis.quality_metrics.avg_wdl_loss,
        robust_loss=domain_analysis.quality_metrics.robust_loss,
        avg_match_rate=domain_analysis.quality_metrics.avg_match_rate,
        avg_ipr=domain_analysis.quality_metrics.avg_ipr,
        std_acpl=domain_analysis.quality_metrics.std_acpl,
        std_match_rate=domain_analysis.quality_metrics.std_match_rate,
        roi_mean=domain_analysis.performance_metrics.roi_mean,
        roi_max=domain_analysis.performance_metrics.roi_max,
        roi_std=domain_analysis.performance_metrics.roi_std,
        step_function_detected=domain_analysis.performance_metrics.step_function_detected,
        step_function_magnitude=domain_analysis.performance_metrics.step_function_magnitude,
        peer_delta_acpl=domain_analysis.performance_metrics.peer_delta_acpl,
        peer_delta_match=domain_analysis.performance_metrics.peer_delta_match,
        longest_streak=domain_analysis.performance_metrics.longest_streak,
        selectivity_score=domain_analysis.performance_metrics.selectivity_score,
        time_management=time_management,
        opening_patterns=opening_patterns,
        performance=performance,
        phase_quality=phase_quality,
        clutch_accuracy=clutch_accuracy,
        endgame=endgame,
        time_complexity=time_complexity,
        benchmark=benchmark,
        risk_score=domain_analysis.risk_assessment.risk_score,
        risk_factors=domain_analysis.risk_assessment.risk_factors,
        confidence_level=domain_analysis.risk_assessment.confidence_level,
        first_game_date=domain_analysis.first_game_date,
        last_game_date=domain_analysis.last_game_date,
        analyzed_at=domain_analysis.analyzed_at or datetime.utcnow(),
        suspicious_games_ids=domain_analysis.suspicious_games_ids,
        tactical={},  # TODO: Implementar si es necesario
        time_patterns=None  # TODO: Implementar si es necesario
    )


def game_analysis_to_domain(sql_analysis: SQLGameAnalysis) -> DomainGameAnalysis:
    """Convierte SQLGameAnalysis a DomainGameAnalysis."""

    quality_metrics = QualityMetrics(
        avg_acpl=sql_analysis.acpl,
        avg_wdl_loss=sql_analysis.wdl_loss,
        robust_loss=sql_analysis.weighted_match_rate,
        avg_match_rate=sql_analysis.match_rate,
        avg_ipr=sql_analysis.ipr
    )

    timing_metrics = TimingMetrics(
        mean_move_time=sql_analysis.mean_move_time,
        time_variance=sql_analysis.time_variance,
        uniformity_score=sql_analysis.uniformity_score,
        lag_spike_count=sql_analysis.lag_spike_count
    )

    opening_metrics = OpeningMetrics(
        mean_entropy=sql_analysis.opening_entropy,
        novelty_depth=sql_analysis.novelty_depth,
        opening_breadth=sql_analysis.opening_breadth,
        second_choice_rate=sql_analysis.second_choice_rate
    )

    endgame_metrics = EndgameMetrics(
        conversion_efficiency=sql_analysis.conversion_efficiency,
        tb_match_rate=sql_analysis.tb_match_rate,
        dtz_deviation=sql_analysis.dtz_deviation
    )

    return DomainGameAnalysis(
        game_id=sql_analysis.game_id,
        quality_metrics=quality_metrics,
        timing_metrics=timing_metrics,
        opening_metrics=opening_metrics,
        endgame_metrics=endgame_metrics,
        suspicious_quality=sql_analysis.suspicious_quality,
        suspicious_timing=sql_analysis.suspicious_timing,
        suspicious_opening=sql_analysis.suspicious_opening,
        overall_suspicion_score=sql_analysis.overall_suspicion_score,
        analyzed_at=sql_analysis.analyzed_at
    )


def domain_to_game_analysis(domain_analysis: DomainGameAnalysis) -> SQLGameAnalysis:
    """Convierte DomainGameAnalysis a SQLGameAnalysis."""

    analyzed_at = domain_analysis.analyzed_at or datetime.utcnow()

    return SQLGameAnalysis(
        game_id=domain_analysis.game_id,
        acpl=domain_analysis.quality_metrics.avg_acpl,
        wdl_loss=domain_analysis.quality_metrics.avg_wdl_loss,
        match_rate=domain_analysis.quality_metrics.avg_match_rate,
        weighted_match_rate=domain_analysis.quality_metrics.robust_loss,
        ipr=domain_analysis.quality_metrics.avg_ipr,
        ipr_z_score=0.0,
        precision_burst_count=0,
        mean_move_time=domain_analysis.timing_metrics.mean_move_time,
        time_variance=domain_analysis.timing_metrics.time_variance,
        time_complexity_corr=0.0,
        lag_spike_count=domain_analysis.timing_metrics.lag_spike_count,
        uniformity_score=domain_analysis.timing_metrics.uniformity_score,
        clutch_accuracy_diff=None,
        opening_entropy=domain_analysis.opening_metrics.mean_entropy,
        novelty_depth=domain_analysis.opening_metrics.novelty_depth,
        second_choice_rate=domain_analysis.opening_metrics.second_choice_rate,
        opening_breadth=domain_analysis.opening_metrics.opening_breadth,
        tb_match_rate=domain_analysis.endgame_metrics.tb_match_rate,
        dtz_deviation=domain_analysis.endgame_metrics.dtz_deviation,
        conversion_efficiency=domain_analysis.endgame_metrics.conversion_efficiency,
        suspicious_quality=domain_analysis.suspicious_quality,
        suspicious_timing=domain_analysis.suspicious_timing,
        suspicious_opening=domain_analysis.suspicious_opening,
        overall_suspicion_score=domain_analysis.overall_suspicion_score,
        analyzed_at=analyzed_at
    )


def update_sql_game_analysis(sql_analysis: SQLGameAnalysis, domain_analysis: DomainGameAnalysis) -> SQLGameAnalysis:
    """Actualiza un análisis SQL existente con datos del dominio."""

    sql_analysis.acpl = domain_analysis.quality_metrics.avg_acpl
    sql_analysis.wdl_loss = domain_analysis.quality_metrics.avg_wdl_loss
    sql_analysis.match_rate = domain_analysis.quality_metrics.avg_match_rate
    sql_analysis.weighted_match_rate = domain_analysis.quality_metrics.robust_loss
    sql_analysis.ipr = domain_analysis.quality_metrics.avg_ipr

    sql_analysis.mean_move_time = domain_analysis.timing_metrics.mean_move_time
    sql_analysis.time_variance = domain_analysis.timing_metrics.time_variance
    sql_analysis.uniformity_score = domain_analysis.timing_metrics.uniformity_score
    sql_analysis.lag_spike_count = domain_analysis.timing_metrics.lag_spike_count

    sql_analysis.opening_entropy = domain_analysis.opening_metrics.mean_entropy
    sql_analysis.novelty_depth = domain_analysis.opening_metrics.novelty_depth
    sql_analysis.opening_breadth = domain_analysis.opening_metrics.opening_breadth
    sql_analysis.second_choice_rate = domain_analysis.opening_metrics.second_choice_rate

    sql_analysis.conversion_efficiency = domain_analysis.endgame_metrics.conversion_efficiency
    sql_analysis.tb_match_rate = domain_analysis.endgame_metrics.tb_match_rate
    sql_analysis.dtz_deviation = domain_analysis.endgame_metrics.dtz_deviation

    sql_analysis.suspicious_quality = domain_analysis.suspicious_quality
    sql_analysis.suspicious_timing = domain_analysis.suspicious_timing
    sql_analysis.suspicious_opening = domain_analysis.suspicious_opening
    sql_analysis.overall_suspicion_score = domain_analysis.overall_suspicion_score

    if domain_analysis.analyzed_at:
        sql_analysis.analyzed_at = domain_analysis.analyzed_at

    return sql_analysis
