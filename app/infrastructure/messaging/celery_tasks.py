"""
New Celery tasks using Application Layer.
Replaces legacy celery_app.py with clean architecture.
"""
import asyncio
import logging
from datetime import datetime
from typing import Optional

from celery import Celery, current_task
from celery.signals import task_failure, task_revoked
from sqlmodel import Session

from ...core.config import get_config, get_optimized_celery_settings
from ...application.container import get_container
from ...application.commands.player_commands import (
    AnalyzePlayerCommand,
    UpdatePlayerProgressCommand
)
from ...application.use_cases.game_use_cases import AnalyzeGameUseCase
from ...domain.entities.game import Game
from ...database import get_session

# Setup with performance optimizations
config = get_config()
optimized_celery_settings = get_optimized_celery_settings()
logger = logging.getLogger(__name__)

# Performance-optimized Celery app
celery_app = Celery(
    "chess_analyzer",
    broker=config.redis.url,
    backend=config.redis.url
)

# Apply optimized configuration
celery_app.conf.update(optimized_celery_settings)


@celery_app.task(
    name="analyze_player_v2",
    bind=True,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3, "countdown": 60},
)
def analyze_player_task(self, username: str, force_refresh: bool = False, months_to_analyze: int = 12):
    """
    New player analysis task using Application Layer.
    Replaces analyze_player_detailed from legacy celery_app.py.
    """
    return asyncio.run(_analyze_player_task_async(username, force_refresh, months_to_analyze))


async def _analyze_player_task_async(username: str, force_refresh: bool = False, months_to_analyze: int = 12):
    """
    Async implementation of player analysis task.
    """
    task = current_task
    task_id = task.request.id if task else None
    logger.info(f"Starting player analysis", extra={
        "username": username,
        "task_id": task_id,
        "force_refresh": force_refresh,
        "months_to_analyze": months_to_analyze
    })

    try:
        container = get_container()

        def _safe_int(value):
            try:
                return int(value) if value is not None else None
            except (TypeError, ValueError):
                return None

        # 1. Iniciar análisis usando Application Layer
        command = AnalyzePlayerCommand(
            username=username,
            force_refresh=force_refresh,
            months_to_analyze=months_to_analyze
        )

        analyze_use_case = container.get_analyze_player_use_case()
        logger.info("About to execute analyze use case")
        result = await analyze_use_case.execute(command)
        logger.info(f"Use case result: success={result.success}")

        if not result.success:
            logger.error(f"Failed to start analysis: {result.error_message}")
            return {"success": False, "error": result.error_message}

        logger.info(f"About to access player_id from result")
        player_id = result.player_id
        logger.info(f"Player ID retrieved: {player_id}")

        # 2. Obtener partidas (mantener lógica existente por ahora)
        # TODO: Mover esto a un GameFetchingService en domain layer
        games = await _fetch_player_games(username, months_to_analyze)
        total_games = len(games)

        if total_games == 0:
            logger.warning(f"No games found for player {username}")
            return {"success": True, "games_analyzed": 0}

        # 3. Actualizar progreso inicial
        progress_command = UpdatePlayerProgressCommand(
            username=username,
            done_games=0,
            total_games=total_games,
            task_id=task_id
        )

        update_progress_use_case = container.get_update_player_progress_use_case()
        await update_progress_use_case.execute(progress_command)

        # 4. Analizar cada partida
        analyze_game_use_case = container.get_analyze_game_use_case()
        analyzed_count = 0

        for i, game_data in enumerate(games):
            try:
                pgn = game_data.get("pgn")
                if not pgn:
                    logger.warning("Skipping game without PGN", extra={
                        "username": username,
                        "index": i
                    })
                    continue

                end_time = game_data.get("end_time")
                try:
                    played_at = datetime.fromisoformat(end_time) if end_time else None
                except ValueError:
                    played_at = None

                white_username = game_data.get("white")
                black_username = game_data.get("black")

                player_result = None
                if username and white_username and username.lower() == white_username.lower():
                    player_result = game_data.get("white_result")
                elif username and black_username and username.lower() == black_username.lower():
                    player_result = game_data.get("black_result")

                # Crear Game entity compatible con el dominio actual
                game = Game(
                    id=None,
                    pgn=pgn,
                    white_username=white_username,
                    black_username=black_username,
                    white_elo=_safe_int(game_data.get("white_rating")),
                    black_elo=_safe_int(game_data.get("black_rating")),
                    time_control=game_data.get("time_control"),
                    termination=None,
                    eco_code=None,
                    opening_key=None,
                    move_times=game_data.get("move_times"),
                    moves=None,
                    created_at=played_at,
                    username=username,
                    player_id=player_id,
                    game_url=game_data.get("url"),
                    result=player_result,
                    played_at=played_at,
                    pgn_data=pgn
                )

                # Guardar partida
                game_repo = container._get_game_repository()
                saved_game = await game_repo.save(game)

                # Analizar partida
                logger.info(
                    "Analyzing game",
                    extra={
                        "username": username,
                        "game_index": i + 1,
                        "total_games": total_games,
                        "saved_game_id": saved_game.id
                    }
                )
                analysis_result = await analyze_game_use_case.execute(saved_game.id)

                if analysis_result and analysis_result.analysis:
                    analyzed_count += 1

                # Actualizar progreso cada 10 partidas
                if (i + 1) % 10 == 0:
                    progress_command = UpdatePlayerProgressCommand(
                        username=username,
                        done_games=i + 1,
                        total_games=total_games,
                        task_id=task_id
                    )
                    await update_progress_use_case.execute(progress_command)

                # Check if task was aborted
                if task and hasattr(task, "is_aborted") and task.is_aborted():
                    logger.info(f"Task aborted for {username}")
                    return {"success": False, "error": "Task aborted"}

            except Exception as e:
                logger.error(f"Error analyzing game {i}: {e}")
                continue

        # 5. Progreso final
        final_progress = UpdatePlayerProgressCommand(
            username=username,
            done_games=total_games,
            total_games=total_games,
            task_id=task_id
        )
        await update_progress_use_case.execute(final_progress)

        # 6. Generar análisis agregado del jugador
        # TODO: Implementar PlayerAnalysisAggregationService
        await _generate_player_analysis(username, player_id, container)

        logger.info(f"Completed analysis for {username}", extra={
            "total_games": total_games,
            "analyzed_games": analyzed_count,
            "task_id": task_id
        })

        return {
            "success": True,
            "username": username,
            "total_games": total_games,
            "analyzed_games": analyzed_count,
            "task_id": task_id
        }

    except Exception as e:
        logger.error(f"Failed to analyze player {username}: {e}", exc_info=True)
        raise


@celery_app.task(
    name="analyze_game_v2",
    bind=True,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 2, "countdown": 30},
)
def analyze_game_task(self, game_id: int, force_reanalysis: bool = False):
    """
    New game analysis task using Application Layer.
    Replaces analyze_game_task from legacy celery_app.py.
    """
    return asyncio.run(_analyze_game_task_async(game_id, force_reanalysis))


async def _analyze_game_task_async(game_id: int, force_reanalysis: bool = False):
    """
    Async implementation of game analysis task.
    """
    task = current_task
    task_id = task.request.id if task else None
    logger.info(f"Starting game analysis", extra={
        "game_id": game_id,
        "task_id": task_id,
        "force_reanalysis": force_reanalysis
    })

    try:
        container = get_container()
        analyze_game_use_case = container.get_analyze_game_use_case()

        result = await analyze_game_use_case.execute(game_id, force_reanalysis)

        if not result:
            return {"success": False, "error": "Game not found"}

        if not result.analysis:
            return {"success": False, "error": "Game could not be analyzed"}

        logger.info(f"Completed game analysis", extra={
            "game_id": game_id,
            "task_id": task_id,
            "cheat_probability": result.analysis.risk_assessment.cheat_probability
        })

        return {
            "success": True,
            "game_id": game_id,
            "analysis": {
                "avg_acpl": result.analysis.quality_metrics.avg_acpl,
                "cheat_probability": result.analysis.risk_assessment.cheat_probability,
                "risk_level": result.analysis.risk_assessment.risk_level
            },
            "task_id": task_id
        }

    except Exception as e:
        logger.error(f"Failed to analyze game {game_id}: {e}", exc_info=True)
        raise


@celery_app.task(name="test_worker_v2")
def test_worker_functionality():
    """Test task to verify worker functionality."""
    logger.info("Testing worker functionality with new architecture")
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


# Helper functions (temporary - will be moved to proper services)

async def _fetch_player_games(username: str, months_to_analyze: int) -> list:
    """
    Temporary helper to fetch games.
    TODO: Move to GameFetchingService in domain layer.
    """
    # For now, import and use existing logic
    from ...utils import fetch_games
    return fetch_games(username, months_to_analyze)  # Remove await - fetch_games is sync


async def _generate_player_analysis(username: str, player_id: int, container):
    """
    Temporary helper to generate player analysis.
    TODO: Move to PlayerAnalysisAggregationService.
    """
    from ...domain.value_objects.metrics import (
        QualityMetrics,
        TimingMetrics,
        OpeningMetrics,
        EndgameMetrics,
        PerformanceMetrics,
        PhaseQuality,
        ClutchAccuracy,
        TimeComplexity,
        Benchmark,
        RiskAssessment
    )
    from ...domain.entities.analysis import PlayerAnalysis

    # Get all games for player
    game_repo = container._get_game_repository()
    analysis_repo = container._get_analysis_repository()

    games = await game_repo.get_by_player_id(player_id)

    # Aggregate metrics from game analyses
    total_acpl = 0
    total_wdl_loss = 0
    total_match_rate = 0
    total_move_time = 0
    analyzed_games = 0
    suspicious_count = 0

    for game in games:
        game_analysis = await analysis_repo.get_game_analysis_by_id(game.id)
        if game_analysis:
            total_acpl += game_analysis.quality_metrics.avg_acpl
            total_wdl_loss += game_analysis.quality_metrics.avg_wdl_loss
            total_match_rate += game_analysis.quality_metrics.avg_match_rate
            total_move_time += getattr(game_analysis.timing_metrics, "avg_move_time", None) or game_analysis.timing_metrics.mean_move_time
            analyzed_games += 1

            risk = getattr(game_analysis, "risk_assessment", None)
            cheat_probability = risk.cheat_probability if risk else 0.0
            if cheat_probability > 0.7:
                suspicious_count += 1

    if analyzed_games == 0:
        return

    # Calculate averages
    avg_acpl = total_acpl / analyzed_games
    avg_wdl_loss = total_wdl_loss / analyzed_games
    avg_match_rate = total_match_rate / analyzed_games
    avg_move_time = total_move_time / analyzed_games

    # Calculate risk
    suspicion_rate = suspicious_count / analyzed_games
    cheat_probability = min(suspicion_rate * 1.5, 1.0)  # Simple heuristic

    risk_level = "low"
    if cheat_probability > 0.3:
        risk_level = "medium"
    if cheat_probability > 0.7:
        risk_level = "high"

    # Create player analysis
    quality_metrics = QualityMetrics(
        avg_acpl=avg_acpl,
        avg_wdl_loss=avg_wdl_loss,
        robust_loss=avg_acpl,
        avg_match_rate=avg_match_rate,
        avg_ipr=max(800.0, 2800.0 - (avg_acpl * 20))
    )

    timing_metrics = TimingMetrics(
        mean_move_time=avg_move_time,
        time_variance=0.0,
        uniformity_score=0.0,
        lag_spike_count=0
    )

    opening_metrics = OpeningMetrics(
        mean_entropy=0.0,
        novelty_depth=None,
        opening_breadth=0,
        second_choice_rate=0.0
    )

    endgame_metrics = EndgameMetrics(
        conversion_efficiency=None,
        tb_match_rate=None,
        dtz_deviation=None
    )

    performance_metrics = PerformanceMetrics(
        trend_acpl=None,
        trend_match_rate=None,
        roi_curve=[],
        roi_mean=None,
        roi_max=None,
        roi_std=None,
        step_function_detected=False,
        step_function_magnitude=None,
        peer_delta_acpl=0.0,
        peer_delta_match=0.0,
        longest_streak=0,
        selectivity_score=0.0
    )

    phase_quality = PhaseQuality(
        opening_acpl=avg_acpl,
        middlegame_acpl=avg_acpl,
        endgame_acpl=avg_acpl,
        opening_blunder_rate=0.0,
        middlegame_blunder_rate=0.0,
        endgame_blunder_rate=0.0,
        blunder_rate=0.0
    )

    clutch_accuracy = ClutchAccuracy(
        avg_clutch_diff=None,
        clutch_games_pct=0.0
    )

    time_complexity = TimeComplexity(time_complexity_corr=None)

    benchmark = Benchmark(percentile_acpl=0, percentile_entropy=0)

    risk_assessment = RiskAssessment(
        risk_score=int(cheat_probability * 100),
        risk_factors={},
        confidence_level=75
    )

    player_analysis = PlayerAnalysis(
        username=username,
        games_analyzed=analyzed_games,
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
        first_game_date=min((game.created_at for game in games if game.created_at), default=None),
        last_game_date=max((game.created_at for game in games if game.created_at), default=None),
        analyzed_at=datetime.utcnow(),
        suspicious_games_ids=[],
        segments=[],
        change_points=[]
    )

    player_service = container._get_player_service()
    await player_service.complete_analysis(username, player_analysis)


# Task error handling
@task_failure.connect
def on_task_failure(sender=None, task_id=None, args=None, kwargs=None, traceback=None, einfo=None, **kwds):
    """Handle task failures."""
    logger.error(f"Task {task_id} failed", extra={
        "task_id": task_id,
        "task_name": sender.name if sender else "unknown",
        "args": args,
        "kwargs": kwargs,
        "error": str(einfo)
    })


@task_revoked.connect
def on_task_revoked(sender=None, request=None, terminated=None, signum=None, expired=None, **kwds):
    """Handle task revocations."""
    task_id = request.id if request else "unknown"
    logger.warning(f"Task {task_id} revoked", extra={
        "task_id": task_id,
        "task_name": sender.name if sender else "unknown",
        "terminated": terminated,
        "expired": expired
    })
