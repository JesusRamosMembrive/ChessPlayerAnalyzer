"""
New Celery tasks using Application Layer.
Replaces legacy celery_app.py with clean architecture.
"""
import logging
from datetime import datetime
from typing import Optional

from celery import Celery, current_task
from celery.signals import task_failure, task_revoked
from sqlmodel import Session

from ...core.config import get_config
from ...application.container import get_container
from ...application.commands.player_commands import (
    AnalyzePlayerCommand,
    UpdatePlayerProgressCommand
)
from ...application.use_cases.game_use_cases import AnalyzeGameUseCase
from ...domain.entities.game import Game
from ...infrastructure.database.connection import get_session

# Setup
config = get_config()
logger = logging.getLogger(__name__)

# Celery app
celery_app = Celery(
    "chess_analyzer",
    broker=config.redis.url,
    backend=config.redis.url
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_disable_rate_limits=False,
    task_ignore_result=False,
)


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
    task_id = current_task.request.id
    logger.info(f"Starting player analysis", extra={
        "username": username,
        "task_id": task_id,
        "force_refresh": force_refresh,
        "months_to_analyze": months_to_analyze
    })

    try:
        container = get_container()

        # 1. Iniciar análisis usando Application Layer
        command = AnalyzePlayerCommand(
            username=username,
            force_refresh=force_refresh,
            months_to_analyze=months_to_analyze
        )

        analyze_use_case = container.get_analyze_player_use_case()
        result = await analyze_use_case.execute(command)

        if not result.success:
            logger.error(f"Failed to start analysis: {result.error_message}")
            return {"success": False, "error": result.error_message}

        player_id = result.player_id

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
                # Crear Game entity
                game = Game(
                    player_id=player_id,
                    username=username,
                    game_url=game_data["url"],
                    time_control=game_data.get("time_control", "unknown"),
                    result=game_data.get("result", "unknown"),
                    played_at=datetime.fromisoformat(game_data["end_time"]) if game_data.get("end_time") else None,
                    pgn_data=game_data.get("pgn")
                )

                # Guardar partida
                game_repo = container._get_game_repository()
                saved_game = await game_repo.save(game)

                # Analizar partida
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
                if self.is_aborted():
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
    task_id = current_task.request.id
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
    return await fetch_games(username, months_to_analyze)


async def _generate_player_analysis(username: str, player_id: int, container):
    """
    Temporary helper to generate player analysis.
    TODO: Move to PlayerAnalysisAggregationService.
    """
    from ...domain.value_objects.metrics import QualityMetrics, TimingMetrics, RiskAssessment
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
            total_move_time += game_analysis.timing_metrics.avg_move_time
            analyzed_games += 1

            if game_analysis.risk_assessment.cheat_probability > 0.7:
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
    player_analysis = PlayerAnalysis(
        player_id=player_id,
        overall_metrics=QualityMetrics(
            avg_acpl=avg_acpl,
            avg_wdl_loss=avg_wdl_loss,
            avg_match_rate=avg_match_rate
        ),
        timing_metrics=TimingMetrics(
            avg_move_time=avg_move_time,
            time_variance=0.0,  # TODO: Calculate properly
            quick_moves_rate=0.0  # TODO: Calculate properly
        ),
        risk_assessment=RiskAssessment(
            cheat_probability=cheat_probability,
            risk_level=risk_level,
            suspicion_flags=[]
        ),
        games_analyzed=analyzed_games,
        total_games=len(games),
        analysis_date=datetime.utcnow()
    )

    # Save analysis
    await analysis_repo.save_player_analysis(player_analysis)


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