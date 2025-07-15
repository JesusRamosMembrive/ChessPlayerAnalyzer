"""Tareas Celery relacionadas con el análisis **longitudinal** de un jugador.

Fase 1: simplemente reexportamos las tareas existentes mientras migramos
la implementación desde ``app.celery_app``.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlmodel import Session, select
from sqlalchemy import func
from celery import exceptions as celery_exceptions

from app.celery_app import (
    celery_app,
    redis_client,
    TASK_MAX_RETRIES,
    TASK_SOFT_TIME_LIMIT,
    TASK_TIME_LIMIT,
)
from app.tasks.utils import safe, export_analysis_to_json
from app.utils import cache_get, cache_set, notify_ws
from app.database import engine
from app import models
from app.models import GameAnalysisDetailed

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper functions for player analysis
# ---------------------------------------------------------------------------

def _is_aborted(task, player_username=None):
    """Check if task has been aborted or cancelled via Redis flag.
    
    Uses reflection to be compatible with different Celery versions
    and checks Redis for immediate cancellation.
    """
    try:
        # Check Redis cancellation flag first (fastest method)
        if player_username:
            cancellation_key = f"cancel:{player_username}"
            if redis_client.get(cancellation_key):
                logger.info(f"Task {task.request.id} detected Redis cancellation flag for {player_username}")
                return True

        # Celery >=5.3 exposes Task.is_aborted()
        if hasattr(task, "is_aborted"):
            return task.is_aborted()
        if hasattr(task.request, "is_aborted"):
            return task.request.is_aborted()
        if getattr(task.request, "stopped", False):
            return True
    except Exception:
        pass
    return False


def _check_sufficient_games(username: str) -> tuple[bool, int]:
    """Check if player has sufficient analyzed games for detailed analysis."""
    with Session(engine) as s:
        analyzed_count = s.exec(
            select(func.count(GameAnalysisDetailed.game_id))
            .join(models.Game)
            .where(
                (models.Game.white_username == username) |
                (models.Game.black_username == username)
            )
        ).one()
        
        logger.info(f"DEBUG PLAYER: Pre-analysis check - Found {analyzed_count} games with existing analysis for {username} in GameAnalysisDetailed table")
        return analyzed_count >= 1, analyzed_count


def _update_player_status(username: str) -> None:
    """Update player status to ready and mark analysis as complete."""
    with Session(engine) as s:
        player = s.get(models.Player, username)
        if player:
            player.status = "ready"
            player.progress = 100
            player.finished_at = datetime.now(timezone.utc)
            s.add(player)
            s.commit()
    
    from app.main import clear_analysis_in_progress
    clear_analysis_in_progress()
    logger.info(f"Analysis completed for user {username} - system ready for new requests")


def _create_or_update_player_record(username: str, total_games: int) -> None:
    """Create or update player record with initial status."""
    with Session(engine) as s:
        player = s.get(models.Player, username)
        if player is None:
            player = models.Player(
                username=username,
                status="pending",
                requested_at=datetime.now(timezone.utc),
                progress=0,
                total_games=total_games,
                done_games=0,
            )
            s.add(player)
        else:
            player.status = "pending"
            player.requested_at = datetime.now(timezone.utc)
            player.progress = 0
            player.total_games = total_games
            player.done_games = 0
        s.commit()
        logger.info(f"DEBUG CELERY: Created/updated player record for {username}")


def _get_or_create_game_record(game_data: dict) -> int:
    """Get existing game record or create a new one. Returns game ID."""
    with Session(engine) as s:
        existing_game = s.exec(
            select(models.Game).where(
                (models.Game.pgn == game_data["pgn"]) &
                (models.Game.white_username == game_data.get("white")) &
                (models.Game.black_username == game_data.get("black"))
            )
        ).first()

        if existing_game and existing_game.id is not None:
            gid = existing_game.id
            logger.info(f"DEBUG CELERY: Found existing game record with ID: {gid}")
            return gid
        else:
            game_db = models.Game(
                pgn=game_data["pgn"],
                move_times=game_data.get("move_times", []),
                white_username=game_data.get("white"),
                black_username=game_data.get("black"),
                white_elo=game_data.get("white_elo"),
                black_elo=game_data.get("black_elo"),
            )
            s.add(game_db)
            s.commit()
            s.refresh(game_db)
            gid = game_db.id
            logger.info(f"DEBUG CELERY: Created new game record with ID: {gid}")
            return gid


def _update_player_task_id(username: str, task_id: str) -> None:
    """Update player record with the task ID for cancellation purposes."""
    with Session(engine) as s:
        player = s.get(models.Player, username)
        if player:
            player.last_task_id = task_id
            s.commit()


# ---------------------------------------------------------------------------
# Migrated task implementations
# ---------------------------------------------------------------------------

@celery_app.task(
    name="analyze_player_detailed_new",
    autoretry_for=(Exception, celery_exceptions.SoftTimeLimitExceeded),
    retry_backoff=True,
    retry_backoff_max=600,
    retry_jitter=True,
    retry_kwargs={"max_retries": TASK_MAX_RETRIES},
    soft_time_limit=TASK_SOFT_TIME_LIMIT,
    time_limit=TASK_TIME_LIMIT,
)
def analyze_player_detailed_impl(username: str):
    """
    Análisis longitudinal detallado de un jugador.
    Se ejecuta después de que todas sus partidas han sido analizadas.
    """
    logger.info(f"DEBUG PLAYER: Starting player detailed analysis for {username}")
    cached = cache_get("analyze_player_detailed", [username], {})
    if cached:
        logger.info("DEBUG PLAYER: Returning cached result for analyze_player_detailed")
        return cached

    try:
        # Verificar que hay suficientes partidas analizadas
        sufficient, analyzed_count = _check_sufficient_games(username)
        
        if not sufficient:
            logging.warning(f"Insuficientes partidas analizadas para {username}: {analyzed_count}")
            return {
                "username": username,
                "status": "insufficient_data",
                "games_analyzed": analyzed_count
            }

        # Ejecutar análisis del jugador
        logger.info(f"DEBUG PLAYER: Starting analysis engine for {username}")
        from app.celery_app import analysis_engine  # Import deferred to avoid cycles
        player_analysis = analysis_engine.analyze_player(username)
        logger.info("DEBUG PLAYER: PlayerAnalysisDetailed result: %s", player_analysis)

        with Session(engine) as s:
           pa = s.get(models.PlayerAnalysisDetailed, username)
           logger.info(f"DEBUG PLAYER: Retrieved player analysis from DB for {username}: risk_score={pa.risk_score}, games_analyzed={pa.games_analyzed} (this count reflects games with completed analysis in GameAnalysisDetailed table)")

           export_analysis_to_json(pa, username, analysis_type="player")

        # Notificar resultado
        notify_ws(username, {
            "type": "player_analysis_complete",
            "risk_score": pa.risk_score,
            "risk_factors": pa.risk_factors
        })

        result = {
            "username": username,
            "risk_score": pa.risk_score,
            "games_analyzed": pa.games_analyzed,
            "analyzed_at": pa.analyzed_at.isoformat()
        }

        _update_player_status(username)
        
        notify_ws(username, {"status": "ready", "progress": 100})

        logger.info(f"DEBUG PLAYER: Final analysis result for {username}: risk_score={result['risk_score']}, games_analyzed={result['games_analyzed']} (total games processed in this analysis session)")
        # Store result in cache
        cache_set("analyze_player_detailed", [username], {}, result)
        return result

    except Exception as e:
        logging.error(f"DEBUG PLAYER: Error en análisis detallado de jugador {username}: {e}")
        raise  # Re-raise for Celery autoretry


# ---------------------------------------------------------------------------
# Wrappers temporales – delegan en versiones legacy
# ---------------------------------------------------------------------------

@celery_app.task(name="process_player_enhanced", bind=True)
def process_player_enhanced(self, *args, **kwargs):  # noqa: D401
    """Delegación temporal hacia la versión legacy con import diferido."""
    from app.celery_app import process_player_enhanced_legacy as _legacy  # import local para evitar ciclos
    return _legacy(*args, **kwargs)


@celery_app.task(name="analyze_player_detailed")
def analyze_player_detailed(_, username: str):  # noqa: D401
    """Wrapper que delega en la nueva implementación migrada."""
    return analyze_player_detailed_impl(username)

__all__ = [
    "process_player_enhanced",
    "analyze_player_detailed",
] 