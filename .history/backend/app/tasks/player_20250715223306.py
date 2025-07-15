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

           export_analysis_to_json(pa, username, "player")

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
def analyze_player_detailed(*args, **kwargs):  # noqa: D401
    """Wrapper que delega en la nueva implementación migrada."""
    return analyze_player_detailed_impl(*args, **kwargs)

__all__ = [
    "process_player_enhanced",
    "analyze_player_detailed",
] 