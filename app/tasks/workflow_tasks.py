"""
Workflow Tasks Module

This module contains orchestration and workflow Celery tasks that coordinate
the analysis pipeline, including the main process_player_enhanced task and
related signal handlers.
"""
from __future__ import annotations

from datetime import datetime, timezone

from celery import chain, group, chord
from celery.exceptions import SoftTimeLimitExceeded
from celery.signals import task_failure, task_revoked
from sqlmodel import Session, select

from app import models
from .shared_config import (
    logger, engine, redis_client, notify_ws, task_progress, fetch_games,
    is_task_aborted, DEFAULT_PRIORITY,
    TASK_SOFT_TIME_LIMIT, TASK_TIME_LIMIT, TASK_MAX_RETRIES
)

# Get celery app instance - will be injected by main celery_app.py
celery_app = None

def register_workflow_tasks(app):
    """Register workflow tasks with the Celery app instance."""
    global celery_app
    celery_app = app

    # Register tasks
    app.task(
        name="process_player_enhanced",
        bind=True,
        autoretry_for=(Exception, SoftTimeLimitExceeded),
        retry_backoff=True,
        retry_backoff_max=600,
        retry_jitter=True,
        retry_kwargs={"max_retries": TASK_MAX_RETRIES},
        soft_time_limit=TASK_SOFT_TIME_LIMIT,
        time_limit=TASK_TIME_LIMIT,
    )(process_player_enhanced)

    app.task(name="process_player")(_deprecated)

def process_player_enhanced(self, username: str, months: int = 12, priority: int = DEFAULT_PRIORITY):
    """
    Main orchestration task for processing a player's games.

    This task:
    1. Downloads player games from external source
    2. Creates Game records in database
    3. Orchestrates parallel analysis chains for each game
    4. Coordinates final player analysis after all games are complete
    """
    logger.info(f"DEBUG CELERY: Starting process_player_enhanced for {username}, months: {months}")

    # 1. DESCARGAR partidas y crear registros Game ──────────────────────────
    logger.info(f"DEBUG CELERY: Fetching games for {username}")
    games = fetch_games(username, months)
    logger.info(f"DEBUG CELERY: Downloaded {len(games)} games")
    logger.info(f"DEBUG CELERY: Sample game structure: {games[0] if games else 'No games'}")

    # Verificar revocación después de descargar partidas
    if not self.request.called_directly and is_task_aborted(self, username):
        logger.info(f"Task {self.request.id} has been revoked after fetching games, stopping execution")
        return {"status": "revoked", "username": username, "games_fetched": len(games)}

    game_ids = []

    with Session(engine) as s:
        player = s.get(models.Player, username)
        if player is None:
            player = models.Player(
                username=username,
                status="pending",
                requested_at=datetime.now(timezone.utc),
                progress=0,
                total_games=len(games),
                done_tasks=0,
                done_games=0,
            )
            s.add(player)
        else:
            player.status = "pending"
            player.requested_at = datetime.now(timezone.utc)
            player.progress = 0
            player.total_games = len(games)
            player.done_tasks = 0
            player.done_games = 0
        s.commit()
        logger.info(f"DEBUG CELERY: Created/updated player record for {username}")

    chains = []          # ← aquí iremos acumulando chain por partida

    # Normalizar prioridad (0-9)
    priority = max(0, min(9, int(priority)))

    total_games = len(games)
    for i, g in enumerate(games):
        if not self.request.called_directly and is_task_aborted(self, username):
            logger.info(f"Task {self.request.id} has been revoked during game processing, stopping execution")
            with Session(engine) as s:
                pl = s.get(models.Player, username)
                if pl:
                    try:
                        pl.status = "error"
                        pl.error = "stopped_by_user"
                        pl.finished_at = datetime.now(timezone.utc)
                        s.add(pl)
                        s.commit()
                    except Exception:
                        s.rollback()
            notify_ws(username, {"status": "stopped"})
            return {"status": "revoked", "username": username, "games_processed": i}

        logger.info(f"DEBUG CELERY: Processing game {i+1}/{len(games)}")
        logger.info(f"DEBUG CELERY: Game data - white: {g.get('white')}, black: {g.get('black')}, white_elo: {g.get('white_elo')}, black_elo: {g.get('black_elo')}")

        with Session(engine) as s:
            existing_game = s.exec(
                select(models.Game).where(
                    (models.Game.pgn == g["pgn"]) &
                    (models.Game.white_username == g.get("white")) &
                    (models.Game.black_username == g.get("black"))
                )
            ).first()

            if existing_game:
                gid = existing_game.id
                game_ids.append(gid)
                logger.info(f"DEBUG CELERY: Found existing game record with ID: {gid}")
            else:
                game_db = models.Game(
                    pgn=g["pgn"],
                    move_times=g.get("move_times", []),
                    white_username=g.get("white"),
                    black_username=g.get("black"),
                    white_elo=g.get("white_elo"),
                    black_elo=g.get("black_elo"),
                )
                s.add(game_db);  s.commit();  s.refresh(game_db)
                gid = game_db.id
                game_ids.append(gid)
                logger.info(f"DEBUG CELERY: Created new game record with ID: {gid}")

        # Get tasks from celery app
        analyze_game_task = celery_app.tasks.get("analyze_game_task")
        analyze_game_detailed = celery_app.tasks.get("analyze_game_detailed")

        if not analyze_game_task or not analyze_game_detailed:
            raise RuntimeError("Required analysis tasks not found in Celery app")

        # Propagar prioridad a las subtareas
        basic = (
            analyze_game_task.s(g["pgn"], gid, move_times=g.get("move_times"), player=username)
            .set(priority=priority)
        )
        if redis_client.get(f"cancel:{username}"):
            logger.info(f"process_player_enhanced detected cancellation before scheduling chain for {username}")
            return {"status": "revoked", "username": username, "games_queued": len(games)}

        detailed = (
            analyze_game_detailed.si(gid, username)
            .set(priority=priority)
        )
        chains.append(chain(basic, detailed))

        # ── Progress update ─────────────────────────────────────
        try:
            task_progress(self, i + 1, total_games, username)
        except Exception:
            pass

    logger.info(f"DEBUG CELERY: Created {len(chains)} analysis chains")

    # Get player analysis task from celery app
    analyze_player_detailed = celery_app.tasks.get("analyze_player_detailed")
    if not analyze_player_detailed:
        raise RuntimeError("analyze_player_detailed task not found in Celery app")

    # 3. group & chord: cuando todas las partidas acaben … ──────────────────
    #    se lanza analyze_player_detailed(username)
    full_workflow = chord(
        group(chains),
        analyze_player_detailed.s(username).set(priority=priority)
    ).set(priority=priority)
    chord_result = full_workflow.apply_async(priority=priority)  # AsyncResult del body
    header_id = chord_result.parent.id if chord_result.parent else chord_result.id

    # ── Guardar el ID del grupo/encabezado para poder revocarlo ────────────
    with Session(engine) as s:
        pl_upd = s.get(models.Player, username)
        if pl_upd:
            pl_upd.last_task_id = header_id
            s.commit()

    result = {
        "username": username,
        "games_queued": len(games),
        "enhanced_analysis": True,
        "task_id": header_id,
    }
    logger.info("DEBUG CELERY: process_player_enhanced result: %s", result)
    return result

def _deprecated(*a, **kw):
    """Deprecated process_player task."""
    raise RuntimeError("Deprecated. Use process_player_enhanced")

# Signal handlers
@task_failure.connect
def on_task_failure(sender=None, task_id=None, args=None, kwargs=None, **k):
    """Handle task failure signals, particularly for process_player_enhanced."""
    if sender and sender.name == "process_player_enhanced":
        username = args[0] if args else None
        with Session(engine) as s:
            pl = s.get(models.Player, username)
            if pl:
                pl.status = "error"
                pl.error = str(k.get("exception", "unknown"))
                s.add(pl); s.commit()
        if username:
            notify_ws(username, {"status": "error"})

@task_revoked.connect
def on_task_revoked(sender=None, request=None, terminated=None, signum=None, expired=None, **k):
    """Maneja la revocación de tareas."""
    logger.info(f"Task {request.id if request else 'unknown'} has been revoked (terminated={terminated}, signum={signum}, expired={expired})")

    if request and request.task == "process_player_enhanced":
        username = request.args[0] if request.args else None
        if username:
            logger.info(f"Updating player {username} status after task revocation")
            with Session(engine) as s:
                pl = s.get(models.Player, username)
                if pl:
                    pl.status = "ready"
                    pl.error = "Analysis stopped by user"
                    pl.finished_at = datetime.now(timezone.utc)
                    s.add(pl)
                    s.commit()
                    logger.info(f"Player {username} status updated to ready after revocation")
            notify_ws(username, {"status": "stopped", "message": "Analysis stopped by user"})