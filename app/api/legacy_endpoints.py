"""
Legacy endpoints extraídos de main.py para mejor organización.
Mantiene compatibilidad con versiones anteriores del API.
"""
from __future__ import annotations

import io
import logging
import time
from datetime import datetime, UTC
from typing import Literal, Optional

import chess.pgn
from celery import chain
from celery.result import AsyncResult
from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select

from app import models
from app.celery_app import (
    analyze_game_detailed,
    analyze_game_task,
    celery_app,
    extract_game_id,
    process_player_enhanced,
)
from app.database import get_session
from app.schemas import AnalyzeGameIn, PlayerMetricsOut, TaskQueuedOut
from app.utils import notify_ws, player_lock, redis_client

logger = logging.getLogger(__name__)

# Router para endpoints legacy
legacy_router = APIRouter()

# Constants para Redis
CLEANUP_IN_PROGRESS_KEY = "cleanup_in_progress"
ANALYSIS_IN_PROGRESS_KEY = "analysis_in_progress"


def is_cleanup_in_progress():
    """Check if cleanup is currently in progress."""
    return redis_client.get(CLEANUP_IN_PROGRESS_KEY) is not None


def is_analysis_in_progress():
    """Check if any analysis is currently in progress."""
    return redis_client.get(ANALYSIS_IN_PROGRESS_KEY) is not None


def set_cleanup_in_progress(username: str):
    """Mark cleanup as in progress for a specific user."""
    redis_client.set(CLEANUP_IN_PROGRESS_KEY, username, ex=300)  # 5 minute timeout


def clear_cleanup_in_progress():
    """Clear cleanup in progress flag."""
    redis_client.delete(CLEANUP_IN_PROGRESS_KEY)


def set_analysis_in_progress(username: str):
    """Mark analysis as in progress for a specific user."""
    redis_client.set(ANALYSIS_IN_PROGRESS_KEY, username, ex=3600)  # 1 hour timeout


def clear_analysis_in_progress():
    """Clear analysis in progress flag."""
    redis_client.delete(ANALYSIS_IN_PROGRESS_KEY)


@legacy_router.post("/analyze", response_model=TaskQueuedOut, tags=["legacy"])
def analyze_game_root(request: AnalyzeGameIn, session: Session = Depends(get_session)):
    """Legacy alias for single-game analysis (POST /analyze)."""
    try:
        # Persist game with minimal info – will be updated by Celery
        game_db = models.Game(pgn=request.pgn, move_times=request.move_times)
        session.add(game_db)
        session.commit()
        session.refresh(game_db)

        game_pgn_obj = chess.pgn.read_game(io.StringIO(request.pgn))
        username = (
            game_pgn_obj.headers.get("White")
            or game_pgn_obj.headers.get("Black")
            or "unknown"
        )

        c = chain(
            analyze_game_task.s(request.pgn, game_db.id, move_times=request.move_times),
            extract_game_id.s(),
            analyze_game_detailed.s(username),
        )
        async_res = c.apply_async()

        return TaskQueuedOut(task_id=async_res.id, game_id=game_db.id, status="queued")

    except Exception as exc:  # noqa: BLE001
        session.rollback()
        raise HTTPException(status_code=500, detail=str(exc))


@legacy_router.get("/tasks/{task_id}")
def task_status(task_id: str):
    """Obtiene el estado de una tarea de Celery."""
    try:
        res = AsyncResult(task_id, app=celery_app)

        if res.state == "PENDING":
            return {"state": res.state}
        elif res.state == "FAILURE":
            return {"state": res.state, "error": str(res.info)}
        else:
            return {"state": res.state, "result": res.result}
    except Exception as e:
        logger.error(f"Error obteniendo estado de tarea {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@legacy_router.delete("/tasks/{task_id}")
def stop_task(task_id: str):
    """Detiene una tarea de Celery si existe y está en ejecución."""
    try:
        logger.info(f"Revoking task/group {task_id} with SIGKILL")
        celery_app.control.revoke(task_id, terminate=True, signal="SIGKILL")

        # Esperar brevemente para que los workers procesen la revocación
        time.sleep(1)

        return {
            "task_id": task_id,
            "state": "REVOKED",
            "message": "Task has been stopped successfully",
        }
    except Exception as e:
        logger.error(f"Error al detener la tarea {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@legacy_router.get("/games/{game_id}")
def get_game(game_id: int, session: Session = Depends(get_session)):
    """Obtiene los detalles de una partida analizada."""
    game = session.get(models.Game, game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")

    return {
        "id": game.id,
        "created_at": game.created_at.isoformat(),
        "pgn": game.pgn,
        "white_username": game.white_username,
        "black_username": game.black_username,
        "eco_code": game.eco_code,
        "opening_key": game.opening_key,
        "moves": [
            {
                "move_number": m.move_number,
                "played": m.played,
                "best": m.best,
                "best_rank": m.best_rank,
                "cp_loss": m.cp_loss,
            }
            for m in game.moves
        ]
        if game.moves
        else [],
    }


@legacy_router.post("/games/{game_id}/stop")
def stop_game_analysis(
    game_id: int, task_id: str, session: Session = Depends(get_session)
):
    """Detiene un análisis de partida en progreso."""
    try:
        # Verificar que la partida existe
        game = session.get(models.Game, game_id)
        if not game:
            raise HTTPException(status_code=404, detail="Game not found")

        # Obtener la tarea y revocarla
        res = AsyncResult(task_id, app=celery_app)

        if res.state in ["PENDING", "STARTED"]:
            # Revocar la tarea (terminate=True para forzar la terminación)
            res.revoke(terminate=True)

            return {
                "game_id": game_id,
                "task_id": task_id,
                "status": "stopped",
                "message": "Analysis has been stopped successfully",
            }
        else:
            # La tarea no está en ejecución o ya ha terminado
            return {
                "game_id": game_id,
                "task_id": task_id,
                "status": res.state,
                "message": "Task cannot be stopped because it's not running",
            }
    except Exception as e:
        logger.error(f"Error al detener el análisis para la partida {game_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# JUGADORES LEGACY
# ============================================================


@legacy_router.get("/players/{username}")
def get_player(username: str, session: Session = Depends(get_session)):
    """Obtiene el estado de análisis de un jugador."""
    player = session.get(models.Player, username)

    if not player:
        # Jugador no existe, retornar estado "not_analyzed"
        return {
            "username": username,
            "status": "not_analyzed",
            "progress": 0,
            "message": "Player not analyzed yet. Use POST to start analysis.",
        }

    return {
        "username": player.username,
        "status": player.status,
        "progress": player.progress,
        "total_games": player.total_games,
        "done_games": (player.done_tasks or 0) // 2,
        "requested_at": (
            player.requested_at.isoformat() if player.requested_at else None
        ),
        "finished_at": (
            player.finished_at.isoformat() if player.finished_at else None
        ),
        "error": player.error,
        "last_task_id": player.last_task_id,
    }


@legacy_router.post("/players/{username}", status_code=status.HTTP_202_ACCEPTED)
def analyze_player(
    username: str,
    months: int = 12,
    session: Session = Depends(get_session),
) -> dict[str, str | int | Literal["pending", "already_processing"]]:
    """
    Lanza (o reaprovecha) el análisis completo de *username*.

    • Idempotente: varias peticiones concurrentes jamás crean dos análisis.
    • Distribuido: usa un lock Redis + row-lock para evitar carreras entre pods.
    • Fiable: si el análisis previo murió, lo detecta y lo relanza.
    """
    if is_cleanup_in_progress():
        cleanup_user = redis_client.get(CLEANUP_IN_PROGRESS_KEY)
        raise HTTPException(
            status_code=423,
            detail=f"System is currently cleaning up player '{cleanup_user}'. Please wait and try again.",
        )

    if is_analysis_in_progress():
        active_user = redis_client.get(ANALYSIS_IN_PROGRESS_KEY)
        if active_user != username:
            raise HTTPException(
                status_code=423,
                detail=f"Analysis already in progress for player '{active_user}'. Only one analysis allowed at a time.",
            )

    # 🔒 EXCLUSIÓN A NIVEL DE CLÚSTER (Redis)
    with player_lock(username):
        # ── 1 · Obtener y BLOQUEAR la fila (segunda barrera) ────────────────
        player = session.exec(
            select(models.Player)
            .where(models.Player.username == username)
            .with_for_update(nowait=True)
        ).first()

        # Flag para saber si hay que disparar una tarea nueva
        relaunch_needed = False

        # ── 2 · Decidir qué hacer según el estado actual ────────────────────
        if player:
            if player.status == "pending":
                # ¿La tarea realmente sigue viva?
                alive = (
                    player.last_task_id
                    and AsyncResult(player.last_task_id).state in {"PENDING", "STARTED"}
                )
                if alive:
                    # Análisis en curso → salida idempotente
                    return {
                        "username": username,
                        "status": "already_processing",
                        "task_id": player.last_task_id,
                        "progress": player.progress,
                    }
                # Tarea zombi → relanzar
                relaunch_needed = True

            elif player.status in {"ready", "error"}:
                # Se permite volver a empezar desde cero
                relaunch_needed = True
        else:
            # Primer análisis de este jugador
            player = models.Player(username=username)
            session.add(player)
            relaunch_needed = True

        # ── 3 · (Re)inicializar registro y publicar tarea ───────────────────
        if relaunch_needed:
            now = datetime.now(UTC)
            player.status = models.PlayerStatus.pending
            player.progress = 0
            player.done_tasks = 0
            player.done_games = 0
            player.total_games = 0
            player.requested_at = now
            player.finished_at = None
            player.error = None
            player.last_task_id = None
            session.add(player)
            session.commit()

            # Publicar tarea única
            task = celery_app.send_task('process_player_enhanced', args=[username, months])
            player.last_task_id = task.id
            session.add(player)
            session.commit()

            set_analysis_in_progress(username)

            # Aviso opcional vía WebSocket
            notify_ws(username, {"status": "pending", "progress": 0})

            return {
                "username": username,
                "status": "pending",
                "task_id": task.id,
                "progress": 0,
            }

        # No deberíamos llegar aquí
        raise HTTPException(500, "Estado inesperado en analyze_player")


@legacy_router.post("/players/{username}/refresh")
def refresh_player(username: str, session: Session = Depends(get_session)):
    """Refresca el análisis de un jugador (vuelve a analizar)."""
    try:
        player = session.get(models.Player, username)
        if not player:
            raise HTTPException(status_code=404, detail="Player not found")

        # Resetear estado
        player.status = "pending"
        player.progress = 0
        player.requested_at = datetime.now(UTC)
        player.error = None
        session.commit()
        # Lanzar tarea
        task = celery_app.send_task('process_player_enhanced', args=[username, 12])

        return {"status": "queued", "username": username, "task_id": task.id}
    except Exception as e:
        logger.error(f"Error en refresh para {username}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@legacy_router.get("/players")
def list_players(
    status: Optional[models.PlayerStatus] = None,  # ?status=pending|ready|error
    session: Session = Depends(get_session),
):
    """
    Devuelve la **lista de jugadores** que existen en BD con su estado
    actual.

    • Si se pasa el query-param `status`, filtra por ese estado
      (`pending`, `ready`, `error`).
    • Ordena por `requested_at` descendente para que los más recientes
      aparezcan primero.
    """
    q = select(models.Player)
    if status:
        q = q.where(models.Player.status == status)

    players = session.exec(q.order_by(models.Player.requested_at.desc())).all()
    return [
        {
            "username": p.username,
            "status": p.status,
            "progress": p.progress,
            "total_games": p.total_games,
            "done_games": (p.done_tasks or 0) // 2,
            "requested_at": (
                p.requested_at.isoformat() if p.requested_at else None
            ),
            "finished_at": (
                p.finished_at.isoformat() if p.finished_at else None
            ),
        }
        for p in players
    ]


@legacy_router.get("/players/active")
def list_active_players(session: Session = Depends(get_session)):
    return {
        "active_count": session.exec(
            select(models.Player).where(
                models.Player.status == models.PlayerStatus.pending
            )
        ).count(),
        "analyses": list_players(status=models.PlayerStatus.pending, session=session),
    }


@legacy_router.get("/metrics/game/{game_id}")
def game_metrics(game_id: int, session: Session = Depends(get_session)):
    """
    Devuelve el análisis detallado de una partida ('GameAnalysisDetailed').

    Nota: mantenemos la misma ruta para no romper al cliente,
    pero internamente ya consulta la tabla nueva.
    """
    ga = session.exec(  # 1· usar el modelo nuevo
        select(models.GameAnalysisDetailed).where(
            models.GameAnalysisDetailed.game_id == game_id
        )
    ).first()

    if not ga:
        raise HTTPException(status_code=404, detail="Analysis not found")

    # 2· construir respuesta con las métricas más relevantes
    return {
        "game_id": ga.game_id,
        # ── Calidad ───────────────────────────────
        "acpl": ga.acpl,
        "match_rate": ga.match_rate,
        "weighted_match_rate": ga.weighted_match_rate,
        "ipr": ga.ipr,
        "ipr_z_score": ga.ipr_z_score,
        # ── Tiempo ────────────────────────────────
        "mean_move_time": ga.mean_move_time,
        "time_variance": ga.time_variance,
        "time_complexity_corr": ga.time_complexity_corr,
        "lag_spike_count": ga.lag_spike_count,
        "uniformity_score": ga.uniformity_score,
        # ── Apertura ──────────────────────────────
        "opening_entropy": ga.opening_entropy,
        "novelty_depth": ga.novelty_depth,
        "second_choice_rate": ga.second_choice_rate,
        "opening_breadth": ga.opening_breadth,
        # ── Final (si se calculó) ─────────────────
        "tb_match_rate": ga.tb_match_rate,
        "dtz_deviation": ga.dtz_deviation,
        "conversion_efficiency": ga.conversion_efficiency,
        # ── Score de sospecha ─────────────────────
        "suspicion_score": ga.overall_suspicion_score,
        # ── Metadata ──────────────────────────────
        "analyzed_at": ga.analyzed_at.isoformat(),
    }


@legacy_router.get("/metrics/player/{username}", response_model=PlayerMetricsOut)
def player_metrics(username: str, session: Session = Depends(get_session)):
    # Debug: Try to find the record with a different approach
    from sqlmodel import select
    stmt = select(models.PlayerAnalysisDetailed).where(models.PlayerAnalysisDetailed.username == username)
    obj = session.exec(stmt).first()

    print(f"DEBUG METRICS: Looking for username '{username}', found: {obj is not None}")
    if obj:
        print(f"DEBUG METRICS: Record found - risk_score: {obj.risk_score}, games_analyzed: {obj.games_analyzed}")
    else:
        # Try to see what records exist
        all_records = session.exec(select(models.PlayerAnalysisDetailed)).all()
        print(f"DEBUG METRICS: Total records in table: {len(all_records)}")
        for record in all_records:
            print(f"DEBUG METRICS: Existing record: username='{record.username}'")

    if not obj:
        raise HTTPException(status_code=404, detail="No metrics yet")

    from app.analysis.engine import ChessAnalysisEngine

    engine = ChessAnalysisEngine()
    games_df = engine._get_player_games_with_analysis(username, session)
    from app.analysis import longitudinal

    long_features = longitudinal.aggregate_longitudinal_features(games_df, None)

    def clean_nan_values(value):
        """Recursively convert NaN, inf, -inf to None for JSON serialization."""
        import math
        import numpy as np

        if isinstance(value, (float, np.floating)):
            if math.isnan(value) or math.isinf(value):
                return None
        elif isinstance(value, (int, np.integer)):
            try:
                if np.isnan(value) or np.isinf(value):
                    return None
            except (TypeError, ValueError):
                pass
        elif isinstance(value, np.ndarray):
            return [clean_nan_values(item) for item in value.tolist()]
        elif isinstance(value, list):
            return [clean_nan_values(item) for item in value]
        elif isinstance(value, dict):
            return {k: clean_nan_values(v) for k, v in value.items()}
        return value

    # Always include risk data for API compatibility
    risk_data = {
        "risk_score": obj.risk_score,
        "risk_factors": obj.risk_factors or {},
        "confidence_level": obj.confidence_level,
        "suspicious_games_count": (
            len(obj.suspicious_games_ids) if obj.suspicious_games_ids else 0
        ),
    }

    response_data = obj.dict()
    response_data["risk"] = risk_data

    cleaned_long_features = clean_nan_values(long_features)

    performance_data = obj.performance or {}
    response_data.update(
        {
            "trend_acpl": performance_data.get("trend_acpl"),
            "trend_match_rate": performance_data.get("trend_match_rate"),
            "roi_curve": performance_data.get("roi_curve"),
            "consistency_score": cleaned_long_features.get("consistency_score"),
        }
    )

    if not response_data.get("favorite_openings") and obj.opening_patterns:
        response_data["favorite_openings"] = []

    response_data = clean_nan_values(response_data)

    return response_data


@legacy_router.delete("/players/{username}", status_code=204)
def delete_player(username: str, session: Session = Depends(get_session)):
    player = session.get(models.Player, username)
    if not player:
        raise HTTPException(404, "Player not found")

    # 1. Limpiar flag de cancelación en Redis si existe
    cancellation_key = f"cancel:{username}"
    redis_client.delete(cancellation_key)
    logger.info(f"Cleaned up Redis cancellation flag for {username}")

    # 2. Borrar todas las partidas asociadas a este jugador
    # (tanto como blancas como negras)
    games_to_delete = session.exec(
        select(models.Game).where(
            (models.Game.white_username == username)
            | (models.Game.black_username == username)
        )
    ).all()

    logger.info(f"Found {len(games_to_delete)} games to delete for player {username}")

    for game in games_to_delete:
        session.delete(game)

    # 3. Borrar el jugador (esto también borrará PlayerAnalysisDetailed por CASCADE)
    session.delete(player)

    # 4. Commit todos los cambios
    session.commit()

    logger.info(
        f"Successfully deleted player {username} and {len(games_to_delete)} associated games"
    )


@legacy_router.post("/players/{username}/stop")
def stop_player_analysis(username: str, session: Session = Depends(get_session)):
    player = session.get(models.Player, username)
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")

    try:
        analysis_in_progress = (
            player.status == "pending" or (player.progress and player.progress > 0)
        )

        if not analysis_in_progress or not player.last_task_id:
            return {
                "username": username,
                "status": player.status,
                "message": "No analysis in progress to stop",
            }

        set_cleanup_in_progress(username)

        cancellation_key = f"cancel:{username}"
        redis_client.set(cancellation_key, "1", ex=3600)

        try:
            header_or_task_id = player.last_task_id
            celery_app.control.revoke(
                header_or_task_id, terminate=True, signal="SIGKILL"
            )
        except Exception as e:
            logger.warning(f"Error revoking header/task {player.last_task_id}: {e}")

        try:
            insp = celery_app.control.inspect()
            active = insp.active() or {}
            reserved = insp.reserved() or {}
            scheduled = insp.scheduled() or {}

            def revoke_matching(task_list):
                for t in task_list:
                    args_s = str(t.get("args", []))
                    kwargs_s = str(t.get("kwargs", {}))
                    if username in args_s or username in kwargs_s:
                        tid = t.get("id")
                        if tid:
                            celery_app.control.revoke(
                                tid, terminate=True, signal="SIGKILL"
                            )

            for _, tasks in active.items():
                revoke_matching(tasks or [])
            for _, tasks in reserved.items():
                revoke_matching(tasks or [])
            for _, tasks in scheduled.items():
                revoke_matching(
                    [
                        x.get("request", {}) if isinstance(x, dict) else {}
                        for x in (tasks or [])
                    ]
                )
        except Exception as e:
            logger.warning(f"Inspect/revoke error: {e}")

        player.status = (
            models.PlayerStatus.error if hasattr(models, "PlayerStatus") else "error"
        )
        player.error = "stopped_by_user"
        session.add(player)
        session.commit()

        notify_ws(username, {"status": "stopped", "message": "Analysis stop requested"})

        return {
            "username": username,
            "task_id": player.last_task_id,
            "status": "stopped",
            "message": "Stop signal sent; running tasks will halt shortly",
        }
    except Exception as e:
        session.rollback()
        logger.error(f"Error al solicitar stop para {username}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        clear_cleanup_in_progress()


@legacy_router.post("/players/{username}/reset")
def reset_player(username: str, session: Session = Depends(get_session)):
    """Fuerza el reset del estado de un jugador para permitir re-análisis."""
    player = session.get(models.Player, username)
    if not player:
        raise HTTPException(404, "Player not found")

    # 1. Limpiar flag de cancelación en Redis
    cancellation_key = f"cancel:{username}"
    redis_client.delete(cancellation_key)
    logger.info(f"Cleaned up Redis cancellation flag for {username}")

    # 3. Resetear a estado inicial
    player.status = models.PlayerStatus.not_analyzed
    player.progress = 0
    player.total_games = None
    player.done_tasks = None
    player.done_games = None
    player.requested_at = None
    player.finished_at = None
    player.error = None
    player.last_task_id = None
    session.commit()

    logger.info(f"Successfully reset player {username} to initial state")
    return {"status": "reset", "username": username}


@legacy_router.post("/maintenance/cleanup-orphaned-games")
def cleanup_orphaned_games(session: Session = Depends(get_session)):
    """Limpia partidas huérfanas de jugadores que ya no existen."""
    # Obtener todos los usernames existentes
    existing_players = session.exec(select(models.Player.username)).all()
    existing_usernames = set(existing_players)

    # Buscar partidas con jugadores que no existen
    all_games = session.exec(select(models.Game)).all()
    orphaned_games = []

    for game in all_games:
        white_exists = (
            game.white_username in existing_usernames
            if game.white_username
            else True
        )
        black_exists = (
            game.black_username in existing_usernames
            if game.black_username
            else True
        )

        # Si alguno de los jugadores no existe, la partida es huérfana
        if not white_exists or not black_exists:
            orphaned_games.append(game)

    # Borrar partidas huérfanas
    for game in orphaned_games:
        session.delete(game)

    session.commit()

    logger.info(f"Cleaned up {len(orphaned_games)} orphaned games")

    return {
        "status": "success",
        "orphaned_games_deleted": len(orphaned_games),
        "total_games_checked": len(all_games),
    }


# ============================================================
# STREAMING (SSE)
# ============================================================


@legacy_router.get("/stream/{username}")
async def stream_updates(username: str):
    """Stream de eventos SSE para actualizaciones en tiempo real."""

    async def event_generator():
        pubsub = redis_client.pubsub()
        await pubsub.subscribe(f"player:{username}")

        try:
            async for msg in pubsub.listen():
                if msg["type"] == "message":
                    yield {"data": msg["data"]}
        finally:
            await pubsub.unsubscribe(f"player:{username}")
            await pubsub.close()

    from sse_starlette.sse import EventSourceResponse

    return EventSourceResponse(event_generator())