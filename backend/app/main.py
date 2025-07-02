# app/main.py
"""
API principal de Chess Analyzer.
Versión simplificada que combina lo mejor del original y el refactor.
"""
import logging
from datetime import datetime, UTC
from typing import List, Optional, Literal

from app import models
from app.celery_app import celery_app, analyze_game_task, process_player_enhanced as process_player
from app.database import get_session
from app.utils import redis_client, notify_ws, player_lock
from celery.result import AsyncResult
from fastapi import Depends, HTTPException, status
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlmodel import Session, select
from sse_starlette.sse import EventSourceResponse
from app.schemas import PlayerMetricsOut


# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear aplicación FastAPI
app = FastAPI(
    title="Chess Analyzer API",
    version="1.0.0",
    description="Análisis de partidas de ajedrez con Stockfish"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos Pydantic para requests
class GameAnalysisRequest(BaseModel):
    pgn: str
    move_times: Optional[List[int]] = None

# ============================================================
# ENDPOINTS BÁSICOS
# ============================================================

@app.get("/")
def root():
    """Endpoint raíz con información de la API."""
    return {
        "name": "Chess Analyzer API",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "POST /analyze",
            "players": "GET/POST /players/{username}",
            "games": "GET /games/{game_id}",
            "metrics": "GET /metrics/game/{game_id}",
            "stop_analysis": {
                "player": "POST /players/{username}/stop",
                "game": "POST /games/{game_id}/stop?task_id={task_id}"
            }
        }
    }

@app.get("/health")
def health_check():
    """Health check para Docker/Kubernetes."""
    return {"status": "healthy", "timestamp": datetime.now(UTC).isoformat()}

# ============================================================
# ANÁLISIS DE PARTIDAS
# ============================================================

@app.post("/analyze")
def analyze(
    req: GameAnalysisRequest,
    session: Session = Depends(get_session),
):
    """Analiza una partida individual con Stockfish."""
    try:
        # Crear registro en BD
        game_db = models.Game(pgn=req.pgn, move_times=req.move_times)
        session.add(game_db)
        session.commit()
        session.refresh(game_db)

        # Lanzar tarea Celery
        task = analyze_game_task.delay(req.pgn, game_db.id, move_times=req.move_times)

        logger.info(f"Análisis iniciado - Game ID: {game_db.id}, Task ID: {task.id}")

        return {
            "game_id": game_db.id,
            "task_id": task.id,
            "state": task.state,
        }
    except Exception as e:
        logger.error(f"Error en analyze: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tasks/{task_id}")
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

@app.delete("/tasks/{task_id}")
def stop_task(task_id: str):
    """Detiene una tarea de Celery si existe y está en ejecución."""
    try:
        logger.info(f"Revoking task/group {task_id} with SIGKILL")
        celery_app.control.revoke(task_id, terminate=True, signal='SIGKILL')

        # Esperar brevemente para que los workers procesen la revocación
        import time as _t
        _t.sleep(1)

        return {
            "task_id": task_id,
            "state": "REVOKED",
            "message": "Task has been stopped successfully"
        }
    except Exception as e:
        logger.error(f"Error al detener la tarea {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/games/{game_id}")
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
                "cp_loss": m.cp_loss
            } for m in game.moves
        ] if game.moves else [],
    }

@app.post("/games/{game_id}/stop")
def stop_game_analysis(game_id: int, task_id: str, session: Session = Depends(get_session)):
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
                "message": "Analysis has been stopped successfully"
            }
        else:
            # La tarea no está en ejecución o ya ha terminado
            return {
                "game_id": game_id,
                "task_id": task_id,
                "status": res.state,
                "message": "Task cannot be stopped because it's not running"
            }
    except Exception as e:
        logger.error(f"Error al detener el análisis para la partida {game_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# JUGADORES
# ============================================================

@app.get("/players/{username}")
def get_player(username: str, session: Session = Depends(get_session)):
    """Obtiene el estado de análisis de un jugador."""
    player = session.get(models.Player, username)

    if not player:
        # Jugador no existe, retornar estado "not_analyzed"
        return {
            "username": username,
            "status": "not_analyzed",
            "progress": 0,
            "message": "Player not analyzed yet. Use POST to start analysis."
        }

    return {
        "username": player.username,
        "status": player.status,
        "progress": player.progress,
        "total_games": player.total_games,
        "done_games": player.done_games,
        "requested_at": player.requested_at.isoformat() if player.requested_at else None,
        "finished_at": player.finished_at.isoformat() if player.finished_at else None,
        "error": player.error,
        "last_task_id": player.last_task_id
    }

@app.post("/players/{username}", status_code=status.HTTP_202_ACCEPTED)
def analyze_player(
    username: str,
    months: int = 6,
    session: Session = Depends(get_session),
) -> dict[str, str | int | Literal["pending", "already_processing"]]:
    """
    Lanza (o reaprovecha) el análisis completo de *username*.

    • Idempotente: varias peticiones concurrentes jamás crean dos análisis.
    • Distribuido: usa un lock Redis + row-lock para evitar carreras entre pods.
    • Fiable: si el análisis previo murió, lo detecta y lo relanza.
    """
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
                    and AsyncResult(player.last_task_id).state
                    in {"PENDING", "STARTED"}
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
            player.status = "pending"
            player.progress = 0
            player.done_games = 0
            player.total_games = 0
            player.requested_at = now
            player.finished_at = None
            player.error = None
            player.last_task_id = None
            session.add(player)
            session.commit()

            # Publicar tarea única
            task = process_player.delay(username, months)
            player.last_task_id = task.id
            session.add(player)
            session.commit()

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

@app.post("/players/{username}/refresh")
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
        task = process_player.delay(username)

        return {
            "status": "queued",
            "username": username,
            "task_id": task.id
        }
    except Exception as e:
        logger.error(f"Error en refresh para {username}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/players")
def list_players(
    status: Optional[models.PlayerStatus] = None,        # ?status=pending|ready|error
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
            "done_games": p.done_games,
            "requested_at": p.requested_at.isoformat() if p.requested_at else None,
            "finished_at": p.finished_at.isoformat() if p.finished_at else None,
        }
        for p in players
    ]

@app.get("/players/active")
def list_active_players(session: Session = Depends(get_session)):
    return {"active_count": session.exec(
                select(models.Player).where(models.Player.status == models.PlayerStatus.pending)
            ).count(),
            "analyses": list_players(status=models.PlayerStatus.pending, session=session)}

# ============================================================
# MÉTRICAS
# ============================================================

@app.get("/metrics/game/{game_id}")
def game_metrics(game_id: int, session: Session = Depends(get_session)):
    """
    Devuelve el análisis detallado de una partida (‘GameAnalysisDetailed’).

    Nota: mantenemos la misma ruta para no romper al cliente,
    pero internamente ya consulta la tabla nueva.
    """
    ga = session.exec(                         # 1· usar el modelo nuevo
        select(models.GameAnalysisDetailed)
        .where(models.GameAnalysisDetailed.game_id == game_id)
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
        # ── Flags y score ─────────────────────────
        "suspicious_quality": ga.suspicious_quality,
        "suspicious_timing": ga.suspicious_timing,
        "suspicious_opening": ga.suspicious_opening,
        "overall_suspicion_score": ga.overall_suspicion_score,
        # ── Metadata ──────────────────────────────
        "analyzed_at": ga.analyzed_at.isoformat(),
    }

@app.get("/metrics/player/{username}", response_model=PlayerMetricsOut)
def player_metrics(username: str, session: Session = Depends(get_session)):
    obj = session.get(models.PlayerAnalysisDetailed, username)
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

    risk_data = None
    if obj.risk_score > 0 or obj.risk_factors:
        risk_data = {
            "risk_score": obj.risk_score,
            "risk_factors": obj.risk_factors,
            "confidence_level": obj.confidence_level,
            "suspicious_games_count": len(obj.suspicious_games_ids) if obj.suspicious_games_ids else 0
        }

    response_data = obj.dict()
    response_data["risk"] = risk_data

    cleaned_long_features = clean_nan_values(long_features)

    performance_data = obj.performance or {}
    response_data.update({
        "trend_acpl": performance_data.get("trend_acpl"),
        "trend_match_rate": performance_data.get("trend_match_rate"),
        "roi_curve": performance_data.get("roi_curve"),
        "consistency_score": cleaned_long_features.get("consistency_score"),
    })

    if not response_data.get("favorite_openings") and obj.opening_patterns:
        response_data["favorite_openings"] = []

    response_data = clean_nan_values(response_data)

    return response_data

@app.delete("/players/{username}", status_code=204)
def delete_player(username: str, session: Session = Depends(get_session)):
    player = session.get(models.Player, username)
    if not player:
        raise HTTPException(404, "Player not found")
    # BORRAR análisis relacionados (foreign keys ON DELETE CASCADE)
    session.delete(player)
    session.commit()


@app.post("/players/{username}/stop")
def stop_player_analysis(username: str, session: Session = Depends(get_session)):
    """Detiene un análisis de jugador en progreso."""
    player = session.get(models.Player, username)
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")

    try:

        # Verificar si hay un análisis en progreso
        analysis_in_progress = (
            player.status == "pending" or (player.progress and player.progress > 0)
        )

        if not analysis_in_progress or not player.last_task_id:
            # No hay análisis activo o no tenemos task_id registrado
            return {
                "username": username,
                "status": player.status,
                "message": "No analysis in progress to stop"
            }

        task_id = player.last_task_id
        logger.info(f"Starting comprehensive task revocation for player {username}, main task: {task_id}")

        # Set cancellation flag in Redis for immediate task checking
        cancellation_key = f"cancel:{username}"
        redis_client.setex(cancellation_key, 3600, "true")  # Expire after 1 hour
        logger.info(f"Set cancellation flag in Redis: {cancellation_key}")

        # Estrategia más agresiva de revocación
        revoked_tasks = []

        # 1. Revocar la tarea principal
        logger.info(f"Revoking main task {task_id}")
        celery_app.control.revoke(task_id, terminate=True, signal='SIGKILL')
        revoked_tasks.append(task_id)

        # 2. Usar inspect para encontrar todas las tareas activas relacionadas con este usuario
        try:
            inspect = celery_app.control.inspect()
            active_tasks = inspect.active()

            if active_tasks:
                for worker, tasks in active_tasks.items():
                    for task_info in tasks:
                        task_name = task_info.get('name', '')
                        task_args = task_info.get('args', [])
                        current_task_id = task_info.get('id', '')

                        # Buscar tareas relacionadas con este usuario
                        user_related = False
                        if task_name in ['analyze_game_task', 'analyze_game_detailed', 'process_player_enhanced']:
                            # Verificar si el usuario está en los argumentos
                            if username in str(task_args):
                                user_related = True

                        if user_related and current_task_id not in revoked_tasks:
                            logger.info(f"Found related task on worker {worker}: {task_name} ({current_task_id})")
                            celery_app.control.revoke(current_task_id, terminate=True, signal='SIGKILL')
                            revoked_tasks.append(current_task_id)

        except Exception as e:
            logger.warning(f"Could not inspect active tasks: {e}")

        # 3. Revocar usando patrones de nombres de tareas
        try:
            # Intentar revocar todas las tareas que podrían estar relacionadas
            task_patterns = [
                f"analyze_game_task.*{username}",
                f"analyze_game_detailed.*{username}",
                f"process_player_enhanced.*{username}"
            ]

            for pattern in task_patterns:
                try:
                    celery_app.control.revoke(pattern, terminate=True, signal='SIGKILL')
                    logger.info(f"Attempted to revoke tasks matching pattern: {pattern}")
                except Exception as e:
                    logger.debug(f"Pattern revocation failed for {pattern}: {e}")

        except Exception as e:
            logger.warning(f"Pattern-based revocation failed: {e}")

        # 4. Esperar un poco más para que los workers procesen las revocaciones
        import time as _t
        _t.sleep(2)

        # 5. Verificar el estado de la tarea principal
        res = AsyncResult(task_id, app=celery_app)
        if res.state == "REVOKED":
            logger.info(f"Main task {task_id} successfully revoked")
        else:
            logger.warning(f"Main task {task_id} state after revocation: {res.state}")

        # 6. Actualizar el estado del jugador
        player.status = "ready"  # Marcamos como ready para permitir un nuevo análisis
        player.error = "Analysis stopped by user"
        player.finished_at = datetime.now(UTC)
        session.add(player)
        session.commit()

        # 7. Notificar por WebSocket
        notify_ws(username, {"status": "stopped", "message": "Analysis stopped by user"})

        # 8. Clean up cancellation flag after a delay to ensure tasks see it
        import time as _t2
        _t2.sleep(1)  # Give tasks time to see the flag
        redis_client.delete(cancellation_key)
        logger.info(f"Cleaned up cancellation flag: {cancellation_key}")

        logger.info(f"Revocation completed for {username}. Total tasks revoked: {len(revoked_tasks)}")

        return {
            "username": username,
            "task_id": task_id,
            "status": "stopped",
            "message": "Analysis has been stopped successfully",
            "revoked_tasks": len(revoked_tasks)
        }
    except Exception as e:
        logger.error(f"Error al detener el análisis para {username}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/players/{username}/reset")
def reset_player(username: str, session: Session = Depends(get_session)):
    """Fuerza el reset del estado de un jugador para permitir re-análisis."""
    player = session.get(models.Player, username)
    if not player:
        raise HTTPException(404, "Player not found")

    # Resetear a estado inicial
    player.status = "not_analyzed"
    player.progress = 0
    player.total_games = None
    player.done_games = None
    player.requested_at = None
    player.finished_at = None
    player.error = None
    player.last_task_id = None
    session.commit()

    return {"status": "reset", "username": username}

# ============================================================
# STREAMING (SSE)
# ============================================================

@app.get("/stream/{username}")
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

    return EventSourceResponse(event_generator())

# ============================================================
# INICIALIZACIÓN
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
