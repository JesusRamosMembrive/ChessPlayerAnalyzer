# app/main.py
"""
API principal de Chess Analyzer.
Versión simplificada que combina lo mejor del original y el refactor.
"""
from datetime import datetime, UTC
from typing import List, Optional, Literal
import logging

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse


from app import models
from app.celery_app import celery_app, analyze_game_task, process_player
from app.database import get_session
from app.utils import redis_client, notify_ws, player_lock

from datetime import datetime, UTC
from fastapi import Depends, HTTPException, status
from sqlmodel import Session, select
from celery.result import AsyncResult

from app.database import get_session
from app import models
from app.celery_app import process_player
from app.utils import player_lock, notify_ws

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
        "error": player.error
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

# ============================================================
# MÉTRICAS
# ============================================================

@app.get("/metrics/game/{game_id}")
def game_metrics(game_id: int, session: Session = Depends(get_session)):
    """Obtiene las métricas calculadas de una partida."""
    gm = session.exec(
        select(models.GameMetrics).where(models.GameMetrics.game_id == game_id)
    ).first()
    
    if not gm:
        raise HTTPException(status_code=404, detail="Metrics not found")
    
    return {
        "game_id": gm.game_id,
        "pct_top1": gm.pct_top1,
        "pct_top3": gm.pct_top3,
        "acl": gm.acl,
        "sigma_total": gm.sigma_total,
        "constant_time": gm.constant_time,
        "pause_spike": gm.pause_spike,
        "suspicious": gm.suspicious,
        "computed_at": gm.computed_at.isoformat()
    }

@app.get("/metrics/player/{username}")
def player_metrics(username: str, session: Session = Depends(get_session)):
    """Obtiene las métricas agregadas de un jugador."""
    pm = session.exec(
        select(models.PlayerMetrics).where(models.PlayerMetrics.username == username)
    ).first()
    
    if not pm:
        raise HTTPException(status_code=404, detail="No metrics yet for this player")
    
    return {
        "username": pm.username,
        "game_count": pm.game_count,
        "opening_entropy": pm.opening_entropy,
        "most_played": pm.most_played,
        "low_entropy": pm.low_entropy,
        "updated_at": pm.updated_at.isoformat()
    }

@app.delete("/players/{username}", status_code=204)
def delete_player(username: str, session: Session = Depends(get_session)):
    player = session.get(models.Player, username)
    if not player:
        raise HTTPException(404, "Player not found")
    # BORRAR análisis relacionados (foreign keys ON DELETE CASCADE)
    session.delete(player)
    session.commit()

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