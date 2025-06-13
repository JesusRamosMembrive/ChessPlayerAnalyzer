from datetime import datetime, UTC
from typing import List
import json

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from celery.result import AsyncResult
from sqlmodel import Session, select

from app import models
from app.celery_app import celery_app, analyze_game_task, process_player
from app.database import get_session
from app.utils import redis_client, notify_ws

# Crea las tablas (desarrollo). En producción usa Alembic.
# models.SQLModel.metadata.create_all(engine)

app = FastAPI(title="Chess Analyzer API", version="0.2.1")

class GameAnalysisRequest(BaseModel):
    pgn: str
    move_times: List[int] = None   # ← opcional

@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(
    req: GameAnalysisRequest,
    session: Session = Depends(get_session),
):
    # 1) crea el registro Game en la BD
    game_db = models.Game(pgn=req.pgn, move_times=req.move_times)
    session.add(game_db)
    session.commit()
    session.refresh(game_db)

    # 2) lanza la tarea Celery
    task = analyze_game_task.delay(req.pgn, game_db.id)

    # 3) respuesta inmediata
    return {
        "game_id": game_db.id,
        "task_id": task.id,
        "state": task.state,
    }


@app.get("/tasks/{task_id}")
def task_status(task_id: str):
    res = AsyncResult(task_id, app=celery_app)
    if res.state == "PENDING":
        return {"state": res.state}
    if res.state == "FAILURE":
        raise HTTPException(status_code=500, detail=str(res.info))
    return {"state": res.state, "result": res.result}


@app.get("/games/{game_id}")
def get_game(game_id: int, session: Session = Depends(get_session)):
    game = session.get(models.Game, game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    return {
        "id": game.id,
        "created_at": game.created_at,
        "pgn": game.pgn,
        "moves": [m.dict(exclude={"game"}) for m in game.moves],
    }

@app.get("/metrics/game/{game_id}")
def metrics_game(game_id: int, session: Session = Depends(get_session)):
    gm = session.exec(select(models.GameMetrics).where(models.GameMetrics.game_id == game_id)).first()
    if not gm:
        raise HTTPException(status_code=404, detail="Metrics not found")
    return gm


@app.get("/metrics/player/{username}")
def player_metrics(username: str, session: Session = Depends(get_session)):
    pm = session.exec(select(models.PlayerMetrics)
                      .where(models.PlayerMetrics.username == username)).first()
    if not pm:
        raise HTTPException(404, "No metrics yet")
    return pm


@app.get("/players/{username}", response_model=models.Player)
def player(username: str, session: Session = Depends(get_session)):
    p = session.get(models.Player, username)
    if p is None:
        # crear registro y lanzar tarea
        p = models.Player(username=username, status="pending")
        p.requested_at = datetime.now(UTC)
        session.add(p)
        session.commit()
        process_player.delay(username)
        notify_ws(username, {"status": "pending"})
    return p

@app.post("/players/{username}/refresh")
def refresh(username: str):
    process_player.delay(username)
    return {"status": "queued"}

@app.get("/stream/{username}")
async def stream(username: str):
    async def event_generator():
        pubsub = redis_client.pubsub()
        await pubsub.subscribe(f"player:{username}")

        async for msg in pubsub.listen():
            if msg["type"] != "message":
                continue
            yield {"data": msg["data"]}     # ya es str
    return EventSourceResponse(event_generator())