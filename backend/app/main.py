from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from celery.result import AsyncResult
from sqlmodel import Session

from app.celery_app import celery_app, analyze_game_task
from app.database import get_session, engine
from app import models  # ensure models are registered

# Crea las tablas (desarrollo). En producción usa Alembic.
# models.SQLModel.metadata.create_all(engine)

app = FastAPI(title="Chess Analyzer API", version="0.2.1")


class GameAnalysisRequest(BaseModel):
    pgn: str


@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(req: GameAnalysisRequest, session: Session = Depends(get_session)):
    game = models.Game(pgn=req.pgn)
    session.add(game)
    session.commit()
    session.refresh(game)

    task = analyze_game_task.delay(req.pgn, game.id)
    return {"task_id": task.id, "game_id": game.id, "state": task.state}


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