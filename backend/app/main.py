from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from celery.result import AsyncResult
from sqlmodel import Session, select
from sqlalchemy import or_

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
    game_db = models.Game(pgn=req.pgn, move_times=None)
    session.add(game_db)
    session.commit()
    session.refresh(game_db)

    task = analyze_game_task.delay(req.pgn, game_db.id)
    return {"task_id": task.id, "game_id": game_db.id, "state": task.state}


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


@app.post("/players/{username}/stop")
def stop_player_analysis(username: str, session: Session = Depends(get_session)):
    """Stop analysis for a player and remove all traces from database."""
    try:
        games = session.exec(
            select(models.Game).where(
                or_(models.Game.white_username == username,
                    models.Game.black_username == username)
            )
        ).all()
        
        if not games:
            raise HTTPException(status_code=404, detail=f"No games found for player {username}")
        
        for game in games:
            try:
                celery_app.control.revoke(f"analyze_game_task_{game.id}", terminate=True)
                celery_app.control.revoke(f"compute_game_metrics_{game.id}", terminate=True)
            except Exception as e:
                print(f"Warning: Could not cancel tasks for game {game.id}: {e}")
        
        game_ids = [game.id for game in games]
        
        for move_analysis in session.exec(
            select(models.MoveAnalysis).where(models.MoveAnalysis.game_id.in_(game_ids))
        ).all():
            session.delete(move_analysis)
        
        for game_metrics in session.exec(
            select(models.GameMetrics).where(models.GameMetrics.game_id.in_(game_ids))
        ).all():
            session.delete(game_metrics)
        
        for game in games:
            session.delete(game)
        
        player_metrics = session.exec(
            select(models.PlayerMetrics).where(models.PlayerMetrics.username == username)
        ).first()
        if player_metrics:
            session.delete(player_metrics)
        
        session.commit()
        
        return {"message": f"Analysis stopped and all data removed for player {username}"}
        
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Error stopping analysis: {str(e)}")
