from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select
from celery import chain
import io
import chess.pgn


from app import models
from app.database import get_session
from app.celery_app import analyze_game_task, analyze_game_detailed, extract_game_id, celery_app
from celery.result import AsyncResult
from app.schemas import (
    AnalyzeGameIn,
    TaskQueuedOut,
    GameOut,
    GameAnalysisStatusOut,
    GameAnalysisCancelOut,
)
import hashlib
from app.utils import redis_client

router = APIRouter()

@router.get(
    "/{game_id}",
    response_model=GameOut,
    summary="Obtener los detalles de una partida analizada",
    description="Devuelve la información almacenada y las jugadas evaluadas de la partida identificada por **game_id**.",
    responses={404: {"description": "Partida no encontrada"}},
)
async def get_game(game_id: int, session: Session = Depends(get_session)):
    """Get details of an analyzed game."""
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

@router.post(
    "/analyze",
    response_model=TaskQueuedOut,
    summary="Encolar análisis de una partida",
    description="Crea un registro de partida y lanza una tarea Celery para analizar el PGN proporcionado.",
    status_code=status.HTTP_202_ACCEPTED,
)
async def analyze_game(
    req: AnalyzeGameIn,
    session: Session = Depends(get_session)
):
    """Analyze a single game with Stockfish."""
    # Dedupe: evitar análisis duplicados de la misma partida
    dedupe_key = f"dedupe:analyze_game:{hashlib.sha256(req.pgn.encode('utf-8')).hexdigest()}"
    if not redis_client.setnx(dedupe_key, "1"):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Duplicate analysis request")
    redis_client.expire(dedupe_key, 3600)

    try:
        # Create game record in database
        game_db = models.Game(pgn=req.pgn, move_times=req.move_times)
        session.add(game_db)
        session.commit()
        session.refresh(game_db)

        game_pgn_obj = chess.pgn.read_game(io.StringIO(req.pgn))
        username = game_pgn_obj.headers.get("White") or game_pgn_obj.headers.get("Black") or "unknown"

        c = chain(
            analyze_game_task.s(req.pgn, game_db.id, move_times=req.move_times),
            extract_game_id.s(),
            analyze_game_detailed.s(username),
        )
        async_res = c.apply_async()

        return TaskQueuedOut(game_id=game_db.id, task_id=async_res.id, status="queued")
    except Exception as e:
        session.rollback()
        redis_client.delete(dedupe_key)
        raise HTTPException(status_code=500, detail=str(e))

@router.get(
    "/{game_id}/status/{task_id}",
    response_model=GameAnalysisStatusOut,
    summary="Consultar estado del análisis de partida",
    description="Devuelve el estado actual de la tarea de análisis asociada.",
    responses={404: {"description": "Partida no encontrada"}},
)
async def get_game_analysis_status(
    game_id: int,
    task_id: str,
    session: Session = Depends(get_session)
):
    """Get the status of a game analysis task."""
    # Verify game exists
    game = session.get(models.Game, game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    
    # Get task status
    result = AsyncResult(task_id, app=celery_app)
    
    return {
        "game_id": game_id,
        "task_id": task_id,
        "status": result.state,
        "result": result.result if result.ready() else None
    }

@router.post(
    "/{game_id}/cancel/{task_id}",
    response_model=GameAnalysisCancelOut,
    summary="Cancelar el análisis de una partida",
    description="Revoca la tarea Celery en ejecución y marca el análisis como cancelado.",
    responses={404: {"description": "Partida no encontrada"}},
)
async def cancel_game_analysis(
    game_id: int,
    task_id: str,
    session: Session = Depends(get_session)
):
    """Cancel a running game analysis."""
    # Verify game exists
    game = session.get(models.Game, game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    
    # Revoke the Celery task
    celery_app.control.revoke(task_id, terminate=True)
    
    return {
        "status": "cancelled",
        "game_id": game_id,
        "task_id": task_id
    }
