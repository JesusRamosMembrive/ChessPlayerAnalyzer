from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select

from app import models
from app.database import get_session
from app.celery_app import analyze_game_task, celery_app
from celery.result import AsyncResult

router = APIRouter()

@router.get("/{game_id}")
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

@router.post("/analyze")
async def analyze_game(
    pgn: str,
    move_times: Optional[List[int]] = None,
    session: Session = Depends(get_session)
):
    """Analyze a single game with Stockfish."""
    try:
        # Create game record in database
        game_db = models.Game(pgn=pgn, move_times=move_times)
        session.add(game_db)
        session.commit()
        session.refresh(game_db)

        # Start Celery task for analysis
        task = analyze_game_task.delay(pgn, game_db.id, move_times=move_times)

        return {
            "game_id": game_db.id,
            "task_id": task.id,
            "status": "queued"
        }
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{game_id}/status/{task_id}")
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

@router.post("/{game_id}/cancel/{task_id}")
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
