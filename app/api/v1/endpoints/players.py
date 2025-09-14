from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select

from app import models
from app.database import get_session
from app.celery_app import celery_app, process_player_enhanced
from app.utils import redis_client, notify_ws, player_lock
from app.schemas import (
    PlayerStatusOut,
    PlayerAnalyzeOut,
    PlayerDeleteOut,
    PlayerListItemOut,
)

router = APIRouter()

@router.get(
    "/{username}",
    response_model=PlayerStatusOut,
    summary="Obtener estado de análisis de jugador",
    description="Devuelve el progreso y estado actual del análisis para **username**.",
    responses={404: {"description": "Jugador no encontrado"}},
)
async def get_player(username: str, session: Session = Depends(get_session)):
    """Get player analysis status."""
    player = session.get(models.Player, username)

    if not player:
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
        "done_games": (player.done_tasks or 0) // 2,
        "requested_at": player.requested_at.isoformat() if player.requested_at else None,
        "finished_at": player.finished_at.isoformat() if player.finished_at else None,
        "error": player.error,
        "last_task_id": player.last_task_id
    }

@router.post(
    "/{username}",
    response_model=PlayerAnalyzeOut,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Encolar análisis de jugador",
    description="Inicia el procesamiento de partidas de **username** en los últimos *months* meses.",
)
async def analyze_player(
    username: str,
    months: int = 12,
    session: Session = Depends(get_session),
):
    """Start analyzing a player's games."""
    with player_lock(username):
        player = session.get(models.Player, username)
        
        if player and player.status == "pending":
            return {
                "status": "already_processing",
                "username": username,
                "task_id": player.last_task_id,
                "progress": player.progress
            }
            
        if not player:
            player = models.Player(username=username, status="pending", done_tasks=0, done_games=0)
            session.add(player)
        else:
            player.status = "pending"
            player.progress = 0
            player.error = None
            player.done_tasks = 0
            player.done_games = 0
            
        player.requested_at = datetime.now(timezone.utc)
        player.finished_at = None
        session.commit()
        session.refresh(player)
        
        task_id = "temp-task-id"  # Temporary fix
        player.last_task_id = task_id

        # Try to create the actual task
        try:
            task = celery_app.send_task('process_player_enhanced', args=[username, months])
            if task and hasattr(task, 'id') and task.id:
                task_id = task.id
                player.last_task_id = task_id
        except Exception as e:
            print(f"DEBUG: Error creating task: {e}")
        session.commit()
        
        return {
            "status": "queued",
            "username": username,
            "task_id": task_id
        }

@router.delete(
    "/{username}",
    response_model=PlayerDeleteOut,
    summary="Eliminar análisis de jugador",
    description="Borra al jugador y todos sus datos de análisis.",
    responses={404: {"description": "Jugador no encontrado"}},
)
async def delete_player(username: str, session: Session = Depends(get_session)):
    """Delete a player and their analysis data."""
    player = session.get(models.Player, username)
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")
    
    session.delete(player)
    session.commit()
    return {"status": "deleted", "username": username}

@router.get(
    "/",
    response_model=List[PlayerListItemOut],
    summary="Listar jugadores",
    description="Devuelve la lista de jugadores analizados o en cola con opción de filtrar por *status*.",
)
async def list_players(
    status: Optional[models.PlayerStatus] = None,
    session: Session = Depends(get_session),
):
    """List all players with optional status filter."""
    query = select(models.Player)
    if status:
        query = query.where(models.Player.status == status)
        
    players = session.exec(query.order_by(models.Player.requested_at.desc())).all()
    
    return [
        {
            "username": p.username,
            "status": p.status,
            "progress": p.progress,
            "total_games": p.total_games,
            "done_games": (p.done_tasks or 0) // 2,
            "requested_at": p.requested_at.isoformat() if p.requested_at else None,
            "finished_at": p.finished_at.isoformat() if p.finished_at else None,
        }
        for p in players
    ]
