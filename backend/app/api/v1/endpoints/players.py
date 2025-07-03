from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select

from app import models
from app.database import get_session
from app.celery_app import process_player_enhanced as process_player
from app.utils import redis_client, notify_ws, player_lock

router = APIRouter()

@router.get("/{username}")
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
        "done_games": player.done_games,
        "requested_at": player.requested_at.isoformat() if player.requested_at else None,
        "finished_at": player.finished_at.isoformat() if player.finished_at else None,
        "error": player.error,
        "last_task_id": player.last_task_id
    }

@router.post("/{username}", status_code=status.HTTP_202_ACCEPTED)
async def analyze_player(
    username: str,
    months: int = 6,
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
            player = models.Player(username=username, status="pending")
            session.add(player)
        else:
            player.status = "pending"
            player.progress = 0
            player.error = None
            
        player.requested_at = datetime.now(timezone.utc)
        player.finished_at = None
        session.commit()
        session.refresh(player)
        
        task = process_player.delay(username, months)
        player.last_task_id = task.id
        session.commit()
        
        return {
            "status": "queued",
            "username": username,
            "task_id": task.id
        }

@router.delete("/{username}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_player(username: str, session: Session = Depends(get_session)):
    """Delete a player and their analysis data."""
    player = session.get(models.Player, username)
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")
    
    session.delete(player)
    session.commit()
    return {"status": "deleted", "username": username}

@router.get("/")
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
            "done_games": p.done_games,
            "requested_at": p.requested_at.isoformat() if p.requested_at else None,
            "finished_at": p.finished_at.isoformat() if p.finished_at else None,
        }
        for p in players
    ]
