# app/api/v1/endpoints/players.py - BULLDOZER TOTAL
"""
BULLDOZER TOTAL: Players endpoints ultra-simplificados.

FILOSOFÍA:
- Endpoints simples que usan BULLDOZER API directamente
- Sin complejidad innecesaria, sin locks complejos
- Backward compatibility mantenida
"""
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select

from app.database import get_session
from app.celery_tasks import process_player_bulldozer
from app.models import PlayerProgress, GameAnalysis
from app.bulldozer_api import (
    start_player_analysis_bulldozer,
    get_player_status_bulldozer,
    get_player_metrics_bulldozer,
    get_game_analysis_bulldozer
)
from app.schemas import (
    PlayerStatusOut,
    PlayerAnalyzeOut,
    PlayerDeleteOut,
    PlayerListItemOut,
    PlayerMetricsOut,
)
from app.utils import notify_ws

import logging
logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/{username}",
    response_model=PlayerStatusOut,
    summary="Obtener estado de análisis de jugador",
    description="Devuelve el progreso y estado actual del análisis para **username**.",
    responses={404: {"description": "Jugador no encontrado"}},
)
async def get_player(username: str, session: Session = Depends(get_session)):
    """Get player analysis status - BULLDOZER version."""
    try:
        # Use database session directly - BULLDOZER simple approach
        stmt = select(PlayerProgress).where(PlayerProgress.username == username)
        progress = session.exec(stmt).first()

        if not progress:
            return {
                "username": username,
                "status": "not_analyzed",
                "progress": 0,
                "total_games": 0,
                "done_games": 0,
                "requested_at": None,
                "finished_at": None,
                "error": None,
                "last_task_id": None
            }

        return {
            "username": progress.username,
            "status": progress.status,
            "progress": progress.progress,
            "total_games": progress.total_games,
            "done_games": progress.done_games,
            "requested_at": progress.requested_at.isoformat() if progress.requested_at else None,
            "finished_at": progress.finished_at.isoformat() if progress.finished_at else None,
            "error": progress.error_message,
            "last_task_id": progress.last_task_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"BULLDOZER: Error getting player status for {username}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving player status: {str(e)}"
        )


@router.post(
    "/{username}",
    response_model=PlayerAnalyzeOut,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Iniciar análisis de jugador",
    description="Inicia el análisis completo de todas las partidas de **username**.",
    responses={
        202: {"description": "Análisis iniciado correctamente"},
        423: {"description": "Análisis en progreso para otro usuario"},
        409: {"description": "Usuario ya está siendo analizado"}
    },
)
async def analyze_player(
    username: str,
    force_reanalysis: bool = False,
    session: Session = Depends(get_session)
):
    """Start player analysis - BULLDOZER version."""
    try:
        logger.info(f"BULLDOZER: Starting analysis request for {username}")

        # Check if already in progress
        existing = session.exec(
            select(PlayerProgress).where(PlayerProgress.username == username)
        ).first()

        if existing and existing.status == "pending":
            raise HTTPException(
                status_code=409,
                detail=f"Analysis already in progress for {username}"
            )

        if existing and existing.status == "ready" and not force_reanalysis:
            raise HTTPException(
                status_code=409,
                detail=f"Player {username} already analyzed. Use force_reanalysis=true to re-analyze."
            )

        # Start BULLDOZER analysis using Celery
        task = process_player_bulldozer.delay(username, 12)  # 12 months default

        result = {
            "message": f"BULLDOZER analysis started for player {username}",
            "task_id": task.id,
            "username": username,
            "status": "pending"
        }

        # Notify via WebSocket
        try:
            notify_ws(username, {
                'type': 'analysis_started',
                'username': username,
                'task_id': task.id,
                'bulldozer': True
            })
        except Exception as e:
            logger.warning(f"BULLDOZER: Failed to send WebSocket notification: {e}")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"BULLDOZER: Error starting analysis for {username}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error starting analysis: {str(e)}"
        )


@router.get(
    "/",
    response_model=List[PlayerListItemOut],
    summary="Listar jugadores",
    description="Lista jugadores con filtros opcionales.",
)
async def list_players(
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    session: Session = Depends(get_session)
):
    """List players - BULLDOZER version."""
    try:
        query = select(PlayerProgress)
        if status:
            query = query.where(PlayerProgress.status == status)

        query = query.offset(offset).limit(limit)
        players = session.exec(query).all()

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

    except Exception as e:
        logger.error(f"BULLDOZER: Error listing players: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing players: {str(e)}"
        )


@router.delete(
    "/{username}",
    response_model=PlayerDeleteOut,
    summary="Eliminar jugador",
    description="Elimina completamente un jugador y todos sus análisis.",
)
async def delete_player(username: str, session: Session = Depends(get_session)):
    """Delete player - BULLDOZER version."""
    try:
        # Delete all game analyses for the player
        analyses = session.exec(
            select(GameAnalysis).where(GameAnalysis.analyzed_username == username)
        ).all()

        for analysis in analyses:
            session.delete(analysis)

        # Delete player progress
        progress = session.exec(
            select(PlayerProgress).where(PlayerProgress.username == username)
        ).first()

        if progress:
            session.delete(progress)

        session.commit()

        logger.info(f"BULLDOZER: Deleted player {username} and {len(analyses)} analyses")

        return {
            "username": username,
            "deleted": True,
            "analyses_deleted": len(analyses)
        }

    except Exception as e:
        logger.error(f"BULLDOZER: Error deleting player {username}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting player: {str(e)}"
        )