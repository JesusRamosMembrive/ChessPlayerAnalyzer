# app/api/v1/endpoints/players.py
"""
Players endpoints for the chess analyzer API.
Unified architecture - standard implementation.
"""
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select

from app.database import get_session
from app.celery_tasks import process_player_enhanced
from app.models import Player, PlayerStatus
from app.schemas import (
    PlayerStatusOut,
    PlayerAnalyzeOut,
    PlayerDeleteOut,
    PlayerListItemOut,
    PlayerMetricsOut,
)
from app.utils import notify_ws, player_lock
from app.services.analysis_lock import get_analysis_lock_service

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
    """Get player analysis status."""

    try:
        player = session.exec(
            select(Player).where(Player.username == username)
        ).first()

        if not player:
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
            "username": player.username,
            "status": player.status.value if hasattr(player.status, 'value') else player.status,
            "progress": player.progress,
            "total_games": player.total_games,
            "done_games": player.done_games,
            "requested_at": player.requested_at.isoformat() if player.requested_at else None,
            "finished_at": player.finished_at.isoformat() if player.finished_at else None,
            "error": player.error,
            "last_task_id": player.last_task_id
        }
    except Exception as e:
        logger.error(f"Error getting player status for {username}: {e}")
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
    """Start player analysis."""

    # Check analysis preconditions using unified lock service
    analysis_lock_service = get_analysis_lock_service()
    preconditions = analysis_lock_service.check_analysis_preconditions(username)

    if not preconditions["can_proceed"]:
        # Return the first conflict found
        conflict = preconditions["conflicts"][0]
        raise HTTPException(
            status_code=423,
            detail=conflict["message"]
        )

    try:
        # Establecer lock de análisis usando el servicio unificado
        analysis_lock_service.set_global_analysis_lock(username)

        # Iniciar análisis usando task estándar
        task = process_player_enhanced.delay(username, force_reanalysis)

        # Actualizar player en BD
        player = session.exec(
            select(Player).where(Player.username == username)
        ).first()

        if not player:
            player = Player(
                username=username,
                status=PlayerStatus.pending,
                requested_at=datetime.now(timezone.utc),
                last_task_id=task.id
            )
            session.add(player)
        else:
            player.status = PlayerStatus.pending
            player.last_task_id = task.id
            player.requested_at = datetime.now(timezone.utc)
            session.add(player)
        session.commit()

        result = {
            "message": f"Analysis started for player {username}",
            "task_id": task.id,
            "username": username,
            "status": "pending"
        }

        # Notificar vía WebSocket
        try:
            notify_ws(username, {
                'type': 'analysis_started',
                'username': username,
                'task_id': result['task_id']
            })
        except Exception as e:
            logger.warning(f"Failed to send WebSocket notification: {e}")

        return result

    except Exception as e:
        # Limpiar lock en caso de error usando el servicio unificado
        analysis_lock_service.clear_global_analysis_lock()
        logger.error(f"Error starting analysis for {username}: {e}")
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
    """List players."""

    try:
        query = select(Player)
        if status:
            query = query.where(Player.status == status)

        query = query.offset(offset).limit(limit)
        players = session.exec(query).all()

        return [
            {
                "username": p.username,
                "status": p.status.value if hasattr(p.status, 'value') else p.status,
                "progress": p.progress,
                "total_games": p.total_games,
                "done_games": p.done_games,
                "requested_at": p.requested_at.isoformat() if p.requested_at else None,
                "finished_at": p.finished_at.isoformat() if p.finished_at else None,
            }
            for p in players
        ]

    except Exception as e:
        logger.error(f"Error listing players: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing players: {str(e)}"
        )


@router.delete(
    "/{username}",
    status_code=204,
    summary="Eliminar jugador",
    description="Elimina un jugador y todos sus datos.",
)
async def delete_player(username: str, session: Session = Depends(get_session)):
    """Delete player."""

    try:
        player = session.exec(
            select(Player).where(Player.username == username)
        ).first()

        if not player:
            raise HTTPException(404, "Player not found")

        session.delete(player)
        session.commit()

        # Notificar vía WebSocket
        try:
            notify_ws(username, {
                'type': 'player_deleted',
                'username': username
            })
        except Exception as e:
            logger.warning(f"Failed to send WebSocket notification: {e}")

        return  # 204 No Content debe estar vacío

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting player {username}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting player: {str(e)}"
        )


@router.post(
    "/{username}/refresh",
    response_model=PlayerAnalyzeOut,
    summary="Refrescar análisis de jugador",
    description="Vuelve a analizar un jugador.",
)
async def refresh_player(username: str, session: Session = Depends(get_session)):
    """Refresh player analysis."""

    # Verificar que el jugador existe
    player = session.exec(
        select(Player).where(Player.username == username)
    ).first()

    if not player:
        raise HTTPException(status_code=404, detail="Player not found")

    # Forzar re-análisis
    return await analyze_player(username, force_reanalysis=True, session=session)


@router.post(
    "/{username}/stop",
    summary="Detener análisis de jugador",
    description="Detiene el análisis en progreso de un jugador.",
)
async def stop_player_analysis(username: str, session: Session = Depends(get_session)):
    """Stop player analysis."""

    try:
        # Obtener estado del jugador
        player = session.exec(
            select(Player).where(Player.username == username)
        ).first()

        if not player:
            raise HTTPException(status_code=404, detail="Player not found")

        task_id = player.last_task_id
        if not task_id:
            raise HTTPException(status_code=400, detail="No active task found")

        # Revocar task usando celery estándar
        from app.celery_tasks import celery_app
        celery_app.control.revoke(task_id, terminate=True)

        # Limpiar locks Redis usando el servicio unificado
        analysis_lock_service = get_analysis_lock_service()
        analysis_lock_service.clear_global_analysis_lock()
        analysis_lock_service.clear_cleanup_lock()

        # Notificar vía WebSocket
        try:
            notify_ws(username, {
                'type': 'analysis_stopped',
                'username': username,
                'task_id': task_id
            })
        except Exception as e:
            logger.warning(f"Failed to send WebSocket notification: {e}")

        return {"message": f"Analysis stopped for player {username}", "task_id": task_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping analysis for {username}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error stopping analysis: {str(e)}"
        )