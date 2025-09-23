# app/api/v1/endpoints/players_v2.py
"""
Endpoints de players adaptados para usar el sistema V1/V2 según configuración.
Compatible con la interfaz React existente.
"""
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select

from app.database import get_session
from app.adapters import analysis_adapter
from app.config_v2 import config_v2
from app.schemas import (
    PlayerStatusOut,
    PlayerAnalyzeOut,
    PlayerDeleteOut,
    PlayerListItemOut,
    PlayerMetricsOut,
)
from app.utils import redis_client, notify_ws, player_lock

import logging
logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/{username}",
    response_model=PlayerStatusOut,
    summary="Obtener estado de análisis de jugador (V1/V2)",
    description="Devuelve el progreso y estado actual del análisis para **username**. Compatible con V1 y V2.",
    responses={404: {"description": "Jugador no encontrado"}},
)
async def get_player_v2(username: str, session: Session = Depends(get_session)):
    """Get player analysis status using V1 or V2 based on configuration."""

    if config_v2.DEBUG_V2:
        logger.info(f"GET /players/{username} - Using engine version: {config_v2.get_engine_version()}")

    try:
        status_data = analysis_adapter.get_player_status(username, session)
        return status_data
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
    summary="Iniciar análisis de jugador (V1/V2)",
    description="Inicia el análisis completo de todas las partidas de **username**. Compatible con V1 y V2.",
    responses={
        202: {"description": "Análisis iniciado correctamente"},
        423: {"description": "Análisis en progreso para otro usuario"},
        409: {"description": "Usuario ya está siendo analizado"}
    },
)
async def analyze_player_v2(
    username: str,
    force_reanalysis: bool = False,
    session: Session = Depends(get_session)
):
    """Start player analysis using V1 or V2 based on configuration."""

    if config_v2.DEBUG_V2:
        logger.info(f"POST /players/{username} - Using engine version: {config_v2.get_engine_version()}")

    # Verificar locks Redis (mismo comportamiento que V1)
    CLEANUP_IN_PROGRESS_KEY = "cleanup_in_progress"
    ANALYSIS_IN_PROGRESS_KEY = "analysis_in_progress"

    # Check if cleanup is in progress
    if redis_client.get(CLEANUP_IN_PROGRESS_KEY):
        cleanup_user = redis_client.get(CLEANUP_IN_PROGRESS_KEY).decode('utf-8')
        raise HTTPException(
            status_code=423,
            detail=f"Cleanup in progress for user {cleanup_user}. Please wait."
        )

    # Check if analysis is in progress for another user
    if redis_client.get(ANALYSIS_IN_PROGRESS_KEY):
        active_user = redis_client.get(ANALYSIS_IN_PROGRESS_KEY).decode('utf-8')
        if active_user != username:
            raise HTTPException(
                status_code=423,
                detail=f"Analysis in progress for user {active_user}. Please wait."
            )

    try:
        # Establecer lock de análisis
        redis_client.setex(ANALYSIS_IN_PROGRESS_KEY, 7200, username)  # 2 horas

        # Iniciar análisis usando el adaptador
        result = analysis_adapter.start_player_analysis(username, force_reanalysis)

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
        # Limpiar lock en caso de error
        redis_client.delete(ANALYSIS_IN_PROGRESS_KEY)
        logger.error(f"Error starting analysis for {username}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error starting analysis: {str(e)}"
        )


@router.get(
    "/",
    response_model=List[PlayerListItemOut],
    summary="Listar jugadores (V1/V2)",
    description="Lista jugadores con filtros opcionales. Compatible con V1 y V2.",
)
async def list_players_v2(
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    session: Session = Depends(get_session)
):
    """List players using V1 or V2 based on configuration."""

    if config_v2.DEBUG_V2:
        logger.info(f"GET /players - Using engine version: {config_v2.get_engine_version()}")

    try:
        # Por ahora, mantener comportamiento V1 para listado
        # TODO: Implementar listado V2 cuando sea necesario
        from app import models as models_v1

        query = select(models_v1.Player)
        if status:
            query = query.where(models_v1.Player.status == status)

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
    summary="Eliminar jugador (V1/V2)",
    description="Elimina un jugador y todos sus datos. Compatible con V1 y V2.",
)
async def delete_player_v2(username: str, session: Session = Depends(get_session)):
    """Delete player using V1 or V2 based on configuration."""

    if config_v2.DEBUG_V2:
        logger.info(f"DELETE /players/{username} - Using engine version: {config_v2.get_engine_version()}")

    try:
        deleted = analysis_adapter.delete_player(username, session)
        if not deleted:
            raise HTTPException(404, "Player not found")

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
    summary="Refrescar análisis de jugador (V1/V2)",
    description="Vuelve a analizar un jugador. Compatible con V1 y V2.",
)
async def refresh_player_v2(username: str, session: Session = Depends(get_session)):
    """Refresh player analysis using V1 or V2 based on configuration."""

    if config_v2.DEBUG_V2:
        logger.info(f"POST /players/{username}/refresh - Using engine version: {config_v2.get_engine_version()}")

    # Verificar que el jugador existe
    status_data = analysis_adapter.get_player_status(username, session)
    if status_data.get('status') == 'not_analyzed':
        raise HTTPException(status_code=404, detail="Player not found")

    # Forzar re-análisis
    return await analyze_player_v2(username, force_reanalysis=True, session=session)


@router.post(
    "/{username}/stop",
    summary="Detener análisis de jugador (V1/V2)",
    description="Detiene el análisis en progreso de un jugador. Compatible con V1 y V2.",
)
async def stop_player_analysis_v2(username: str, session: Session = Depends(get_session)):
    """Stop player analysis using V1 or V2 based on configuration."""

    if config_v2.DEBUG_V2:
        logger.info(f"POST /players/{username}/stop - Using engine version: {config_v2.get_engine_version()}")

    try:
        # Obtener estado del jugador
        status_data = analysis_adapter.get_player_status(username, session)

        if status_data.get('status') == 'not_analyzed':
            raise HTTPException(status_code=404, detail="Player not found")

        task_id = status_data.get('last_task_id')
        if not task_id:
            raise HTTPException(status_code=400, detail="No active task found")

        # Revocar task según la versión
        if config_v2.get_engine_version() == "v2":
            from app.celery_tasks_v2 import celery_app_v2
            celery_app_v2.control.revoke(task_id, terminate=True)
        else:
            from app.celery_app import celery_app
            celery_app.control.revoke(task_id, terminate=True)

        # Limpiar locks Redis
        redis_client.delete("analysis_in_progress")
        redis_client.delete("cleanup_in_progress")

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