# app/api/v1/endpoints/metrics.py
"""
Endpoints de métricas - arquitectura unificada.
Compatible con la interfaz React existente.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from app.database import get_session
from app.models import Player, Game, AnalysisResult
from app.schemas import PlayerMetricsOut

import logging
logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/player/{username}",
    response_model=PlayerMetricsOut,
    summary="Obtener métricas de jugador",
    description="Devuelve las métricas completas de análisis para **username**.",
    responses={404: {"description": "No hay métricas disponibles para este jugador"}},
)
async def get_player_metrics(username: str, session: Session = Depends(get_session)):
    """Get player metrics."""

    try:
        player = session.exec(
            select(Player).where(Player.username == username)
        ).first()

        if not player or not player.aggregated_metrics:
            raise HTTPException(status_code=404, detail="No metrics yet")

        # Las métricas ya están en el formato correcto para React
        return player.aggregated_metrics

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting metrics for player {username}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving player metrics: {str(e)}"
        )


@router.get(
    "/game/{game_id}",
    summary="Obtener métricas de partida",
    description="Devuelve las métricas de análisis para una partida específica.",
    responses={404: {"description": "Partida no encontrada"}},
)
async def get_game_metrics(game_id: int, session: Session = Depends(get_session)):
    """Get game metrics."""

    try:
        game = session.get(Game, game_id)
        if not game:
            raise HTTPException(status_code=404, detail="Game not found")

        # Buscar análisis results para esta partida
        analysis_results = session.exec(
            select(AnalysisResult).where(AnalysisResult.game_id == game_id)
        ).all()

        if not analysis_results:
            raise HTTPException(status_code=404, detail="No analysis available for this game")

        # Para compatibilidad con React, devolver las métricas del primer análisis
        # (o combinar si hay múltiples jugadores)
        primary_analysis = analysis_results[0]

        response_data = {
            "game_id": game.id,
            "white_username": game.white_username,
            "black_username": game.black_username,
            "created_at": game.created_at.isoformat() if game.created_at else None,
            "analyzed_at": primary_analysis.analyzed_at.isoformat(),
            "engine_depth": primary_analysis.engine_depth,
            # Las métricas están en el campo JSON
            **primary_analysis.metrics
        }

        return response_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting game metrics for {game_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving game metrics: {str(e)}"
        )