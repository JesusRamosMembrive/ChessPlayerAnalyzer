# app/api/v1/endpoints/metrics_v2.py
"""
Endpoints de métricas adaptados para usar el sistema V1/V2 según configuración.
Compatible con la interfaz React existente.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from app.database import get_session
from app.adapters import analysis_adapter
from app.config_v2 import config_v2
from app.schemas import PlayerMetricsOut

import logging
logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/player/{username}",
    response_model=PlayerMetricsOut,
    summary="Obtener métricas de jugador (V1/V2)",
    description="Devuelve las métricas completas de análisis para **username**. Compatible con V1 y V2.",
    responses={404: {"description": "No hay métricas disponibles para este jugador"}},
)
async def get_player_metrics_v2(username: str, session: Session = Depends(get_session)):
    """Get player metrics using V1 or V2 based on configuration."""

    if config_v2.DEBUG_V2:
        logger.info(f"GET /metrics/player/{username} - Using engine version: {config_v2.get_engine_version()}")

    try:
        metrics_data = analysis_adapter.get_player_metrics(username, session)
        return metrics_data

    except ValueError as e:
        if "No metrics yet" in str(e):
            raise HTTPException(status_code=404, detail="No metrics yet")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Error getting metrics for player {username}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving player metrics: {str(e)}"
        )


@router.get(
    "/game/{game_id}",
    summary="Obtener métricas de partida (V1/V2)",
    description="Devuelve las métricas de análisis para una partida específica. Compatible con V1 y V2.",
    responses={404: {"description": "Partida no encontrada"}},
)
async def get_game_metrics_v2(game_id: int, session: Session = Depends(get_session)):
    """Get game metrics using V1 or V2 based on configuration."""

    if config_v2.DEBUG_V2:
        logger.info(f"GET /metrics/game/{game_id} - Using engine version: {config_v2.get_engine_version()}")

    try:
        # Por ahora, mantener comportamiento V1 para métricas de partida
        # TODO: Implementar métricas de partida V2 cuando sea necesario
        from app import models as models_v1

        game = session.get(models_v1.Game, game_id)
        if not game:
            raise HTTPException(status_code=404, detail="Game not found")

        detailed = session.get(models_v1.GameAnalysisDetailed, game_id)
        if not detailed:
            raise HTTPException(status_code=404, detail="No analysis available for this game")

        # Convertir a formato de respuesta
        response_data = {
            "game_id": game.id,
            "white_username": game.white_username,
            "black_username": game.black_username,
            "created_at": game.created_at.isoformat() if game.created_at else None,
            "eco_code": game.eco_code,
            "opening_key": game.opening_key,
            # Métricas de análisis detallado
            "acpl": detailed.acpl,
            "wdl_loss": detailed.wdl_loss,
            "match_rate": detailed.match_rate,
            "weighted_match_rate": detailed.weighted_match_rate,
            "ipr": detailed.ipr,
            "ipr_z_score": detailed.ipr_z_score,
            "precision_burst_count": detailed.precision_burst_count,
            "mean_move_time": detailed.mean_move_time,
            "time_variance": detailed.time_variance,
            "time_complexity_corr": detailed.time_complexity_corr,
            "lag_spike_count": detailed.lag_spike_count,
            "uniformity_score": detailed.uniformity_score,
            "clutch_accuracy_diff": detailed.clutch_accuracy_diff,
            "opening_entropy": detailed.opening_entropy,
            "novelty_depth": detailed.novelty_depth,
            "second_choice_rate": detailed.second_choice_rate,
            "opening_breadth": detailed.opening_breadth,
            "tb_match_rate": detailed.tb_match_rate,
            "dtz_deviation": detailed.dtz_deviation,
            "conversion_efficiency": detailed.conversion_efficiency,
            "suspicious_quality": detailed.suspicious_quality,
            "suspicious_timing": detailed.suspicious_timing,
            "suspicious_opening": detailed.suspicious_opening,
            "overall_suspicion_score": detailed.overall_suspicion_score,
            "analyzed_at": detailed.analyzed_at.isoformat() if detailed.analyzed_at else None,
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