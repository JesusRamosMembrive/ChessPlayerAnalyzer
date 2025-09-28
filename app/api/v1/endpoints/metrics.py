# app/api/v1/endpoints/metrics.py - BULLDOZER TOTAL
"""
BULLDOZER TOTAL: Endpoints de métricas ultra-simplificados.

FILOSOFÍA:
- Usa BULLDOZER API directamente
- Métricas completas con análisis longitudinal
- Compatible con React frontend
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from app.database import get_session
from app.models import GameAnalysis
from app.schemas import PlayerMetricsOut
from sqlmodel import select

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
    """Get player metrics - BULLDOZER version."""
    try:
        # Use BULLDOZER engine with session
        from app.analysis.bulldozer_engine import get_player_analysis_summary

        summary = get_player_analysis_summary(username, session)

        if "error" in summary:
            raise HTTPException(status_code=404, detail=summary["error"])

        # Frontend compatibility fix for missing benchmark fields
        logger.info(f"DEBUG ENDPOINT: benchmark before fix: {summary.get('benchmark')}")
        if "benchmark" in summary and summary["benchmark"] is not None:
            if "percentile_match_rate" not in summary["benchmark"]:
                summary["benchmark"]["percentile_match_rate"] = 50  # Default placeholder
                logger.info(f"DEBUG ENDPOINT: Added percentile_match_rate, benchmark now: {summary['benchmark']}")
        else:
            logger.info(f"DEBUG ENDPOINT: No benchmark found or benchmark is None")

        return summary

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"BULLDOZER: Error getting metrics for {username}: {e}")
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
        # BULLDOZER: Get analysis directly by ID
        analysis = session.get(GameAnalysis, game_id)
        if not analysis:
            raise HTTPException(status_code=404, detail="Game analysis not found")

        # Extract player info from PGN headers
        import chess.pgn
        import io

        try:
            game = chess.pgn.read_game(io.StringIO(analysis.pgn))
            white_username = game.headers.get("White", "") if game else ""
            black_username = game.headers.get("Black", "") if game else ""
        except:
            white_username = ""
            black_username = ""

        response_data = {
            "game_id": analysis.id,
            "white_username": white_username,
            "black_username": black_username,
            "analyzed_username": analysis.analyzed_username,
            "analyzed_at": analysis.analyzed_at.isoformat() if analysis.analyzed_at else None,
            # All metrics are in the analysis JSON field
            **analysis.analysis
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