from typing import Dict, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from app import models
from app.database import get_session
from app.analysis.engine import ChessAnalysisEngine
from app.analysis.causal_fairness import demographic_parity, equalized_odds
from app.schemas import (
    GameMetricsOut,
    PlayerMetricsSummaryOut,
    FairnessMetricsIn,
    FairnessMetricsOut,
)

router = APIRouter()

@router.get(
    "/metrics/game/{game_id}",
    response_model=GameMetricsOut,
    summary="Obtener métricas detalladas de una partida",
    description="Devuelve métricas de calidad, tiempo y sospecha generadas por el motor de análisis para la partida identificada por **game_id**.",
    responses={
        404: {"description": "Partida o análisis detallado no encontrado"}
    },
)
async def get_game_metrics(game_id: int, session: Session = Depends(get_session)):
    """Get detailed metrics for an analyzed game."""
    game = session.get(models.Game, game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    
    # Check if detailed analysis exists
    detailed = session.get(models.GameAnalysisDetailed, game_id)
    if not detailed:
        raise HTTPException(status_code=404, detail="Detailed analysis not available")
    
    return {
        "game_id": game_id,
        "acpl": detailed.acpl,
        "match_rate": detailed.match_rate,
        "weighted_match_rate": detailed.weighted_match_rate,
        "ipr": detailed.ipr,
        "ipr_z_score": detailed.ipr_z_score,
        "precision_burst_count": detailed.precision_burst_count,
        "mean_move_time": detailed.mean_move_time,
        "time_variance": detailed.time_variance,
        "time_complexity_corr": detailed.time_complexity_corr,
        "anomaly_score": detailed.anomaly_score,
        "suspicion_score": detailed.overall_suspicion_score,
        "analyzed_at": detailed.analyzed_at.isoformat()
    }

@router.get(
    "/metrics/player/{username}",
    response_model=PlayerMetricsSummaryOut,
    summary="Obtener métricas agregadas de un jugador",
    description="Devuelve las métricas promedio y estadísticas resumidas del jugador **username** basadas en todas sus partidas analizadas.",
    responses={
        404: {"description": "Análisis de jugador no encontrado"}
    },
)
async def get_player_metrics(
    username: str,
    session: Session = Depends(get_session)
):
    """Get aggregated metrics for a player."""
    player = session.get(models.Player, username)
    if not player or not player.analysis:
        raise HTTPException(status_code=404, detail="Player analysis not found")
    
    analysis = player.analysis
    
    return {
        "username": username,
        "games_analyzed": analysis.games_analyzed,
        "avg_acpl": analysis.avg_acpl,
        "avg_match_rate": analysis.avg_match_rate,
        "avg_ipr": analysis.avg_ipr,
        "std_acpl": analysis.std_acpl,
        "std_match_rate": analysis.std_match_rate,
        "roi_mean": analysis.roi_mean,
        "roi_max": analysis.roi_max,
        "roi_std": analysis.roi_std,
        "step_function_detected": analysis.step_function_detected,
        "step_function_magnitude": analysis.step_function_magnitude,
        "peer_delta_acpl": analysis.peer_delta_acpl,
        "peer_delta_match": analysis.peer_delta_match,
        "longest_streak": analysis.longest_streak,
        "selectivity_score": analysis.selectivity_score
    }


@router.post(
    "/metrics/fairness",
    response_model=FairnessMetricsOut,
    summary="Calcular métricas de fairness",
    description=(
        "Calcula diferencias de demographic parity y equalized odds para las "
        "predicciones proporcionadas."
    ),
)
async def compute_fairness_metrics(payload: FairnessMetricsIn) -> FairnessMetricsOut:
    """Compute fairness metrics for given predictions and sensitive groups."""
    dp = demographic_parity(payload.y_pred, payload.sensitive_features)["parity_diff"]
    eo = equalized_odds(payload.y_true, payload.y_pred, payload.sensitive_features)
    return FairnessMetricsOut(
        demographic_parity=dp,
        equalized_odds_tpr_diff=eo["tpr_diff"],
        equalized_odds_fpr_diff=eo["fpr_diff"],
    )
