from typing import Dict, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from app import models
from app.database import get_session
from app.analysis.engine import ChessAnalysisEngine

router = APIRouter()

@router.get("/metrics/game/{game_id}")
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
        "suspicious_quality": detailed.suspicious_quality,
        "suspicious_timing": detailed.suspicious_timing,
        "suspicious_opening": detailed.suspicious_opening,
        "overall_suspicion_score": detailed.overall_suspicion_score,
        "analyzed_at": detailed.analyzed_at.isoformat()
    }

@router.get("/metrics/player/{username}")
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
