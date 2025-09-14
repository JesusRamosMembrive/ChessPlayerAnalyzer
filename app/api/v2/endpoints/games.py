"""
API v2 Games Endpoints
Enhanced game operations with performance optimizations.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Query, File, UploadFile
from sqlmodel import Session, select
from pydantic import BaseModel, Field

from app import models
from app.database import get_session
from app.analysis.game_analyzer import GameAnalyzer
from app.analysis.engine_facade import EngineApi
from app.schemas import GameOut

router = APIRouter()

class GameAnalysisV2(BaseModel):
    """Enhanced game analysis result."""
    game_id: int
    analysis_timestamp: datetime
    processing_time_ms: float
    speedup_factor: float

    # Enhanced metrics
    quality_insights: Dict[str, Any]
    tactical_analysis: Dict[str, Any]
    time_management: Dict[str, Any]
    critical_moments: List[Dict[str, Any]]

    # AI-powered insights
    pattern_recognition: Dict[str, Any]
    improvement_suggestions: List[str]
    similarity_to_masters: Optional[float] = None

@router.post(
    "/analyze-pgn",
    response_model=GameAnalysisV2,
    summary="Analyze PGN with enhanced performance",
    description="Analyze a single PGN game with optimized engine (4.6x speedup).",
    status_code=status.HTTP_201_CREATED
)
async def analyze_pgn_v2(
    pgn_content: str = Field(..., description="PGN content to analyze"),
    depth: int = Field(15, ge=1, le=25, description="Analysis depth"),
    include_deep_insights: bool = Field(False, description="Include computationally expensive insights"),
    session: Session = Depends(get_session)
):
    """Analyze PGN with enhanced performance using optimized engine."""
    game_analyzer = GameAnalyzer()
    engine_api = EngineApi()

    start_time = datetime.now()

    try:
        # Use the optimized analysis engine
        analysis_result = await engine_api.analyze_pgn_optimized(
            pgn_content=pgn_content,
            depth=depth,
            include_deep_insights=include_deep_insights
        )

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return GameAnalysisV2(
            game_id=analysis_result.get("game_id", 0),
            analysis_timestamp=datetime.now(),
            processing_time_ms=processing_time,
            speedup_factor=analysis_result.get("speedup_factor", 1.0),
            quality_insights=analysis_result.get("quality_insights", {}),
            tactical_analysis=analysis_result.get("tactical_analysis", {}),
            time_management=analysis_result.get("time_management", {}),
            critical_moments=analysis_result.get("critical_moments", []),
            pattern_recognition=analysis_result.get("pattern_recognition", {}),
            improvement_suggestions=analysis_result.get("improvement_suggestions", []),
            similarity_to_masters=analysis_result.get("similarity_to_masters")
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"PGN analysis failed: {str(e)}"
        )

# Backward compatibility
@router.post(
    "/",
    response_model=GameOut,
    summary="Create game (v1 compatibility)",
    include_in_schema=False
)
async def create_game_v1_compat(
    pgn_content: str,
    session: Session = Depends(get_session)
):
    """Backward compatibility wrapper."""
    from app.api.v1.endpoints.games import create_game as create_game_v1
    return await create_game_v1(pgn_content, session)