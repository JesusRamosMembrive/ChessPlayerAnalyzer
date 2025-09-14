"""
API v2 Players Endpoints
Enhanced player analysis leveraging the new modular architecture.
"""

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from sqlmodel import Session, select
from pydantic import BaseModel, Field

from app import models
from app.database import get_session
from app.analysis.engine_facade import EngineApi
from app.analysis.data_provider import DataProvider
from app.analysis.player_analyzer import PlayerAnalyzer
from app.celery_app import process_player_enhanced as process_player
from app.utils import redis_client, notify_ws, player_lock
from app.schemas import (
    PlayerStatusOut,
    PlayerAnalyzeOut,
    PlayerDeleteOut,
    PlayerListItemOut,
)

router = APIRouter()

# V2-specific schemas
class PlayerAnalysisRequestV2(BaseModel):
    """Enhanced player analysis request with more options."""
    include_openings: bool = Field(True, description="Include opening analysis")
    include_time_analysis: bool = Field(True, description="Include time management analysis")
    include_longitudinal: bool = Field(True, description="Include longitudinal trends")
    include_fairness: bool = Field(False, description="Include fairness/bias analysis")
    depth_limit: Optional[int] = Field(None, ge=1, le=25, description="Stockfish depth limit")
    priority: str = Field("normal", regex="^(low|normal|high|urgent)$", description="Analysis priority")
    enable_streaming: bool = Field(False, description="Enable real-time streaming updates")

class PlayerAnalysisStatusV2(BaseModel):
    """Enhanced player status with detailed progress information."""
    username: str
    status: str
    progress: float = Field(ge=0, le=100, description="Progress percentage")
    games_analyzed: int
    games_total: int
    current_phase: str
    estimated_completion: Optional[datetime] = None
    performance_metrics: Optional[Dict[str, float]] = None
    analysis_config: Optional[PlayerAnalysisRequestV2] = None
    created_at: datetime
    updated_at: datetime
    analysis_url: Optional[str] = None  # Streaming endpoint URL

class PlayerInsightsV2(BaseModel):
    """Enhanced player insights with advanced analytics."""
    username: str
    summary: Dict[str, Any]
    strengths: List[str]
    weaknesses: List[str]
    improvement_suggestions: List[str]
    peer_comparison: Optional[Dict[str, Any]] = None
    trend_analysis: Optional[Dict[str, Any]] = None
    opening_repertoire: Optional[Dict[str, Any]] = None
    time_management: Optional[Dict[str, Any]] = None

# Enhanced endpoints with new architecture
@router.post(
    "/{username}/analyze",
    response_model=PlayerAnalysisStatusV2,
    summary="Start enhanced player analysis",
    description="Start comprehensive player analysis with advanced configuration options using the new modular engine.",
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        202: {"description": "Analysis started successfully"},
        409: {"description": "Analysis already in progress"},
        422: {"description": "Invalid analysis configuration"},
    },
)
async def analyze_player_v2(
    username: str,
    config: PlayerAnalysisRequestV2,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
):
    """Enhanced player analysis with modular architecture."""
    # Check if player already exists and is being analyzed
    existing_player = session.get(models.Player, username)
    if existing_player and existing_player.status == "analyzing":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Player {username} analysis already in progress"
        )

    # Initialize the enhanced analysis using the new modular architecture
    engine_api = EngineApi()
    data_provider = DataProvider()

    try:
        # Create or update player record
        if not existing_player:
            player = models.Player(
                username=username,
                status="pending",
                created_at=datetime.now(timezone.utc)
            )
            session.add(player)
            session.commit()
            session.refresh(player)
        else:
            existing_player.status = "pending"
            existing_player.updated_at = datetime.now(timezone.utc)
            session.commit()
            player = existing_player

        # Estimate analysis scope
        game_count = await data_provider.get_player_game_count(username)
        estimated_completion = None
        if game_count > 0:
            # Use performance gains to estimate completion time
            # With 4.6x speedup, analysis should be significantly faster
            base_time_per_game = 2.0  # seconds (pre-optimization)
            optimized_time_per_game = base_time_per_game / 4.6  # Post-optimization

            total_estimated_seconds = game_count * optimized_time_per_game
            estimated_completion = datetime.now(timezone.utc).timestamp() + total_estimated_seconds

        # Configure analysis parameters
        analysis_params = {
            "username": username,
            "include_openings": config.include_openings,
            "include_time_analysis": config.include_time_analysis,
            "include_longitudinal": config.include_longitudinal,
            "include_fairness": config.include_fairness,
            "depth_limit": config.depth_limit or 15,
            "priority": config.priority,
            "enable_streaming": config.enable_streaming,
        }

        # Start background analysis task with priority queue
        task_priority = {"low": 9, "normal": 5, "high": 3, "urgent": 1}[config.priority]

        # Use the new modular task system
        background_tasks.add_task(
            process_player,
            username,
            **analysis_params
        )

        # Prepare streaming URL if enabled
        streaming_url = f"/api/v2/streaming/players/{username}" if config.enable_streaming else None

        return PlayerAnalysisStatusV2(
            username=username,
            status="pending",
            progress=0.0,
            games_analyzed=0,
            games_total=game_count,
            current_phase="initialization",
            estimated_completion=datetime.fromtimestamp(estimated_completion) if estimated_completion else None,
            performance_metrics=None,
            analysis_config=config,
            created_at=player.created_at,
            updated_at=datetime.now(timezone.utc),
            analysis_url=streaming_url
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to start analysis: {str(e)}"
        )

@router.get(
    "/{username}/status",
    response_model=PlayerAnalysisStatusV2,
    summary="Get enhanced player analysis status",
    description="Get detailed analysis status with performance metrics and progress tracking.",
    responses={404: {"description": "Player not found"}},
)
async def get_player_status_v2(username: str, session: Session = Depends(get_session)):
    """Get enhanced player analysis status."""
    player = session.get(models.Player, username)

    if not player:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Player {username} not found"
        )

    # Get detailed progress from Redis cache
    progress_data = await redis_client.get(f"player:progress:{username}")
    progress_info = {}
    if progress_data:
        import json
        progress_info = json.loads(progress_data)

    # Calculate performance metrics if analysis is ongoing
    performance_metrics = None
    if player.status in ["analyzing", "completed"]:
        performance_metrics = {
            "games_per_minute": progress_info.get("games_per_minute", 0),
            "avg_analysis_time": progress_info.get("avg_analysis_time_ms", 0),
            "speedup_factor": progress_info.get("speedup_factor", 1.0),
            "estimated_remaining_time": progress_info.get("estimated_remaining_seconds", 0)
        }

    return PlayerAnalysisStatusV2(
        username=username,
        status=player.status,
        progress=progress_info.get("progress_percentage", 0),
        games_analyzed=progress_info.get("games_analyzed", 0),
        games_total=progress_info.get("games_total", 0),
        current_phase=progress_info.get("current_phase", "unknown"),
        estimated_completion=progress_info.get("estimated_completion"),
        performance_metrics=performance_metrics,
        analysis_config=None,  # Could be retrieved from cache if needed
        created_at=player.created_at,
        updated_at=player.updated_at,
        analysis_url=f"/api/v2/streaming/players/{username}" if progress_info.get("streaming_enabled") else None
    )

@router.get(
    "/{username}/insights",
    response_model=PlayerInsightsV2,
    summary="Get enhanced player insights and analytics",
    description="Get comprehensive player insights with advanced analytics, trends, and recommendations.",
    responses={404: {"description": "Player analysis not found"}},
)
async def get_player_insights_v2(
    username: str,
    include_peer_comparison: bool = Query(False, description="Include peer group comparison"),
    include_trends: bool = Query(True, description="Include trend analysis"),
    session: Session = Depends(get_session)
):
    """Get enhanced player insights using the new modular architecture."""
    player = session.get(models.Player, username)

    if not player or player.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Completed analysis for player {username} not found"
        )

    # Use the new PlayerAnalyzer for enhanced insights
    player_analyzer = PlayerAnalyzer()

    try:
        # Get comprehensive player insights
        insights = await player_analyzer.get_comprehensive_insights(
            username=username,
            include_peer_comparison=include_peer_comparison,
            include_trends=include_trends
        )

        return PlayerInsightsV2(
            username=username,
            summary=insights.get("summary", {}),
            strengths=insights.get("strengths", []),
            weaknesses=insights.get("weaknesses", []),
            improvement_suggestions=insights.get("improvement_suggestions", []),
            peer_comparison=insights.get("peer_comparison") if include_peer_comparison else None,
            trend_analysis=insights.get("trend_analysis") if include_trends else None,
            opening_repertoire=insights.get("opening_repertoire", {}),
            time_management=insights.get("time_management", {})
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate insights: {str(e)}"
        )

@router.get(
    "/",
    response_model=List[PlayerAnalysisStatusV2],
    summary="List all players with enhanced status",
    description="Get list of all players with detailed analysis status information.",
)
async def list_players_v2(
    status_filter: Optional[str] = Query(None, regex="^(pending|analyzing|completed|failed)$"),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    session: Session = Depends(get_session)
):
    """List players with enhanced status information."""
    query = select(models.Player)

    if status_filter:
        query = query.where(models.Player.status == status_filter)

    query = query.offset(offset).limit(limit)
    players = session.exec(query).all()

    # Convert to enhanced status format
    result = []
    for player in players:
        # Get progress info from cache
        progress_data = await redis_client.get(f"player:progress:{player.username}")
        progress_info = {}
        if progress_data:
            import json
            progress_info = json.loads(progress_data)

        result.append(PlayerAnalysisStatusV2(
            username=player.username,
            status=player.status,
            progress=progress_info.get("progress_percentage", 0),
            games_analyzed=progress_info.get("games_analyzed", 0),
            games_total=progress_info.get("games_total", 0),
            current_phase=progress_info.get("current_phase", "unknown"),
            estimated_completion=progress_info.get("estimated_completion"),
            performance_metrics=None,  # Skip for list view to improve performance
            analysis_config=None,
            created_at=player.created_at,
            updated_at=player.updated_at,
            analysis_url=None
        ))

    return result

# Backward compatibility - delegate to v1 endpoints
@router.get(
    "/{username}",
    response_model=PlayerStatusOut,
    summary="Get player status (v1 compatibility)",
    description="Backward compatible endpoint that returns v1 format.",
    include_in_schema=False,  # Hide from OpenAPI docs to encourage v2 usage
)
async def get_player_v1_compat(username: str, session: Session = Depends(get_session)):
    """Backward compatibility wrapper for v1 API."""
    # Import v1 endpoint and delegate
    from app.api.v1.endpoints.players import get_player as get_player_v1
    return await get_player_v1(username, session)