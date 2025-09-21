"""
New FastAPI router using Application Layer.
Clean endpoints that delegate to handlers.
"""
from typing import Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel

from ....application.handlers.player_handlers import PlayerHandlers
from ....application.handlers.game_handlers import GameHandlers

# Response models
class AnalyzePlayerResponse(BaseModel):
    success: bool
    player_id: Optional[int] = None
    task_id: Optional[str] = None
    message: str

class PlayerStatusResponse(BaseModel):
    username: str
    status: str
    progress: dict
    task_id: Optional[str] = None
    error_message: Optional[str] = None

class PlayerAnalysisResponse(BaseModel):
    username: str
    analysis: dict

class PlayerGamesResponse(BaseModel):
    username: str
    games: list
    pagination: dict

class GameAnalysisResponse(BaseModel):
    game_id: int
    username: str
    game_url: str
    time_control: str
    result: str
    played_at: Optional[str] = None
    analysis: Optional[dict] = None
    moves: Optional[list] = None

# Request models
class AnalyzePlayerRequest(BaseModel):
    force_refresh: bool = False
    priority: int = 5
    months_to_analyze: int = 12

class RefreshPlayerRequest(BaseModel):
    delete_existing_data: bool = True

class DeletePlayerRequest(BaseModel):
    confirm_deletion: bool = False

# Router
router = APIRouter(prefix="/v2", tags=["players-v2"])

# Dependency injection
def get_player_handlers() -> PlayerHandlers:
    return PlayerHandlers()

def get_game_handlers() -> GameHandlers:
    return GameHandlers()


# Player endpoints
@router.post("/players/{username}/analyze", response_model=AnalyzePlayerResponse)
async def analyze_player(
    username: str,
    request: AnalyzePlayerRequest,
    handlers: PlayerHandlers = Depends(get_player_handlers)
):
    """Initiate analysis of a player using new architecture."""
    try:
        result = await handlers.analyze_player(
            username=username,
            force_refresh=request.force_refresh,
            priority=request.priority,
            months_to_analyze=request.months_to_analyze
        )
        return AnalyzePlayerResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/players/{username}/status", response_model=PlayerStatusResponse)
async def get_player_status(
    username: str,
    include_progress_details: bool = True,
    handlers: PlayerHandlers = Depends(get_player_handlers)
):
    """Get current status of a player analysis."""
    try:
        result = await handlers.get_player_status(
            username=username,
            include_progress_details=include_progress_details
        )
        return PlayerStatusResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/players/{username}/analysis", response_model=PlayerAnalysisResponse)
async def get_player_analysis(
    username: str,
    include_games: bool = False,
    include_suspicious_games: bool = False,
    handlers: PlayerHandlers = Depends(get_player_handlers)
):
    """Get complete analysis of a player."""
    try:
        result = await handlers.get_player_analysis(
            username=username,
            include_games=include_games,
            include_suspicious_games=include_suspicious_games
        )
        return PlayerAnalysisResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/players/{username}/refresh", response_model=AnalyzePlayerResponse)
async def refresh_player_analysis(
    username: str,
    request: RefreshPlayerRequest,
    handlers: PlayerHandlers = Depends(get_player_handlers)
):
    """Refresh analysis of an existing player."""
    try:
        result = await handlers.refresh_player_analysis(
            username=username,
            delete_existing_data=request.delete_existing_data
        )
        return AnalyzePlayerResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/players/{username}")
async def delete_player(
    username: str,
    request: DeletePlayerRequest,
    handlers: PlayerHandlers = Depends(get_player_handlers)
):
    """Delete a player and all associated data."""
    try:
        result = await handlers.delete_player(
            username=username,
            confirm_deletion=request.confirm_deletion
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Game endpoints
@router.get("/players/{username}/games", response_model=PlayerGamesResponse)
async def get_player_games(
    username: str,
    limit: int = 20,
    offset: int = 0,
    include_analysis: bool = False,
    time_control_filter: Optional[str] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    only_analyzed: bool = False,
    handlers: GameHandlers = Depends(get_game_handlers)
):
    """Get games of a player with filters."""
    try:
        result = await handlers.get_player_games(
            username=username,
            limit=limit,
            offset=offset,
            include_analysis=include_analysis,
            time_control_filter=time_control_filter,
            date_from=date_from,
            date_to=date_to,
            only_analyzed=only_analyzed
        )
        return PlayerGamesResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/games/{game_id}", response_model=GameAnalysisResponse)
async def get_game_analysis(
    game_id: int,
    include_moves: bool = True,
    include_detailed_metrics: bool = True,
    handlers: GameHandlers = Depends(get_game_handlers)
):
    """Get analysis of a specific game."""
    try:
        result = await handlers.get_game_analysis(
            game_id=game_id,
            include_moves=include_moves,
            include_detailed_metrics=include_detailed_metrics
        )
        return GameAnalysisResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/games/suspicious")
async def get_suspicious_games(
    username: Optional[str] = None,
    risk_threshold: int = 70,
    limit: int = 50,
    include_analysis: bool = True,
    handlers: GameHandlers = Depends(get_game_handlers)
):
    """Get games with high cheat probability."""
    try:
        result = await handlers.get_suspicious_games(
            username=username,
            risk_threshold=risk_threshold,
            limit=limit,
            include_analysis=include_analysis
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/games/{game_id}/analyze")
async def analyze_game(
    game_id: int,
    force_reanalysis: bool = False,
    handlers: GameHandlers = Depends(get_game_handlers)
):
    """Analyze a specific game."""
    try:
        result = await handlers.analyze_game(
            game_id=game_id,
            force_reanalysis=force_reanalysis
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Health check for v2 API
@router.get("/health")
async def health_check_v2():
    """Health check for v2 API using new architecture."""
    try:
        # Test container initialization
        from ....application.container import get_container
        container = get_container()

        # Test that we can get a use case
        use_case = container.get_get_player_status_use_case()

        return {
            "status": "healthy",
            "version": "v2",
            "architecture": "clean",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")