from fastapi import APIRouter

# Import all endpoint routers
from .endpoints import (
    players,
    games,
    analysis,
    health,
    tasks
)

# Create the API router
api_router = APIRouter(
    tags=["v1"]
)

# Include all endpoint routers with proper prefixes
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(players.router, prefix="/players", tags=["players"])
api_router.include_router(games.router, prefix="/games", tags=["games"])
api_router.include_router(analysis.router, prefix="/analysis", tags=["analysis"])
api_router.include_router(tasks.router, prefix="/tasks", tags=["tasks"])
