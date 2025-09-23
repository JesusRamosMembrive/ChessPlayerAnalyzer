# app/api/v1/api_v2.py
"""
Router principal V2 que decide automáticamente qué versión usar según configuración.
"""
from fastapi import APIRouter

from app.api.v1.endpoints import (
    players,
    metrics,
    health,
    tasks,
    games,
)

import logging
logger = logging.getLogger(__name__)

api_router = APIRouter()

# Include all routers
api_router.include_router(
    players.router,
    prefix="/players",
    tags=["players"]
)

api_router.include_router(
    metrics.router,
    prefix="/metrics",
    tags=["metrics"]
)

# Include other routers
api_router.include_router(
    health.router,
    prefix="/health",
    tags=["health"]
)

api_router.include_router(
    tasks.router,
    prefix="/tasks",
    tags=["tasks"]
)

api_router.include_router(
    games.router,
    prefix="/games",
    tags=["games"]
)

logger.info("API router initialized")