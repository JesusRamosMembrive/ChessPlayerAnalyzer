# app/api/v1/api_v2.py
"""
Router principal V2 que decide automáticamente qué versión usar según configuración.
"""
from fastapi import APIRouter

from app.config_v2 import config_v2
from app.api.v1.endpoints import (
    players_v2,
    metrics_v2,
    # Reutilizar endpoints que no cambian
    health,
    tasks,
    games,
)

import logging
logger = logging.getLogger(__name__)

# Log de configuración al inicio
config_v2.log_config()

api_router_v2 = APIRouter()

# Incluir todos los routers adaptados
api_router_v2.include_router(
    players_v2.router,
    prefix="/players",
    tags=["players-v2"]
)

api_router_v2.include_router(
    metrics_v2.router,
    prefix="/metrics",
    tags=["metrics-v2"]
)

# Incluir routers que no cambian
api_router_v2.include_router(
    health.router,
    prefix="/health",
    tags=["health"]
)

api_router_v2.include_router(
    tasks.router,
    prefix="/tasks",
    tags=["tasks"]
)

api_router_v2.include_router(
    games.router,
    prefix="/games",
    tags=["games"]
)

logger.info(f"API V2 router initialized with engine version: {config_v2.get_engine_version()}")