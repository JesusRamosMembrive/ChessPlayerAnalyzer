"""
Web infrastructure - FastAPI routers using Application Layer.
"""
from .players_v2 import router as players_v2_router

__all__ = [
    "players_v2_router"
]