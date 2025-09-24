# app/main.py
"""
API principal - arquitectura unificada.
Mantiene exactamente la misma interfaz externa para React.
"""
import logging
import os
from datetime import datetime, UTC
from typing import List, Optional, Literal

from fastapi import Depends, HTTPException, status
from sqlmodel import Session, select

# API Router
from app.api.v1.api import api_router

# Common imports
from app.schemas import PlayerMetricsOut, TaskQueuedOut, AnalyzeGameIn
from app.database import get_session, init_db

# Application factories
from app.factories import create_app

# Services
from app.services.analysis_lock import get_analysis_lock_service

# Utils
from app.utils import notify_ws, player_lock

# Configurar logging estructurado (JSON)
from app.logging_config import setup_logging

# Esta llamada es idempotente; si otro módulo ya la ejecutó no tiene efecto.
setup_logging()
logger = logging.getLogger(__name__)

# Get analysis lock service instance
_analysis_lock_service = get_analysis_lock_service()

# Backward compatibility functions (deprecated - use AnalysisLockService directly)
def is_cleanup_in_progress():
    """Check if cleanup is currently in progress."""
    return _analysis_lock_service.is_cleanup_in_progress()

def is_analysis_in_progress():
    """Check if any analysis is currently in progress."""
    return _analysis_lock_service.is_global_analysis_in_progress()

def set_cleanup_in_progress(username: str):
    """Set cleanup in progress for the given username."""
    return _analysis_lock_service.set_cleanup_lock(username)

def clear_cleanup_in_progress():
    """Clear cleanup in progress flag."""
    return _analysis_lock_service.clear_cleanup_lock()

def set_analysis_in_progress(username: str):
    """Set analysis in progress for the given username."""
    return _analysis_lock_service.set_global_analysis_lock(username)

def clear_analysis_in_progress():
    """Clear analysis in progress flag."""
    return _analysis_lock_service.clear_global_analysis_lock()

# Create application using factory
app = create_app()

# Incluir routers principales
app.include_router(api_router, prefix="/api/v1")

# Incluir endpoints de compatibilidad para React (sin prefijo v1)
app.include_router(api_router, prefix="/api")

# Incluir endpoints de compatibilidad para React (nivel raíz)
app.include_router(api_router)

# Imports para endpoints principales
from app.models import Player
from app.celery_tasks import process_player_enhanced

@app.get("/")
async def root():
    """Root endpoint with API version information."""
    return {
        "message": "Chess Player Analyzer API",
        "version": "2.0.0",
        "engine_version": "unified",
        "description": "Advanced chess player analysis using Stockfish engine",
        "endpoints": {
            "players": "/players/{username}",
            "metrics": "/metrics/player/{username}",
            "health": "/health",
            "docs": "/docs"
        },
        "features": [
            "Player game analysis",
            "Statistical metrics",
            "Performance tracking",
            "Real-time progress updates"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for backward compatibility."""
    return {"status": "healthy", "engine_version": "unified"}

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    logger.info("Starting Chess Player Analyzer API...")
    logger.info("Using unified architecture")

    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

# ====================================================================
# NOTA: Los endpoints principales están disponibles en /api/v1/
# Este archivo mantiene solo endpoints de compatibilidad y utilidad
# ====================================================================


# ====================================================================
# ENDPOINTS ADICIONALES (misma funcionalidad que main.py original)
# ====================================================================

@app.post("/analyze", response_model=TaskQueuedOut, tags=["legacy"])
def analyze_game_root(request: AnalyzeGameIn, session: Session = Depends(get_session)):
    """Legacy alias for single-game analysis (POST /analyze)."""
    # Mantener funcionalidad original para compatibilidad
    # TODO: Adaptar a V2 si es necesario
    from app.api.v1.endpoints.analysis import analyze_game
    return analyze_game(request, session)

@app.get("/players")
def list_players(
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    session: Session = Depends(get_session)
):
    """Lista jugadores con filtros."""
    query = select(Player)
    if status:
        query = query.where(Player.status == status)

    query = query.offset(offset).limit(limit)
    players = session.exec(query).all()

    return [
        {
            "username": p.username,
            "status": p.status.value if hasattr(p.status, 'value') else p.status,
            "progress": p.progress,
            "total_games": p.total_games,
            "done_games": p.done_games,
            "requested_at": p.requested_at.isoformat() if p.requested_at else None,
            "finished_at": p.finished_at.isoformat() if p.finished_at else None,
        }
        for p in players
    ]

@app.get("/stream/{username}")
async def stream_updates(username: str):
    """Stream de eventos SSE para actualizaciones en tiempo real."""
    # Mantener funcionalidad original
    from app.api.v1.endpoints.players import stream_updates as original_stream
    return await original_stream(username)

# ====================================================================
# INFORMACIÓN DE CONFIGURACIÓN
# ====================================================================

@app.get("/config/version")
async def get_version_info():
    """Información sobre la versión activa."""
    return {
        "api_version": "2.0.0",
        "engine_version": "unified",
        "architecture": "unified",
        "description": "Unified architecture - V2 is now the only standard"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)