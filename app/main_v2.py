# app/main_v2.py
"""
API principal adaptado para usar V1 o V2 según configuración.
Mantiene exactamente la misma interfaz externa para React.
"""
import logging
import os
from datetime import datetime, UTC
from typing import List, Optional, Literal

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session, select

# Configuración V2
from app.config_v2 import config_v2

# Import versioned API routers based on configuration
if config_v2.USE_V2_ENGINE:
    from app.api.v1.api_v2 import api_router_v2 as api_router
    logger = logging.getLogger(__name__)
    logger.info("Using V2 API router")
else:
    from app.api.v1 import api_router
    logger = logging.getLogger(__name__)
    logger.info("Using V1 API router")

# Common imports
from app.schemas import PlayerMetricsOut, TaskQueuedOut, AnalyzeGameIn
from app.database import get_session, init_db
from app.error_handlers import register_exception_handlers
from app.middleware.rate_limiter import RateLimitMiddleware
from app.middleware.request_logger import RequestLoggingMiddleware
from app.middleware.trace_context import TraceContextMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

# Application Performance Monitoring (APM)
from app.otel import init_otel, instrument_fastapi

# Utils
from app.utils import notify_ws, player_lock, redis_client

# Configurar logging estructurado (JSON)
from app.logging_config import setup_logging

# Esta llamada es idempotente; si otro módulo ya la ejecutó no tiene efecto.
setup_logging()
logger = logging.getLogger(__name__)

# Log de configuración al inicio
config_v2.log_config()

CLEANUP_IN_PROGRESS_KEY = "cleanup_in_progress"
ANALYSIS_IN_PROGRESS_KEY = "analysis_in_progress"

def is_cleanup_in_progress():
    """Check if cleanup is currently in progress."""
    return redis_client.get(CLEANUP_IN_PROGRESS_KEY) is not None

def is_analysis_in_progress():
    """Check if any analysis is currently in progress."""
    return redis_client.get(ANALYSIS_IN_PROGRESS_KEY) is not None

def set_cleanup_in_progress(username: str):
    """Set cleanup in progress for the given username."""
    redis_client.setex(CLEANUP_IN_PROGRESS_KEY, 3600, username)  # 1 hour

def clear_cleanup_in_progress():
    """Clear cleanup in progress flag."""
    redis_client.delete(CLEANUP_IN_PROGRESS_KEY)

def set_analysis_in_progress(username: str):
    """Set analysis in progress for the given username."""
    redis_client.setex(ANALYSIS_IN_PROGRESS_KEY, 7200, username)  # 2 hours

def clear_analysis_in_progress():
    """Clear analysis in progress flag."""
    redis_client.delete(ANALYSIS_IN_PROGRESS_KEY)

# Inicializar OpenTelemetry
init_otel()

# Crear aplicación FastAPI
app = FastAPI(
    title="Chess Player Analyzer API",
    description="API para análisis avanzado de jugadores de ajedrez usando Stockfish",
    version="2.0.0",  # Incrementar versión para indicar soporte V2
    contact={
        "name": "Chess Analyzer Support",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# Instrumentar con OpenTelemetry
instrument_fastapi(app)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios exactos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Añadir middlewares personalizados
app.add_middleware(RateLimitMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(TraceContextMiddleware)

# Registrar manejadores de errores
register_exception_handlers(app)

# Instrumentar con Prometheus
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

# Incluir routers (V1 o V2 según configuración)
app.include_router(api_router, prefix="/api/v1")

# Importar adaptador para endpoints principales
from app.adapters import analysis_adapter

@app.get("/")
async def root():
    """Root endpoint with API version information."""
    return {
        "message": "Chess Player Analyzer API",
        "version": "2.0.0",
        "engine_version": config_v2.get_engine_version(),
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
    return {"status": "healthy", "engine_version": config_v2.get_engine_version()}

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    logger.info("Starting Chess Player Analyzer API...")
    logger.info(f"Using engine version: {config_v2.get_engine_version()}")

    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

# ====================================================================
# ENDPOINTS PRINCIPALES COMPATIBLES CON REACT
# ====================================================================

@app.get("/players/{username}")
def get_player(username: str, session: Session = Depends(get_session)):
    """Obtiene el estado de análisis de un jugador - Compatible V1/V2."""
    try:
        return analysis_adapter.get_player_status(username, session)
    except Exception as e:
        logger.error(f"Error getting player {username}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/players/{username}", status_code=status.HTTP_202_ACCEPTED)
def analyze_player(
    username: str,
    force_reanalysis: bool = False,
    session: Session = Depends(get_session)
):
    """Inicia análisis de jugador - Compatible V1/V2."""

    # Verificar locks (misma lógica que V1)
    if is_cleanup_in_progress():
        cleanup_user = redis_client.get(CLEANUP_IN_PROGRESS_KEY).decode('utf-8')
        raise HTTPException(
            status_code=423,
            detail=f"Cleanup in progress for user {cleanup_user}. Please wait."
        )

    if is_analysis_in_progress():
        active_user = redis_client.get(ANALYSIS_IN_PROGRESS_KEY)
        if active_user != username:
            raise HTTPException(
                status_code=423,
                detail=f"Analysis in progress for user {active_user}. Please wait."
            )

    try:
        set_analysis_in_progress(username)
        result = analysis_adapter.start_player_analysis(username, force_reanalysis)

        # Notificar WebSocket
        try:
            notify_ws(username, {
                'type': 'analysis_started',
                'username': username,
                'task_id': result['task_id']
            })
        except Exception as e:
            logger.warning(f"Failed to send WebSocket notification: {e}")

        return result

    except Exception as e:
        clear_analysis_in_progress()
        logger.error(f"Error analyzing player {username}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/player/{username}", response_model=PlayerMetricsOut)
def player_metrics(username: str, session: Session = Depends(get_session)):
    """Obtiene métricas de jugador - Compatible V1/V2."""
    try:
        return analysis_adapter.get_player_metrics(username, session)
    except ValueError as e:
        if "No metrics yet" in str(e):
            raise HTTPException(status_code=404, detail="No metrics yet")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting metrics for {username}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/players/{username}", status_code=204)
def delete_player(username: str, session: Session = Depends(get_session)):
    """Elimina jugador - Compatible V1/V2."""
    try:
        deleted = analysis_adapter.delete_player(username, session)
        if not deleted:
            raise HTTPException(404, "Player not found")

        # Notificar WebSocket
        try:
            notify_ws(username, {
                'type': 'player_deleted',
                'username': username
            })
        except Exception as e:
            logger.warning(f"Failed to send WebSocket notification: {e}")

        return {"message": f"Player {username} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting player {username}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tasks/{task_id}")
def task_status(task_id: str):
    """Obtiene el estado de una tarea - Compatible V1/V2."""
    try:
        return analysis_adapter.get_task_status(task_id)
    except Exception as e:
        logger.error(f"Error getting task status {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
    # Por ahora, mantener comportamiento V1
    # TODO: Adaptar a V2 cuando sea necesario
    from app import models as models_v1

    query = select(models_v1.Player)
    if status:
        query = query.where(models_v1.Player.status == status)

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
        "engine_version": config_v2.get_engine_version(),
        "use_v2_engine": config_v2.USE_V2_ENGINE,
        "use_v2_models": config_v2.USE_V2_MODELS,
        "use_v2_tasks": config_v2.USE_V2_TASKS,
        "use_v2_analysis": config_v2.USE_V2_ANALYSIS,
        "hybrid_mode": config_v2.HYBRID_MODE,
        "debug_v2": config_v2.DEBUG_V2,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)