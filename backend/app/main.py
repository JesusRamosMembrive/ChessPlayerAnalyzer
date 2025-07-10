# app/main.py
"""
API principal de Chess Analyzer.
Versión simplificada que combina lo mejor del original y el refactor.
"""
import logging
from datetime import datetime, UTC
from typing import List, Optional, Literal

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session, select

# Import versioned API routers and models
from app import models
from app.schemas import PlayerMetricsOut
from app.api.v1.endpoints import health as health_endpoints
from app.api.v1 import api_router as v1_router
from app.database import get_session, init_db
from app.error_handlers import register_exception_handlers
from app.middleware.rate_limiter import RateLimitMiddleware  # nuevo middleware
from app.middleware.request_logger import RequestLoggingMiddleware  # nuevo middleware de logging
from app.middleware.trace_context import TraceContextMiddleware  # añade encabezados de traza
from prometheus_fastapi_instrumentator import Instrumentator

# Application Performance Monitoring (APM)
from app.otel import init_otel, instrument_fastapi

# Added utils and Celery app imports for task control and Redis interactions
from app.utils import notify_ws, player_lock, redis_client
from app.celery_app import celery_app, process_player_enhanced
from celery.result import AsyncResult

# Configurar logging estructurado (JSON)
from app.logging_config import setup_logging

# Esta llamada es idempotente; si otro módulo ya la ejecutó no tiene efecto.
setup_logging()
logger = logging.getLogger(__name__)

CLEANUP_IN_PROGRESS_KEY = "cleanup_in_progress"
ANALYSIS_IN_PROGRESS_KEY = "analysis_in_progress"

def is_cleanup_in_progress():
    """Check if cleanup is currently in progress."""
    return redis_client.get(CLEANUP_IN_PROGRESS_KEY) is not None

def is_analysis_in_progress():
    """Check if any analysis is currently in progress."""
    return redis_client.get(ANALYSIS_IN_PROGRESS_KEY) is not None

def set_cleanup_in_progress(username: str):
    """Mark cleanup as in progress for a specific user."""
    redis_client.set(CLEANUP_IN_PROGRESS_KEY, username, ex=300)  # 5 minute timeout

def clear_cleanup_in_progress():
    """Clear cleanup in progress flag."""
    redis_client.delete(CLEANUP_IN_PROGRESS_KEY)

def set_analysis_in_progress(username: str):
    """Mark analysis as in progress for a specific user."""
    redis_client.set(ANALYSIS_IN_PROGRESS_KEY, username, ex=3600)  # 1 hour timeout

def clear_analysis_in_progress():
    """Clear analysis in progress flag."""
    redis_client.delete(ANALYSIS_IN_PROGRESS_KEY)

# Metadatos de etiquetas para la documentación OpenAPI
tags_metadata = [
    {
        "name": "health",
        "description": "Comprobaciones de estado del servicio y disponibilidad.",
    },
    {
        "name": "players",
        "description": "Crear, consultar y administrar análisis de jugadores.",
        "externalDocs": {
            "description": "Documentación de modelo Player",
            "url": "https://github.com/OvertureLabs/ChessPlayerAnalyzer/blob/main/docs/tasks.md#player-flow"
        }
    },
    {
        "name": "games",
        "description": "Operaciones relacionadas con partidas individuales (PGN)."
    },
    {
        "name": "analysis",
        "description": "Endpoints que devuelven métricas de análisis para partidas y jugadores."
    },
    {
        "name": "tasks",
        "description": "Monitorización y control de tareas asíncronas de Celery."
    },
    {
        "name": "legacy",
        "description": "Rutas mantenidas por compatibilidad que serán deprecadas."
    },
    {
        "name": "v1",
        "description": "Enrutador raíz que agrupa todos los endpoints versión 1."
    },
]

# Crear aplicación FastAPI
app = FastAPI(
    title="Chess Analyzer API",
    version="1.0.0",
    description="""
    # Chess Analyzer API

    Bienvenido a la API de **Chess Analyzer**. Este servicio expone endpoints para:

    * Analizar partidas PGN individuales o colecciones completas (jugadores).
    * Obtener métricas de rendimiento (centipawns perdidos, precisión, etc.).
    * Monitorizar, cancelar y reiniciar tareas de análisis en tiempo real.

    ## Versionado
    Actualmente sólo se encuentra disponible la versión **v1**. Todas las rutas están bajo el prefijo `/api/v1/*`.

    ## Respuestas de ejemplo
    En la documentación de cada endpoint encontrarás ejemplos reales de peticiones y respuestas que facilitan la integración.

    ## Estado y contribuciones
    El código está disponible bajo licencia MIT. ¡Se aceptan *pull-requests* y *issues*!
    """,
    terms_of_service="https://github.com/OvertureLabs/ChessPlayerAnalyzer/blob/main/LICENSE",
    contact={
        "name": "Equipo Chess Analyzer",
        "url": "https://github.com/OvertureLabs/ChessPlayerAnalyzer",
        "email": "support@chessplayeranalyzer.io",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    openapi_url="/api/v1/openapi.json",
    openapi_tags=tags_metadata,
)

# Registrar manejadores de errores personalizados
register_exception_handlers(app)

# Instrumentar FastAPI con OpenTelemetry (debe ser antes de iniciar)
instrument_fastapi(app)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ───────────────────────────────────────────────────────────
# Rate limiting global
# ───────────────────────────────────────────────────────────
# Se puede ajustar mediante variables de entorno:
#   RATE_LIMIT_MAX_REQUESTS (por defecto 100)
#   RATE_LIMIT_WINDOW_SECONDS (por defecto 60)
app.add_middleware(RateLimitMiddleware)

# Middleware de logging de peticiones
app.add_middleware(RequestLoggingMiddleware)

# Middleware que añade traceparent/X-Trace-Id
app.add_middleware(TraceContextMiddleware)

# ───────────────────────────────────────────────────────────
# Métricas Prometheus
# ───────────────────────────────────────────────────────────
# Esto añade el endpoint `/metrics` y registra métricas básicas
Instrumentator().instrument(app).expose(app)

# Include versioned API routers
app.include_router(health_endpoints.router, prefix="/api/v1", tags=["health"])
app.include_router(v1_router, prefix="/api/v1")

# Root endpoint for API discovery
@app.get("/")
async def root():
    """Root endpoint with API version information."""
    return {
        "name": "Chess Analyzer API",
        "version": "1.0.0",
        "documentation": "/api/v1/docs",
        "api_versions": ["v1"],
        "current_version": "v1",
        "endpoints": {
            "v1": {
                "documentation": "/api/v1/docs",
                "openapi_schema": "/api/v1/openapi.json",
                "health": "/api/v1/health"
            }
        }
    }

# Health check endpoint for backward compatibility
@app.get("/health")
async def health_check():
    """Health check endpoint for backward compatibility."""
    return {"status": "healthy", "version": "v1", "timestamp": datetime.now(UTC).isoformat()}

# Initialize database tables on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    init_db()
    logger.info("Application startup: Database initialized")
    # Inicializar instrumentación de SQLAlchemy y Celery
    init_otel()
    logger.info("Application startup: OpenTelemetry SQLAlchemy/Celery initialized")

# ============================================================
# LEGACY ENDPOINTS (for backward compatibility)
# These will be deprecated in a future version
# ============================================================

# Import the legacy endpoints at the bottom of the file to avoid circular imports
from fastapi import APIRouter

# Create a router for legacy endpoints
legacy_router = APIRouter()

# Import and include the versioned routers for legacy compatibility
from app.api.v1.endpoints import games as v1_games
from app.api.v1.endpoints import players as v1_players
from app.api.v1.endpoints import analysis as v1_analysis

# Map legacy routes to versioned endpoints
legacy_router.include_router(v1_games.router, prefix="/games", tags=["legacy"])
legacy_router.include_router(v1_players.router, prefix="/players", tags=["legacy"])
legacy_router.include_router(v1_analysis.router, prefix="/analyze", tags=["legacy"])

# Include the legacy router
app.include_router(legacy_router)

# ────────────────────────────────────────────────────────────────────────────
# Back-compat single-game analyze endpoint using new validation models
# This mirrors /api/v1/games/analyze but keeps the old path used by tests.
# ────────────────────────────────────────────────────────────────────────────

from app.schemas import AnalyzeGameIn, TaskQueuedOut  # pylint: disable=wrong-import-position
from app.celery_app import analyze_game_task  # pylint: disable=wrong-import-position


@app.post("/analyze", response_model=TaskQueuedOut, tags=["legacy"])
def analyze_game_root(request: AnalyzeGameIn, session: Session = Depends(get_session)):
    """Legacy alias for single-game analysis (POST /analyze)."""
    try:
        
        # Persist game with minimal info – will be updated by Celery
        game_db = models.Game(pgn=request.pgn, move_times=request.move_times)
        session.add(game_db)
        session.commit()
        session.refresh(game_db)

        task = analyze_game_task.delay(request.pgn, game_db.id, move_times=request.move_times)
        
        return TaskQueuedOut(
            task_id=task.id,
            game_id=game_db.id,
            status="queued"
        )
        
    except Exception as exc:  # noqa: BLE001
        session.rollback()
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/tasks/{task_id}")
def task_status(task_id: str):
    """Obtiene el estado de una tarea de Celery."""
    try:
        res = AsyncResult(task_id, app=celery_app)

        if res.state == "PENDING":
            return {"state": res.state}
        elif res.state == "FAILURE":
            return {"state": res.state, "error": str(res.info)}
        else:
            return {"state": res.state, "result": res.result}
    except Exception as e:
        logger.error(f"Error obteniendo estado de tarea {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/tasks/{task_id}")
def stop_task(task_id: str):
    """Detiene una tarea de Celery si existe y está en ejecución."""
    try:
        logger.info(f"Revoking task/group {task_id} with SIGKILL")
        celery_app.control.revoke(task_id, terminate=True, signal='SIGKILL')

        # Esperar brevemente para que los workers procesen la revocación
        import time as _t
        _t.sleep(1)

        return {
            "task_id": task_id,
            "state": "REVOKED",
            "message": "Task has been stopped successfully"
        }
    except Exception as e:
        logger.error(f"Error al detener la tarea {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/games/{game_id}")
def get_game(game_id: int, session: Session = Depends(get_session)):
    """Obtiene los detalles de una partida analizada."""
    game = session.get(models.Game, game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")

    return {
        "id": game.id,
        "created_at": game.created_at.isoformat(),
        "pgn": game.pgn,
        "white_username": game.white_username,
        "black_username": game.black_username,
        "eco_code": game.eco_code,
        "opening_key": game.opening_key,
        "moves": [
            {
                "move_number": m.move_number,
                "played": m.played,
                "best": m.best,
                "best_rank": m.best_rank,
                "cp_loss": m.cp_loss
            } for m in game.moves
        ] if game.moves else [],
    }

@app.post("/games/{game_id}/stop")
def stop_game_analysis(game_id: int, task_id: str, session: Session = Depends(get_session)):
    """Detiene un análisis de partida en progreso."""
    try:
        # Verificar que la partida existe
        game = session.get(models.Game, game_id)
        if not game:
            raise HTTPException(status_code=404, detail="Game not found")

        # Obtener la tarea y revocarla
        res = AsyncResult(task_id, app=celery_app)

        if res.state in ["PENDING", "STARTED"]:
            # Revocar la tarea (terminate=True para forzar la terminación)
            res.revoke(terminate=True)

            return {
                "game_id": game_id,
                "task_id": task_id,
                "status": "stopped",
                "message": "Analysis has been stopped successfully"
            }
        else:
            # La tarea no está en ejecución o ya ha terminado
            return {
                "game_id": game_id,
                "task_id": task_id,
                "status": res.state,
                "message": "Task cannot be stopped because it's not running"
            }
    except Exception as e:
        logger.error(f"Error al detener el análisis para la partida {game_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# JUGADORES
# ============================================================

@app.get("/players/{username}")
def get_player(username: str, session: Session = Depends(get_session)):
    """Obtiene el estado de análisis de un jugador."""
    player = session.get(models.Player, username)

    if not player:
        # Jugador no existe, retornar estado "not_analyzed"
        return {
            "username": username,
            "status": "not_analyzed",
            "progress": 0,
            "message": "Player not analyzed yet. Use POST to start analysis."
        }

    return {
        "username": player.username,
        "status": player.status,
        "progress": player.progress,
        "total_games": player.total_games,
        "done_games": player.done_games,
        "requested_at": player.requested_at.isoformat() if player.requested_at else None,
        "finished_at": player.finished_at.isoformat() if player.finished_at else None,
        "error": player.error,
        "last_task_id": player.last_task_id
    }

@app.post("/players/{username}", status_code=status.HTTP_202_ACCEPTED)
def analyze_player(
    username: str,
    months: int = 12,
    session: Session = Depends(get_session),
) -> dict[str, str | int | Literal["pending", "already_processing"]]:
    """
    Lanza (o reaprovecha) el análisis completo de *username*.

    • Idempotente: varias peticiones concurrentes jamás crean dos análisis.
    • Distribuido: usa un lock Redis + row-lock para evitar carreras entre pods.
    • Fiable: si el análisis previo murió, lo detecta y lo relanza.
    """
    if is_cleanup_in_progress():
        cleanup_user = redis_client.get(CLEANUP_IN_PROGRESS_KEY).decode('utf-8')
        raise HTTPException(
            status_code=423, 
            detail=f"System is currently cleaning up player '{cleanup_user}'. Please wait and try again."
        )
    
    if is_analysis_in_progress():
        active_user = redis_client.get(ANALYSIS_IN_PROGRESS_KEY).decode('utf-8')
        if active_user != username:
            raise HTTPException(
                status_code=423,
                detail=f"Analysis already in progress for player '{active_user}'. Only one analysis allowed at a time."
            )
    
    # 🔒 EXCLUSIÓN A NIVEL DE CLÚSTER (Redis)
    with player_lock(username):
        # ── 1 · Obtener y BLOQUEAR la fila (segunda barrera) ────────────────
        player = session.exec(
            select(models.Player)
            .where(models.Player.username == username)
            .with_for_update(nowait=True)
        ).first()

        # Flag para saber si hay que disparar una tarea nueva
        relaunch_needed = False

        # ── 2 · Decidir qué hacer según el estado actual ────────────────────
        if player:
            if player.status == "pending":
                # ¿La tarea realmente sigue viva?
                alive = (
                    player.last_task_id
                    and AsyncResult(player.last_task_id).state
                    in {"PENDING", "STARTED"}
                )
                if alive:
                    # Análisis en curso → salida idempotente
                    return {
                        "username": username,
                        "status": "already_processing",
                        "task_id": player.last_task_id,
                        "progress": player.progress,
                    }
                # Tarea zombi → relanzar
                relaunch_needed = True

            elif player.status in {"ready", "error"}:
                # Se permite volver a empezar desde cero
                relaunch_needed = True
        else:
            # Primer análisis de este jugador
            player = models.Player(username=username)
            session.add(player)
            relaunch_needed = True

        # ── 3 · (Re)inicializar registro y publicar tarea ───────────────────
        if relaunch_needed:
            now = datetime.now(UTC)
            player.status = models.PlayerStatus.pending
            player.progress = 0
            player.done_games = 0
            player.total_games = 0
            player.requested_at = now
            player.finished_at = None
            player.error = None
            player.last_task_id = None
            session.add(player)
            session.commit()

            # Publicar tarea única
            task = process_player_enhanced.delay(username, months)
            player.last_task_id = task.id
            session.add(player)
            session.commit()
            
            set_analysis_in_progress(username)

            # Aviso opcional vía WebSocket
            notify_ws(username, {"status": "pending", "progress": 0})

            return {
                "username": username,
                "status": "pending",
                "task_id": task.id,
                "progress": 0,
            }

        # No deberíamos llegar aquí
        raise HTTPException(500, "Estado inesperado en analyze_player")

@app.post("/players/{username}/refresh")
def refresh_player(username: str, session: Session = Depends(get_session)):
    """Refresca el análisis de un jugador (vuelve a analizar)."""
    try:
        player = session.get(models.Player, username)
        if not player:
            raise HTTPException(status_code=404, detail="Player not found")

        # Resetear estado
        player.status = "pending"
        player.progress = 0
        player.requested_at = datetime.now(UTC)
        player.error = None
        session.commit()
        # Lanzar tarea
        task = process_player_enhanced.delay(username, 12)  # 12 meses por defecto

        return {
            "status": "queued",
            "username": username,
            "task_id": task.id
        }
    except Exception as e:
        logger.error(f"Error en refresh para {username}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/players")
def list_players(
    status: Optional[models.PlayerStatus] = None,        # ?status=pending|ready|error
    session: Session = Depends(get_session),
):
    """
    Devuelve la **lista de jugadores** que existen en BD con su estado
    actual.

    • Si se pasa el query-param `status`, filtra por ese estado
      (`pending`, `ready`, `error`).
    • Ordena por `requested_at` descendente para que los más recientes
      aparezcan primero.
    """
    q = select(models.Player)
    if status:
        q = q.where(models.Player.status == status)

    players = session.exec(q.order_by(models.Player.requested_at.desc())).all()
    return [
        {
            "username": p.username,
            "status": p.status,
            "progress": p.progress,
            "total_games": p.total_games,
            "done_games": p.done_games,
            "requested_at": p.requested_at.isoformat() if p.requested_at else None,
            "finished_at": p.finished_at.isoformat() if p.finished_at else None,
        }
        for p in players
    ]

@app.get("/players/active")
def list_active_players(session: Session = Depends(get_session)):
    return {"active_count": session.exec(
                select(models.Player).where(models.Player.status == models.PlayerStatus.pending)
            ).count(),
            "analyses": list_players(status=models.PlayerStatus.pending, session=session)}

# ============================================================
# MÉTRICAS
# ============================================================

@app.get("/metrics/game/{game_id}")
def game_metrics(game_id: int, session: Session = Depends(get_session)):
    """
    Devuelve el análisis detallado de una partida (‘GameAnalysisDetailed’).

    Nota: mantenemos la misma ruta para no romper al cliente,
    pero internamente ya consulta la tabla nueva.
    """
    ga = session.exec(                         # 1· usar el modelo nuevo
        select(models.GameAnalysisDetailed)
        .where(models.GameAnalysisDetailed.game_id == game_id)
    ).first()

    if not ga:
        raise HTTPException(status_code=404, detail="Analysis not found")

    # 2· construir respuesta con las métricas más relevantes
    return {
        "game_id": ga.game_id,
        # ── Calidad ───────────────────────────────
        "acpl": ga.acpl,
        "match_rate": ga.match_rate,
        "weighted_match_rate": ga.weighted_match_rate,
        "ipr": ga.ipr,
        "ipr_z_score": ga.ipr_z_score,
        # ── Tiempo ────────────────────────────────
        "mean_move_time": ga.mean_move_time,
        "time_variance": ga.time_variance,
        "time_complexity_corr": ga.time_complexity_corr,
        "lag_spike_count": ga.lag_spike_count,
        "uniformity_score": ga.uniformity_score,
        # ── Apertura ──────────────────────────────
        "opening_entropy": ga.opening_entropy,
        "novelty_depth": ga.novelty_depth,
        "second_choice_rate": ga.second_choice_rate,
        "opening_breadth": ga.opening_breadth,
        # ── Final (si se calculó) ─────────────────
        "tb_match_rate": ga.tb_match_rate,
        "dtz_deviation": ga.dtz_deviation,
        "conversion_efficiency": ga.conversion_efficiency,
        # ── Flags y score ─────────────────────────
        "suspicious_quality": ga.suspicious_quality,
        "suspicious_timing": ga.suspicious_timing,
        "suspicious_opening": ga.suspicious_opening,
        "overall_suspicion_score": ga.overall_suspicion_score,
        # ── Metadata ──────────────────────────────
        "analyzed_at": ga.analyzed_at.isoformat(),
    }

@app.get("/metrics/player/{username}", response_model=PlayerMetricsOut)
def player_metrics(username: str, session: Session = Depends(get_session)):
    obj = session.get(models.PlayerAnalysisDetailed, username)
    if not obj:
        raise HTTPException(status_code=404, detail="No metrics yet")

    from app.analysis.engine import ChessAnalysisEngine
    engine = ChessAnalysisEngine()
    games_df = engine._get_player_games_with_analysis(username, session)
    from app.analysis import longitudinal
    long_features = longitudinal.aggregate_longitudinal_features(games_df, None)

    def clean_nan_values(value):
        """Recursively convert NaN, inf, -inf to None for JSON serialization."""
        import math
        import numpy as np
        if isinstance(value, (float, np.floating)):
            if math.isnan(value) or math.isinf(value):
                return None
        elif isinstance(value, (int, np.integer)):
            try:
                if np.isnan(value) or np.isinf(value):
                    return None
            except (TypeError, ValueError):
                pass
        elif isinstance(value, np.ndarray):
            return [clean_nan_values(item) for item in value.tolist()]
        elif isinstance(value, list):
            return [clean_nan_values(item) for item in value]
        elif isinstance(value, dict):
            return {k: clean_nan_values(v) for k, v in value.items()}
        return value

    risk_data = None
    if obj.risk_score > 0 or obj.risk_factors:
        risk_data = {
            "risk_score": obj.risk_score,
            "risk_factors": obj.risk_factors,
            "confidence_level": obj.confidence_level,
            "suspicious_games_count": len(obj.suspicious_games_ids) if obj.suspicious_games_ids else 0
        }

    response_data = obj.dict()
    response_data["risk"] = risk_data

    cleaned_long_features = clean_nan_values(long_features)

    performance_data = obj.performance or {}
    response_data.update({
        "trend_acpl": performance_data.get("trend_acpl"),
        "trend_match_rate": performance_data.get("trend_match_rate"),
        "roi_curve": performance_data.get("roi_curve"),
        "consistency_score": cleaned_long_features.get("consistency_score"),
    })

    if not response_data.get("favorite_openings") and obj.opening_patterns:
        response_data["favorite_openings"] = []

    response_data = clean_nan_values(response_data)

    return response_data

@app.delete("/players/{username}", status_code=204)
def delete_player(username: str, session: Session = Depends(get_session)):
    player = session.get(models.Player, username)
    if not player:
        raise HTTPException(404, "Player not found")
    
    # 1. Limpiar flag de cancelación en Redis si existe
    cancellation_key = f"cancel:{username}"
    redis_client.delete(cancellation_key)
    logger.info(f"Cleaned up Redis cancellation flag for {username}")
    
    # 2. Borrar todas las partidas asociadas a este jugador
    # (tanto como blancas como negras)
    games_to_delete = session.exec(
        select(models.Game).where(
            (models.Game.white_username == username) |
            (models.Game.black_username == username)
        )
    ).all()
    
    logger.info(f"Found {len(games_to_delete)} games to delete for player {username}")
    
    for game in games_to_delete:
        session.delete(game)
    
    # 3. Borrar el jugador (esto también borrará PlayerAnalysisDetailed por CASCADE)
    session.delete(player)
    
    # 4. Commit todos los cambios
    session.commit()
    
    logger.info(f"Successfully deleted player {username} and {len(games_to_delete)} associated games")


@app.post("/players/{username}/stop")
def stop_player_analysis(username: str, session: Session = Depends(get_session)):
    """Detiene un análisis de jugador en progreso y elimina todos los rastros de la base de datos."""
    player = session.get(models.Player, username)
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")

    try:
        # Verificar si hay un análisis en progreso
        analysis_in_progress = (
            player.status == "pending" or (player.progress and player.progress > 0)
        )

        if not analysis_in_progress or not player.last_task_id:
            # No hay análisis activo o no tenemos task_id registrado
            return {
                "username": username,
                "status": player.status,
                "message": "No analysis in progress to stop"
            }

        task_id = player.last_task_id
        logger.info(f"Starting complete cleanup sequence for player {username}, main task: {task_id}")

        set_cleanup_in_progress(username)
        logger.info(f"Cleanup started for user {username} - blocking new requests")

        import subprocess
        import signal
        import psutil

        logger.info(f"STEP 2: Terminating all Celery worker processes")

        try:
            celery_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['cmdline']:
                        cmdline_str = ' '.join(proc.info['cmdline'])
                        if 'celery' in cmdline_str and 'worker' in cmdline_str and 'app.celery_app' in cmdline_str:
                            celery_processes.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
            
            for proc in celery_processes:
                try:
                    logger.info(f"Sending SIGKILL to Celery worker process {proc.pid}")
                    proc.send_signal(signal.SIGKILL)
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    logger.warning(f"Could not kill process {proc.pid}: {e}")
            
            import time as _t
            _t.sleep(3)
            
            logger.info(f"Terminated {len(celery_processes)} Celery worker processes")
            
        except Exception as e:
            logger.error(f"Error terminating Celery workers: {e}")
            import time as _t
            _t.sleep(2)

        logger.info(f"STEP 3: Deleting all tasks from Redis/Celery queues")

        try:
            celery_app.control.purge()
            logger.info("Purged all Celery queues")
            
            task_keys = redis_client.keys("celery-task-meta-*")
            if task_keys:
                redis_client.delete(*task_keys)
                logger.info(f"Deleted {len(task_keys)} task metadata keys from Redis")
            
            queue_keys = redis_client.keys("*queue*")
            if queue_keys:
                redis_client.delete(*queue_keys)
                logger.info(f"Deleted {len(queue_keys)} queue keys from Redis")
                
        except Exception as e:
            logger.error(f"Error deleting tasks from queues: {e}")

        logger.info(f"All workers terminated and tasks deleted for user {username}, proceeding with database cleanup")

        logger.info(f"STEP 4: Starting database cleanup for player {username}")
        
        games_to_delete = session.exec(
            select(models.Game).where(
                (models.Game.white_username == username) |
                (models.Game.black_username == username)
            )
        ).all()
        
        logger.info(f"Found {len(games_to_delete)} games to delete for player {username}")
        
        for game in games_to_delete:
            session.delete(game)
        
        # Delete the player record (this will cascade to PlayerAnalysisDetailed)
        session.delete(player)
        
        logger.info(
            f"Successfully deleted player {username} and "
            f"{len(games_to_delete)} associated games"
        )

        session.commit()

        logger.info(f"STEP 5: Cleanup sequence completed - workers terminated, tasks deleted, database cleaned")

        # Notificar por WebSocket
        notify_ws(username, {"status": "stopped", "message": "Analysis stopped and all data removed"})

        logger.info(f"Complete cleanup completed for {username}. Games deleted: {len(games_to_delete)}")

        return {
            "username": username,
            "task_id": task_id,
            "status": "stopped",
            "message": "Analysis stopped and all data removed successfully",
            "games_deleted": len(games_to_delete)
        }
    except Exception as e:
        # Revertir la transacción abierta para no dejar la sesión en estado indeterminado
        session.rollback()
        logger.error(f"Error al detener el análisis para {username}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        clear_cleanup_in_progress()
        clear_analysis_in_progress()
        logger.info(f"Cleanup flags cleared for user {username}")

@app.post("/players/{username}/reset")
def reset_player(username: str, session: Session = Depends(get_session)):
    """Fuerza el reset del estado de un jugador para permitir re-análisis."""
    player = session.get(models.Player, username)
    if not player:
        raise HTTPException(404, "Player not found")

    # 1. Limpiar flag de cancelación en Redis
    cancellation_key = f"cancel:{username}"
    redis_client.delete(cancellation_key)
    logger.info(f"Cleaned up Redis cancellation flag for {username}")

    # 2. Opcional: borrar partidas existentes para forzar re-descarga
    # (comentado por defecto, descomenta si quieres borrar todo)
    # games_to_delete = session.exec(
    #     select(models.Game).where(
    #         (models.Game.white_username == username) |
    #         (models.Game.black_username == username)
    #     )
    # ).all()
    # for game in games_to_delete:
    #     session.delete(game)
    # logger.info(f"Deleted {len(games_to_delete)} existing games for {username}")

    # 3. Resetear a estado inicial
    player.status = models.PlayerStatus.not_analyzed
    player.progress = 0
    player.total_games = None
    player.done_games = None
    player.requested_at = None
    player.finished_at = None
    player.error = None
    player.last_task_id = None
    session.commit()

    logger.info(f"Successfully reset player {username} to initial state")
    return {"status": "reset", "username": username}

@app.post("/maintenance/cleanup-orphaned-games")
def cleanup_orphaned_games(session: Session = Depends(get_session)):
    """Limpia partidas huérfanas de jugadores que ya no existen."""
    # Obtener todos los usernames existentes
    existing_players = session.exec(select(models.Player.username)).all()
    existing_usernames = set(existing_players)
    
    # Buscar partidas con jugadores que no existen
    all_games = session.exec(select(models.Game)).all()
    orphaned_games = []
    
    for game in all_games:
        white_exists = game.white_username in existing_usernames if game.white_username else True
        black_exists = game.black_username in existing_usernames if game.black_username else True
        
        # Si alguno de los jugadores no existe, la partida es huérfana
        if not white_exists or not black_exists:
            orphaned_games.append(game)
    
    # Borrar partidas huérfanas
    for game in orphaned_games:
        session.delete(game)
    
    session.commit()
    
    logger.info(f"Cleaned up {len(orphaned_games)} orphaned games")
    
    return {
        "status": "success",
        "orphaned_games_deleted": len(orphaned_games),
        "total_games_checked": len(all_games)
    }

# ============================================================
# STREAMING (SSE)
# ============================================================

@app.get("/stream/{username}")
async def stream_updates(username: str):
    """Stream de eventos SSE para actualizaciones en tiempo real."""
    async def event_generator():
        pubsub = redis_client.pubsub()
        await pubsub.subscribe(f"player:{username}")

        try:
            async for msg in pubsub.listen():
                if msg["type"] == "message":
                    yield {"data": msg["data"]}
        finally:
            await pubsub.unsubscribe(f"player:{username}")
            await pubsub.close()

    from sse_starlette.sse import EventSourceResponse
    return EventSourceResponse(event_generator())

# ============================================================
# INICIALIZACIÓN
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
