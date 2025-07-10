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

        import time

        # --- Overall timings and constants ---
        POLL_INTERVAL_S = 5
        # GRACEFUL_WAIT_S = 30 # Initial wait after SIGTERM
        # FORCEFUL_WAIT_S = 60 # Max wait after SIGKILL (now part of quiescence)
        INITIAL_SIGTERM_GRACE_S = 10 # Short wait after initial SIGTERM
        INITIAL_SIGKILL_GRACE_S = 5  # Short wait after initial SIGKILL
        MAX_QUIESCENCE_WAIT_S = 60   # Max time for the worker quiescence loop
        QUIESCENCE_POLL_INTERVAL_S = 5

        # STEP 1: Set Redis cancellation flag
        logger.info(f"STEP 1: Setting Redis cancellation flag for {username}")
        cancellation_key = f"cancel:{username}" # Define it early for use in finally if needed
        try:
            # Set TTL long enough to cover the whole stop operation + buffer
            redis_client.set(cancellation_key, "true", ex=MAX_QUIESCENCE_WAIT_S + INITIAL_SIGTERM_GRACE_S + INITIAL_SIGKILL_GRACE_S + 60)
        except Exception as e:
            logger.error(f"Error setting Redis cancellation flag for {username}: {e}")

        # STEP 2: Identify all relevant tasks and initial revocation pass
        logger.info(f"STEP 2: Identifying tasks and initial revocation for {username}")
        all_potentially_related_task_ids = set()
        if task_id: # Main task ID from player record
            all_potentially_related_task_ids.add(task_id)
            logger.info(f"Main task {task_id} added for handling.")

        try:
            inspect = celery_app.control.inspect(timeout=2.0) # Short timeout for inspect
            if inspect:
                # Consolidate inspection of active, scheduled, reserved
                task_sources = {
                    "active": inspect.active(),
                    "scheduled": inspect.scheduled(),
                    "reserved": inspect.reserved()
                }
                for source_name, tasks_by_worker in task_sources.items():
                    if tasks_by_worker:
                        for worker_name, tasks_on_worker in tasks_by_worker.items():
                            for task_info in tasks_on_worker:
                                if _username_in_task_details(task_info, username):
                                    task_info_id = task_info.get('id') or task_info.get('request', {}).get('id')
                                    if task_info_id and task_info_id not in all_potentially_related_task_ids:
                                        all_potentially_related_task_ids.add(task_info_id)
                                        logger.info(f"Identified related task {task_info_id} ({source_name} on {worker_name}) for {username}.")
            else:
                logger.warning("Could not inspect Celery workers during initial task identification.")
        except Exception as e:
            logger.error(f"Error during initial Celery task inspection: {e}.")

        if not all_potentially_related_task_ids:
            logger.info(f"No potentially related Celery tasks found for {username}. Proceeding to cleanup.")
        else:
            logger.info(f"Attempting to stop {len(all_potentially_related_task_ids)} tasks: {list(all_potentially_related_task_ids)}")
            # Initial SIGTERM pass
            logger.info(f"STEP 2a: Sending SIGTERM to identified tasks.")
            for t_id in all_potentially_related_task_ids:
                try: celery_app.control.revoke(t_id, terminate=True)
                except Exception as e: logger.error(f"Error sending SIGTERM to task {t_id}: {e}")

            logger.info(f"Waiting {INITIAL_SIGTERM_GRACE_S}s for SIGTERM to be processed.")
            time.sleep(INITIAL_SIGTERM_GRACE_S)

            # Initial SIGKILL pass for all initially identified tasks
            # (as some might be PENDING and SIGTERM doesn't remove them from AsyncResult's view quickly)
            logger.info(f"STEP 2b: Sending SIGKILL to all initially identified tasks.")
            for t_id in all_potentially_related_task_ids:
                try: celery_app.control.revoke(t_id, terminate=True, signal='SIGKILL')
                except Exception as e: logger.error(f"Error sending SIGKILL to task {t_id}: {e}")

            logger.info(f"Waiting {INITIAL_SIGKILL_GRACE_S}s for SIGKILL to take effect.")
            time.sleep(INITIAL_SIGKILL_GRACE_S)

        # STEP 3: Worker Quiescence Loop - Ensure no active tasks for the user
        logger.info(f"STEP 3: Entering worker quiescence check for {username} (up to {MAX_QUIESCENCE_WAIT_S}s).")
        quiescence_deadline = time.time() + MAX_QUIESCENCE_WAIT_S
        user_tasks_remain_active = False

        while time.time() < quiescence_deadline:
            user_tasks_found_this_iteration = False
            active_tasks_on_workers = None
            try:
                inspect = celery_app.control.inspect(timeout=1.0) # Short timeout for inspect call
                if inspect:
                    active_tasks_on_workers = inspect.active()
                else: # Workers not responding to inspect
                    logger.warning("Celery inspect returned None (no workers responding?). Assuming quiescence for this iteration.")
                    # This means we can't confirm tasks are running, so we might proceed if this state persists.
                    # However, if tasks were running on a worker that just became unresponsive, they might still be writing.
                    # This is a tricky state. For now, treat as "no tasks found".
                    pass # user_tasks_found_this_iteration remains False

            except Exception as e: # Covers redis.exceptions.TimeoutError from inspect, etc.
                logger.error(f"Error during celery inspect.active() call: {e}. Assuming quiescence for this iteration.")
                # Similar to inspect returning None, if we can't query, we can't find tasks.
                pass # user_tasks_found_this_iteration remains False

            if active_tasks_on_workers:
                for worker_name, tasks_on_worker_list in active_tasks_on_workers.items():
                    if not tasks_on_worker_list: continue # Skip worker if it reports no active tasks
                    for active_task_info in tasks_on_worker_list:
                        if _username_in_task_details(active_task_info, username):
                            user_tasks_found_this_iteration = True
                            active_task_id = active_task_info.get('id') or active_task_info.get('request', {}).get('id')
                            logger.warning(f"User '{username}' task {active_task_id or 'unknown_id'} still active on worker {worker_name}. Re-issuing SIGKILL.")
                            if active_task_id: # Only revoke if we have an ID
                                try:
                                    celery_app.control.revoke(active_task_id, terminate=True, signal='SIGKILL', destination=[worker_name])
                                except Exception as e_revoke:
                                    logger.error(f"Error re-issuing SIGKILL to {active_task_id} on {worker_name}: {e_revoke}")
                            break # Found an active task for the user on this worker
                    if user_tasks_found_this_iteration: break # Break from worker loop

            user_tasks_remain_active = user_tasks_found_this_iteration # Update overall status

            if not user_tasks_remain_active:
                logger.info(f"No active tasks for user '{username}' found in this quiescence check. Verification successful.")
                break # Exit quiescence loop

            logger.info(f"User tasks still active. Waiting {QUIESCENCE_POLL_INTERVAL_S}s before next quiescence check...")
            time.sleep(QUIESCENCE_POLL_INTERVAL_S)

        if user_tasks_remain_active: # Check after loop finishes (either by break or timeout)
            logger.error(f"CRITICAL: Workers still report active tasks for user '{username}' after {MAX_QUIESCENCE_WAIT_S}s of quiescence checks. Database deletion will be ABORTED to prevent data corruption.")
            # No need to rollback session here as we haven't done DB operations yet in this path
            clear_cleanup_in_progress() # Clear our own flag
            # Clear analysis_in_progress if it was set for this user, to allow retries or other operations.
            # This depends on how ANALYSIS_IN_PROGRESS_KEY is managed. Assuming it might be set.
            if redis_client.get(ANALYSIS_IN_PROGRESS_KEY) == username.encode(): # Check if this user set it
                 clear_analysis_in_progress()
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Could not stop all Celery tasks for user {username} after multiple attempts. Player data has not been deleted. Please try again later or contact support if the issue persists.")

        # If we reach here, user_tasks_remain_active is False.
        logger.info(f"Worker quiescence confirmed for user '{username}'. Proceeding with cleanup.")

        # STEP 4: Comprehensive Redis and queue cleanup (previously Step 7)
        logger.info(f"STEP 4: Comprehensive Redis and queue cleanup")

        try:
            # Reduced scope of key deletion to be less aggressive.
            # Only delete task metadata specifically, not all "*queue*" or "*{username}*" keys.
            # The "cancel:{username}" key will expire via its TTL.

            celery_task_meta_keys = redis_client.keys("celery-task-meta-*")
            if celery_task_meta_keys:
                # Filter further for task IDs we were monitoring if possible, though complex.
                # For now, deleting all task meta is broad but common during such cleanups.
                redis_client.delete(*celery_task_meta_keys)
                logger.info(f"Deleted {len(celery_task_meta_keys)} celery-task-meta-* keys from Redis.")

            # Consider if specific user-related cache keys for analysis results should be cleared.
            # Example: if player analysis results are cached with keys like "player_analysis:{username}"
            # This is application-specific. For now, no other user-specific keys are deleted here
            # beyond what the tasks themselves or other parts of the app manage.
            # The `cancel:{username}` key is left to expire by its TTL.

            # The aggressive deletion of `*queue*` and `*{username}*` keys (except cancel flag) is removed
            # to prevent unintended side effects on other parts of the system or other users.
            # `celery_app.control.purge()` was also avoided for similar reasons.

            # Removing the generic sleep here, specific waits are handled in task polling.
            # import time as _t
            # _t.sleep(3)

        except Exception as e:
            logger.error(f"Error during targeted Redis cleanup: {e}")

        # The old "STEP 4: Verifying all tasks are stopped before database cleanup" and its try-except block
        # are removed as this verification is now part of the multi-stage task termination logic (Steps 3-6).

        logger.info(f"STEP 8: Starting database cleanup for player {username}") # Adjusted step number
        
        games_to_delete = session.exec(
            select(models.Game).where(
                (models.Game.white_username == username) |
                (models.Game.black_username == username)
            )
        ).all()
        
        logger.info(f"Found {len(games_to_delete)} games to delete for player {username}")
        
        for game in games_to_delete:
            # Explicitly delete MoveAnalysis records for the current game FIRST
            move_analyses_to_delete = session.exec(
                select(models.MoveAnalysis).where(models.MoveAnalysis.game_id == game.id)
            ).all()
            if move_analyses_to_delete:
                logger.info(f"Found {len(move_analyses_to_delete)} move analysis records to delete for game {game.id}")
                for ma in move_analyses_to_delete:
                    session.delete(ma)
            else:
                logger.info(f"No move analysis records found for game {game.id}, or they are already marked for deletion via cascade.")

            logger.info(f"Marking game record for deletion: game_id {game.id}")
            session.delete(game) # Mark the game for deletion
        
        # Delete the player record (this will cascade to PlayerAnalysisDetailed if configured)
        logger.info(f"Marking player record for deletion: {username}")
        session.delete(player)
        
        logger.info(
            f"Attempting to commit deletions for player {username} and {len(games_to_delete)} associated games (plus their move analyses)."
        )

        session.commit() # This is where the error was happening. Now it should succeed.

        logger.info(f"Database cleanup successful for player {username}.")
        logger.info(f"STEP 5: Cleanup sequence completed - tasks handled, Redis cleaned, database records deleted.") # Adjusted log

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


def _username_in_task_details(task_info: dict, username_to_check: str) -> bool:
    """
    Checks if the username appears in the string representation of a task's args or kwargs.
    `task_info` is a dictionary like one item from `inspect.active()[worker_name]`.
    """
    if not task_info or not isinstance(task_info, dict):
        return False

    # Celery task args/kwargs from inspect can be complex. Checking string representation is a common approach.
    task_args_repr = str(task_info.get('args', '[]'))
    task_kwargs_repr = str(task_info.get('kwargs', '{}'))

    if username_to_check in task_args_repr or username_to_check in task_kwargs_repr:
        return True

    # Sometimes the request object itself is nested
    request_info = task_info.get('request')
    if isinstance(request_info, dict):
        request_args_repr = str(request_info.get('args', '[]'))
        request_kwargs_repr = str(request_info.get('kwargs', '{}'))
        if username_to_check in request_args_repr or username_to_check in request_kwargs_repr:
            return True

    return False


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
