# app/main.py
"""
API principal de Chess Analyzer.
Versión refactorizada con responsabilidades separadas.
"""
import logging
from datetime import datetime, UTC

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session

# Import versioned API routers
from app.api.v1 import api_router as v1_router
from app.api.v1.endpoints import health as health_endpoints
from app.database import init_db
from app.error_handlers import register_exception_handlers
from app.middleware.rate_limiter import RateLimitMiddleware
from app.middleware.request_logger import RequestLoggingMiddleware
from app.middleware.trace_context import TraceContextMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

# Application Performance Monitoring (APM)
from app.otel import init_otel, instrument_fastapi

# Configurar logging estructurado (JSON)
from app.logging_config import setup_logging

# Esta llamada es idempotente; si otro módulo ya la ejecutó no tiene efecto.
setup_logging()
logger = logging.getLogger(__name__)

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

# ────────────────────────────────────────────────────────────────────────────
# Legacy endpoints for backward compatibility
# ────────────────────────────────────────────────────────────────────────────

# Import and include legacy endpoints from separate module
from app.api.legacy_endpoints import legacy_router
app.include_router(legacy_router, tags=["legacy"])

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
# INICIALIZACIÓN
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)