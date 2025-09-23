"""
Chess Player Analyzer - Production FastAPI Application.
Clean Architecture implementation with backward compatibility.

This replaces the legacy main.py (983 lines) with a clean, maintainable version
while preserving all essential functionality.
"""
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# New clean architecture imports
from .infrastructure.web.routers.players_v2 import router as players_v2_router
from .infrastructure.messaging.celery_tasks import celery_app

# Legacy compatibility imports (to be removed after migration)
from .api.v1 import api_router as v1_router
from .api.v1.endpoints import players as legacy_players_router
from .database import init_db

# Core configuration and performance monitoring
from .core.config import get_config
from .core.monitoring import performance_monitor, perform_comprehensive_health_check
from .application.container import get_container, cleanup_container

# Setup logging
from .logging_config import setup_logging
setup_logging()

logger = logging.getLogger(__name__)

# Global state for backward compatibility
_app_state = {
    "startup_time": None,
    "container": None
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management with clean architecture."""
    # Startup
    startup_time = datetime.utcnow()
    _app_state["startup_time"] = startup_time

    logger.info("🚀 Starting Chess Analyzer with Clean Architecture")

    # Initialize database
    try:
        init_db()
        logger.info("📊 Database initialized")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise

    # Initialize dependency injection container
    try:
        container = get_container()
        _app_state["container"] = container
        logger.info("🔧 Dependency injection container initialized")
    except Exception as e:
        logger.error(f"❌ Container initialization failed: {e}")
        raise

    # Test core services
    try:
        # Test that we can get a use case (validates full DI chain)
        status_use_case = container.get_get_player_status_use_case()
        logger.info("✅ Core services validated")
    except Exception as e:
        logger.error(f"❌ Core services validation failed: {e}")
        raise

    logger.info(f"🎉 Chess Analyzer started successfully in {(datetime.utcnow() - startup_time).total_seconds():.2f}s")

    yield

    # Shutdown
    logger.info("🛑 Shutting down Chess Analyzer")
    await cleanup_container()
    logger.info("✅ Shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    config = get_config()

    # Create FastAPI app with clean architecture
    app = FastAPI(
        title="Chess Player Analyzer",
        description="Advanced chess player analysis with cheat detection using Clean Architecture",
        version="2.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers - New architecture first, then legacy for compatibility
    app.include_router(players_v2_router, prefix="/api")
    app.include_router(v1_router, prefix="/api/v1")  # Legacy compatibility
    app.include_router(legacy_players_router.router, prefix="/players", tags=["legacy-players"])  # Direct legacy routes

    # Root endpoints
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "Chess Player Analyzer",
            "version": "2.0.0",
            "architecture": "Clean Architecture with DDD & CQRS",
            "status": "production",
            "startup_time": _app_state["startup_time"].isoformat() if _app_state["startup_time"] else None,
            "documentation": {
                "swagger": "/docs",
                "redoc": "/redoc",
                "openapi": "/openapi.json"
            },
            "api": {
                "v2": "/api/v2 (recommended - clean architecture)",
                "v1": "/api/v1 (legacy compatibility)"
            }
        }

    @app.get("/health")
    async def health_check():
        """Enhanced health check endpoint."""
        try:
            # Check database connectivity
            from .database import engine
            with engine.connect() as conn:
                conn.execute("SELECT 1")
            db_status = "healthy"
        except Exception as e:
            db_status = f"unhealthy: {str(e)}"

        # Check DI container
        container_status = "healthy" if _app_state["container"] else "unhealthy"

        # Check Redis connectivity
        try:
            from .utils import redis_client
            redis_client.ping()
            redis_status = "healthy"
        except Exception as e:
            redis_status = f"unhealthy: {str(e)}"

        overall_healthy = all(
            status == "healthy"
            for status in [db_status, container_status, redis_status]
        )

        status_code = 200 if overall_healthy else 503

        return JSONResponse(
            status_code=status_code,
            content={
                "status": "healthy" if overall_healthy else "degraded",
                "timestamp": datetime.utcnow().isoformat(),
                "uptime_seconds": (
                    (datetime.utcnow() - _app_state["startup_time"]).total_seconds()
                    if _app_state["startup_time"] else 0
                ),
                "services": {
                    "database": db_status,
                    "container": container_status,
                    "redis": redis_status
                },
                "version": "2.0.0",
                "architecture": "clean"
            }
        )

    @app.get("/health/comprehensive")
    async def comprehensive_health_check():
        """Comprehensive health check with performance metrics."""
        health_data = await perform_comprehensive_health_check()
        status_code = 200 if health_data["status"] == "healthy" else 503
        return JSONResponse(status_code=status_code, content=health_data)

    @app.get("/metrics")
    async def get_metrics():
        """Get performance metrics summary."""
        return performance_monitor.get_metrics_summary()

    @app.get("/metrics/cache")
    async def get_cache_metrics():
        """Get cache performance metrics."""
        from .core.performance import performance_optimizer
        return performance_optimizer.get_cache_stats()

    # Legacy endpoints for backward compatibility (simplified)
    @app.post("/analyze", status_code=status.HTTP_202_ACCEPTED, tags=["legacy-compatibility"])
    async def legacy_analyze_redirect(request: dict):
        """Legacy endpoint - redirects to new architecture."""
        return {
            "message": "This endpoint has been moved to /api/v2/players/{username}/analyze",
            "new_endpoint": "/api/v2/players/{username}/analyze",
            "method": "POST",
            "migration_guide": "/docs#migration",
            "deprecated": True
        }

    @app.get("/players/{username}", tags=["legacy-compatibility"])
    async def legacy_player_status_redirect(username: str):
        """Legacy endpoint - redirects to new architecture."""
        return {
            "message": f"This endpoint has been moved to /api/v2/players/{username}/status",
            "new_endpoint": f"/api/v2/players/{username}/status",
            "migration_guide": "/docs#migration",
            "deprecated": True
        }

    # Error handlers
    @app.exception_handler(404)
    async def not_found_handler(request, exc):
        return JSONResponse(
            status_code=404,
            content={
                "error": "Endpoint not found",
                "message": "This endpoint may have been moved to the new API structure",
                "suggestions": {
                    "v2_api": "/api/v2/",
                    "documentation": "/docs",
                    "health": "/health"
                }
            }
        )

    @app.exception_handler(500)
    async def internal_error_handler(request, exc):
        logger.error(f"Internal server error: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": "An unexpected error occurred. Please check logs.",
                "timestamp": datetime.utcnow().isoformat()
            }
        )

    return app


# Create the app instance
app = create_app()

# Export for production servers
__all__ = ["app", "celery_app"]


# Development server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main_production:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )