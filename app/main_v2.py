"""
New FastAPI application using clean architecture.
This replaces main.py with the new Application Layer.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# New infrastructure
from .infrastructure.web.routers.players_v2 import router as players_v2_router
from .infrastructure.messaging.celery_tasks import celery_app

# Core configuration
from .core.config import get_config

# Application layer
from .application.container import get_container, cleanup_container

# Setup logging
from .logging_config import setup_logging
setup_logging()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting Chess Analyzer v2 with clean architecture")

    # Initialize container
    container = get_container()
    logger.info("Dependency injection container initialized")

    # Test database connection
    try:
        player_repo = container._get_player_repository()
        logger.info("Database connection verified")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down Chess Analyzer v2")
    await cleanup_container()


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    config = get_config()

    app = FastAPI(
        title="Chess Player Analyzer v2",
        description="Clean architecture implementation with CQRS pattern",
        version="2.0.0",
        lifespan=lifespan
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(players_v2_router)

    # Health check for v2
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "version": "2.0.0",
            "architecture": "clean",
            "message": "Chess Analyzer v2 is running"
        }

    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "Chess Player Analyzer v2",
            "version": "2.0.0",
            "architecture": "Clean Architecture with CQRS",
            "documentation": "/docs",
            "health": "/health",
            "api_prefix": "/v2"
        }

    return app


# Create app instance
app = create_app()

# Export for production servers
__all__ = ["app", "celery_app"]