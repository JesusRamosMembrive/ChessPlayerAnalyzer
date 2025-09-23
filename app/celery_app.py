"""
Chess Player Analyzer - Production Celery Application.
Clean Architecture implementation replacing legacy celery_app.py (883 lines).

This module provides backward compatibility while using the new clean architecture.
"""
import logging

# Import the new clean architecture Celery app and tasks
from .infrastructure.messaging.celery_tasks import (
    celery_app,
    analyze_player_task,
    analyze_game_task,
    test_worker_functionality
)

# Setup logging
from .logging_config import setup_logging
setup_logging()

logger = logging.getLogger(__name__)

# Backward compatibility aliases for legacy code
process_player_enhanced = analyze_player_task
analyze_player_detailed = analyze_player_task
analyze_game_detailed = analyze_game_task

# Legacy function for extracting game ID (simple compatibility)
def extract_game_id(game_url: str) -> str:
    """Extract game ID from Chess.com URL for backward compatibility."""
    if "/game/" in game_url:
        return game_url.split("/game/")[-1].split("/")[0]
    return game_url

# Log the migration
logger.info("🔄 Using new Clean Architecture Celery implementation")
logger.info("📦 Legacy celery_app.py (883 lines) replaced with clean architecture")

# Export everything for backward compatibility
__all__ = [
    "celery_app",
    "analyze_player_task",
    "analyze_game_task",
    "test_worker_functionality",
    # Legacy aliases
    "process_player_enhanced",
    "analyze_player_detailed"
]

# The celery_app instance is ready to use
# Run with: celery -A app.celery_app worker --loglevel=info