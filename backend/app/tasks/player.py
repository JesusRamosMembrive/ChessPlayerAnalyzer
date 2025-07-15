"""Tareas Celery relacionadas con el análisis **longitudinal** de un jugador.

Fase 1: simplemente reexportamos las tareas existentes mientras migramos
la implementación desde ``app.celery_app``.
"""
from __future__ import annotations

from app.celery_app import (
    celery_app,
    redis_client,
)
from app.tasks.utils import safe, export_analysis_to_json  # noqa: F401 se usará más adelante

# ---------------------------------------------------------------------------
# Wrappers temporales – delegan en versiones legacy
# ---------------------------------------------------------------------------

@celery_app.task(name="process_player_enhanced", bind=True)
def process_player_enhanced(self, *args, **kwargs):  # noqa: D401
    """Delegación temporal hacia la versión legacy con import diferido."""
    from app.celery_app import process_player_enhanced_legacy as _legacy  # import local para evitar ciclos
    return _legacy(*args, **kwargs)


@celery_app.task(name="analyze_player_detailed")
def analyze_player_detailed(*args, **kwargs):  # noqa: D401
    """Delegación temporal hacia la versión legacy con import diferido."""
    from app.celery_app import analyze_player_detailed_legacy as _legacy  # import local para evitar ciclos
    return _legacy(*args, **kwargs)

__all__ = [
    "process_player_enhanced",
    "analyze_player_detailed",
] 