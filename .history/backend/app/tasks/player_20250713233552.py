"""Tareas Celery relacionadas con el análisis **longitudinal** de un jugador.

Fase 1: simplemente reexportamos las tareas existentes mientras migramos
la implementación desde ``app.celery_app``.
"""
from __future__ import annotations

from app.celery_app import (
    process_player_enhanced as process_player_enhanced,  # noqa: F401 reexport
    analyze_player_detailed as analyze_player_detailed,  # noqa: F401 reexport
)

__all__ = [
    "process_player_enhanced",
    "analyze_player_detailed",
] 