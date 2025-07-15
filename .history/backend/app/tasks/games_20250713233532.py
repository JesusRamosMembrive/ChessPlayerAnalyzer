"""Tareas Celery relacionadas con el análisis **individual** de partidas.

Paso 1 del refactor: reexportar funciones que ya viven en ``app.celery_app``
para poder importar desde otros módulos sin depender de un fichero gigante.
Más adelante se moverá la implementación aquí y se limpiará ``celery_app.py``.
"""
from __future__ import annotations

from app.celery_app import (
    analyze_game_task as analyze_game_task,  # noqa: F401 reexport
    analyze_game_detailed as analyze_game_detailed,  # noqa: F401 reexport
)

__all__ = [
    "analyze_game_task",
    "analyze_game_detailed",
] 