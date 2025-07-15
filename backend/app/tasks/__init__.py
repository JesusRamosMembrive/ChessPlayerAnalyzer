from __future__ import annotations

"""Paquete de tareas Celery desglosadas.

Cada submódulo agrupa tareas relacionadas para mantener el código limpio.
"""

from .games import analyze_game_task, analyze_game_detailed  # noqa: F401 reexport
from .player import process_player_enhanced, analyze_player_detailed  # noqa: F401 reexport

__all__ = [
    "analyze_game_task",
    "analyze_game_detailed",
    "process_player_enhanced",
    "analyze_player_detailed",
] 