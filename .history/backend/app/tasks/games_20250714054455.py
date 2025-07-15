"""Tareas Celery relacionadas con el análisis **individual** de partidas.

Paso 1 del refactor: reexportar funciones que ya viven en ``app.celery_app``
para poder importar desde otros módulos sin depender de un fichero gigante.
Más adelante se moverá la implementación aquí y se limpiará ``celery_app.py``.
"""
from __future__ import annotations

from app.celery_app import (
    celery_app,
    analyze_game_task_legacy,
    analyze_game_detailed_legacy,
    redis_client,
    ENGINE_PATH,
    MAX_DEPTH,
    TB_PATH,
)
from app.tasks.utils import safe, export_analysis_to_json  # noqa: F401 (se usarán en siguientes etapas)

# ---------------------------------------------------------------------------
# Wrappers temporales → llaman a las implementaciones _legacy_  -------------
# ---------------------------------------------------------------------------

@celery_app.task(name="analyze_game_task", bind=True)
def analyze_game_task(self, *args, **kwargs):  # noqa: D401 (firma Celery)
    """Wrapper que delega en la versión legacy.

    Se crea para mantener el nombre público mientras migramos la lógica
    real a este módulo.
    """
    return analyze_game_task_legacy(*args, **kwargs)


@celery_app.task(name="analyze_game_detailed")
def analyze_game_detailed(*args, **kwargs):  # noqa: D401
    """Wrapper que delega en la versión legacy (por ahora)."""
    return analyze_game_detailed_legacy(*args, **kwargs)

__all__ = [
    "analyze_game_task",
    "analyze_game_detailed",
] 