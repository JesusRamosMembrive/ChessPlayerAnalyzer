"""Tareas Celery relacionadas con el análisis **individual** de partidas.

Paso 1 del refactor: reexportar funciones que ya viven en ``app.celery_app``
para poder importar desde otros módulos sin depender de un fichero gigante.
Más adelante se moverá la implementación aquí y se limpiará ``celery_app.py``.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

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
# Helper trasladado desde celery_app.analyze_game_task_legacy
# ---------------------------------------------------------------------------

def _is_aborted(task, player_username: str | None = None) -> bool:  # noqa: D401 (helper interno)
    """Detecta si la tarea Celery ha sido revocada.

    1. Comprueba primero una clave Redis específica del usuario (cancel:<username>)
       para reaccionar rápidamente a las cancelaciones a nivel de aplicación.
    2. Si el worker soporta :py:meth:`Task.is_aborted`, la utiliza.
    3. De forma retro-compatible inspecciona :pyattr:`task.request.is_aborted` o
       la marca *stopped*.
    """
    try:
        # Cancelación vía Redis (más rápida y simple)
        if player_username:
            cancellation_key = f"cancel:{player_username}"
            if redis_client.get(cancellation_key):
                logger.info(
                    "Task %s detected Redis cancellation flag for %s",
                    getattr(task.request, "id", "<no-id>"),
                    player_username,
                )
                return True

        # Celery >=5.3
        if hasattr(task, "is_aborted") and callable(task.is_aborted):
            return task.is_aborted()

        # Celery <5.3 – intentar a través de request
        if hasattr(task.request, "is_aborted") and callable(task.request.is_aborted):
            return task.request.is_aborted()
        # A veces Celery marca .stopped
        if getattr(task.request, "stopped", False):
            return True

    except Exception:
        # Nunca lanzar excepción desde chequeo de cancelación
        pass

    return False

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