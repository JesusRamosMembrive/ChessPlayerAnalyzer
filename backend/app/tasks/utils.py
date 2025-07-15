from __future__ import annotations

"""Utilidades compartidas para las tareas Celery.

Se extraen desde ``app.celery_app`` para reducir tamaño y acoplamiento.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import logging
from typing import Any

from app.utils import sa_to_dict  # función existente para serializar modelos SQLAlchemy

logger = logging.getLogger(__name__)

__all__ = [
    "export_analysis_to_json",
    "safe",
]


def export_analysis_to_json(data_obj: Any, username: str, *, analysis_type: str = "analysis") -> str | None:
    """Exporta un objeto de análisis (SQLModel) a un JSON legible en *debug_results*.

    Args:
        data_obj: Objeto SQLAlchemy/SQLModel a serializar.
        username: Nombre de usuario para el fichero.
        analysis_type: Cadena descriptiva del tipo de análisis ("game", "player", etc.).

    Returns
    -------
    Ruta al archivo generado o ``None`` si falló.
    """
    try:
        debug_dir = Path("debug_results")
        debug_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(datetime.now(timezone.utc).timestamp())
        filename = f"{username}_{timestamp}.json"
        filepath = debug_dir / filename

        data_dict = sa_to_dict(data_obj)

        with filepath.open("w", encoding="utf-8") as f:
            json.dump(data_dict, f, ensure_ascii=False, indent=2, default=str)

        logger.info("DEBUG EXPORT: Saved %s analysis to %s", analysis_type, filepath)
        return str(filepath)

    except Exception as exc:  # noqa: BLE001 (queremos atrapar todo)
        logger.error("DEBUG EXPORT: Failed to export %s analysis for %s: %s", analysis_type, username, exc)
        return None


def safe(value):
    """Convierte valores posiblemente *None* o *NaN* en ``float`` seguro.

    Celery serializa los datos usando JSON; asegurar que los números sean
    flotantes o 0.0 evita excepciones de serialización.
    """
    import numpy as np  # import local para evitar dependencia dura si no se usa

    try:
        # Maneja np.nan: ``np.nan != np.nan`` es *True*
        if value is None or (isinstance(value, float) and value != value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0 