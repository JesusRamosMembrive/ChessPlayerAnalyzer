from __future__ import annotations

"""Configuración unificada de logging estructurado.

Este módulo expone ``setup_logging`` que configura el *root logger* para
emitir logs en formato JSON a **stdout** usando los niveles definidos en
la variable de entorno ``LOG_LEVEL`` (por defecto ``INFO``).

Se basa en *python-json-logger* y puede invocarse múltiples veces sin
reconfigurar gracias a *functools.lru_cache*.
"""

import logging.config
import os
from functools import lru_cache
from typing import Any, Dict


@lru_cache(maxsize=1)
def setup_logging() -> None:
    """Inicializa la configuración global de logging.

    • Formato JSON (clave/valor) para fácil ingesta en sistemas como
      Elastic, Loki o Datadog.
    • Nivel configurable con ``LOG_LEVEL``.
    • No desactiva loggers existentes, simplemente unifica formato.
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    logging_config: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                # Añadimos identificadores de traza para correlación
                "fmt": "%(asctime)s %(levelname)s %(name)s %(trace_id)s %(span_id)s %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "json",
                "stream": "ext://sys.stdout",
            },
        },
        "root": {
            "handlers": ["console"],
            "level": log_level,
        },
    }

    logging.config.dictConfig(logging_config) 