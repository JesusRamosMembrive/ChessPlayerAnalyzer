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
    • Fallback to standard logging if JSON logger is not available.
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Try JSON logging first, fallback to standard logging
    try:
        # Check if pythonjsonlogger is available
        import pythonjsonlogger.jsonlogger

        logging_config: Dict[str, Any] = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                    "fmt": "%(asctime)s %(levelname)s %(name)s %(message)s",
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
    except ImportError:
        # Fallback to standard logging
        logging_config: Dict[str, Any] = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "handlers": ["console"],
                "level": log_level,
            },
        }

    logging.config.dictConfig(logging_config) 