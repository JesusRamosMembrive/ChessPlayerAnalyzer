"""alerting.py – Gestión de alertas y notificaciones externas.

Este módulo define utilidades sencillas para enviar alertas a un webhook
externo (por ejemplo, Slack, Discord, MS Teams, etc.) cuando se producen
registros de nivel **ERROR** o superior.  El destino y la activación se
controlan mediante variables de entorno:

• ALERT_WEBHOOK_URL   → URL completa del webhook (https://hooks.slack.com/…)
• ALERT_ENABLED       → "1" o "true" para forzar envío aunque no sea producción.
• ALERT_MIN_LEVEL     → Nivel mínimo (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                       por defecto «ERROR».

Dependencias: usa `requests`, ya incluida en requirements.txt.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

import requests

__all__ = [
    "send_alert",
    "CriticalAlertHandler",
]

# ---------------------------------------------------------------------------
# Función de envío genérica --------------------------------------------------
# ---------------------------------------------------------------------------

def _webhook_url() -> Optional[str]:
    url = os.getenv("ALERT_WEBHOOK_URL")
    return url.strip() if url else None


def _alert_enabled(levelno: int) -> bool:
    """Determina si hay que enviar la alerta según configuración."""
    env_flag = os.getenv("ALERT_ENABLED", "1").lower() in {"1", "true", "yes"}
    min_level_name = os.getenv("ALERT_MIN_LEVEL", "ERROR").upper()
    try:
        min_level_no = getattr(logging, min_level_name)
    except AttributeError:
        min_level_no = logging.ERROR

    return env_flag and levelno >= min_level_no and _webhook_url() is not None


def send_alert(title: str, message: str, *, level: str = "ERROR", extra: Dict[str, Any] | None = None) -> None:
    """Envía un payload JSON sencillo al webhook configurado.

    El contenido depende de la plataforma receptora; para Slack, un bloque
    de texto bastará si el webhook es _Incoming Webhook_.
    """
    url = _webhook_url()
    if not url:
        return  # Silencioso si no hay configuración

    payload = {
        "text": f"*{title}*\n{message}",
        "level": level,
    }
    if extra:
        payload.update(extra)

    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as exc:  # noqa: BLE001 (no queremos fallar por alerta)
        # Log interno pero no interrumpir flujo
        logging.getLogger(__name__).warning("Fallo enviando alerta: %s", exc)


# ---------------------------------------------------------------------------
# Manejador de logging -------------------------------------------------------
# ---------------------------------------------------------------------------

class CriticalAlertHandler(logging.Handler):
    """Logging handler que envía alertas al detectar eventos críticos."""

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401 (firma)
        if not _alert_enabled(record.levelno):
            return

        try:
            msg = self.format(record)
            title = f"[{record.levelname}] {record.name}"
            send_alert(title, msg, level=record.levelname)
        except Exception:  # noqa: BLE001 (proteger cualquier error)
            # Nunca lanzar excepción desde logging
            self.handleError(record) 