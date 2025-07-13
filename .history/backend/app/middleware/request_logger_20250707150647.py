from __future__ import annotations

"""Middleware de registro de peticiones y respuestas.

Registra método, ruta, host de cliente, código de estado y tiempo de
procesamiento de cada petición.  La información se envía al logger
configurado en la aplicación para una trazabilidad sencilla.
"""

import logging
import time
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

__all__ = [
    "RequestLoggingMiddleware",
]

# Logger dedicado para las peticiones HTTP
logger = logging.getLogger("chess-analyzer.request")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware que escribe en el log cada petición HTTP."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:  # type: ignore[override]
        start_time = time.perf_counter()

        response = await call_next(request)

        duration_ms = (time.perf_counter() - start_time) * 1000
        client_host = request.client.host if request.client else "unknown"

        # Formato: GET /path - 200 (123.45 ms) from 127.0.0.1
        logger.info(
            "%s %s - %s (%.2f ms) from %s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            client_host,
        )

        return response 