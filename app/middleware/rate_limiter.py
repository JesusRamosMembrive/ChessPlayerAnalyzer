from __future__ import annotations

"""Middleware de limitación de peticiones (rate-limiting).

Implementa un algoritmo sliding-window muy simple en memoria para restringir
el número de peticiones que un mismo cliente (IP) puede realizar en un
intervalo de tiempo dado.  No pretende sustituir soluciones dedicadas como
Redis/SlowAPI, pero es suficiente para proteger la API frente a abusos
ocasionales y permite mantener la compatibilidad sin incorporar dependencias
externas.
"""

from collections import defaultdict, deque
from typing import Deque, Dict
import os
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

__all__ = [
    "RateLimitMiddleware",
]


class _InMemoryLimiter:
    """Estructura interna para llevar la cuenta de peticiones por IP."""

    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        # Mapa IP -> deque con *timestamps* de peticiones dentro de la ventana
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)

    def register_hit(self, key: str) -> int:
        """Registra una petición y devuelve el número restante permitido.

        Si ya se ha superado el límite, devuelve un número negativo.
        """
        now = time.time()
        bucket = self._hits[key]

        # Limpieza de timestamps fuera de la ventana
        window_start = now - self.window_seconds
        while bucket and bucket[0] < window_start:
            bucket.popleft()

        if len(bucket) >= self.max_requests:
            # Límite excedido
            return -1

        bucket.append(now)
        return self.max_requests - len(bucket)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware que aplica limitación de tasa por IP a toda la aplicación."""

    def __init__(
        self,
        app,
        max_requests: int | None = None,
        window_seconds: int | None = None,
    ) -> None:  # noqa: D401, ANN001 (firma impuesta por Starlette)
        super().__init__(app)
        # Permitir configuración mediante variables de entorno
        self.max_requests = max_requests or int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "100"))
        self.window_seconds = window_seconds or int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
        self._limiter = _InMemoryLimiter(self.max_requests, self.window_seconds)

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[override]
        client_host = request.client.host if request.client else "unknown"
        remaining = self._limiter.register_hit(client_host)

        if remaining < 0:
            # Limite superado -> devolver 429
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Se excedió el límite de peticiones. Inténtalo de nuevo más tarde.",
                },
                headers={
                    "Retry-After": str(self.window_seconds),
                    "X-RateLimit-Limit": str(self.max_requests),
                    "X-RateLimit-Remaining": "0",
                },
            )

        # Continuar procesamiento
        response = await call_next(request)
        # Incluir cabeceras informativas sobre el límite
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response 