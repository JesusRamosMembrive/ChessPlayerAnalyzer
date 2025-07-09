from __future__ import annotations

"""Middleware que añade encabezados de traza a cada respuesta HTTP.

Incluye:
• ``traceparent`` (W3C Trace Context)
• ``X-Trace-Id`` – facilita correlación con logs y otras herramientas
"""

import logging
from typing import Callable

from opentelemetry import trace
from opentelemetry.trace import Span
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

__all__ = ["TraceContextMiddleware"]


class TraceContextMiddleware(BaseHTTPMiddleware):
    """Inserta identificadores de traza en las cabeceras de respuesta."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:  # type: ignore[override]
        response = await call_next(request)

        span: Span | None = trace.get_current_span()
        if span is not None:
            ctx = span.get_span_context()
            if ctx.trace_id != 0:  # hay contexto válido
                trace_id = f"{ctx.trace_id:032x}"
                span_id = f"{ctx.span_id:016x}"
                # Cabeceras estándar / personalizadas
                response.headers["traceparent"] = f"00-{trace_id}-{span_id}-01"
                response.headers["X-Trace-Id"] = trace_id
        return response 