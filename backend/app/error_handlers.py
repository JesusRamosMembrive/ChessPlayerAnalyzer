from __future__ import annotations

"""Manejadores de excepciones y middleware de error unificado para la API.

Proporciona un conjunto de manejadores que generan respuestas JSON
consistentes en caso de errores.

Úsalo desde *main.py* con::

    from app.error_handlers import register_exception_handlers
    register_exception_handlers(app)
"""

from datetime import datetime, timezone
import logging
from typing import Any, Dict

from fastapi import HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_payload(message: str, code: int, *, errors: Any | None = None) -> Dict[str, Any]:
    """Crea el payload JSON estándar para respuestas de error."""
    payload: Dict[str, Any] = {
        "status": "error",
        "message": message,
        "code": code,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if errors is not None:
        payload["errors"] = errors
    return payload

# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

async def http_exception_handler(request: Request, exc: HTTPException):  # type: ignore[arg-type]
    """Maneja *fastapi.HTTPException* y derivados."""
    logger.warning("HTTPException %s -> %s", exc.status_code, exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=_build_payload(str(exc.detail), exc.status_code),
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError):  # type: ignore[arg-type]
    """Maneja errores de validación Pydantic/FastAPI (código 422)."""
    logger.warning("Validation error on %s: %s", request.url.path, exc.errors())
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=_build_payload(
            "Error de validación en los parámetros de la solicitud",
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            errors=exc.errors(),
        ),
    )


async def generic_exception_handler(request: Request, exc: Exception):  # noqa: BLE001
    """Captura cualquier excepción no gestionada y responde 500."""
    logger.error("Unhandled exception on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_build_payload(
            "Error interno del servidor",
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        ),
    )

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_exception_handlers(app):  # type: ignore[valid-type]
    """Registra los manejadores de excepción sobre una instancia de *FastAPI*."""
    app.add_exception_handler(HTTPException, http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore[arg-type]
    # Debe ir al final para no sobre-escribir los específicos
    app.add_exception_handler(Exception, generic_exception_handler)  # type: ignore[arg-type] 