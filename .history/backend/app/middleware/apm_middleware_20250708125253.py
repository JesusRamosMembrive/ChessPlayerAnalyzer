"""
Middleware personalizado para Application Performance Monitoring (APM).

Captura métricas detalladas de rendimiento, errores y comportamiento
de las solicitudes HTTP.
"""

import time
import logging
from typing import Callable, Dict, Any
import json

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.datastructures import Headers

from app.apm import (
    add_breadcrumb,
    set_tag,
    set_context,
    capture_exception,
    apm_config
)

logger = logging.getLogger(__name__)


class APMMiddleware(BaseHTTPMiddleware):
    """
    Middleware que captura métricas detalladas para el APM.
    
    Funcionalidades:
    - Mide el tiempo de respuesta de cada endpoint
    - Captura información del request/response
    - Agrega breadcrumbs para trazabilidad
    - Establece tags y contexto para categorización
    - Maneja errores y excepciones automáticamente
    """
    
    def __init__(self, app, **kwargs):
        super().__init__(app, **kwargs)
        self.app = app
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Procesa cada request y captura métricas para el APM.
        """
        if not apm_config.enabled:
            return await call_next(request)
            
        # Ignorar rutas que no queremos monitorear
        if any(path in request.url.path for path in apm_config.ignored_paths):
            return await call_next(request)
        
        # Iniciar medición de tiempo
        start_time = time.perf_counter()
        
        # Capturar información del request
        request_info = {
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "headers": dict(request.headers),
            "client": request.client.host if request.client else "unknown",
        }
        
        # Establecer contexto del request
        set_context("request", {
            "url": str(request.url),
            "method": request.method,
            "path": request.url.path,
            "query_string": str(request.url.query),
            "client_ip": request_info["client"],
        })
        
        # Establecer tags básicos
        set_tag("http.method", request.method)
        set_tag("http.path", request.url.path)
        set_tag("http.scheme", request.url.scheme)
        
        # Agregar breadcrumb para el inicio del request
        add_breadcrumb(
            message=f"{request.method} {request.url.path}",
            category="http.request",
            level="info",
            data={
                "method": request.method,
                "url": str(request.url),
                "client": request_info["client"],
            }
        )
        
        # Variables para capturar el response
        response = None
        status_code = 500
        response_headers = {}
        
        try:
            # Procesar el request
            response = await call_next(request)
            status_code = response.status_code
            response_headers = dict(response.headers)
            
            # Para StreamingResponse, necesitamos capturar el contenido de manera especial
            if isinstance(response, StreamingResponse):
                # Crear un generador que capture el contenido mientras lo transmite
                async def stream_with_metrics():
                    content_length = 0
                    async for chunk in response.body_iterator:
                        content_length += len(chunk)
                        yield chunk
                    
                    # Agregar métricas después de transmitir todo
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    self._log_response_metrics(
                        request,
                        status_code,
                        duration_ms,
                        content_length,
                        response_headers
                    )
                
                # Reemplazar el body_iterator original
                response.body_iterator = stream_with_metrics()
                return response
            
            # Para respuestas normales
            duration_ms = (time.perf_counter() - start_time) * 1000
            content_length = int(response_headers.get("content-length", 0))
            
            self._log_response_metrics(
                request,
                status_code,
                duration_ms,
                content_length,
                response_headers
            )
            
            return response
            
        except Exception as e:
            # Calcular duración hasta el error
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            # Capturar la excepción en el APM
            capture_exception(
                e,
                extra={
                    "request": request_info,
                    "duration_ms": duration_ms,
                }
            )
            
            # Agregar breadcrumb del error
            add_breadcrumb(
                message=f"Error en {request.method} {request.url.path}: {str(e)}",
                category="http.error",
                level="error",
                data={
                    "exception_type": type(e).__name__,
                    "exception_message": str(e),
                    "duration_ms": duration_ms,
                }
            )
            
            # Re-lanzar la excepción para que otros middlewares la manejen
            raise
            
    def _log_response_metrics(
        self,
        request: Request,
        status_code: int,
        duration_ms: float,
        content_length: int,
        response_headers: Dict[str, Any]
    ):
        """
        Registra las métricas del response en el APM.
        """
        # Establecer tags adicionales
        set_tag("http.status_code", str(status_code))
        set_tag("http.response_time_ms", str(int(duration_ms)))
        set_tag("http.content_length", str(content_length))
        
        # Determinar si es un error
        is_error = status_code >= 400
        if is_error:
            set_tag("error", "true")
            level = "error"
        else:
            level = "info"
        
        # Agregar breadcrumb del response
        add_breadcrumb(
            message=f"{request.method} {request.url.path} - {status_code} ({duration_ms:.2f} ms)",
            category="http.response",
            level=level,
            data={
                "status_code": status_code,
                "duration_ms": duration_ms,
                "content_length": content_length,
                "content_type": response_headers.get("content-type", "unknown"),
            }
        )
        
        # Establecer contexto del response
        set_context("response", {
            "status_code": status_code,
            "headers": response_headers,
            "duration_ms": duration_ms,
            "content_length": content_length,
        })
        
        # Log detallado para análisis
        logger.info(
            "APM: %s %s - %s (%.2f ms, %d bytes) from %s",
            request.method,
            request.url.path,
            status_code,
            duration_ms,
            content_length,
            request.client.host if request.client else "unknown"
        )
        
        # Métricas de rendimiento adicionales
        if duration_ms > 1000:  # Requests lentos (> 1 segundo)
            add_breadcrumb(
                message=f"Slow request detected: {request.url.path}",
                category="performance",
                level="warning",
                data={
                    "duration_ms": duration_ms,
                    "threshold_ms": 1000,
                }
            )
            
        # Detectar respuestas grandes
        if content_length > 1_000_000:  # > 1MB
            add_breadcrumb(
                message=f"Large response detected: {content_length} bytes",
                category="performance",
                level="warning",
                data={
                    "content_length": content_length,
                    "threshold_bytes": 1_000_000,
                }
            ) 