"""
Configuración del Application Performance Monitoring (APM) con Sentry.

Este módulo configura Sentry SDK para rastreo de errores y rendimiento
tanto en FastAPI como en Celery.
"""

import os
import logging
from typing import Optional, Dict, Any

import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.starlette import StarletteIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
from sentry_sdk.integrations.celery import CeleryIntegration
from sentry_sdk.integrations.redis import RedisIntegration
from sentry_sdk.integrations.logging import LoggingIntegration

logger = logging.getLogger(__name__)


class APMConfig:
    """Configuración centralizada del APM."""
    
    def __init__(self):
        self.dsn = os.getenv("SENTRY_DSN", "")
        self.environment = os.getenv("SENTRY_ENVIRONMENT", "development")
        self.release = os.getenv("SENTRY_RELEASE", "chess-analyzer@1.0.0")
        self.traces_sample_rate = float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.1"))
        self.profiles_sample_rate = float(os.getenv("SENTRY_PROFILES_SAMPLE_RATE", "0.1"))
        self.debug = os.getenv("SENTRY_DEBUG", "false").lower() == "true"
        self.enabled = os.getenv("SENTRY_ENABLED", "true").lower() == "true"
        
        # Configuración específica de rendimiento
        self.enable_tracing = os.getenv("SENTRY_ENABLE_TRACING", "true").lower() == "true"
        self.max_breadcrumbs = int(os.getenv("SENTRY_MAX_BREADCRUMBS", "100"))
        self.attach_stacktrace = os.getenv("SENTRY_ATTACH_STACKTRACE", "true").lower() == "true"
        
        # Filtros de transacciones para evitar ruido
        self.ignored_paths = [
            "/health",
            "/api/v1/health",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
        ]


apm_config = APMConfig()


def before_send_transaction(event: Dict[str, Any], hint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Filtra transacciones antes de enviarlas a Sentry.
    
    Ignora rutas de health checks y métricas para reducir ruido.
    """
    if "transaction" in event:
        transaction_name = event["transaction"]
        
        # Ignorar rutas específicas
        for ignored_path in apm_config.ignored_paths:
            if ignored_path in transaction_name:
                return None
                
    return event


def before_send(event: Dict[str, Any], hint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Procesa eventos antes de enviarlos a Sentry.
    
    Puede filtrar información sensible o agregar contexto adicional.
    """
    # Filtrar información sensible si es necesario
    if "request" in event and "data" in event["request"]:
        # Eliminar campos sensibles como contraseñas
        sensitive_fields = ["password", "token", "api_key", "secret"]
        request_data = event["request"]["data"]
        
        if isinstance(request_data, dict):
            for field in sensitive_fields:
                if field in request_data:
                    request_data[field] = "[REDACTED]"
    
    return event


def init_sentry_fastapi():
    """
    Inicializa Sentry para la aplicación FastAPI.
    
    Configura integraciones específicas para web y base de datos.
    """
    if not apm_config.enabled or not apm_config.dsn:
        logger.info("Sentry APM está deshabilitado o no configurado")
        return
    
    try:
        sentry_sdk.init(
            dsn=apm_config.dsn,
            environment=apm_config.environment,
            release=apm_config.release,
            debug=apm_config.debug,
            
            # Configuración de sampling
            traces_sample_rate=apm_config.traces_sample_rate,
            profiles_sample_rate=apm_config.profiles_sample_rate,
            
            # Integraciones
            integrations=[
                FastApiIntegration(
                    transaction_style="endpoint",
                    failed_request_status_codes={400, 401, 403, 404, 500, 501, 502, 503, 504},
                ),
                StarletteIntegration(
                    transaction_style="endpoint",
                ),
                SqlalchemyIntegration(),
                RedisIntegration(),
                LoggingIntegration(
                    level=logging.INFO,  # Captura logs INFO y superiores
                    event_level=logging.ERROR,  # Solo envía como eventos los ERROR
                ),
            ],
            
            # Configuración adicional
            max_breadcrumbs=apm_config.max_breadcrumbs,
            attach_stacktrace=apm_config.attach_stacktrace,
            send_default_pii=False,  # No enviar información personal identificable
            
            # Callbacks personalizados
            before_send=before_send,
            before_send_transaction=before_send_transaction,
            
            # Configuración de rendimiento
            enable_tracing=apm_config.enable_tracing,
        )
        
        logger.info(f"Sentry APM inicializado para FastAPI - Entorno: {apm_config.environment}")
        
    except Exception as e:
        logger.error(f"Error al inicializar Sentry: {e}")


def init_sentry_celery():
    """
    Inicializa Sentry para los workers de Celery.
    
    Configura integraciones específicas para tareas asíncronas.
    """
    if not apm_config.enabled or not apm_config.dsn:
        logger.info("Sentry APM está deshabilitado o no configurado para Celery")
        return
    
    try:
        sentry_sdk.init(
            dsn=apm_config.dsn,
            environment=apm_config.environment,
            release=apm_config.release,
            debug=apm_config.debug,
            
            # Configuración de sampling
            traces_sample_rate=apm_config.traces_sample_rate,
            profiles_sample_rate=apm_config.profiles_sample_rate,
            
            # Integraciones
            integrations=[
                CeleryIntegration(
                    monitor_beat_tasks=True,  # Monitorear tareas periódicas
                    propagate_traces=True,    # Propagar trazas entre tareas
                ),
                SqlalchemyIntegration(),
                RedisIntegration(),
                LoggingIntegration(
                    level=logging.INFO,
                    event_level=logging.ERROR,
                ),
            ],
            
            # Configuración adicional
            max_breadcrumbs=apm_config.max_breadcrumbs,
            attach_stacktrace=apm_config.attach_stacktrace,
            send_default_pii=False,
            
            # Callbacks personalizados
            before_send=before_send,
            
            # Configuración de rendimiento
            enable_tracing=apm_config.enable_tracing,
        )
        
        logger.info(f"Sentry APM inicializado para Celery - Entorno: {apm_config.environment}")
        
    except Exception as e:
        logger.error(f"Error al inicializar Sentry para Celery: {e}")


def capture_message(message: str, level: str = "info", **kwargs):
    """
    Captura un mensaje personalizado en Sentry.
    
    Args:
        message: Mensaje a capturar
        level: Nivel del mensaje (debug, info, warning, error, fatal)
        **kwargs: Contexto adicional para el mensaje
    """
    if apm_config.enabled:
        sentry_sdk.capture_message(message, level=level, **kwargs)


def capture_exception(exception: Exception = None, **kwargs):
    """
    Captura una excepción en Sentry con contexto adicional.
    
    Args:
        exception: Excepción a capturar (si no se proporciona, captura la actual)
        **kwargs: Contexto adicional para la excepción
    """
    if apm_config.enabled:
        sentry_sdk.capture_exception(exception, **kwargs)


def add_breadcrumb(message: str, category: str = None, level: str = "info", data: dict = None):
    """
    Agrega un breadcrumb para seguimiento de eventos.
    
    Args:
        message: Mensaje del breadcrumb
        category: Categoría del evento (ej: "auth", "database", "analysis")
        level: Nivel del breadcrumb
        data: Datos adicionales
    """
    if apm_config.enabled:
        sentry_sdk.add_breadcrumb(
            message=message,
            category=category,
            level=level,
            data=data or {}
        )


def set_user_context(user_id: str = None, username: str = None, email: str = None, **kwargs):
    """
    Establece el contexto del usuario para las trazas.
    
    Args:
        user_id: ID del usuario
        username: Nombre de usuario
        email: Email del usuario
        **kwargs: Datos adicionales del usuario
    """
    if apm_config.enabled:
        sentry_sdk.set_user({
            "id": user_id,
            "username": username,
            "email": email,
            **kwargs
        })


def set_tag(key: str, value: str):
    """
    Establece un tag para categorizar eventos.
    
    Args:
        key: Nombre del tag
        value: Valor del tag
    """
    if apm_config.enabled:
        sentry_sdk.set_tag(key, value)


def set_context(name: str, context: dict):
    """
    Establece contexto adicional para los eventos.
    
    Args:
        name: Nombre del contexto
        context: Diccionario con datos del contexto
    """
    if apm_config.enabled:
        sentry_sdk.set_context(name, context)


# Decorador para transacciones personalizadas
def measure_performance(operation: str, description: str = None):
    """
    Decorador para medir el rendimiento de funciones específicas.
    
    Args:
        operation: Tipo de operación (ej: "db.query", "http.request", "task.analysis")
        description: Descripción opcional de la operación
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not apm_config.enabled:
                return func(*args, **kwargs)
                
            with sentry_sdk.start_transaction(
                op=operation,
                name=func.__name__,
                description=description or func.__doc__
            ) as transaction:
                try:
                    result = func(*args, **kwargs)
                    transaction.set_status("ok")
                    return result
                except Exception as e:
                    transaction.set_status("internal_error")
                    raise
        
        return wrapper
    return decorator 