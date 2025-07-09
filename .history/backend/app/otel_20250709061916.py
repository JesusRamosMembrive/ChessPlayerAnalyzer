"""
Configuración de OpenTelemetry para trazas de FastAPI, Celery y SQLAlchemy hacia Jaeger.
"""
import os
import logging

from opentelemetry import trace  # type: ignore
from opentelemetry.sdk.resources import Resource  # type: ignore
from opentelemetry.sdk.trace import TracerProvider  # type: ignore
from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore
from opentelemetry.exporter.jaeger.thrift import JaegerExporter  # type: ignore

from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor  # type: ignore
from opentelemetry.instrumentation.celery import CeleryInstrumentor  # type: ignore
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor  # type: ignore
from opentelemetry.instrumentation.requests import RequestsInstrumentor  # type: ignore
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor  # type: ignore

from app.database import engine

logger = logging.getLogger(__name__)


# Variables globales para evitar reinicialización
_otel_initialized = False
_fastapi_instrumented = False
_sqlalchemy_instrumented = False
_celery_instrumented = False

def init_otel_base():
    """
    Inicializa el TracerProvider de OpenTelemetry con Jaeger.
    Se puede llamar múltiples veces sin problemas.
    """
    global _otel_initialized
    if _otel_initialized:
        return
    
    try:
        # Leer configuración desde variables de entorno
        service_name = os.getenv("OTEL_SERVICE_NAME", "chess-analyzer")
        jaeger_host = os.getenv("OTEL_EXPORTER_JAEGER_AGENT_HOST", "localhost")
        jaeger_port = int(os.getenv("OTEL_EXPORTER_JAEGER_AGENT_PORT", "6831"))
        
        logger.info(f"OTEL Config - Service: {service_name}, Host: {jaeger_host}, Port: {jaeger_port}")

        # Crear proveedor de trazas con metadata de recurso
        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(provider)

        # Configurar exportador Jaeger
        jaeger_exporter = JaegerExporter(
            agent_host_name=jaeger_host,
            agent_port=jaeger_port,
        )
        span_processor = BatchSpanProcessor(jaeger_exporter)
        provider.add_span_processor(span_processor)

        logger.info(f"OTEL Jaeger inicializado en {jaeger_host}:{jaeger_port} para servicio {service_name}")
        _otel_initialized = True
    except Exception as e:
        logger.warning(f"No se pudo inicializar OpenTelemetry: {e}")

def instrument_fastapi(app):
    """
    Instrumenta FastAPI. Debe llamarse ANTES de que la aplicación inicie.
    """
    global _fastapi_instrumented
    if _fastapi_instrumented:
        return
        
    init_otel_base()
    try:
        FastAPIInstrumentor.instrument_app(app)
        logger.info("OTEL: FastAPI instrumentado")
        _fastapi_instrumented = True
    except Exception as e:
        logger.warning(f"No se pudo instrumentar FastAPI: {e}")

def init_otel():
    """
    Inicializa OpenTelemetry y aplica instrumentaciones para SQLAlchemy y Celery.
    """
    global _sqlalchemy_instrumented, _celery_instrumented
    
    init_otel_base()
    
    # SQLAlchemy
    if not _sqlalchemy_instrumented:
        try:
            SQLAlchemyInstrumentor().instrument(engine=engine)
            logger.info("OTEL: SQLAlchemy instrumentado")
            _sqlalchemy_instrumented = True
        except Exception as e:
            logger.warning(f"No se pudo instrumentar SQLAlchemy: {e}")
    
    # Celery
    if not _celery_instrumented:
        try:
            CeleryInstrumentor().instrument(propagate=True)
            logger.info("OTEL: Celery instrumentado")
            _celery_instrumented = True
        except Exception as e:
            logger.warning(f"No se pudo instrumentar Celery: {e}") 

    # Requests
    try:
        RequestsInstrumentor().instrument()
        logger.info("OTEL: Requests instrumentado")
    except Exception as e:
        logger.warning(f"No se pudo instrumentar Requests: {e}")

    # HTTPX (cliente async/sync)
    try:
        HTTPXClientInstrumentor().instrument()
        logger.info("OTEL: HTTPX instrumentado")
    except Exception as e:
        logger.warning(f"No se pudo instrumentar HTTPX: {e}") 