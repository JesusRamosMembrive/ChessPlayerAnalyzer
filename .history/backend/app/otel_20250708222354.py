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

from app.database import engine

logger = logging.getLogger(__name__)


# Variable global para evitar reinicialización
_otel_initialized = False

def init_otel_base():
    """
    Inicializa el TracerProvider de OpenTelemetry con Jaeger.
    Se puede llamar múltiples veces sin problemas.
    """
    global _otel_initialized
    if _otel_initialized:
        return
    
    # Leer configuración desde variables de entorno
    service_name = os.getenv("OTEL_SERVICE_NAME", "chess-analyzer")
    jaeger_host = os.getenv("OTEL_EXPORTER_JAEGER_AGENT_HOST", "localhost")
    jaeger_port = int(os.getenv("OTEL_EXPORTER_JAEGER_AGENT_PORT", "6831"))

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

def instrument_fastapi(app):
    """
    Instrumenta FastAPI. Debe llamarse ANTES de que la aplicación inicie.
    """
    init_otel_base()
    FastAPIInstrumentor.instrument_app(app)
    logger.info("OTEL: FastAPI instrumentado")

def init_otel():
    """
    Inicializa OpenTelemetry y aplica instrumentaciones para SQLAlchemy y Celery.
    """
    init_otel_base()
    
    # SQLAlchemy
    SQLAlchemyInstrumentor().instrument(engine=engine)
    logger.info("OTEL: SQLAlchemy instrumentado")

    # Celery
    CeleryInstrumentor().instrument()
    logger.info("OTEL: Celery instrumentado") 