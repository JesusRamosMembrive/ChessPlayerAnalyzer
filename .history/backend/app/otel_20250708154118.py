"""
Configuración de OpenTelemetry para trazas de FastAPI, Celery y SQLAlchemy hacia Jaeger.
"""
import os
import logging

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter

from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.celery import CeleryInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

from app.database import engine

logger = logging.getLogger(__name__)


def init_otel(app=None):
    """
    Inicializa OpenTelemetry con Jaeger y aplica instrumentaciones.
    - OTEL_EXPORTER_JAEGER_AGENT_HOST: host del agente Jaeger (default localhost)
    - OTEL_EXPORTER_JAEGER_AGENT_PORT: puerto (default 6831)
    - OTEL_SERVICE_NAME: nombre del servicio (default chess-analyzer)

    Si se pasa una instancia FastAPI en `app`, también instrumenta HTTP.
    """
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

    # Instrumentaciones automáticas
    # FastAPI
    if app:
        FastAPIInstrumentor.instrument_app(app)
        logger.info("OTEL: FastAPI instrumentado")

    # SQLAlchemy
    SQLAlchemyInstrumentor().instrument(engine=engine)
    logger.info("OTEL: SQLAlchemy instrumentado")

    # Celery
    CeleryInstrumentor().instrument()
    logger.info("OTEL: Celery instrumentado") 