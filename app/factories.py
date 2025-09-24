# app/factories.py
"""
Application and Worker Factory Functions.

This module centralizes all initialization logic for FastAPI app and Celery worker
to avoid side effects when importing modules and improve testability.
"""
import os
import logging
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from celery import Celery
from kombu import Queue
from prometheus_fastapi_instrumentator import Instrumentator

from app.otel import init_otel, instrument_fastapi
from app.error_handlers import register_exception_handlers
from app.middleware.rate_limiter import RateLimitMiddleware
from app.middleware.request_logger import RequestLoggingMiddleware
from app.middleware.trace_context import TraceContextMiddleware
from app.logging_config import setup_logging


def create_app(enable_telemetry: bool = True, enable_cors: bool = True) -> FastAPI:
    """
    Create and configure FastAPI application.

    Args:
        enable_telemetry: Whether to initialize OpenTelemetry and Prometheus
        enable_cors: Whether to enable CORS middleware

    Returns:
        Configured FastAPI application instance
    """
    # Initialize telemetry if enabled
    if enable_telemetry:
        init_otel()

    # Create FastAPI app
    app = FastAPI(
        title="Chess Player Analyzer API",
        description="API para análisis avanzado de jugadores de ajedrez usando Stockfish",
        version="2.0.0",
        contact={
            "name": "Chess Analyzer Support",
            "email": "support@example.com",
        },
        license_info={
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT",
        },
    )

    # Configure telemetry
    if enable_telemetry:
        instrument_fastapi(app)

        # Instrument with Prometheus
        instrumentator = Instrumentator()
        instrumentator.instrument(app).expose(app)

    # Configure CORS
    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # En producción, especificar dominios exactos
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Add custom middlewares
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(TraceContextMiddleware)

    # Register exception handlers
    register_exception_handlers(app)

    return app


def create_worker(enable_telemetry: bool = True, enable_logging: bool = True) -> Celery:
    """
    Create and configure Celery worker.

    Args:
        enable_telemetry: Whether to initialize OpenTelemetry
        enable_logging: Whether to setup structured JSON logging

    Returns:
        Configured Celery application instance
    """
    # Setup logging if enabled
    if enable_logging:
        setup_logging()

    # Initialize telemetry if enabled
    if enable_telemetry:
        init_otel()

    # Environment configuration
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    task_soft_time_limit = int(os.getenv("TASK_SOFT_TIME_LIMIT", "1800"))  # 30 min default
    task_time_limit = int(os.getenv("TASK_TIME_LIMIT", "1860"))  # hard limit (soft + 1 min)
    task_max_retries = int(os.getenv("TASK_MAX_RETRIES", "3"))  # default max retries

    # Create Celery app
    celery_app = Celery("chess_tasks", broker=redis_url, backend=redis_url)

    # Configure task queues with priority support
    celery_app.conf.task_default_queue = "default"
    celery_app.conf.task_queues = (
        Queue("default", max_priority=10),
    )

    # Configure task settings
    celery_app.conf.update(
        # When a worker is lost (OOM/timeout) we want the broker to re-queue the task
        task_reject_on_worker_lost=True,
        # Force ACK *after* the task finishes so it can be retried on crash
        task_acks_late=True,
        # Apply global time limits – individual tasks can override these
        task_soft_time_limit=task_soft_time_limit,
        task_time_limit=task_time_limit,
        # Global retry defaults (used by autoretry_for)
        task_default_retry_delay=60,  # seconds between automatic retries
        task_max_retries=task_max_retries,
        # Result expiration
        result_expires=3600,  # 1 hour
        # Worker settings
        worker_prefetch_multiplier=1,
    )

    return celery_app


def get_logger(name: str) -> logging.Logger:
    """
    Get logger instance for the given name.

    This is a helper function that can be used after setup_logging() has been called.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)