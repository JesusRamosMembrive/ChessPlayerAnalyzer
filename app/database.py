# app/database.py
"""Performance-optimized PostgreSQL connection with clean architecture."""
import logging
import time
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from sqlmodel import SQLModel, create_engine
from itertools import cycle
from typing import List
from sqlalchemy.orm import sessionmaker
from sqlmodel import Session as SQLModelSession
from sqlalchemy.engine import Engine

from .core.config import get_config, get_optimized_database_settings

# Get configuration from centralized config
config = get_config()
optimized_settings = get_optimized_database_settings()


def _create_engine_retry(url: str, **kwargs) -> Engine:
    """Performance-optimized engine creation with exponential backoff retries."""
    attempt = 0
    delay = config.database.initial_retry_delay
    log = logging.getLogger(__name__)

    while True:
        try:
            eng: Engine = create_engine(url, **kwargs)

            # Quick connection test
            with eng.connect() as conn:
                conn.execute(text("SELECT 1"))

            # Dispose connections to prevent forked workers from inheriting FDs
            eng.dispose()

            if attempt:
                log.info("Database connection restored after %d attempt(s)", attempt)
            return eng

        except OperationalError as exc:
            attempt += 1
            if attempt > config.database.max_retries:
                log.error("Failed to connect to database after %d attempts: %s", attempt - 1, exc)
                raise

            log.warning(
                "Database connection error (attempt %d/%d). Retrying in %.1f s...",
                attempt,
                config.database.max_retries,
                delay,
            )
            time.sleep(delay)
            delay *= 2  # exponential backoff

# Performance-optimized global engine with retry logic
engine = _create_engine_retry(
    config.database.url,
    **optimized_settings
)

# Performance-optimized read replicas
read_engines: List[Engine] = []
if config.database.read_replica_urls:
    for url in [u.strip() for u in config.database.read_replica_urls.split(",") if u.strip()]:
        read_engines.append(
            _create_engine_retry(url, **optimized_settings)
        )

# Ciclo round-robin para balanceo; fallback a primaria si no hay réplicas
_read_cycle = cycle(read_engines) if read_engines else None


def _next_read_engine():
    """Devuelve la siguiente réplica en orden o la primaria si no hay."""
    if _read_cycle:
        return next(_read_cycle)
    return engine  # type: ignore[name-defined]


class RoutingSession(SQLModelSession):
    """Session que envía consultas SELECT a una réplica de lectura."""

    def get_bind(self, mapper=None, clause=None, **kw):  # noqa: D401
        # Durante flush/emisión de cambios → primaria
        if self._flushing:
            return engine  # type: ignore[name-defined]

        # Si la cláusula es un SELECT y existen réplicas → réplica
        if clause is not None and getattr(clause, "__visit_name__", "") == "select":
            return _next_read_engine()

        # Por defecto (INSERT/UPDATE/DELETE, DDL, etc.) → primaria
        return engine  # type: ignore[name-defined]


# Factoría de sesiones usando la clase de enrutamiento
SessionLocal = sessionmaker(bind=engine, class_=RoutingSession, autocommit=False, autoflush=False)


def get_session():
    """Dependencia FastAPI que abre y cierra la sesión usando RoutingSession."""
    with SessionLocal() as session:
        yield session


def init_db():
    """Initialize database tables."""
    from app import models  # Import models here to avoid circular imports
    SQLModel.metadata.create_all(engine)
