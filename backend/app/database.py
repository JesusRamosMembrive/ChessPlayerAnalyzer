# app/database.py
"""Conexión global a PostgreSQL y helper para obtener sesiones."""
import os
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

# La URL debe coincidir con docker-compose.yml
DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://chess:chess@postgres:5432/chessdb",  # <-- chessdb, no chess!
)

# Configuración de pool de conexiones; valores predeterminados razonables pero override mediante variables de entorno
POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10"))  # conexiones persistentes en el pool
MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "20"))  # conexiones adicionales temporales
POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "30"))  # segundos para esperar una conexión libre
POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "1800"))  # reciclar después de 30 min

# Configuración para reintentos de conexión
MAX_RETRIES = int(os.getenv("DB_MAX_RETRIES", "5"))  # número máximo de reintentos
INITIAL_RETRY_DELAY = float(os.getenv("DB_RETRY_DELAY", "2.0"))  # segundos entre reintentos inicial


def _create_engine_retry(url: str, **kwargs) -> Engine:
    """Crea un Engine con reintentos exponenciales.

    Parametros (via kwargs) se pasan directamente a ``create_engine``.
    El algoritmo usa espera exponencial (2x) con un máximo de ``MAX_RETRIES``.
    Si no se logra establecer conexión, se vuelve a lanzar la excepción para
    que el proceso se caiga de forma explícita (útil para orquestadores).
    """

    attempt = 0
    delay = INITIAL_RETRY_DELAY
    log = logging.getLogger(__name__)

    while True:
        try:
            eng: Engine = create_engine(url, **kwargs)

            # Intento rápido de comprobar la conexión.
            with eng.connect() as conn:
                conn.execute(text("SELECT 1"))

            # Cerramos todas las conexiones abiertas para que los workers forked
            # (p.ej. Celery) no hereden FDs de la conexión creada en el proceso
            # padre. Se crearán conexiones nuevas y seguras en cada hijo cuando
            # sea necesario.
            eng.dispose()

            if attempt:  # Hubo reintentos previos
                log.info("Conexión a la base de datos restablecida tras %d intento(s)", attempt)
            return eng

        except OperationalError as exc:
            attempt += 1
            if attempt > MAX_RETRIES:
                log.error("No se pudo conectar a la base de datos tras %d intentos: %s", attempt - 1, exc)
                raise

            log.warning(
                "Error al conectar con la base de datos (intento %d/%d). Reintentando en %.1f s…", 
                attempt,
                MAX_RETRIES,
                delay,
            )
            time.sleep(delay)
            delay *= 2  # back-off exponencial

# Engine global con pooling configurado explícitamente con reintentos
engine = _create_engine_retry(
    DB_URL,
    echo=False,
    pool_pre_ping=True,
    pool_size=POOL_SIZE,
    max_overflow=MAX_OVERFLOW,
    pool_timeout=POOL_TIMEOUT,
    pool_recycle=POOL_RECYCLE,
)

# Variable de entorno con URLs separadas por coma para las réplicas
READ_REPLICA_URLS = os.getenv("READ_REPLICA_URLS", "")

read_engines: List[Engine] = []  # lista de engines de solo-lectura
if READ_REPLICA_URLS:
    for url in [u.strip() for u in READ_REPLICA_URLS.split(",") if u.strip()]:
        read_engines.append(
            _create_engine_retry(
                url,
                echo=False,
                pool_pre_ping=True,
                pool_size=POOL_SIZE,
                max_overflow=MAX_OVERFLOW,
                pool_timeout=POOL_TIMEOUT,
                pool_recycle=POOL_RECYCLE,
            )
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
