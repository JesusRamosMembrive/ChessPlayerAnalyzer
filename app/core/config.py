"""
Configuración centralizada de la aplicación.
Todas las variables de entorno se definen aquí para evitar dispersión.
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class DatabaseConfig:
    """Configuración de base de datos."""
    url: str
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 1800
    max_retries: int = 5
    initial_retry_delay: float = 2.0
    read_replica_urls: str = ""


@dataclass(frozen=True)
class RedisConfig:
    """Configuración de Redis."""
    url: str


@dataclass(frozen=True)
class StockfishConfig:
    """Configuración de Stockfish."""
    path: str
    depth: int = 12


@dataclass(frozen=True)
class CeleryConfig:
    """Configuración de Celery."""
    redis_url: str
    soft_time_limit: int = 1800  # 30 minutos
    time_limit: int = 1860       # 31 minutos
    max_retries: int = 3


@dataclass(frozen=True)
class TracingConfig:
    """Configuración de OpenTelemetry."""
    enabled: bool = False
    service_name: str = "chess-analyzer"
    jaeger_host: str = "localhost"
    jaeger_port: int = 6831


@dataclass(frozen=True)
class ExternalConfig:
    """Configuración de servicios externos."""
    syzygy_path: Path
    archive_dir: str = "archives"


@dataclass(frozen=True)
class LoggingConfig:
    """Configuración de logging."""
    level: str = "INFO"


@dataclass(frozen=True)
class AppConfig:
    """Configuración principal de la aplicación."""
    database: DatabaseConfig
    redis: RedisConfig
    stockfish: StockfishConfig
    celery: CeleryConfig
    tracing: TracingConfig
    external: ExternalConfig
    logging: LoggingConfig

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Crea configuración desde variables de entorno."""
        return cls(
            database=DatabaseConfig(
                url=os.getenv(
                    "DATABASE_URL",
                    "postgresql+psycopg://chess:chess@postgres:5432/chessdb"
                ),
                pool_size=int(os.getenv("DB_POOL_SIZE", "10")),
                max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "20")),
                pool_timeout=int(os.getenv("DB_POOL_TIMEOUT", "30")),
                pool_recycle=int(os.getenv("DB_POOL_RECYCLE", "1800")),
                max_retries=int(os.getenv("DB_MAX_RETRIES", "5")),
                initial_retry_delay=float(os.getenv("DB_RETRY_DELAY", "2.0")),
                read_replica_urls=os.getenv("READ_REPLICA_URLS", "")
            ),
            redis=RedisConfig(
                url=os.getenv("REDIS_URL", "redis://redis:6379/0")
            ),
            stockfish=StockfishConfig(
                path=os.getenv("STOCKFISH_PATH", "stockfish"),
                depth=int(os.getenv("STOCKFISH_DEPTH", "12"))
            ),
            celery=CeleryConfig(
                redis_url=os.getenv("REDIS_URL", "redis://redis:6379/0"),
                soft_time_limit=int(os.getenv("TASK_SOFT_TIME_LIMIT", "1800")),
                time_limit=int(os.getenv("TASK_TIME_LIMIT", "1860")),
                max_retries=int(os.getenv("TASK_MAX_RETRIES", "3"))
            ),
            tracing=TracingConfig(
                enabled=os.getenv("ENABLE_TRACING", "").lower() in ("1", "true", "yes", "on"),
                service_name=os.getenv("OTEL_SERVICE_NAME", "chess-analyzer"),
                jaeger_host=os.getenv("OTEL_EXPORTER_JAEGER_AGENT_HOST", "localhost"),
                jaeger_port=int(os.getenv("OTEL_EXPORTER_JAEGER_AGENT_PORT", "6831"))
            ),
            external=ExternalConfig(
                syzygy_path=Path(os.getenv("SYZYGY_PATH", "/data/syzygy")),
                archive_dir=os.getenv("FETCH_ARCHIVE_DIR", "archives")
            ),
            logging=LoggingConfig(
                level=os.getenv("LOG_LEVEL", "INFO").upper()
            )
        )


# Instancia global de configuración
config = AppConfig.from_env()


# Funciones de conveniencia para compatibilidad
def get_database_url() -> str:
    """Obtiene la URL de la base de datos."""
    return config.database.url


def get_redis_url() -> str:
    """Obtiene la URL de Redis."""
    return config.redis.url


def get_stockfish_config() -> tuple[str, int]:
    """Obtiene configuración de Stockfish (path, depth)."""
    return config.stockfish.path, config.stockfish.depth


def is_tracing_enabled() -> bool:
    """Verifica si el tracing está habilitado."""
    return config.tracing.enabled