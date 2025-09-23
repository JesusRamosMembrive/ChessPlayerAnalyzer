"""
Configuración centralizada de la aplicación.
Todas las variables de entorno se definen aquí para evitar dispersión.
Performance-optimized configuration with clean architecture.
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

# from .performance import performance_optimizer  # Removed to fix circular import


@dataclass(frozen=True)
class DatabaseConfig:
    """Performance-optimized database configuration."""
    url: str
    pool_size: int = 20  # Optimized from 10
    max_overflow: int = 30  # Optimized from 20
    pool_timeout: int = 30
    pool_recycle: int = 3600  # Optimized from 1800
    pool_pre_ping: bool = True  # New optimization
    max_retries: int = 5
    initial_retry_delay: float = 2.0
    read_replica_urls: str = ""
    connect_args: Dict[str, Any] = None

    def __post_init__(self):
        if self.connect_args is None:
            # Apply performance optimizations based on database type
            if "postgresql" in self.url or "psycopg" in self.url:
                # PostgreSQL specific optimizations
                object.__setattr__(self, 'connect_args', {
                    "connect_timeout": 20
                })
            else:
                # SQLite specific optimizations (fallback)
                object.__setattr__(self, 'connect_args', {
                    "check_same_thread": False,
                    "timeout": 20,
                    "pragmas": {
                        "journal_mode": "WAL",
                        "cache_size": -1 * 64000,  # 64MB cache
                        "foreign_keys": 1,
                        "ignore_check_constraints": 0,
                        "synchronous": 0,
                        "temp_store": 2,
                        "mmap_size": 268435456,  # 256MB mmap
                    }
                })


@dataclass(frozen=True)
class RedisConfig:
    """Performance-optimized Redis configuration."""
    url: str
    max_connections: int = 50  # Optimized from default 10
    retry_on_timeout: bool = True
    health_check_interval: int = 30
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    socket_keepalive: bool = True

    def get_optimized_settings(self) -> Dict[str, Any]:
        """Get Redis optimization settings."""
        return {
            "max_connections": 50,
            "retry_on_timeout": True,
            "health_check_interval": 30,
            "socket_timeout": 5,
            "socket_connect_timeout": 5,
            "socket_keepalive": True,
            "socket_keepalive_options": {}
        }


@dataclass(frozen=True)
class StockfishConfig:
    """Configuración de Stockfish."""
    path: str
    depth: int = 12


@dataclass(frozen=True)
class CeleryConfig:
    """Performance-optimized Celery configuration."""
    redis_url: str
    soft_time_limit: int = 1500  # 25 minutes (optimized)
    time_limit: int = 1800       # 30 minutes (optimized)
    max_retries: int = 3
    worker_prefetch_multiplier: int = 4  # Optimized from 1
    worker_max_tasks_per_child: int = 1000
    worker_concurrency: int = 4
    task_acks_late: bool = True
    task_reject_on_worker_lost: bool = True
    result_expires: int = 3600  # 1 hour
    result_compression: str = "gzip"

    def get_optimized_settings(self) -> Dict[str, Any]:
        """Get Celery optimization settings."""
        return {
            "task_serializer": "json",
            "result_serializer": "json",
            "accept_content": ["json"],
            "result_accept_content": ["json"],
            "worker_prefetch_multiplier": 4,
            "worker_max_tasks_per_child": 1000,
            "worker_disable_rate_limits": False,
            "worker_send_task_events": True,
            "result_expires": 3600,
            "result_persistent": True,
            "result_compression": "gzip",
            "task_time_limit": 30 * 60,
            "task_soft_time_limit": 25 * 60,
            "task_acks_late": True,
            "task_reject_on_worker_lost": True,
            "worker_concurrency": 4,
            "broker_connection_retry_on_startup": True,
            "broker_connection_retry": True,
            "worker_send_task_events": True,
            "task_send_sent_event": True,
        }


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
                pool_size=int(os.getenv("DB_POOL_SIZE", "20")),  # Optimized default
                max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "30")),  # Optimized default
                pool_timeout=int(os.getenv("DB_POOL_TIMEOUT", "30")),
                pool_recycle=int(os.getenv("DB_POOL_RECYCLE", "3600")),  # Optimized default
                pool_pre_ping=os.getenv("DB_POOL_PRE_PING", "true").lower() == "true",
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
                soft_time_limit=int(os.getenv("TASK_SOFT_TIME_LIMIT", "1500")),  # Optimized
                time_limit=int(os.getenv("TASK_TIME_LIMIT", "1800")),  # Optimized
                max_retries=int(os.getenv("TASK_MAX_RETRIES", "3")),
                worker_prefetch_multiplier=int(os.getenv("WORKER_PREFETCH_MULTIPLIER", "4")),
                worker_max_tasks_per_child=int(os.getenv("WORKER_MAX_TASKS_PER_CHILD", "1000")),
                worker_concurrency=int(os.getenv("WORKER_CONCURRENCY", "4")),
                task_acks_late=os.getenv("TASK_ACKS_LATE", "true").lower() == "true",
                task_reject_on_worker_lost=os.getenv("TASK_REJECT_ON_WORKER_LOST", "true").lower() == "true",
                result_expires=int(os.getenv("RESULT_EXPIRES", "3600")),
                result_compression=os.getenv("RESULT_COMPRESSION", "gzip")
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


def get_config() -> AppConfig:
    """Get the global application configuration."""
    return config


def get_optimized_database_settings() -> Dict[str, Any]:
    """Get optimized database settings for SQLAlchemy engine."""
    return {
        "pool_size": config.database.pool_size,
        "max_overflow": config.database.max_overflow,
        "pool_timeout": config.database.pool_timeout,
        "pool_recycle": config.database.pool_recycle,
        "pool_pre_ping": config.database.pool_pre_ping,
        "echo": False,
        "future": True,
        "connect_args": config.database.connect_args
    }


def get_optimized_redis_settings() -> Dict[str, Any]:
    """Get optimized Redis settings for connections."""
    return config.redis.get_optimized_settings()


def get_optimized_celery_settings() -> Dict[str, Any]:
    """Get optimized Celery settings for worker configuration."""
    return config.celery.get_optimized_settings()