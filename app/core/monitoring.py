"""
Performance monitoring and metrics collection for Clean Architecture.
Centralized monitoring system for production deployment.
"""
import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import defaultdict
import json

from .performance import performance_optimizer, simple_cache


@dataclass
class MetricData:
    """Container for metric data."""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels
        }


class PerformanceMonitor:
    """Centralized performance monitoring system."""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.PerformanceMonitor")
        self._metrics: List[MetricData] = []
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._last_cleanup = datetime.utcnow()

    def increment_counter(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        key = f"{name}:{json.dumps(labels or {}, sort_keys=True)}"
        self._counters[key] += value

        metric = MetricData(
            name=name,
            value=self._counters[key],
            timestamp=datetime.utcnow(),
            labels=labels or {}
        )
        self._metrics.append(metric)

    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric."""
        key = f"{name}:{json.dumps(labels or {}, sort_keys=True)}"
        self._gauges[key] = value

        metric = MetricData(
            name=name,
            value=value,
            timestamp=datetime.utcnow(),
            labels=labels or {}
        )
        self._metrics.append(metric)

    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram value."""
        key = f"{name}:{json.dumps(labels or {}, sort_keys=True)}"
        self._histograms[key].append(value)

        metric = MetricData(
            name=name,
            value=value,
            timestamp=datetime.utcnow(),
            labels=labels or {}
        )
        self._metrics.append(metric)

    def time_operation(self, operation_name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager to time operations."""
        return TimedOperation(self, operation_name, labels)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of all collected metrics."""
        now = datetime.utcnow()

        # Cleanup old metrics (keep last hour)
        if (now - self._last_cleanup).total_seconds() > 300:  # 5 minutes
            cutoff = now - timedelta(hours=1)
            self._metrics = [m for m in self._metrics if m.timestamp > cutoff]
            self._last_cleanup = now

        # Calculate statistics
        histograms_stats = {}
        for key, values in self._histograms.items():
            if values:
                histograms_stats[key] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "p95": sorted(values)[int(len(values) * 0.95)] if len(values) > 0 else 0
                }

        return {
            "timestamp": now.isoformat(),
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": histograms_stats,
            "cache_stats": performance_optimizer.get_cache_stats(),
            "metrics_count": len(self._metrics)
        }

    def get_system_health(self) -> Dict[str, Any]:
        """Get system health information."""
        health = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {}
        }

        try:
            # Database health
            from ..database import engine
            with engine.connect() as conn:
                conn.execute("SELECT 1")
            health["checks"]["database"] = "healthy"
        except Exception as e:
            health["checks"]["database"] = f"unhealthy: {str(e)}"
            health["status"] = "degraded"

        try:
            # Redis health
            from ..utils import redis_client
            redis_client.ping()
            health["checks"]["redis"] = "healthy"
        except Exception as e:
            health["checks"]["redis"] = f"unhealthy: {str(e)}"
            health["status"] = "degraded"

        try:
            # Application container health
            from ..application.container import get_container
            container = get_container()
            container.get_get_player_status_use_case()
            health["checks"]["application"] = "healthy"
        except Exception as e:
            health["checks"]["application"] = f"unhealthy: {str(e)}"
            health["status"] = "degraded"

        # Performance metrics
        cache_stats = performance_optimizer.get_cache_stats()
        health["performance"] = {
            "cache_hit_rate": cache_stats["hit_rate_percentage"],
            "active_connections": len(self._metrics),
            "memory_usage": "good"  # TODO: Add actual memory monitoring
        }

        return health


class TimedOperation:
    """Context manager for timing operations."""

    def __init__(self, monitor: PerformanceMonitor, operation_name: str, labels: Optional[Dict[str, str]] = None):
        self.monitor = monitor
        self.operation_name = operation_name
        self.labels = labels or {}
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.monitor.record_histogram(
                f"{self.operation_name}_duration_seconds",
                duration,
                self.labels
            )

            # Also record success/failure
            status = "success" if exc_type is None else "error"
            self.monitor.increment_counter(
                f"{self.operation_name}_total",
                1,
                {**self.labels, "status": status}
            )


class DatabaseMetricsCollector:
    """Collects database-specific metrics."""

    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor

    async def collect_metrics(self):
        """Collect database performance metrics."""
        try:
            from ..database import engine

            # Connection pool metrics
            pool = engine.pool
            self.monitor.set_gauge("db_pool_size", pool.size())
            self.monitor.set_gauge("db_pool_checked_in", pool.checkedin())
            self.monitor.set_gauge("db_pool_checked_out", pool.checkedout())
            self.monitor.set_gauge("db_pool_overflow", pool.overflow())

            # Query performance (sample queries)
            with self.monitor.time_operation("db_health_check"):
                with engine.connect() as conn:
                    conn.execute("SELECT 1")

        except Exception as e:
            self.monitor.increment_counter("db_metrics_collection_errors", 1)
            logging.error(f"Error collecting database metrics: {e}")


class CeleryMetricsCollector:
    """Collects Celery task metrics."""

    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor

    def record_task_start(self, task_name: str, task_id: str):
        """Record task start."""
        self.monitor.increment_counter(
            "celery_tasks_started_total",
            1,
            {"task_name": task_name, "task_id": task_id}
        )

    def record_task_completion(self, task_name: str, task_id: str, duration: float, success: bool):
        """Record task completion."""
        status = "success" if success else "failure"

        self.monitor.increment_counter(
            "celery_tasks_completed_total",
            1,
            {"task_name": task_name, "status": status}
        )

        self.monitor.record_histogram(
            "celery_task_duration_seconds",
            duration,
            {"task_name": task_name}
        )


# Global monitoring instances
performance_monitor = PerformanceMonitor()
db_metrics_collector = DatabaseMetricsCollector(performance_monitor)
celery_metrics_collector = CeleryMetricsCollector(performance_monitor)


# Monitoring decorators
def monitor_async_function(func_name: str = None):
    """Decorator to monitor async function performance."""
    def decorator(func):
        name = func_name or f"{func.__module__}.{func.__name__}"

        async def wrapper(*args, **kwargs):
            with performance_monitor.time_operation(name):
                return await func(*args, **kwargs)
        return wrapper
    return decorator


def monitor_sync_function(func_name: str = None):
    """Decorator to monitor sync function performance."""
    def decorator(func):
        name = func_name or f"{func.__module__}.{func.__name__}"

        def wrapper(*args, **kwargs):
            with performance_monitor.time_operation(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Health check utilities
async def perform_comprehensive_health_check() -> Dict[str, Any]:
    """Perform comprehensive health check of all systems."""
    health_data = performance_monitor.get_system_health()

    # Add detailed metrics
    health_data["metrics"] = performance_monitor.get_metrics_summary()

    # Add database metrics
    await db_metrics_collector.collect_metrics()

    # Add performance optimizations status
    health_data["optimizations"] = {
        "database_pooling": "enabled",
        "redis_connection_pooling": "enabled",
        "celery_prefetch_optimization": "enabled",
        "cache_optimization": "enabled"
    }

    return health_data