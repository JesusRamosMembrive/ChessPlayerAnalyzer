from datetime import datetime, timezone
from fastapi import APIRouter, Depends
from sqlmodel import Session
import redis
from celery import Celery
import os
import time
try:
    import psutil
except ImportError:
    psutil = None

from app.database import get_session
from app.schemas import HealthOut
from app.apm import apm_config, add_breadcrumb, set_context
from app.utils import redis_client
from app.celery_app import celery_app

router = APIRouter()

@router.get(
    "/health",
    response_model=HealthOut,
    summary="Comprobación de salud del servicio",
    description="Devuelve información de estado y conexión a servicios subyacentes.",
)
async def health_check(session: Session = Depends(get_session)):
    """Health check endpoint for the API."""
    start_time = time.time()
    
    # APM: Agregar breadcrumb para health check
    add_breadcrumb(
        message="Health check iniciado",
        category="health",
        level="info"
    )
    
    # Test database connection
    db_status = "unknown"
    db_response_time = None
    try:
        db_start = time.time()
        session.execute("SELECT 1")
        db_response_time = (time.time() - db_start) * 1000  # ms
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    # Test Redis connection
    redis_status = "unknown"
    redis_response_time = None
    try:
        redis_start = time.time()
        redis_client.ping()
        redis_response_time = (time.time() - redis_start) * 1000  # ms
        redis_status = "connected"
    except Exception as e:
        redis_status = f"error: {str(e)}"
    
    # Test Celery connection
    celery_status = "unknown"
    active_workers = 0
    try:
        # Obtener información de workers activos
        inspect = celery_app.control.inspect()
        stats = inspect.stats()
        if stats:
            active_workers = len(stats)
            celery_status = "connected"
        else:
            celery_status = "no workers"
    except Exception as e:
        celery_status = f"error: {str(e)}"
    
    # Obtener métricas del sistema
    system_metrics = {}
    if psutil:
        try:
            system_metrics = {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
            }
        except:
            pass
    
    # Determinar estado general
    overall_status = "healthy"
    if db_status != "connected" or redis_status != "connected":
        overall_status = "degraded"
    if db_status.startswith("error") and redis_status.startswith("error"):
        overall_status = "unhealthy"
    
    # APM: Establecer contexto con métricas
    set_context("health_check", {
        "database": {
            "status": db_status,
            "response_time_ms": db_response_time
        },
        "redis": {
            "status": redis_status,
            "response_time_ms": redis_response_time
        },
        "celery": {
            "status": celery_status,
            "active_workers": active_workers
        },
        "system": system_metrics
    })
    
    total_time = (time.time() - start_time) * 1000  # ms
    
    return {
        "status": overall_status,
        "version": "v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database": db_status,
        "services": {
            "database": db_status == "connected",
            "celery": celery_status == "connected",
            "redis": redis_status == "connected"
        },
        "metrics": {
            "response_times_ms": {
                "total": round(total_time, 2),
                "database": round(db_response_time, 2) if db_response_time else None,
                "redis": round(redis_response_time, 2) if redis_response_time else None
            },
            "celery": {
                "active_workers": active_workers
            },
            "system": system_metrics,
            "apm": {
                "enabled": apm_config.enabled,
                "environment": apm_config.environment,
                "traces_sample_rate": apm_config.traces_sample_rate
            }
        }
    }
