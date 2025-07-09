from datetime import datetime, timezone
from fastapi import APIRouter, Depends
from sqlmodel import Session
import redis
from celery import Celery
import os
import time
import psutil

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
    # Test database connection
    try:
        session.execute("SELECT 1")
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "version": "v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database": db_status,
        "services": {
            "database": db_status == "connected",
            "celery": True,  # Would need actual Celery health check
            "redis": True    # Would need actual Redis health check
        }
    }
