"""
Shared configuration and utilities for all Celery tasks.
This module contains common imports, constants, and helper functions used across different task modules.
"""
from __future__ import annotations

import io
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import chess.engine
import chess.pgn

# Configurar logging estructurado JSON
from app.logging_config import setup_logging
setup_logging()

from app.analysis.engine import ChessAnalysisEngine
from app.database import engine

# Application Performance Monitoring (APM)
from app.otel import init_otel

logger = logging.getLogger(__name__)

# Inicializar OpenTelemetry (solo una vez en worker)
init_otel()

from app.utils import (
    fetch_games,
    notify_ws,
    update_progress,
    task_progress,
    sa_to_dict,
    redis_client,
    cache_get,
    cache_set,
)
from celery import Celery
from celery import chain, group, chord
from celery import current_task
from celery.signals import task_failure, task_revoked
from sqlmodel import Session, select
from sqlalchemy import func
from sqlalchemy.orm import selectinload

from app.models import GameAnalysisDetailed
from app import models
from app.analysis.engine import prepare_moves_dataframe
from app.utils import TB_PATH

from app.analysis import (
    aggregate_quality_features as q_feats,
    aggregate_time_features    as t_feats,
    aggregate_opening_features as o_feats,
    aggregate_endgame_features as e_feats,
)
from app.analysis.bayesian import BayesianSuspicionModel
from app.analysis.clustering import recompute_and_update_clusters

from kombu import Queue  # Añadido para configurar colas con prioridad
from celery.exceptions import SoftTimeLimitExceeded
from celery.schedules import crontab

# Configuración de prioridades (0 = más alta)
HIGH_PRIORITY   = 0
DEFAULT_PRIORITY = 5
LOW_PRIORITY    = 9

# Configuración del motor
ENGINE_PATH = os.getenv("STOCKFISH_PATH", "stockfish")
MAX_DEPTH = int(os.getenv("STOCKFISH_DEPTH", "12"))

# ──────────────────────────────────────────────────────────────
#  Task timeout & retry configuration (env-driven)
# ──────────────────────────────────────────────────────────────
TASK_SOFT_TIME_LIMIT = int(os.getenv("TASK_SOFT_TIME_LIMIT", "1800"))  # 30 min default
TASK_TIME_LIMIT      = int(os.getenv("TASK_TIME_LIMIT", "1860"))      # hard limit (soft + 1 min)
TASK_MAX_RETRIES     = int(os.getenv("TASK_MAX_RETRIES", "3"))         # default max retries

# Inicializar el motor de análisis (configurar rutas según tu sistema)
analysis_engine = ChessAnalysisEngine(
    reference_book_path=Path("/data/reference_book.bin") if Path("/data/reference_book.bin").exists() else None,
    tablebase_path=Path("/data/syzygy") if Path("/data/syzygy").exists() else None,
)

engine_helper = ChessAnalysisEngine()

def safe(v):
    """
    Helper function to safely convert values to float.
    v puede ser None o np.nan; devuélvelo como 0.0 si no es numérico
    """
    try:
        return float(v) if v == v else 0.0      # np.nan != np.nan
    except (TypeError, ValueError):
        return 0.0

def export_analysis_to_json(data_obj, username: str, analysis_type: str = "analysis"):
    """
    Export analysis results to JSON file in debug_results directory.

    Args:
        data_obj: SQLAlchemy object to export (GameAnalysisDetailed or PlayerAnalysisDetailed)
        username: Player username for filename
        analysis_type: Type of analysis for logging ("game" or "player")
    """
    try:
        debug_dir = Path("debug_results")
        debug_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(datetime.now(timezone.utc).timestamp())

        filename = f"{username}_{timestamp}.json"
        filepath = debug_dir / filename

        data_dict = sa_to_dict(data_obj)

        with filepath.open("w", encoding="utf-8") as f:
            json.dump(data_dict, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"DEBUG EXPORT: Saved {analysis_type} analysis to {filepath}")
        return str(filepath)

    except Exception as e:
        logger.error(f"DEBUG EXPORT: Failed to export {analysis_type} analysis for {username}: {e}")
        return None

def is_task_aborted(task, player_username=None):
    """
    Helper para comprobar revocación de forma segura.
    Devuelve True si el worker indica que la tarea fue revocada.
    Usa reflection para ser compatible con versiones de Celery donde
    no existen is_aborted / revoked en request.
    También verifica Redis para cancellación inmediata.
    """
    try:
        # Check Redis cancellation flag first (fastest method)
        if player_username:
            cancellation_key = f"cancel:{player_username}"
            if redis_client.get(cancellation_key):
                logger.info(f"Task {task.request.id} detected Redis cancellation flag for {player_username}")
                return True

        # Celery >=5.3 expone Task.is_aborted()
        if hasattr(task, "is_aborted"):
            return task.is_aborted()
        # Celery <5.3: intentar consultarlo en request (no siempre disponible)
        if hasattr(task.request, "is_aborted"):
            return task.request.is_aborted()
        if getattr(task.request, "stopped", False):
            return True
    except Exception:
        pass
    return False