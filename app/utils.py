# app/utils.py
from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from typing import List, Dict
# Database imports moved to function level to avoid side effects
import numpy as np

from celery import current_task, Task  # noqa: E402 (circular import safe here)

# Import new Redis service, Analysis Lock service, and Http client
from app.infrastructure.redis_service import get_redis_service
from app.services.analysis_lock import get_analysis_lock_service
from app.infrastructure.http_client import get_http_client

# ──────────────────────────────────────────────────────────────────────────────
#  Configuration & Backward Compatibility
# ──────────────────────────────────────────────────────────────────────────────
# Get service instances
_redis_service = get_redis_service()
_analysis_lock_service = get_analysis_lock_service()
_http_client = get_http_client()

# Maintain backward compatibility with direct redis_client usage
redis_client = _redis_service.client

import os
from pathlib import Path

TB_PATH = Path(os.getenv("SYZYGY_PATH", "/data/syzygy"))  # default, cámbialo

# ──────────────────────────────────────────────────────────────────────────────
#  1. Descarga de partidas
# ──────────────────────────────────────────────────────────────────────────────
def fetch_games(username: str, months: int = 12) -> List[Dict]:
    """
    Devuelve una lista de dicts con **pgn** y **move_times** de los últimos
    `months` meses del jugador `username`.

    REFACTOR: Ahora usa HttpClient interno (fase 1D).
    Mantiene 100% backward compatibility.
    """
    logging.info(f"fetch_games: Starting for {username}, months={months}")

    # Use HttpClient for Chess.com API calls
    games = _http_client['fetch_games_from_chesscom'](username, months)

    # Save archive using HttpClient
    _http_client['save_games_archive'](games, username)

    logging.info(f"fetch_games: {username} → {len(games)} partidas")
    return games


# ──────────────────────────────────────────────────────────────────────────────
#  2. Progreso y notificaciones
# ──────────────────────────────────────────────────────────────────────────────
def notify_ws(username: str, payload: dict) -> None:
    """
    Publica JSON en el canal Redis «player:<username>».
    Los listeners (SSE / WebSocket) lo reenvían a los clientes.
    """
    channel = f"player:{username}"
    _redis_service.publish(channel, payload)


def update_progress(username: str, *, increment: int = 1) -> None:
    """Atomic progress update for player analysis."""
    # Import database components locally to avoid side effects at module level
    from app.database import engine
    from app import models
    from sqlmodel import Session, select

    with Session(engine) as s:
        pl = s.exec(
            select(models.Player)
            .where(models.Player.username == username)
            .with_for_update()
        ).one_or_none()
        if not pl:
            return

        pl.done_games = (pl.done_games or 0) + increment
        expected = (pl.total_games or 0) * 2  # básico + detallado
        pl.progress = int(pl.done_games / expected * 100) if expected else 0

        if expected and pl.done_games >= expected:
            pl.status = "ready"
            pl.finished_at = datetime.now(UTC)

        progress_now = pl.progress
        status_now = pl.status

        s.add(pl)
        s.commit()

    notify_ws(username, {"progress": progress_now, "status": status_now})


# ---------------------------------------------------------------
#  3. Task-level progress reporting
# ---------------------------------------------------------------

def task_progress(task: Task | None, current: int, total: int, username: str | None = None) -> None:
    """Report in-flight Celery task progress.

    Parameters
    ----------
    task : celery.Task | None
        The bound task instance (``self``) or ``current_task`` when not bound.
    current : int
        Units completed so far.
    total : int
        Total units to process.
    username : str | None, optional
        If provided, a WebSocket/Redis message will also be sent so that
        clients can receive live updates.
    """
    if total <= 0:
        percent = 0
    else:
        percent = int(current / total * 100)

    try:
        if task is None:
            task = current_task
        if task is not None:
            task.update_state(state="PROGRESS", meta={"current": current, "total": total, "percent": percent})
    except Exception as exc:
        logging.debug(f"task_progress: could not update_state – {exc}")

    if username:
        try:
            notify_ws(username, {
                "type": "task_progress",
                "task_id": task.request.id if task else None,
                "current": current,
                "total": total,
                "percent": percent,
            })
        except Exception as exc:
            logging.debug(f"task_progress: could not notify_ws – {exc}")

@contextmanager
def player_lock(username: str, timeout: int = 900, block: int = 5):
    """
    Lock distribuido de Redis para impedir que dos pods/procesos
    inicien el mismo análisis simultáneamente.

    * `timeout` → segundos tras los cuales el lock expira automáticamente
                  (p.ej. 15 min).
    * `block`   → segundos que un segundo hilo espera antes de abortar con
                  HTTP 423 (Locked).

    Note: This now uses the unified AnalysisLockService internally.
    """
    with _analysis_lock_service.player_lock(username, timeout=timeout, blocking_timeout=block):
        yield


def sa_to_dict(obj, _seen=None):
    """
    Convierte recursivamente un objeto SQLAlchemy en un dict serializable.
    Incluye todos los atributos de columna y todas las relaciones,
    evitando ciclos mediante el conjunto `_seen`.
    """
    # Import inspection locally to avoid side effects at module level
    from sqlalchemy.inspection import inspect

    if _seen is None:
        _seen = set()

    if obj is None or id(obj) in _seen:
        return None

    _seen.add(id(obj))
    mapper = inspect(obj.__class__)

    data = {c.key: getattr(obj, c.key) for c in mapper.column_attrs}

    for rel in mapper.relationships:
        value = getattr(obj, rel.key)
        if value is None:
            data[rel.key] = None
        elif rel.uselist:
            data[rel.key] = [sa_to_dict(i, _seen) for i in value]
        else:
            data[rel.key] = sa_to_dict(value, _seen)

    return data


def pretty_print_sa(obj):
    """
    Imprime en pantalla todo el contenido del objeto SQLAlchemy (y sub‑objetos)
    con formato JSON legible.
    """
    import json
    print(json.dumps(sa_to_dict(obj), indent=2, ensure_ascii=False, default=str))

# BULLDOZER REFACTOR: clean_json_numbers() eliminated
# Replaced with fail-fast validation in app/validation.py
# Reason: Sanitization hides real mathematical problems instead of fixing them

# ──────────────────────────────────────────────────────────────────────────────
#  Task result caching helpers
# ──────────────────────────────────────────────────────────────────────────────

def cache_get(task_name: str, args: list | tuple, kwargs: dict) -> dict | None:
    """Return cached task result or None if missing/invalid."""
    return _redis_service.cache_get(task_name, args, kwargs)


def cache_set(task_name: str, args: list | tuple, kwargs: dict, result: dict, ttl: int = 86_400) -> None:
    """Store task result in Redis with TTL (default 24h)."""
    _redis_service.cache_set(task_name, args, kwargs, result, ttl)
