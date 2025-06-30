# app/utils.py
from __future__ import annotations

import json
import logging
import math
import pathlib
import re
from contextlib import contextmanager
from datetime import datetime, UTC
from typing import List, Dict

import redis
import requests
from sqlalchemy.inspection import inspect
from app import models
from app.database import engine
from sqlmodel import Session, select
import numpy as np

import os
from pathlib import Path
# ──────────────────────────────────────────────────────────────────────────────
#  Configuración común
# ──────────────────────────────────────────────────────────────────────────────
REDIS_URL = "redis://redis:6379/0"
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

CLK_RGX = re.compile(r"\[%clk\s+([\d:.]+)]")
UA = "chess-analyzer/0.2 (+https://github.com/tu_usuario)"
TB_PATH = Path(os.getenv("SYZYGY_PATH", "/data/syzygy"))  # default, cámbialo

# ──────────────────────────────────────────────────────────────────────────────
#  1. Descarga de partidas
# ──────────────────────────────────────────────────────────────────────────────
def _sec(t: str) -> int:
    """Convierte «hh:mm:ss.f» o «m:ss.f»  → segundos (int)."""
    parts = list(map(float, t.split(":")))
    parts = [0] * (3 - len(parts)) + parts
    if len(parts) == 3:
        h, m, s = parts
    else:
        h, m, s = 0, *parts
    return int(h * 3600 + m * 60 + s)


def fetch_games(username: str, months: int = 6) -> List[Dict]:
    """
    Devuelve una lista de dicts con **pgn** y **move_times** de los últimos
    `months` meses del jugador `username`.
    """
    logging.info(f"fetch_games: Starting for {username}, months={months}")

    s = requests.Session()
    s.headers["User-Agent"] = UA

    arch_url = f"https://api.chess.com/pub/player/{username}/games/archives"
    logging.info(f"fetch_games: Fetching archives from {arch_url}")

    try:
        resp = s.get(arch_url, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error(f"fetch_games: Error fetching archives: {e}")
        raise

    archives = resp.json()["archives"][-months:]  # los más recientes
    logging.info(f"fetch_games: Found {len(archives)} archives to process")

    games: list[dict] = []

    for idx, url in enumerate(archives):
        logging.info(f"fetch_games: Processing archive {idx + 1}/{len(archives)}: {url}")
        try:
            data = s.get(url, timeout=10).json()
            logging.info(f"fetch_games: Found {len(data.get('games', []))} games in archive")

            for g in data["games"]:
                pgn = g["pgn"]
                clocks = CLK_RGX.findall(pgn)
                move_times = [
                    _sec(clocks[i - 1]) - _sec(clocks[i])
                    for i in range(1, len(clocks))
                ] if clocks else []

                games.append({
                    "pgn": pgn,
                    "move_times": move_times,
                    "white": g["white"]["username"],
                    "black": g["black"]["username"],
                    "end_time": datetime.fromtimestamp(g["end_time"], UTC).isoformat(),
                })
        except Exception as e:
            logging.error(f"fetch_games: Error processing archive {url}: {e}")
            continue

    # ---------------------------------------------------------------
    #  Guardar copia local de las partidas descargadas
    # ---------------------------------------------------------------
    try:
        archive_dir = pathlib.Path(
            os.getenv("FETCH_ARCHIVE_DIR", "archives")
        ).expanduser().resolve()  # ← ABSOLUTA

        archive_dir.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = archive_dir / f"{username}_{stamp}.json"

        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(games, fh, ensure_ascii=False, indent=2)

        logging.info(f"fetch_games: Saved {len(games)} games to {out_path}")
    except Exception as exc:
        # No queremos que un fallo de disco interrumpa el análisis
        logging.warning(f"fetch_games: could not archive games → {exc}")

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
    redis_client.publish(channel, json.dumps(payload))


def update_progress(username: str, *, increment: int = 1) -> None:
    """Atomic progress update for player analysis."""
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

@contextmanager
def player_lock(username: str, timeout: int = 900, block: int = 5):
    """
    Lock distribuido de Redis para impedir que dos pods/procesos
    inicien el mismo análisis simultáneamente.

    * `timeout` → segundos tras los cuales el lock expira automáticamente
                  (p.ej. 15 min).
    * `block`   → segundos que un segundo hilo espera antes de abortar con
                  HTTP 423 (Locked).
    """
    lock = redis_client.lock(f"lock:player:{username}", timeout=timeout)
    if not lock.acquire(blocking=True, blocking_timeout=block):
        raise RuntimeError(f"player {username!r} is already locked")
    try:
        yield
    finally:
        # Si el código dentro del with explota no dejamos el lock colgado
        lock.release()


def sa_to_dict(obj, _seen=None):
    """
    Convierte recursivamente un objeto SQLAlchemy en un dict serializable.
    Incluye todos los atributos de columna y todas las relaciones,
    evitando ciclos mediante el conjunto `_seen`.
    """
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
    print(json.dumps(sa_to_dict(obj), indent=2, ensure_ascii=False, default=str))

def clean_json_numbers(obj):
    """
    Reemplaza NaN/inf por None y convierte numpy.* a tipos Python nativos
    para que psycopg pueda serializar a JSON.
    """
    if isinstance(obj, dict):
        return {k: clean_json_numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_json_numbers(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        obj = obj.item()           # np.float64 → float, np.int64 → int
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj