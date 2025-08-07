import os
import sys
from pathlib import Path

import pytest
from sqlmodel import SQLModel

# Asegurar que `app` se puede importar durante los tests
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Base de datos local en fichero (sin necesidad de Postgres)
os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")

# Usar Stockfish del PATH por defecto; puedes sobreescribir esta ruta si lo necesitas
os.environ.setdefault("STOCKFISH_PATH", "stockfish")
os.environ.setdefault("STOCKFISH_DEPTH", "1")

# Redis en memoria para tests (evita depender de un servidor Redis)
try:
    import fakeredis  # type: ignore
    import app.utils as _utils
    _utils.redis_client = fakeredis.FakeRedis(decode_responses=True)
except Exception:
    # Si fakeredis no está disponible, se intentará usar REDIS_URL real
    pass

from app.database import engine

# Si usamos SQLite, re‑crear el engine con check_same_thread=False
try:
    import app.database as _db
    if str(_db.DB_URL).startswith("sqlite"):
        _db.engine.dispose()
        _db.engine = _db._create_engine_retry(
            _db.DB_URL,
            echo=False,
            pool_pre_ping=True,
            connect_args={"check_same_thread": False},
        )
        # Reconfigurar el sessionmaker para que use el nuevo engine
        _db.SessionLocal.configure(bind=_db.engine)
except Exception:
    pass

from app.celery_app import celery_app
celery_app.conf.update(task_always_eager=True, task_eager_propagates=True)

from app.main import app
from fastapi.testclient import TestClient

# Fuente de datos local para evitar red en tests de jugador
LOCAL_GAMES_JSON = os.getenv("LOCAL_GAMES_JSON")
if LOCAL_GAMES_JSON:
    import json
    import app.utils as _utils

    def _fetch_games_local(username: str, months: int = 12):  # pragma: no cover
        with open(LOCAL_GAMES_JSON, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        # Se espera una lista de dicts con al menos 'pgn' y 'move_times'
        return data

    _utils.fetch_games = _fetch_games_local

@pytest.fixture(scope="session", autouse=True)
def setup_db():
    SQLModel.metadata.create_all(engine)
    yield
    SQLModel.metadata.drop_all(engine)

@pytest.fixture(scope="session")
def client():
    return TestClient(app)
