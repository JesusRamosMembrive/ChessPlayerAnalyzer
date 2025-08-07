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

from app.celery_app import celery_app
celery_app.conf.update(task_always_eager=True, task_eager_propagates=True)

from app.main import app
from fastapi.testclient import TestClient

@pytest.fixture(scope="session", autouse=True)
def setup_db():
    SQLModel.metadata.create_all(engine)
    yield
    SQLModel.metadata.drop_all(engine)

@pytest.fixture(scope="session")
def client():
    return TestClient(app)
