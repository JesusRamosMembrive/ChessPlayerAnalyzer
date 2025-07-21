import os
import sys
from pathlib import Path

import pytest
from sqlmodel import SQLModel

# Asegurar que `app` se puede importar durante los tests
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost/testdb")
os.environ.setdefault("STOCKFISH_PATH", "/usr/games/stockfish")
os.environ.setdefault("STOCKFISH_DEPTH", "1")

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
