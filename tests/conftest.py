"""
Configuración global de pytest para el proyecto.
"""
import os
import tempfile
from typing import Generator
import pytest
from sqlmodel import SQLModel, Session, create_engine
from fastapi.testclient import TestClient

# Configurar variables de entorno para tests antes de importar app
os.environ["DATABASE_URL"] = "sqlite:///test.db"
os.environ["REDIS_URL"] = "redis://localhost:6379/1"
os.environ["STOCKFISH_PATH"] = "/usr/games/stockfish"
os.environ["STOCKFISH_DEPTH"] = "1"
os.environ["ENABLE_TRACING"] = "false"

from app.main import app
from app.database import engine


@pytest.fixture(name="session")
def session_fixture() -> Generator[Session, None, None]:
    """
    Fixture para sesión de base de datos en tests.
    Crea tablas al inicio y las limpia al final.
    """
    # Crear todas las tablas
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        yield session

    # Limpiar después del test
    SQLModel.metadata.drop_all(engine)


@pytest.fixture(name="client")
def client_fixture() -> TestClient:
    """
    Fixture para cliente de testing de FastAPI.
    """
    return TestClient(app)


@pytest.fixture
def sample_pgn() -> str:
    """
    PGN de muestra para tests.
    """
    return '''[Event "Test Game"]
[Site "Test"]
[Date "2023.01.01"]
[Round "1"]
[White "TestPlayer1"]
[Black "TestPlayer2"]
[Result "1-0"]
[WhiteElo "1500"]
[BlackElo "1500"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 1-0'''


@pytest.fixture
def sample_moves_data() -> list:
    """
    Datos de muestra para análisis de movimientos.
    """
    return [
        {
            "move_number": 1,
            "played": "e4",
            "best": "e4",
            "cp_loss": 0,
            "eval_before": 15,
            "eval_after": 15
        },
        {
            "move_number": 2,
            "played": "Nf6",
            "best": "Nc6",
            "cp_loss": 25,
            "eval_before": -15,
            "eval_after": 10
        }
    ]


@pytest.fixture
def mock_stockfish_analysis():
    """
    Mock para análisis de Stockfish.
    """
    return {
        "best_move": "e4",
        "evaluation": 15,
        "mate": None,
        "depth": 12
    }