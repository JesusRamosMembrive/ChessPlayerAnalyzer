# -----------------------------------------------------------------------------
# File: backend/app/database.py
# -----------------------------------------------------------------------------
"""Conexión global a PostgreSQL y helper para obtener sesiones."""
import os
from sqlmodel import SQLModel, create_engine, Session

DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://chess:chess@postgres:5432/chessdb",
)

engine = create_engine(DB_URL, echo=False, pool_pre_ping=True)


def get_session():
    """Dependencia FastAPI que abre y cierra la sesión por petición."""
    with Session(engine) as session:
        yield session