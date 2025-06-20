from sqlmodel import SQLModel, Session
from app.database import engine
from app import models
from sqlalchemy import text, inspect
import logging

logger = logging.getLogger(__name__)

logger.info("Creando tablas...")
SQLModel.metadata.create_all(engine)
logger.info("✅ Tablas creadas exitosamente")



def ensure_indices() -> None:
    """
    Crea los índices sólo si la tabla existe.
    De este modo evitamos UndefinedTable la primera vez que se levanta el stack.
    """
    insp = inspect(engine)
    ddl = []

    if insp.has_table("game"):
        ddl += [
            "CREATE INDEX IF NOT EXISTS ix_game_white_username  ON game (white_username)",
            "CREATE INDEX IF NOT EXISTS ix_game_black_username  ON game (black_username)",
        ]

    if insp.has_table("move_analysis"):
        ddl.append(
            "CREATE INDEX IF NOT EXISTS ix_moveanalysis_game_id ON move_analysis (game_id)"
        )

    # Ejecutamos cada sentencia fuera de transacción para compat. Postgres
    with engine.begin() as conn:          # autocommit=true
        for stmt in ddl:
            conn.exec_driver_sql(stmt)
ensure_indices()