import sys
import logging
from sqlmodel import SQLModel
from app.database import engine
from sqlalchemy import inspect

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("=== MIGRATE CONTAINER STARTING ===")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Python path: {sys.path}")
        
        logger.info("Testing database connection...")
        from sqlalchemy import text
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()
            logger.info(f"Database connection successful: {version}")
        
        logger.info("Importing database models...")
        from app.models import (
            Game, MoveAnalysis, GameAnalysisDetailed, 
            Player, PlayerAnalysisDetailed, ReferenceStats
        )
        logger.info("All models imported successfully")
        
        logger.info("Creating database tables...")
        SQLModel.metadata.create_all(engine)
        logger.info("✅ Tables created successfully")
        
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        logger.info(f"Tables in database: {tables}")
        
        if not tables:
            raise Exception("No tables were created despite successful metadata.create_all()")
        
        logger.info("Creating database indices...")
        ensure_indices()
        logger.info("✅ Indices created successfully")
        
        logger.info("=== MIGRATE CONTAINER COMPLETED SUCCESSFULLY ===")
        
    except Exception as e:
        logger.error(f"❌ MIGRATE CONTAINER FAILED: {e}")
        import traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        sys.exit(1)

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

    with engine.begin() as conn:
        for stmt in ddl:
            conn.exec_driver_sql(stmt)

if __name__ == "__main__":
    main()
