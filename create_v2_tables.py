#!/usr/bin/env python3
"""
Script para crear las tablas V2 del refactor.
"""
import sys
import os
sys.path.append('/app')

def create_v2_tables():
    try:
        print("🔄 Importing dependencies...")
        from app.database import init_db
        from app.models_v2 import Game, AnalysisResult, Player, ReferenceStats
        from sqlmodel import SQLModel, create_engine

        # Obtener URL de base de datos
        db_url = os.getenv('DATABASE_URL')
        if not db_url:
            raise ValueError("DATABASE_URL environment variable not set")

        print(f"🔗 Connecting to database: {db_url}")

        # Inicializar base de datos V1 existente
        print("🔄 Initializing V1 database...")
        init_db()

        # Crear tablas V2
        print("🔄 Creating V2 tables...")
        engine = create_engine(db_url)
        SQLModel.metadata.create_all(engine)

        print("✅ All V1 and V2 tables created successfully!")
        return True

    except Exception as e:
        print(f"❌ Error creating V2 tables: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = create_v2_tables()
    sys.exit(0 if success else 1)