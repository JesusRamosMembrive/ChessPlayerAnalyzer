#!/usr/bin/env python3
"""
BULLDOZER TOTAL: Create ultra-simplified database schema.

Creates the two simple tables needed for BULLDOZER architecture:
1. game_analysis_bulldozer - Everything in one table
2. player_progress_bulldozer - Simple progress tracking

Run this script to set up the BULLDOZER database.
"""
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from sqlmodel import SQLModel, create_engine
from app.models import GameAnalysis, PlayerProgress
from app.database import get_database_url

def create_bulldozer_tables():
    """Create BULLDOZER tables in the database."""
    print("🏗️ BULLDOZER TOTAL: Creating ultra-simplified database schema...")

    # Get database URL
    database_url = get_database_url()
    print(f"📍 Connecting to: {database_url}")

    # Create engine
    engine = create_engine(database_url, echo=True)

    try:
        print("🔨 Creating BULLDOZER tables...")

        # This will create both tables defined in models.py
        SQLModel.metadata.create_all(engine)

        print("✅ BULLDOZER tables created successfully!")
        print("")
        print("📋 Created tables:")
        print("   - game_analysis_bulldozer (single table for all analysis)")
        print("   - player_progress_bulldozer (simple progress tracking)")
        print("")
        print("🚀 BULLDOZER database ready!")

        return True

    except Exception as e:
        print(f"❌ Failed to create BULLDOZER tables: {e}")
        return False

def verify_tables():
    """Verify that BULLDOZER tables were created correctly."""
    try:
        from sqlmodel import Session, select

        database_url = get_database_url()
        engine = create_engine(database_url)

        with Session(engine) as session:
            # Test basic table access
            stmt = select(GameAnalysis).limit(1)
            session.exec(stmt).all()

            stmt = select(PlayerProgress).limit(1)
            session.exec(stmt).all()

        print("✅ BULLDOZER tables verification passed!")
        return True

    except Exception as e:
        print(f"❌ BULLDOZER tables verification failed: {e}")
        return False

if __name__ == "__main__":
    print("🧱 BULLDOZER TOTAL Database Setup")
    print("=" * 50)

    # Create tables
    success = create_bulldozer_tables()

    if success:
        print("🔍 Verifying tables...")
        verify_success = verify_tables()

        if verify_success:
            print("")
            print("🎉 BULLDOZER TOTAL setup complete!")
            print("💥 Ultra-simplified architecture ready!")
            print("")
            print("Next steps:")
            print("1. Run FASE 2B tests")
            print("2. Update API endpoints")
            print("3. Test end-to-end analysis")

            sys.exit(0)
        else:
            print("⚠️ Table verification failed")
            sys.exit(1)
    else:
        print("💥 BULLDOZER setup failed")
        sys.exit(1)