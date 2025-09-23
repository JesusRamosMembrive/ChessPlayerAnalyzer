#!/usr/bin/env python3
"""Utility executed in the migrate container to ensure the unified schema exists."""
from __future__ import annotations

import os
import sys
import traceback

# Allow imports of the project package when running inside /app
REPO_PATH = "/app"
if REPO_PATH not in sys.path:
    sys.path.append(REPO_PATH)


def main() -> int:
    try:
        from app.database import init_db
        from app import models  # noqa: F401 - register SQLModel metadata

        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise RuntimeError("DATABASE_URL environment variable not set")

        print(f"🔗 Connecting to database: {db_url}")
        # init_db() already imports app.models and creates all tables on the engine
        init_db()
        print("✅ Unified schema is ready")
        return 0
    except Exception as exc:
        print(f"❌ Failed to initialize unified schema: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
