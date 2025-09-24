#!/usr/bin/env python3
"""
Simple dependency installer script.
Alternative to setup_dev.sh for cross-platform compatibility.
"""
import sys
import subprocess
import os

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"  → {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"    ❌ Error: {e}")
        if e.stdout:
            print(f"    stdout: {e.stdout}")
        if e.stderr:
            print(f"    stderr: {e.stderr}")
        return False

def main():
    print("🚀 Chess Player Analyzer - Development Setup")
    print("=" * 50)

    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return 1

    # Check if in virtual environment
    if not os.environ.get('VIRTUAL_ENV'):
        print("⚠️  Warning: Not in a virtual environment")
        print("   Recommended: python3 -m venv .venv && source .venv/bin/activate")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return 1

    print("\n📦 Installing dependencies...")

    # Install production dependencies
    if not run_command("pip install -r requirements.txt", "Installing production dependencies"):
        return 1

    # Install development dependencies
    if not run_command("pip install -r requirements-dev.txt", "Installing development dependencies"):
        return 1

    print("\n🧪 Testing installation...")

    # Test core imports
    try:
        import fastapi
        import redis
        import pytest
        print("  ✅ Core dependencies working")
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return 1

    # Test refactor modules (skip database-dependent modules for now)
    try:
        from app.infrastructure.redis_service import RedisService
        print("  ✅ RedisService import working")

        # Skip factories for now due to database connection issues
        # from app.factories import create_app, create_worker
        print("  ⚠️  Factories skipped (database connection required)")
        print("  ℹ️  This is expected - factories will work once database is available")
    except ImportError as e:
        print(f"  ❌ Refactor import failed: {e}")
        return 1

    print("\n🎉 Development environment ready!")
    print("\nNext steps:")
    print("  python3 tests/refactor/run_refactor_tests.py  # Run refactor tests")
    print("  uvicorn app.main:app --reload                # Start API")
    print("  celery -A app.celery_tasks:celery_app worker --loglevel=info  # Start worker")

    return 0

if __name__ == "__main__":
    sys.exit(main())