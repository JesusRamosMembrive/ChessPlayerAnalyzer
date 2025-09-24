#!/bin/bash
# setup_dev.sh - Development environment setup script

set -e

echo "🚀 Chess Player Analyzer - Development Setup"
echo "============================================"

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Warning: You're not in a virtual environment"
    echo "   It's recommended to create one:"
    echo "   python3 -m venv .venv"
    echo "   source .venv/bin/activate"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "📦 Installing dependencies..."

# Install production dependencies first
echo "  → Installing production dependencies..."
pip install -r requirements.txt

# Install development dependencies
echo "  → Installing development dependencies..."
pip install -r requirements-dev.txt

echo ""
echo "🧪 Testing installation..."

# Test that key modules can be imported
python3 -c "
import fastapi
import redis
import pytest
print('✅ Core dependencies working')
"

# Test our refactor modules
python3 -c "
from app.infrastructure.redis_service import RedisService
from app.factories import create_app, create_worker
print('✅ Refactor modules working')
"

echo ""
echo "🎉 Development environment ready!"
echo ""
echo "Next steps:"
echo "  1. Run tests: python3 tests/refactor/run_refactor_tests.py"
echo "  2. Start API: uvicorn app.main:app --reload"
echo "  3. Start worker: celery -A app.celery_tasks:celery_app worker --loglevel=info"
echo ""
echo "Available commands:"
echo "  pytest tests/refactor/                    # Run refactor tests"
echo "  python3 tests/refactor/run_refactor_tests.py  # Run refactor test suite"
echo "  black app/ tests/                        # Format code"
echo "  flake8 app/ tests/                       # Lint code"
echo ""