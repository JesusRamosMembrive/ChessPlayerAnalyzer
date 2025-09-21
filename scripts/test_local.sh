#!/bin/bash
# Script para ejecutar tests localmente sin Docker

set -e

echo "🧪 Ejecutando tests locales..."

# Configurar variables de entorno para tests
export DATABASE_URL="sqlite:///test.db"
export REDIS_URL="redis://localhost:6379/1"
export STOCKFISH_PATH="/usr/games/stockfish"
export STOCKFISH_DEPTH="1"
export ENABLE_TRACING="false"

# Limpiar base de datos de test anterior
rm -f test.db

echo "📦 Verificando dependencias..."
pip install -r requirements.txt -r requirements-dev.txt > /dev/null

echo "🏃‍♂️ Ejecutando tests unitarios (rápidos)..."
python -m pytest tests/unit/ -v --tb=short

echo "🔗 Ejecutando tests de integración..."
python -m pytest tests/integration/ -v --tb=short

echo "✅ Todos los tests completados!"