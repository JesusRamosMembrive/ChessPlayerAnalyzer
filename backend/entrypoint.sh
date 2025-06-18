#!/usr/bin/env bash
set -euo pipefail
cd /code

# Espera a Postgres …
until pg_isready -h "${POSTGRES_HOST:-postgres}" -p "${POSTGRES_PORT:-5432}" -U "${POSTGRES_USER:-chess}" >/dev/null 2>&1; do
  sleep 1
done

mode="${1:-api}"     # ← si no pasan argumento, asumimos 'api'

case "$mode" in
  api)
      alembic upgrade head
      exec uvicorn app.main:app --host 0.0.0.0 --port 8000
      ;;
  worker)
      alembic upgrade head
      shift
      exec celery -A app.celery_app worker --loglevel=info "$@"
      ;;
  *)
      exec "$@"
      ;;
esac
