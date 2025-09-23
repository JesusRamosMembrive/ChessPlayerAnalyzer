# Repository Guidelines

## Project Structure & Module Organization
- `app/` holds all runtime code: `api/` HTTP entrypoints, `services/` domain orchestration, `analysis/` Stockfish metrics, `domain/` entities, and `core/` infrastructure glue.
- Schema change tooling lives in `alembic/` and `migrations/`; keep revisions atomic and check them in alongside updated models.
- `docs/` captures ops guides, while `test/` stores datasets and exploratory scripts; formal suites reside in `tests/` mirroring package names.
- CLI utilities (`player_cli.py`, `player_analyze_cli.py`) and helper scripts (`bulk_upload.py`, `check_status.py`) stay in the repository root for quick inspection.

## Build, Test, and Development Commands
- `docker-compose up --build` launches the default stack (FastAPI, Celery, Postgres, Redis, Jaeger); add `-d` for detached runs.
- Use `docker compose -f docker-compose.v2.yml up -d` to exercise the unified V2 pipeline; inspect services with `docker compose -f docker-compose.v2.yml logs -f backend_v2`.
- Local CLI exploration: `python player_cli.py <username>` for quick checks, or `python player_analyze_cli.py <username>` for richer output.
- Run suites with `pytest` (configured for `-vv`); scope runs via `pytest tests/services -k task_name` and rely on `requirements-dev.txt` for dev dependencies.

## Coding Style & Naming Conventions
- Target Python 3.12 (see `Dockerfile`); use four-space indents, PEP 8 spacing, and group imports as stdlib/third-party/local.
- Modules and functions stay `snake_case`, classes `PascalCase`, and constants `UPPER_SNAKE_CASE` to match the existing tree.
- Favor type hints and pydantic models for new APIs; update `schemas.py` and `analysis` calculators together so contracts stay synchronized.
- Leverage structured logging via `logging_config.py` and avoid stray `print`; tracing toggles (`ENABLE_TRACING`, `OTEL_*`) must remain consistent across services.

## Testing Guidelines
- Prefer fast unit tests with `pytest` and `httpx.AsyncClient`; gate slower Celery or Postgres exercises behind markers (`pytest -m "not slow"`).
- Keep fixtures alongside tests or under `tests/fixtures`; heavier JSON payloads can reference `test/data` rather than duplicating files.
- Document any temporary skips or external dependencies in the PR so reviewers understand coverage trade-offs.

## Commit & Pull Request Guidelines
- Mirror recent history with short, imperative commit subjects (`Refine timing metrics`); add bodies when extra context helps.
- Bundle related migrations, env vars, and config updates with the code that consumes them.
- PRs should outline intent, validation steps (`pytest`, relevant docker commands), linked issues, and screenshots or metrics when behavior changes user flows.
- Call out riskier changes (schema migrations, tablebase expectations, tracing toggles) in the PR checklist to alert deployers.
