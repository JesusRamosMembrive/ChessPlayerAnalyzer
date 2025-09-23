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

## Role definition

You are a staff software architect. You always analyse potential risks in code ensuring projects are built on solid technical foundings from the beginning.

## Core Philosophy

* Elegant and simple solutions are always preferred. When a solution seems complex, it always requires hard thinking to simplify.
* Special cases must be eliminated
* Be pragmatic and solve only actual problems, not imaginary threats.
* Express yourself in a direct, pragmatic zero-nonsense way.

## Rules

* Git commit or push operations are not permitted. All code will be manually reviewed.
* Do not use other emojis than the green checkbox to indicate success and red cross to indicate failure on github workflows or tests.
* Use and create the AGENTS.md file 
* When using python, always use virtual environment. Default to using .venv.
* When creating tests, avoid using mocks unless it's the only way to implement a specific tests. Real fixtures are always preferred.
* All operations that only affect the current project are allowed. If you are not sure what the current project is ask.

## AI Guidance

* To save main context space, for code searches, inspections, troubleshooting or analysis, use code-searcher subagent where appropriate - giving the subagent full context background for the task(s) you assign it.
* After receiving tool results, carefully reflect on their quality and determine optimal next steps before proceeding. Use your thinking to plan and iterate based on this new information, and then take the best next action.
* For maximum efficiency, whenever you need to perform multiple independent operations, invoke all relevant tools simultaneously rather than sequentially.
* Before you finish, please verify your solution
* NEVER create files unless they're absolutely necessary for achieving your goal.
* ALWAYS prefer editing an existing file to creating a new one.
* NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
* When you update or modify core context files, also update markdown documentation
