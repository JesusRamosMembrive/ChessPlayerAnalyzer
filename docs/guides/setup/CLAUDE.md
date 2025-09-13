# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Docker Development
- **Start all services**: `docker-compose up --build`
- **Start with dev profile**: `docker-compose --profile dev up --build`
- **Start with ML/PyTorch support**: `docker-compose --profile ml up --build`
- **Stop services**: `docker-compose down`

### Testing
- **Run tests**: `pytest` (with verbose output enabled in pytest.ini)
- **Test configuration**: Uses pytest.ini with `-vv` addopts and filters deprecation warnings

### Database
- **Database migrations**: Handled automatically by the `migrate` service on startup via `python -m app.init_db`
- **Manual migration**: `alembic upgrade head` (when running outside Docker)

### Development URLs
- **API**: http://localhost:8000
- **Jaeger UI** (tracing): http://localhost:16686
- **Database**: localhost:5432 (chess/chess/chessdb)
- **Redis**: localhost:6379

## Architecture Overview

This is a chess analysis application with microservices architecture:

### Core Services
- **backend**: FastAPI REST API (port 8000)
- **celery**: Standard Celery worker for analysis tasks
- **celery-torch**: Specialized PyTorch worker for ML tasks (queue: torch)
- **postgres**: Database for games, players, and analysis results
- **redis**: Message broker for Celery + real-time notifications
- **jaeger**: Distributed tracing (development only)

### Key Workflow
1. **Player Analysis Request** → `POST /players/{username}`
2. **Game Download** → Fetches games from Chess.com API
3. **Analysis Pipeline** → Celery chains: `analyze_game_task` → `analyze_game_detailed`
4. **Aggregation** → Final `analyze_player_detailed` calculates player metrics
5. **Results** → Available via API endpoints and SSE streaming

### Database Models
- **Game**: Stores PGN, move times, ECO codes
- **MoveAnalysis**: Per-move Stockfish evaluations
- **GameAnalysisDetailed**: Advanced game metrics
- **Player**: Player metadata and analysis status
- **PlayerAnalysisDetailed**: Aggregated player metrics

### Analysis Modules (app/analysis/)
- **engine.py**: Stockfish integration and core analysis
- **quality.py**: Move quality metrics
- **timing.py**: Time management analysis
- **openings.py**: Opening repertoire analysis
- **endgame.py**: Endgame evaluation with Syzygy tablebase support
- **longitudinal.py**: Temporal performance analysis
- **ml_classifier.py**: Machine learning classifications
- **bayesian.py**: Bayesian statistical analysis

## Environment Variables

### Required
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `STOCKFISH_PATH`: Path to Stockfish binary
- `STOCKFISH_DEPTH`: Analysis depth (default: 12)

### Optional
- `SYZYGY_PATH`: Path to Syzygy tablebase files (for endgame metrics)
- `ENABLE_TRACING`: Enable OpenTelemetry tracing (true/false)
- `OTEL_SERVICE_NAME`: Service name for tracing
- `OTEL_EXPORTER_JAEGER_AGENT_HOST`: Jaeger host
- `OTEL_EXPORTER_JAEGER_AGENT_PORT`: Jaeger port

## API Structure

### Versioned API
- Base path: `/api/v1/`
- Routers organized in `app/api/v1/endpoints/`
- Main endpoints: games, players, analysis, health, tasks

### Key Endpoints
- `POST /players/{username}`: Start player analysis
- `GET /players/{username}`: Get analysis status
- `GET /metrics/player/{username}`: Get player metrics
- `GET /stream/{username}`: SSE progress updates
- `POST /analyze`: Analyze single game

## CLI Tools
- **player_cli.py**: Basic command-line interface
- **player_analyze_cli.py**: Enhanced CLI with rich formatting
- **bulk_upload.py**: Batch game analysis

## Testing
- Tests located in `tests/` directory
- Current tests: `test_fairness.py`, `test_garch_model.py`
- CI configured to handle pytest exit code 5 (no tests collected)

## Development Notes
- Uses SQLModel for database models with SQLAlchemy backend
- Celery task routing: default queue for standard tasks, torch queue for ML
- OpenTelemetry instrumentation for FastAPI, Celery, and SQLAlchemy
- Structured JSON logging configured
- Rate limiting and request logging middleware included