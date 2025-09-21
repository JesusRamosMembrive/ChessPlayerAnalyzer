# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Docker Development (Recommended)
```bash
# Start all services (API, Celery worker, PostgreSQL, Redis, Jaeger)
docker-compose up --build

# Start only specific services
docker-compose up postgres redis
docker-compose up backend celery

# View logs for specific service
docker-compose logs -f backend
docker-compose logs -f celery
```

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Initialize database
python -m app.init_db

# Start FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Start Celery worker
celery -A app.celery_app worker --loglevel=info
```

### Testing
```bash
# Run tests (pytest configured to handle zero tests gracefully)
pytest -vv

# The CI allows exit code 5 (no tests collected) as success
```

### Environment Variables
Required for local development:
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `STOCKFISH_PATH`: Path to Stockfish binary (typically `/usr/games/stockfish`)
- `STOCKFISH_DEPTH`: Analysis depth (default: 12, use 1 for testing)

Optional for tracing:
- `ENABLE_TRACING`: Set to "true" to enable OpenTelemetry tracing
- `OTEL_SERVICE_NAME`: Service name for tracing
- `OTEL_EXPORTER_JAEGER_AGENT_HOST`: Jaeger host (default: jaeger)

## Architecture Overview

This is a **chess game analysis platform** that analyzes Chess.com players and their games using Stockfish engine. The system follows a **microservices architecture** with async task processing.

### Core Components

**FastAPI Backend** (`app/main.py`):
- REST API for game/player analysis requests
- Real-time progress streaming via Server-Sent Events
- Database operations with SQLModel/PostgreSQL

**Celery Worker** (`app/celery_app.py`):
- Async task processing for game analysis
- Stockfish integration for move evaluation
- Complex task chains: `analyze_game_task` → `analyze_game_detailed` → `analyze_player_detailed`

**Analysis Pipeline**:
1. **Download**: Fetch player games from Chess.com API
2. **Game Analysis**: Stockfish evaluates each move (`MoveAnalysis`)
3. **Detailed Metrics**: Calculate advanced statistics (`GameAnalysisDetailed`)
4. **Player Aggregation**: Generate player-level metrics (`PlayerAnalysisDetailed`)

### Key Directories

```
app/
├── analysis/           # Chess metrics calculation modules
│   ├── quality/        # Move quality, blunders, accuracy
│   ├── timing/         # Time management analysis
│   └── openings/       # Opening repertoire analysis
├── api/v1/endpoints/   # FastAPI route handlers
├── domain/             # Business logic and domain models
├── services/           # External service integrations
└── middleware/         # Request/response middleware
```

### Database Models (`app/models.py`)

- **Player**: Chess.com player with analysis status
- **Game**: Individual chess game with PGN and metadata
- **MoveAnalysis**: Per-move Stockfish evaluation
- **GameAnalysisDetailed**: Game-level aggregated metrics
- **PlayerAnalysisDetailed**: Player-level longitudinal analysis

### Task Management

The system uses **Celery task chains** to ensure proper sequencing:
- Tasks are idempotent and use `celery_once` to prevent duplicates
- Progress tracking via Redis notifications
- Error handling preserves partial results

### API Endpoints

Key endpoints in `app/main.py`:
- `POST /players/{username}`: Start player analysis
- `GET /players/{username}`: Get analysis status/progress
- `GET /metrics/player/{username}`: Retrieve player metrics
- `GET /stream/{username}`: SSE progress stream
- `POST /analyze`: Analyze single game

### External Dependencies

- **Stockfish**: Chess engine for position evaluation
- **Chess.com API**: Source for player games and archives
- **Syzygy Tablebases**: Optional endgame accuracy (requires ~150GB+ disk space)

### Monitoring & Observability

- **OpenTelemetry**: Distributed tracing (Jaeger UI at http://localhost:16686)
- **Prometheus**: Metrics collection via `prometheus-fastapi-instrumentator`
- **Structured Logging**: JSON logs with correlation IDs

### CLI Tools

- `player_cli.py`: Basic command-line interface
- `player_analyze_cli.py`: Enhanced CLI with progress visualization
- `bulk_upload.py`: Batch game analysis utility

## Development Notes

- The codebase is primarily in **Spanish** (README, comments)
- Uses **SQLModel** for type-safe database operations
- **Redis** serves dual purpose: Celery broker + SSE notifications
- **Alembic** handles database migrations
- CI pipeline treats pytest exit code 5 (no tests) as success
- Tracing is enabled by default in docker-compose for development