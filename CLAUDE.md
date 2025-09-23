# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ChessPlayerAnalyzer is a FastAPI application that analyzes Chess.com players and games using Stockfish engine analysis. The system uses a microservices architecture with Celery workers for heavy analysis tasks, PostgreSQL for data storage, and Redis for task queuing and real-time notifications.

## Architecture

The codebase uses a unified V2 architecture with simplified models:

- **FastAPI Backend** (`app/main.py`): REST API with multiple endpoint compatibility layers
- **Celery Workers** (`app/celery_tasks.py`): Asynchronous game and player analysis
- **Analysis Engine** (`app/analysis/engine.py`): Core analysis logic using Stockfish
- **Database Models** (`app/models.py`): Simplified 2-table design (Game, AnalysisResult)
- **Analysis Modules** (`app/analysis/`): Specialized metrics calculators

### Key Components

- **Game**: Stores PGN and basic metadata
- **AnalysisResult**: Unified table storing all analysis metrics as JSON
- **Player**: Status tracking and aggregated metrics
- **AnalysisEngine**: Orchestrates Stockfish analysis and metric calculation

## Development Commands

### Container Operations
```bash
# Start all services
docker-compose up --build

# Start V2 testing environment
docker-compose -f docker-compose.v2.yml up --build

# Stop and cleanup
docker-compose down
docker system prune
```

### Database Operations
```bash
# Run migrations
alembic upgrade head

# Create V2 tables (manual setup)
python create_v2_tables.py
```

### Testing
```bash
# Run specific test
python3 -m pytest tests/unit/test_analysis/test_quality.py::TestCalculateACPL::test_acpl_perfect_game -v

# Run integration tests
python test_sprint3_integration.py
```

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run API server (development)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Run Celery worker
celery -A app.celery_tasks:celery_app worker --loglevel=info --concurrency=1
```

## Key Environment Variables

- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection for Celery broker/backend
- `STOCKFISH_PATH`: Path to Stockfish engine binary
- `STOCKFISH_DEPTH`: Analysis depth (default: 12)
- `ENABLE_TRACING`: Enable OpenTelemetry tracing (true/false)
- `SYZYGY_PATH`: Optional tablebase path for endgame analysis

## API Endpoints Structure

The API maintains compatibility across multiple prefixes:
- `/api/v1/` - Main API routes
- `/api/` - Compatibility layer
- `/` (root) - Legacy compatibility

### Main Workflow
1. `POST /players/{username}` - Start player analysis
2. `GET /stream/{username}` - SSE progress updates
3. `GET /players/{username}` - Check analysis status
4. `GET /metrics/player/{username}` - Get aggregated metrics

## Analysis Pipeline

1. **Game Download**: Fetch games from Chess.com API
2. **Game Storage**: Store PGN and metadata in `Game` table
3. **Analysis Execution**: Run Stockfish analysis via `AnalysisEngine`
4. **Metrics Calculation**: Calculate quality, timing, opening metrics
5. **Result Storage**: Store all metrics in `AnalysisResult.metrics` JSON field
6. **Player Aggregation**: Compute longitudinal metrics and risk scores

## Testing Considerations

- Use `STOCKFISH_DEPTH=1` for faster testing
- The system handles Chess.com API rate limits automatically
- Analysis results are deterministic for the same game/depth
- WebSocket notifications are optional and fail gracefully

## Analysis Modules

- `quality.py`: ACPL, match rates, blunder analysis
- `timing.py`: Move time patterns and variance
- `openings.py`: Opening repertoire and entropy
- `endgame.py`: Tablebase matching and conversion efficiency
- `longitudinal.py`: Player-level trends and risk assessment

## Monitoring

- OpenTelemetry tracing to Jaeger (optional)
- Prometheus metrics via FastAPI Instrumentator
- Structured JSON logging throughout
- Redis-based progress tracking

## Database Schema

The V2 schema uses a simplified design:
- `game_v2`: PGN storage and basic metadata
- `analysis_result_v2`: All analysis metrics in JSON format
- `player_v2`: Status tracking and aggregated metrics
- `reference_stats_v2`: ELO-based benchmarking data

All metrics are stored in the `AnalysisResult.metrics` JSON field with a well-defined structure documented in `app/models.py`.


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
* Use and create the CLAUDE.md file inside the .claude directory in the project
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
