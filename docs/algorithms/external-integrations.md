# External Integrations and APIs

**Status**: Active
**Last Updated**: 2025-09-13
**Category**: System Integrations

## Overview

ChessPlayerAnalyzer integrates with multiple external systems to provide comprehensive chess analysis. This document details all external APIs, services, and dependencies that the application relies on for data acquisition, analysis, and enhanced functionality.

## Core External Integrations

### 1. Chess.com Public API

**Purpose**: Primary source for chess game data and player statistics
**Implementation**: `app/utils.py:51-123`
**Rate Limiting**: Built-in throttling with User-Agent identification

#### API Endpoints Used

```python
# Archive discovery endpoint
GET https://api.chess.com/pub/player/{username}/games/archives

# Monthly game data endpoint
GET https://api.chess.com/pub/player/{username}/games/{YYYY}/{MM}
```

#### Key Features

- **Monthly Game Archives**: Retrieves historical games organized by month
- **PGN Format**: Complete game notation with move timing data (`[%clk]` annotations)
- **Player Metadata**: Includes usernames, ratings, game results, timestamps
- **User-Agent**: `chess-analyzer/0.2 (+https://github.com/tu_usuario)` for API identification

#### Data Processing

```python
def fetch_games(username: str, months: int = 12) -> List[Dict]:
    """
    Retrieves games from Chess.com API with move timing extraction

    Returns:
        List of dicts with keys: pgn, move_times, white, black, end_time
    """
```

#### Local Archiving

All fetched games are automatically archived locally:
- **Directory**: Configurable via `FETCH_ARCHIVE_DIR` (default: `archives/`)
- **Format**: `{username}_{timestamp}.json`
- **Retention**: Permanent local backup for analysis reproducibility

### 2. Stockfish Chess Engine

**Purpose**: Position evaluation and move analysis
**Implementation**: Distributed across analysis modules
**Version**: Latest stable via Docker container

#### Configuration

```yaml
# docker-compose.yml environment
STOCKFISH_PATH: /usr/games/stockfish
STOCKFISH_DEPTH: 12  # Analysis depth setting
```

#### Integration Points

1. **Position Evaluation**: Centipawn analysis for move quality assessment
2. **Move Scoring**: Best move identification and alternative move comparison
3. **Tactical Analysis**: Blunder, mistake, and inaccuracy detection
4. **Opening Analysis**: ECO code validation and opening preparation assessment

#### Usage Pattern

```python
# Typical Stockfish integration (pseudocode)
import chess.engine

with chess.engine.SimpleEngine.popen_uci(ENGINE_PATH) as engine:
    result = engine.analyse(board, chess.engine.Limit(depth=12))
    evaluation = result["score"].relative.score()
```

### 3. Syzygy Endgame Tablebases

**Purpose**: Perfect endgame analysis for positions ≤7 pieces
**Implementation**: `app/analysis/endgame.py`
**Path Configuration**: `SYZYGY_PATH` environment variable

#### Tablebase Integration

```python
from chess.syzygy import Tablebase

# Tablebase lookup for perfect play evaluation
def probe_tablebase(board: chess.Board) -> Optional[int]:
    """
    Returns:
        1: Win for side to move
        0: Draw
        -1: Loss for side to move
        None: Position not in tablebase
    """
```

#### Analysis Features

- **Perfect Play Detection**: Identifies optimal moves in endgame positions
- **Tablebase Match Rate**: Measures player accuracy against perfect play
- **WDL (Win/Draw/Loss)**: Evaluation for positions with ≤7 pieces
- **DTZ (Distance to Zero)**: Move count to conversion/mate

### 4. ECO Opening Classification

**Purpose**: Opening identification and classification
**Implementation**: `app/analysis/eco_table.py`
**Data Source**: Static ECO database with opening names

#### ECO Database Structure

```python
# ECO code mapping
ECO_NAMES: dict[str, str] = {
    "A00": "Uncommon Opening",
    "C45": "Scotch Game: Scotch Gambit",
    # ... ~500 opening classifications
}
```

#### Chess.com ECO Integration

Game data includes ECO URLs for opening information:
```json
{
    "ECO": "C45",
    "ECOUrl": "https://www.chess.com/openings/Scotch-Game-Scotch-Gambit"
}
```

## Infrastructure Dependencies

### 5. Redis Integration

**Purpose**: Caching, message brokering, and real-time notifications
**Implementation**: `app/utils.py:30-31`

#### Redis Usage Patterns

1. **Celery Message Broker**: Task queue management
2. **Result Caching**: `cache_get()`, `cache_set()` functions
3. **WebSocket Notifications**: Real-time progress updates
4. **Distributed Locking**: Player analysis mutex (`player_lock()`)

#### Configuration

```python
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
```

### 6. PostgreSQL Database

**Purpose**: Primary data storage and persistence
**Implementation**: SQLAlchemy/SQLModel ORM
**Connection**: `postgresql+psycopg://chess:chess@postgres:5432/chessdb`

#### External Data Storage

- **Game Data**: Parsed PGN games from Chess.com
- **Analysis Results**: Computed metrics and evaluations
- **Player Profiles**: Aggregated statistics and trends
- **Reference Stats**: Benchmarking data for analysis

### 7. OpenTelemetry & Jaeger Tracing

**Purpose**: Distributed observability and performance monitoring
**Implementation**: `app/otel.py`

#### Tracing Configuration

```yaml
environment:
  ENABLE_TRACING: "true"
  OTEL_SERVICE_NAME: chess-analyzer-api
  OTEL_EXPORTER_JAEGER_AGENT_HOST: jaeger
  OTEL_EXPORTER_JAEGER_AGENT_PORT: 6831
```

#### Monitored Operations

- **API Requests**: FastAPI endpoint tracing
- **Database Queries**: SQLAlchemy operation tracing
- **Celery Tasks**: Distributed task execution tracking
- **External API Calls**: Chess.com API request monitoring

## Development and Deployment Services

### 8. GitHub Integration

**Purpose**: Source control and CI/CD automation
**Implementation**: `.github/workflows/ci.yml`

#### GitHub Actions Features

- **Automated Testing**: Python test suite execution
- **Security Scanning**: Dependency vulnerability checks
- **Code Quality**: Linting and formatting validation
- **Deployment Triggers**: Production deployment automation

### 9. Container Registry

**Purpose**: Docker image storage and distribution
**Implementation**: Multi-stage Dockerfile with BuildKit caching

#### Container Images

1. **Base Application**: Python 3.12 + Stockfish + app dependencies
2. **ML/PyTorch**: Specialized container with PyTorch 2.3.1 for ML workloads
3. **Development**: Volume-mounted development environment

## External Service Dependencies

### 10. Machine Learning Services (Optional)

**Purpose**: Advanced pattern detection and player profiling
**Implementation**: `app/ml/` modules with dedicated torch queue

#### ML Pipeline Integration

- **Model Storage**: Persistent model checkpoints in `model_store/`
- **Training Data**: Player performance datasets
- **Inference**: Real-time player classification and anomaly detection
- **Queue Isolation**: Separate Celery queue for computationally intensive ML tasks

### 11. Cloud Deployment Platforms

**Purpose**: Production hosting and scaling
**Recommended Platforms**: Railway, Render, Fly.io

#### Platform-Specific Integrations

1. **Railway**: Direct Docker deployment with PostgreSQL addon
2. **Render**: Web service + PostgreSQL + Redis deployment
3. **Fly.io**: Multi-region deployment with persistent volumes
4. **Vercel/Netlify**: Frontend deployment for Next.js UI

## Security and Compliance

### Authentication & API Keys

- **Chess.com API**: Public API, no authentication required
- **User-Agent**: Identifies application for Chess.com rate limiting
- **Redis**: Internal network only, no external authentication
- **Database**: Containerized with internal credentials

### Rate Limiting and Abuse Prevention

```python
# Chess.com API respect
s.headers["User-Agent"] = UA
# Built-in request throttling and error handling
```

### Data Privacy

- **Local Archiving**: All Chess.com data cached locally for analysis
- **No Personal Data**: Only public chess game data processed
- **Retention Policy**: Configurable archive retention periods

## Monitoring and Health Checks

### External Service Health

```python
# Health check endpoints monitor external dependencies
@app.get("/health/external")
async def external_health():
    return {
        "chess_com_api": "reachable",
        "stockfish": "available",
        "redis": "connected",
        "database": "operational"
    }
```

### Error Handling and Resilience

1. **Chess.com API Failures**: Graceful degradation with cached data
2. **Stockfish Unavailability**: Fallback to cached evaluations
3. **Redis Connection Loss**: In-memory fallback for caching
4. **Database Connectivity**: Connection pooling with retry logic

## Configuration Management

### Environment Variables

```bash
# External service configuration
STOCKFISH_PATH=/usr/games/stockfish
STOCKFISH_DEPTH=12
SYZYGY_PATH=/data/syzygy
FETCH_ARCHIVE_DIR=archives
REDIS_URL=redis://redis:6379/0
DATABASE_URL=postgresql+psycopg://chess:chess@postgres:5432/chessdb

# Observability
ENABLE_TRACING=true
OTEL_SERVICE_NAME=chess-analyzer
OTEL_EXPORTER_JAEGER_AGENT_HOST=jaeger
```

### Service Discovery

- **Docker Compose**: Service name resolution (`redis`, `postgres`, `jaeger`)
- **Kubernetes**: Service mesh integration for production deployments
- **Load Balancing**: Built-in container orchestration load balancing

## Integration Testing

### External Service Mocking

```python
# Test integration with external services
def test_chess_com_api_integration():
    """Test Chess.com API data fetching with mock responses"""

def test_stockfish_engine_integration():
    """Test Stockfish engine communication and evaluation"""

def test_tablebase_integration():
    """Test Syzygy tablebase probing functionality"""
```

### Integration Test Suite

- **API Endpoint Tests**: Validate external API communication
- **Engine Tests**: Verify Stockfish integration and evaluation accuracy
- **Database Tests**: Ensure data persistence and retrieval functionality
- **Cache Tests**: Validate Redis caching and performance improvements

## Performance Optimization

### Caching Strategies

1. **Chess.com API**: 24-hour TTL for player game data
2. **Stockfish Analysis**: Persistent caching of position evaluations
3. **Database Queries**: Redis-backed query result caching
4. **Static Assets**: CDN caching for frontend resources

### Connection Pooling

- **Database**: SQLAlchemy connection pooling with configurable limits
- **Redis**: Connection pool for high-throughput operations
- **HTTP Clients**: Session reuse for Chess.com API requests

## Future Integration Roadmap

### Planned Integrations

1. **Lichess API**: Alternative game data source for broader coverage
2. **Chess24 API**: Additional professional game analysis capabilities
3. **Cloud Storage**: AWS S3/Google Cloud integration for large dataset storage
4. **Machine Learning APIs**: External ML services for advanced pattern recognition

### Enhancement Opportunities

1. **Real-time Streaming**: WebSocket integration for live game analysis
2. **Mobile APIs**: Native mobile app integration endpoints
3. **Social Features**: Integration with chess community platforms
4. **Tournament Data**: FIDE/tournament database integration

---

## See Also

- [Architecture Overview](../architecture/overview.md) - System design and component interaction
- [API Documentation](../api/endpoints.md) - Internal API endpoints and usage
- [Deployment Guide](../guides/deployment.md) - Production deployment configurations
- [Performance Analysis](performance.md) - System performance metrics and optimization