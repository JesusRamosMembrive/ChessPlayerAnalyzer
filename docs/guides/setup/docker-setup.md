# Docker Configuration Guide

This guide covers the complete Docker setup and configuration for ChessPlayerAnalyzer development and production environments.

## Overview

ChessPlayerAnalyzer uses Docker Compose to orchestrate multiple services:

- **PostgreSQL**: Primary database
- **Redis**: Cache and message broker
- **Backend**: FastAPI application
- **Celery**: Task processing workers
- **Celery-Torch**: ML-specific worker queue
- **Jaeger**: Distributed tracing (optional)

## Development Setup

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- 8GB+ available RAM
- 20GB+ available disk space

### Quick Start

```bash
# Clone repository
git clone <repository-url>
cd ChessPlayerAnalyzer

# Start development environment
docker-compose --profile dev up -d

# Initialize database
docker-compose exec backend python -m app.init_db

# Check service status
docker-compose ps
```

### Development Profiles

#### Core Development (`--profile dev`)

Includes essential services for API development:

```bash
docker-compose --profile dev up -d
```

Services: `postgres`, `redis`, `migrate`, `backend`, `celery`

#### ML Development (`--profile ml`)

Includes PyTorch ML pipeline:

```bash
docker-compose --profile ml up -d
```

Additional services: `celery-torch` with PyTorch 2.3.1

## Service Configuration

### PostgreSQL Database

```yaml
postgres:
  image: postgres:15-alpine
  environment:
    POSTGRES_USER: chess
    POSTGRES_PASSWORD: chess
    POSTGRES_DB: chessdb
  volumes:
    - postgres_data:/var/lib/postgresql/data
  ports:
    - "5432:5432"
```

**Connection String**: `postgresql+psycopg://chess:chess@postgres:5432/chessdb`

### Redis Cache/Broker

```yaml
redis:
  image: redis:7-alpine
  ports:
    - "6379:6379"
```

**Connection String**: `redis://redis:6379/0`

### Backend API

```yaml
backend:
  build:
    context: .
    dockerfile: Dockerfile
    target: app-base
  command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
  volumes:
    - ./app:/app/app              # Hot reload
    - ./archives:/app/archives    # Game archives
    - ./debug_results:/app/debug_results
  environment:
    DATABASE_URL: postgresql+psycopg://chess:chess@postgres:5432/chessdb
    REDIS_URL: redis://redis:6379/0
    STOCKFISH_PATH: /usr/games/stockfish
    STOCKFISH_DEPTH: 12
  ports:
    - "8000:8000"
```

### Celery Workers

#### Default Queue Worker

```yaml
celery:
  command: celery -A app.celery_app worker --loglevel=info -Q default
  volumes:
    - ./app:/app/app
    - ./debug_results:/app/debug_results
```

**Handles**: General analysis tasks, API processing, data fetching

#### PyTorch ML Worker

```yaml
celery-torch:
  build:
    target: torch
    args:
      PYTORCH_IMAGE: pytorch/pytorch:2.3.1-cpu-py3.12
  command: celery -A app.celery_app worker --loglevel=info -Q torch
  volumes:
    - ./model_store:/app/model_store  # Persistent ML models
```

**Handles**: ML training, inference, PyTorch model operations

## Environment Variables

### Core Configuration

```bash
# Database
DATABASE_URL=postgresql+psycopg://chess:chess@postgres:5432/chessdb

# Redis
REDIS_URL=redis://redis:6379/0

# Chess Engine
STOCKFISH_PATH=/usr/games/stockfish
STOCKFISH_DEPTH=12

# Syzygy Tablebases (optional)
SYZYGY_PATH=/data/syzygy

# Archive Storage
FETCH_ARCHIVE_DIR=archives

# Rate Limiting
RATE_LIMIT_MAX_REQUESTS=100
RATE_LIMIT_WINDOW_SECONDS=60
```

### Observability (Optional)

```bash
# Distributed Tracing
ENABLE_TRACING=true
OTEL_SERVICE_NAME=chess-analyzer-api
OTEL_EXPORTER_JAEGER_AGENT_HOST=jaeger
OTEL_EXPORTER_JAEGER_AGENT_PORT=6831

# Service Names
OTEL_SERVICE_NAME=chess-analyzer-api      # Backend
OTEL_SERVICE_NAME=chess-analyzer-worker   # Celery
OTEL_SERVICE_NAME=chess-analyzer-torch    # ML Worker
```

## Volume Management

### Persistent Data

```yaml
volumes:
  postgres_data:          # Database storage
  model_store:           # ML models (torch profile)
```

### Development Volumes

```yaml
volumes:
  - ./app:/app/app                    # Code hot-reload
  - ./alembic/versions:/app/alembic/versions  # DB migrations
  - ./archives:/app/archives          # Game data cache
  - ./debug_results:/app/debug_results    # Analysis outputs
```

## Development Workflow

### Daily Development

```bash
# Start services
docker-compose --profile dev up -d

# View logs
docker-compose logs -f backend
docker-compose logs -f celery

# Restart specific service
docker-compose restart backend

# Execute commands
docker-compose exec backend python -c "print('Hello')"
docker-compose exec postgres psql -U chess -d chessdb

# Stop services
docker-compose down
```

### Database Operations

```bash
# Reset database (⚠️ destroys data)
docker-compose down -v
docker-compose --profile dev up -d
docker-compose exec backend python -m app.init_db

# Database backup
docker-compose exec postgres pg_dump -U chess chessdb > backup.sql

# Database restore
docker-compose exec -T postgres psql -U chess chessdb < backup.sql
```

### ML Development

```bash
# Start with ML services
docker-compose --profile ml up -d

# Monitor ML worker
docker-compose logs -f celery-torch

# Check model storage
docker-compose exec celery-torch ls -la /app/model_store

# Execute ML tasks
docker-compose exec backend python -c "
from app.celery_app import train_model_task
result = train_model_task.apply_async(queue='torch')
print(f'Task ID: {result.id}')
"
```

## Production Configuration

### Production Compose Override

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  backend:
    command: gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
    environment:
      DEBUG: false
      ENABLE_TRACING: false
      RATE_LIMIT_MAX_REQUESTS: 50
    volumes: []  # Remove development volumes

  celery:
    environment:
      CELERY_WORKER_PREFETCH_MULTIPLIER: 1
    volumes: []  # Remove development volumes

  postgres:
    environment:
      POSTGRES_PASSWORD_FILE: /run/secrets/postgres_password
    secrets:
      - postgres_password

secrets:
  postgres_password:
    external: true
```

### Production Deployment

```bash
# Production startup
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Health check
curl http://localhost:8000/health

# Production logs
docker-compose logs --tail=100 -f
```

## Troubleshooting

### Common Issues

#### Port Conflicts

```bash
# Check port usage
netstat -tulpn | grep :5432
netstat -tulpn | grep :6379
netstat -tulpn | grep :8000

# Use different ports
docker-compose up -d -p 15432:5432 postgres
```

#### Permission Issues

```bash
# Fix volume permissions
sudo chown -R $USER:$USER ./archives
sudo chown -R $USER:$USER ./debug_results

# Container user issues
docker-compose exec backend whoami
docker-compose exec backend ls -la /app
```

#### Memory Issues

```bash
# Check container memory usage
docker stats

# Limit container memory
services:
  backend:
    mem_limit: 2g
  celery:
    mem_limit: 4g
```

### Service Health Checks

```bash
# Database health
docker-compose exec postgres pg_isready -U chess

# Redis health
docker-compose exec redis redis-cli ping

# Backend health
curl http://localhost:8000/health

# Celery health
docker-compose exec backend celery -A app.celery_app inspect ping
```

### Log Analysis

```bash
# Service-specific logs
docker-compose logs backend
docker-compose logs celery
docker-compose logs postgres
docker-compose logs redis

# Follow logs in real-time
docker-compose logs -f --tail=50 backend

# Filter logs
docker-compose logs backend | grep ERROR
docker-compose logs celery | grep -i "task"
```

## Performance Optimization

### Resource Limits

```yaml
services:
  backend:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
```

### Database Optimization

```yaml
postgres:
  environment:
    POSTGRES_INITDB_ARGS: >
      --encoding=UTF8
      --lc-collate=C
      --lc-ctype=C
  command: >
    postgres
    -c shared_preload_libraries=pg_stat_statements
    -c pg_stat_statements.track=all
    -c max_connections=200
    -c shared_buffers=256MB
    -c effective_cache_size=1GB
```

### Redis Optimization

```yaml
redis:
  command: >
    redis-server
    --maxmemory 512mb
    --maxmemory-policy allkeys-lru
    --save 900 1
    --appendonly yes
```

## Security Considerations

### Development Security

- Use `.env` files for sensitive configuration
- Never commit secrets to version control
- Isolate services with Docker networks
- Regularly update base images

### Production Security

```yaml
services:
  backend:
    security_opt:
      - no-new-privileges:true
    read_only: true
    tmpfs:
      - /tmp
    cap_drop:
      - ALL
```

### Network Security

```yaml
networks:
  backend:
    driver: bridge
    internal: true
  frontend:
    driver: bridge

services:
  postgres:
    networks:
      - backend  # Internal only

  backend:
    networks:
      - backend  # Database access
      - frontend # External API access
```

## Monitoring Setup

### Jaeger Tracing

```yaml
jaeger:
  image: jaegertracing/all-in-one:1.57
  ports:
    - "16686:16686"
  environment:
    COLLECTOR_OTLP_ENABLED: true
```

Access Jaeger UI: http://localhost:16686

### Health Monitoring

```bash
# Add to crontab for production monitoring
*/5 * * * * curl -f http://localhost:8000/health || echo "Backend down" | mail -s "ChessAnalyzer Alert" admin@yourdomain.com
```

---

## See Also

- [Development Guide](../development.md) - Complete development workflow
- [Deployment Guide](../deployment.md) - Production deployment
- [Troubleshooting](../troubleshooting.md) - Common issues and solutions