# Chess Player Analyzer - Production Deployment Guide

## 🚀 Clean Architecture Deployment

This guide covers the deployment of the Chess Player Analyzer application using the new Clean Architecture implementation (v2.0.0).

## 📋 Prerequisites

### System Requirements
- **Python**: 3.9+
- **PostgreSQL**: 13+
- **Redis**: 6+
- **Docker**: 20+
- **Docker Compose**: 2.0+

### Hardware Recommendations
- **CPU**: 4+ cores
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 50GB+ SSD
- **Network**: Stable internet connection for Chess.com API

## 🏗️ Architecture Overview

### Clean Architecture Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Presentation Layer                       │
│  FastAPI Routers │ REST Endpoints │ WebSocket Handlers      │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                   Application Layer                         │
│  Use Cases │ Command/Query Handlers │ CQRS Implementation  │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                      Domain Layer                           │
│  Entities │ Value Objects │ Domain Services │ Repositories │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                  Infrastructure Layer                       │
│  Database │ Redis │ Celery │ External APIs │ File System   │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 Performance Optimizations

### Database Optimizations
- **Connection Pool**: 20 connections (increased from 10)
- **Max Overflow**: 30 connections (increased from 20)
- **Pool Recycle**: 3600 seconds (1 hour)
- **SQLite Optimizations**: WAL mode, 64MB cache, mmap enabled

### Redis Optimizations
- **Max Connections**: 50 (increased from 10)
- **Connection Pooling**: Enabled with retry logic
- **Health Checks**: 30-second intervals
- **Socket Keep-Alive**: Enabled

### Celery Optimizations
- **Worker Prefetch**: 4 (increased from 1)
- **Max Tasks per Child**: 1000
- **Result Compression**: gzip enabled
- **Task Time Limits**: 30/25 minutes (hard/soft)

## 🚀 Quick Start with Docker Compose

### 1. Clone and Setup
```bash
git clone <repository-url>
cd ChessPlayerAnalyzer
cp .env.example .env
```

### 2. Configure Environment
Edit `.env` file:
```bash
# Database Configuration
DATABASE_URL=postgresql+psycopg://chess:chess@postgres:5432/chessdb
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=30
DB_POOL_RECYCLE=3600

# Redis Configuration
REDIS_URL=redis://redis:6379/0

# Celery Configuration
WORKER_PREFETCH_MULTIPLIER=4
WORKER_MAX_TASKS_PER_CHILD=1000
TASK_SOFT_TIME_LIMIT=1500
TASK_TIME_LIMIT=1800

# Stockfish Configuration
STOCKFISH_PATH=stockfish
STOCKFISH_DEPTH=12

# Performance Monitoring
ENABLE_TRACING=false
LOG_LEVEL=INFO
```

### 3. Start Services
```bash
# Start all services
docker-compose up -d

# Check service health
curl http://localhost:8000/health

# Check comprehensive health with metrics
curl http://localhost:8000/health/comprehensive
```

## 📊 Monitoring and Health Checks

### Health Check Endpoints

#### Basic Health Check
```bash
GET /health
```
Returns basic service status.

#### Comprehensive Health Check
```bash
GET /health/comprehensive
```
Returns detailed health information with performance metrics.

#### Performance Metrics
```bash
GET /metrics
```
Returns performance metrics summary.

#### Cache Metrics
```bash
GET /metrics/cache
```
Returns cache performance statistics.

### Example Health Check Response
```json
{
  "status": "healthy",
  "timestamp": "2024-01-20T10:30:00Z",
  "uptime_seconds": 3600,
  "services": {
    "database": "healthy",
    "container": "healthy",
    "redis": "healthy"
  },
  "performance": {
    "cache_hit_rate": 85.2,
    "active_connections": 15,
    "memory_usage": "good"
  },
  "optimizations": {
    "database_pooling": "enabled",
    "redis_connection_pooling": "enabled",
    "celery_prefetch_optimization": "enabled",
    "cache_optimization": "enabled"
  }
}
```

## 🔄 API Endpoints

### V2 Clean Architecture API (Recommended)
```bash
# Player status
GET /api/v2/players/{username}/status

# Analyze player
POST /api/v2/players/{username}/analyze

# Get analysis results
GET /api/v2/players/{username}/analysis
```

### V1 Legacy API (Backward Compatibility)
```bash
# Legacy endpoints redirect to v2
GET /api/v1/players/{username}
POST /api/v1/analyze
```

## 🛠️ Development Deployment

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Setup database
python -m app.database init

# Run development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Start Celery worker
celery -A app.celery_app worker --loglevel=info
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app tests/

# Run specific test category
pytest tests/unit/
pytest tests/integration/
```

## 🏭 Production Deployment

### Docker Production Setup
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  app:
    build: .
    environment:
      - DATABASE_URL=postgresql+psycopg://chess:chess@postgres:5432/chessdb
      - REDIS_URL=redis://redis:6379/0
      - LOG_LEVEL=INFO
      - ENABLE_TRACING=true
    depends_on:
      - postgres
      - redis
    ports:
      - "8000:8000"

  worker:
    build: .
    command: celery -A app.celery_app worker --loglevel=info --concurrency=4
    environment:
      - DATABASE_URL=postgresql+psycopg://chess:chess@postgres:5432/chessdb
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: chessdb
      POSTGRES_USER: chess
      POSTGRES_PASSWORD: chess
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

### Kubernetes Deployment
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: chess-analyzer
spec:
  replicas: 3
  selector:
    matchLabels:
      app: chess-analyzer
  template:
    metadata:
      labels:
        app: chess-analyzer
    spec:
      containers:
      - name: app
        image: chess-analyzer:2.0.0
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: chess-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: chess-secrets
              key: redis-url
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

## 📈 Performance Tuning

### Database Performance
```sql
-- PostgreSQL optimizations
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET work_mem = '4MB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
SELECT pg_reload_conf();
```

### Redis Performance
```bash
# redis.conf optimizations
maxmemory 512mb
maxmemory-policy allkeys-lru
tcp-keepalive 60
timeout 0
```

### Celery Scaling
```bash
# Scale workers based on CPU cores
celery -A app.celery_app worker --concurrency=4 --prefetch-multiplier=4

# Use different queues for different task types
celery -A app.celery_app worker -Q analysis,maintenance --concurrency=2
```

## 🔐 Security Considerations

### Environment Variables
- Never commit `.env` files
- Use secrets management in production
- Rotate credentials regularly

### Network Security
- Use HTTPS in production
- Configure CORS properly
- Implement rate limiting

### Database Security
- Use strong passwords
- Enable SSL connections
- Regular backups

## 📝 Logging and Debugging

### Log Levels
```bash
# Development
LOG_LEVEL=DEBUG

# Production
LOG_LEVEL=INFO

# Critical issues only
LOG_LEVEL=ERROR
```

### Debugging Tips
```bash
# Check container logs
docker-compose logs app
docker-compose logs worker

# Monitor Redis
redis-cli monitor

# Monitor PostgreSQL
SELECT * FROM pg_stat_activity;
```

## 🔄 Migration from Legacy

### Migration Checklist
- [ ] Backup existing data
- [ ] Update environment variables
- [ ] Test new endpoints
- [ ] Update client applications
- [ ] Monitor performance
- [ ] Remove legacy code

### Rollback Plan
If issues occur:
1. Switch back to legacy docker-compose
2. Restore database backup
3. Investigate issues
4. Plan fixes for next deployment

## 📞 Support and Troubleshooting

### Common Issues

#### Database Connection Issues
```bash
# Check database connectivity
docker-compose exec postgres psql -U chess -d chessdb -c "SELECT 1;"
```

#### Redis Connection Issues
```bash
# Check Redis connectivity
docker-compose exec redis redis-cli ping
```

#### High Memory Usage
```bash
# Monitor memory usage
docker stats
```

### Performance Issues
1. Check `/metrics` endpoint
2. Monitor database queries
3. Check Celery task queue
4. Review cache hit rates

For additional support, check the application logs and health endpoints.