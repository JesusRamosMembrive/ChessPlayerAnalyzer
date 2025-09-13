# Guía de Troubleshooting - ChessPlayerAnalyzer

**Audiencia:** Desarrolladores, DevOps, Support
**Última actualización:** 2025-09-13
**Estado:** Completo

Guía completa de resolución de problemas basada en issues reales encontrados durante el desarrollo, con soluciones probadas y procedimientos de debugging.

## <¯ Categorías de Problemas

Este documento cubre problemas comunes organizados por categorías:

1. **Docker y Contenedores** - Problemas de build, networking, volumes
2. **Base de Datos** - PostgreSQL, migrations, corrupción
3. **Celery y Workers** - Tasks fallidas, queue issues, ML worker
4. **API y Backend** - FastAPI errors, endpoints, validación
5. **Frontend** - Next.js, conexión API, build issues
6. **Análisis y Cálculos** - ACPL, métricas, Stockfish
7. **Performance** - Memory leaks, slow queries, timeouts

## =3 Problemas de Docker

### Error: Port Already in Use
```bash
# Síntomas
Error starting userland proxy: listen tcp4 0.0.0.0:8000: bind: address already in use

# Diagnosis
docker-compose ps
lsof -i :8000  # Linux/Mac
netstat -ano | findstr :8000  # Windows

# Solución
docker-compose down
# O matar proceso específico
kill -9 $(lsof -ti:8000)  # Linux/Mac
```

### Error: Container Build Fails
```bash
# Síntomas
failed to solve with frontend dockerfile.v0

# Diagnosis
docker system df  # Check disk space
docker images --filter "dangling=true"

# Solución
docker system prune -a  # Clean up
docker-compose build --no-cache backend
docker-compose build --no-cache celery-torch
```

### Error: Volume Mount Issues
```bash
# Síntomas
bind mount failed: no such file or directory

# Diagnosis
ls -la debug_results/
ls -la model_store/

# Solución
mkdir -p debug_results model_store
chmod 755 debug_results model_store
docker-compose down -v && docker-compose --profile dev up
```

### Error: Network Connectivity
```bash
# Síntomas
backend | psycopg2.OperationalError: could not connect to server

# Diagnosis
docker network ls
docker-compose ps
docker-compose logs postgres

# Solución
docker-compose down
docker network prune
docker-compose --profile dev up --build
```

## =Ä Problemas de Base de Datos

### Error: Database Connection Failed
```bash
# Síntomas
could not connect to server: Connection refused

# Diagnosis
docker-compose exec postgres pg_isready -U chess -d chessdb
docker-compose logs postgres

# Solución
# Wait for healthy status
docker-compose ps postgres
# Force recreate if needed
docker-compose down -v
docker-compose --profile dev up postgres -d
```

### Error: Migration Failures
```bash
# Síntomas
alembic.util.exc.CommandError: Target database is not up to date

# Diagnosis
docker-compose exec backend alembic current
docker-compose exec backend alembic history

# Solución
docker-compose exec backend alembic upgrade head
# Si falla, reset completo:
docker-compose down -v
docker-compose --profile dev up migrate backend
```

### Error: Database Corruption
```sql
-- Síntomas
ERROR: relation "player" does not exist

-- Diagnosis
docker-compose exec postgres psql -U chess -d chessdb
\dt
SELECT COUNT(*) FROM pg_stat_activity;

-- Solución
-- Reset completo de DB
docker-compose down -v
docker volume rm $(docker volume ls -q | grep postgres)
docker-compose --profile dev up
```

### Error: Performance Issues
```sql
-- Síntomas
Slow query performance, timeouts

-- Diagnosis
SELECT
    query,
    calls,
    total_time,
    mean_time,
    max_time
FROM pg_stat_statements
ORDER BY total_time DESC
LIMIT 10;

-- Solución
-- Create missing indexes
CREATE INDEX CONCURRENTLY idx_game_username ON game(username);
CREATE INDEX CONCURRENTLY idx_game_created_at ON game(created_at);
ANALYZE; -- Update statistics
```

## = Problemas de Celery

### Error: Worker Not Processing Tasks
```bash
# Síntomas
Tasks stuck in PENDING state

# Diagnosis
docker-compose exec celery celery -A app.celery_app inspect active
docker-compose exec celery celery -A app.celery_app inspect registered
docker-compose logs celery

# Solución
# Restart worker
docker-compose restart celery

# Purge stuck tasks
docker-compose exec celery celery -A app.celery_app purge

# Check Redis connection
docker-compose exec redis redis-cli ping
```

### Error: ML Worker Down
```bash
# Síntomas
torch queue tasks failing

# Diagnosis
docker-compose ps celery-torch
docker-compose logs celery-torch

# Solución
# Start ML profile
docker-compose --profile ml up celery-torch -d

# Check PyTorch installation
docker-compose exec celery-torch python -c "import torch; print(torch.__version__)"

# Test inference
docker-compose exec celery-torch python -c "
from app.ml_tasks import infer_model
import random
vector = [random.random() for _ in range(128)]
result = infer_model.delay(vector)
print(result.get())
"
```

### Error: Memory Issues
```bash
# Síntomas
celery | MemoryError: Unable to allocate array

# Diagnosis
docker stats chess-analyzer-celery-1
docker-compose exec celery free -h

# Solución
# Increase memory limits in docker-compose.yml
services:
  celery:
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G

# Restart with limits
docker-compose down && docker-compose --profile dev up
```

### Error: Task Routing Issues
```bash
# Síntomas
Tasks going to wrong queue

# Diagnosis
docker-compose exec celery celery -A app.celery_app inspect active_queues
docker-compose exec celery-torch celery -A app.celery_app inspect active_queues

# Solución
# Verify queue names in tasks
grep -r "queue=" app/
# Restart workers with correct queue config
docker-compose restart celery celery-torch
```

## = Problemas de API/Backend

### Error: FastAPI Import Errors
```bash
# Síntomas
ModuleNotFoundError: No module named 'app.something'

# Diagnosis
docker-compose exec backend python -c "import sys; print(sys.path)"
docker-compose exec backend ls -la app/

# Solución
# Check PYTHONPATH in docker-compose.yml
environment:
  PYTHONPATH: /app

# Rebuild if needed
docker-compose build backend
```

### Error: 422 Validation Errors
```bash
# Síntomas
{"detail": [{"loc": ["body", "username"], "msg": "field required"}]}

# Diagnosis
curl -X POST http://localhost:8000/api/v1/players/test \
  -H "Content-Type: application/json" \
  -d '{"invalid": "data"}'

# Solución
# Check request format
curl -X POST http://localhost:8000/api/v1/players/test \
  -H "Content-Type: application/json"
# Correct format for most endpoints
```

### Error: 500 Internal Server Errors
```bash
# Síntomas
{"detail": "Internal server error"}

# Diagnosis
docker-compose logs backend | grep -A 10 -B 5 "ERROR"
docker-compose logs backend | grep "Traceback"

# Solución varies by specific error
# Common fixes:
# 1. Check database connection
# 2. Verify Celery broker
# 3. Check environment variables
# 4. Restart services
```

### Error: Player Analysis Stop Bug
Este es un bug conocido documentado en `docs/guides/troubleshooting/STOP_ANALYSIS_FIX.md`:

```python
# Problema: stop_player_analysis re-añade el player después de borrarlo
# Archivo: app/main.py lines 786-791

# Solución requerida:
# Eliminar líneas 786-791 y reemplazar con:
session.commit()

# Añadir rollback al exception handler:
except Exception as e:
    logger.error(f"Error al detener el análisis para {username}: {e}")
    session.rollback()  # ADD THIS LINE
    raise HTTPException(status_code=500, detail=str(e))
```

## < Problemas de Frontend

### Error: Next.js Build Failures
```bash
# Síntomas
Error: Failed to compile

# Diagnosis
cd "UI React/ChessPlayerAnalyzerReact"
npm run build

# Solución
# Clear cache
rm -rf .next/ node_modules/
npm install
npm run build

# Check TypeScript errors
npx tsc --noEmit
```

### Error: API Connection Issues
```bash
# Síntomas
fetch failed, connection refused

# Diagnosis
# Check if backend is running
curl http://localhost:8000/api/v1/health

# Check API base URL in frontend
grep -r "localhost:8000" "UI React/ChessPlayerAnalyzerReact/lib/"

# Solución
# Verify backend is accessible
docker-compose ps backend

# Update API configuration
# In lib/chess-api.ts:
const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
```

### Error: CORS Issues
```bash
# Síntomas
Access to fetch blocked by CORS policy

# Diagnosis
# Check browser console for CORS errors
# Verify FastAPI CORS middleware

# Solución
# Add CORS middleware in app/main.py:
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## =Ê Problemas de Análisis

### Error: ACPL Valores Absurdos
Este es un problema conocido documentado en `docs/guides/troubleshooting/ERRORES_Y_SOLUCIONES.md`:

```python
# Problema: ACPL ~5600, IPR ~-666, quality_score ~-2191
# Causa: delta_eval extremo (99491) por evaluaciones de mate

# Solución en app/analysis/quality.py:
def acpl(game_df, player_color='white', cap_cp: int = 1500):
    # Usar delta_eval con cap para evitar outliers
    if 'delta_eval' in game_df.columns:
        values = game_df['delta_eval'].clip(upper=cap_cp)
        return float(values.mean())
    # Fallback actual para eval_cp_before/after
```

### Error: Stockfish Not Found
```bash
# Síntomas
[WARNING] Stockfish not found, quality metrics will use fallback values

# Diagnosis
which stockfish
echo $STOCKFISH_PATH

# Solución por plataforma:
# Linux:
sudo apt install stockfish
export STOCKFISH_PATH=/usr/games/stockfish

# macOS:
brew install stockfish
export STOCKFISH_PATH=/opt/homebrew/bin/stockfish

# Windows:
# Download from stockfishchess.org
# Set path to installation directory

# Testing:
python test/run_local_analysis.py --engine-enable --engine-path $STOCKFISH_PATH
```

### Error: NaN en Métricas
```bash
# Síntomas
mean_move_time=NaN, acpl=NaN en resultados

# Diagnosis
# Check datos de entrada
python test/run_local_analysis.py --input test_data.json --verbose

# Solución
# Habilitar engine para métricas completas
python test/run_local_analysis.py \
    --input test_data.json \
    --engine-enable \
    --suppress-warnings
```

## ¡ Problemas de Performance

### Error: Memory Leaks
```bash
# Síntomas
containers consume increasing memory over time

# Diagnosis
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Solución
# Add memory limits
services:
  backend:
    deploy:
      resources:
        limits:
          memory: 2G
  celery:
    deploy:
      resources:
        limits:
          memory: 4G

# Monitor and restart periodically
docker-compose restart backend celery
```

### Error: Slow Analysis
```bash
# Síntomas
Player analysis takes hours to complete

# Diagnosis
# Check Celery task status
curl http://localhost:8000/api/v1/tasks/{task_id}

# Check worker load
docker-compose exec celery celery -A app.celery_app inspect stats

# Solución
# Scale workers
docker-compose --profile dev up --scale celery=3

# Optimize Stockfish depth
# In docker-compose.yml:
environment:
  STOCKFISH_DEPTH: 8  # Reduce from 12

# Use local testing for development
python test/run_local_analysis.py --engine-depth 8
```

### Error: Database Timeouts
```bash
# Síntomas
psycopg2.OperationalError: canceling statement due to user request

# Diagnosis
# Check slow queries
docker-compose exec postgres psql -U chess -d chessdb
SELECT query, total_time FROM pg_stat_statements ORDER BY total_time DESC;

# Solución
# Add connection pooling limits
DATABASE_URL: postgresql+psycopg://chess:chess@postgres:5432/chessdb?pool_size=20&max_overflow=30

# Optimize queries with indexes
CREATE INDEX CONCURRENTLY idx_game_analysis_status ON game(analysis_status);
CREATE INDEX CONCURRENTLY idx_player_updated_at ON player(updated_at);
```

## =
 Debugging Tools

### Script de Testing Local
El script `test/run_local_analysis.py` es fundamental para debugging:

```bash
# Test básico sin Docker
python test/run_local_analysis.py \
    --input debug_results/sample_game.json \
    --engine-disable \
    --verbose

# Test completo con Stockfish
python test/run_local_analysis.py \
    --input debug_results/sample_game.json \
    --engine-enable \
    --engine-path /usr/games/stockfish \
    --csv

# Test por lotes
python test/run_local_analysis.py \
    --input-dir debug_results/ \
    --pattern "Affan_khan123_*.json" \
    --out-dir test/results/ \
    --summary-only
```

### Split de Datasets
Para debugging con datasets grandes:

```bash
# Dividir en partidas individuales
python test/split_json.py \
    --input archives/large_dataset.json \
    --mode per-game \
    --out-dir test/split_data

# Dividir en chunks
python test/split_json.py \
    --input archives/large_dataset.json \
    --mode chunks \
    --chunk-size 10 \
    --out-dir test/split_data
```

### Logs Estructurados
```bash
# API logs con contexto
docker-compose logs backend | grep -E "(ERROR|task_id|username)" | tail -50

# Celery task tracking
docker-compose logs celery | jq 'select(.levelname == "ERROR")'

# Database queries
docker-compose exec postgres tail -f /var/log/postgresql/postgresql.log
```

### Jaeger Tracing
Accede a http://localhost:16686 para:
- Ver traces de requests completos
- Debug performance bottlenecks
- Correlacionar errores entre servicios
- Analizar Celery task execution

### Health Checks
```bash
# Services health
curl http://localhost:8000/api/v1/health

# Database
docker-compose exec postgres pg_isready -U chess -d chessdb

# Redis
docker-compose exec redis redis-cli ping

# Celery workers
docker-compose exec celery celery -A app.celery_app inspect ping
```

## =Ë Checklist de Troubleshooting

### Problema General
1.  Verificar logs: `docker-compose logs [service]`
2.  Check status: `docker-compose ps`
3.  Health checks: `curl http://localhost:8000/api/v1/health`
4.  Restart service: `docker-compose restart [service]`
5.  Reset completo: `docker-compose down -v && docker-compose --profile dev up`

### Problema de Performance
1.  Monitor resources: `docker stats`
2.  Check database: slow queries, indexes
3.  Profile con testing local: `time python test/run_local_analysis.py`
4.  Scale workers: `--scale celery=3`
5.  Jaeger traces: http://localhost:16686

### Problema de Análisis
1.  Test local: `python test/run_local_analysis.py --engine-disable`
2.  Verify Stockfish: `which stockfish`
3.  Check data format: samples en `debug_results/`
4.  Engine test: `--engine-enable --engine-path`
5.  Review fixes: `docs/guides/troubleshooting/ERRORES_Y_SOLUCIONES.md`

## =Ú Referencias

- [Development Guide](development.md) - Setup y comandos básicos
- [Local Testing](troubleshooting/LOCAL_TESTING.md) - Testing sin Docker
- [Stop Analysis Fix](troubleshooting/STOP_ANALYSIS_FIX.md) - Bug conocido
- [Error Analysis](troubleshooting/ERRORES_Y_SOLUCIONES.md) - Problemas de cálculo
- [Architecture](../architecture/overview.md) - Diseño del sistema

## = Historial de Cambios

- **2025-09-13:** Documentación inicial de troubleshooting
- **2025-09-13:** Problemas comunes basados en issues reales
- **2025-09-13:** Herramientas de debugging y checklists

---

**Próxima fase:** Deployment y CI/CD
**Responsable:** Equipo de DevOps/SRE