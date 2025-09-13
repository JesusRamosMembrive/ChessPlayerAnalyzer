# Configuración y Middleware - ChessPlayerAnalyzer

**Audiencia:** DevOps, Desarrolladores
**Última actualización:** 2025-09-13
**Estado:** Completo

Documentación completa de la configuración del sistema, middleware de aplicación y sistemas de observabilidad de ChessPlayerAnalyzer.

## 🎯 Objetivo

Los módulos de configuración proporcionan:
- **Logging estructurado** JSON para observabilidad
- **Distributed tracing** con OpenTelemetry y Jaeger
- **Error handling** unificado con respuestas consistentes
- **Middleware stack** de seguridad y monitoreo
- **Configuración centralizada** via variables de entorno
- **Observabilidad** end-to-end para debugging

## ⚙️ Configuración Principal

### 1. **logging_config.py** - Logging Estructurado
Configuración unificada de logging JSON para toda la aplicación.

#### Configuración Principal
```python
@lru_cache(maxsize=1)
def setup_logging() -> None:
    """
    Inicializa logging estructurado JSON.

    Características:
    - Formato JSON para fácil ingesta en ELK, Loki, Datadog
    - Nivel configurable con LOG_LEVEL
    - Incluye trace_id y span_id para correlación
    - Idempotente (puede llamarse múltiples veces)
    """
```

#### Formato de Output
```json
{
  "asctime": "2025-09-13T10:30:00.123Z",
  "levelname": "INFO",
  "name": "app.celery_app",
  "trace_id": "abc123def456789",
  "span_id": "def456789abc123",
  "message": "Starting player analysis",
  "username": "hikaru",
  "operation": "analyze_start"
}
```

#### Variables de Entorno
```bash
LOG_LEVEL=INFO                    # DEBUG, INFO, WARNING, ERROR
```

#### Uso en Módulos
```python
from app.logging_config import setup_logging
import logging

# Configurar logging (llamada una vez)
setup_logging()
logger = logging.getLogger(__name__)

# Logging estructurado con contexto
logger.info("Processing task", extra={
    "task_id": task_id,
    "username": username,
    "operation": "analysis_start"
})
```

### 2. **otel.py** - OpenTelemetry y Distributed Tracing
Instrumentación completa para trazas distribuidas con Jaeger.

#### Configuración Base
```python
def init_otel_base():
    """
    Inicializa TracerProvider con Jaeger.

    Configuración:
    - Service discovery automático
    - Batch span processing para performance
    - Metadata de recurso configurable
    - Graceful degradation si Jaeger no disponible
    """
```

#### Instrumentación de Servicios
```python
def instrument_fastapi(app):
    """Instrumenta FastAPI con trazas automáticas"""
    FastAPIInstrumentor.instrument_app(app)

def instrument_sqlalchemy():
    """Instrumenta SQLAlchemy para DB tracing"""
    SQLAlchemyInstrumentor().instrument(engine=engine)

def instrument_celery():
    """Instrumenta Celery para task tracing"""
    CeleryInstrumentor().instrument()

def instrument_requests():
    """Instrumenta requests HTTP externos"""
    RequestsInstrumentor().instrument()
```

#### Variables de Entorno
```bash
ENABLE_TRACING=true               # Activar/desactivar tracing
OTEL_SERVICE_NAME=chess-analyzer-api
OTEL_EXPORTER_JAEGER_AGENT_HOST=jaeger
OTEL_EXPORTER_JAEGER_AGENT_PORT=6831
```

#### Arquitectura de Tracing
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   Celery        │    │   SQLAlchemy    │
│   (HTTP)        │────│   (Tasks)       │────│   (Database)    │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                     OpenTelemetry                              │
│                   BatchSpanProcessor                           │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│     Jaeger      │  ◄── UI: http://localhost:16686
│   (Collector)   │
└─────────────────┘
```

### 3. **error_handlers.py** - Manejo de Errores
Sistema unificado de manejo de excepciones con respuestas JSON consistentes.

#### Handlers Implementados
```python
async def http_exception_handler(request: Request, exc: HTTPException):
    """Maneja HTTPException de FastAPI"""
    return JSONResponse(
        status_code=exc.status_code,
        content=_build_payload(str(exc.detail), exc.status_code)
    )

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Maneja errores de validación Pydantic"""
    return JSONResponse(
        status_code=422,
        content=_build_payload("Validation error", 422, errors=exc.errors())
    )

async def general_exception_handler(request: Request, exc: Exception):
    """Catch-all para excepciones no manejadas"""
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content=_build_payload("Internal server error", 500)
    )
```

#### Formato de Error Estándar
```json
{
  "status": "error",
  "message": "Validation error",
  "code": 422,
  "timestamp": "2025-09-13T10:30:00.123Z",
  "errors": [
    {
      "loc": ["body", "username"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

#### Registro en FastAPI
```python
from app.error_handlers import register_exception_handlers

app = FastAPI()
register_exception_handlers(app)
```

## 🛡️ Middleware Stack

### Orden de Middleware (FastAPI)
```python
app = FastAPI()

# 1. CORS (si necesario)
app.add_middleware(CORSMiddleware, ...)

# 2. Rate Limiting
app.add_middleware(RateLimitMiddleware)

# 3. Request Logging
app.add_middleware(RequestLoggingMiddleware)

# 4. Trace Context
app.add_middleware(TraceContextMiddleware)

# 5. Error Handlers
register_exception_handlers(app)

# 6. OpenTelemetry Instrumentation
instrument_fastapi(app)
```

### Flujo de Request/Response
```
┌─────────────────┐
│ HTTP Request    │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ CORS Middleware │ ◄── Headers CORS si configurado
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Rate Limiter    │ ◄── Sliding window, headers X-RateLimit-*
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Request Logger  │ ◄── Log JSON estructurado
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Trace Context   │ ◄── Headers X-Trace-Id, traceparent
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ OpenTelemetry   │ ◄── Automatic span creation
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ FastAPI Router  │ ◄── Endpoint execution
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Error Handlers  │ ◄── Exception handling si error
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ HTTP Response   │
└─────────────────┘
```

## 📊 Observabilidad

### Métricas Prometheus
```python
from prometheus_fastapi_instrumentator import Instrumentator

# Configuración automática
Instrumentator().instrument(app).expose(app)

# Endpoint: /metrics
# Métricas automáticas:
# - http_requests_total
# - http_request_duration_seconds
# - http_requests_in_progress
```

### Jaeger Tracing
**UI:** http://localhost:16686

**Trazas típicas:**
- **HTTP Request** → FastAPI endpoint → Celery task → Database query
- **Player Analysis** → fetch_games → analyze_game_task → analyze_game_detailed
- **Error flows** con stack traces automáticos

### Structured Logging
**Destinos comunes:**
- **stdout** → Docker logs → Centralized logging
- **ELK Stack** → Elasticsearch + Kibana
- **Grafana Loki** → Log aggregation + alerting
- **Datadog** → APM integration

## 🔧 Configuración por Entorno

### Development
```bash
# Logging
LOG_LEVEL=DEBUG

# Tracing (habilitado)
ENABLE_TRACING=true
OTEL_SERVICE_NAME=chess-analyzer-dev
OTEL_EXPORTER_JAEGER_AGENT_HOST=localhost
OTEL_EXPORTER_JAEGER_AGENT_PORT=6831

# Rate limiting (permisivo)
RATE_LIMIT_MAX_REQUESTS=1000
RATE_LIMIT_WINDOW_SECONDS=60
```

### Production
```bash
# Logging
LOG_LEVEL=INFO

# Tracing (habilitado con sampling)
ENABLE_TRACING=true
OTEL_SERVICE_NAME=chess-analyzer-prod
OTEL_EXPORTER_JAEGER_AGENT_HOST=jaeger.monitoring.svc.cluster.local
OTEL_EXPORTER_JAEGER_AGENT_PORT=6831

# Rate limiting (restrictivo)
RATE_LIMIT_MAX_REQUESTS=100
RATE_LIMIT_WINDOW_SECONDS=60

# Error handling
HIDE_STACK_TRACES=true
```

### Testing
```bash
# Logging
LOG_LEVEL=WARNING

# Tracing (deshabilitado)
ENABLE_TRACING=false

# Rate limiting (deshabilitado)
RATE_LIMIT_MAX_REQUESTS=999999
```

## 🚨 Alerting y Monitoring

### Métricas Clave
```python
# Request metrics
http_request_duration_seconds_bucket{le="1.0",method="POST",path="/api/v1/players/{username}"}

# Error rates
rate(http_requests_total{status=~"5.."}[5m])

# Rate limiting
rate_limit_exceeded_total

# Database
db_connection_pool_size
db_query_duration_seconds
```

### Log-based Alerts
```json
// Error rate spike
{
  "query": "sum(rate(log_entries{level=\"ERROR\"}[5m])) > 10"
}

// High latency
{
  "query": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 5"
}

// Database issues
{
  "query": "increase(db_connection_errors_total[5m]) > 0"
}
```

## 🔍 Debugging

### Trace Correlation
```bash
# Encontrar todas las operaciones de un request
curl -H "X-Trace-Id: abc123-def456" http://localhost:16686/api/traces

# Logs correlacionados
grep "abc123-def456" /var/log/chess-analyzer.log
```

### Performance Analysis
```python
# Spans lentos en Jaeger
operation:"analyze_game_task" AND duration:>30s

# Database queries lentas
operation:"sqlalchemy.query" AND duration:>1s
```

### Error Investigation
```python
# Trazas con errores
error:true AND service:"chess-analyzer-api"

# Stack traces en logs
level:"ERROR" AND trace_id:"abc123-def456"
```

## 📚 Referencias

- [Logging Config](../../app/logging_config.py) - Configuración logging
- [OpenTelemetry](../../app/otel.py) - Distributed tracing
- [Error Handlers](../../app/error_handlers.py) - Exception handling
- [Middleware](../../app/middleware/) - Request/response processing

## 🔄 Historial de Cambios

- **2025-09-13:** Documentación inicial de configuración y middleware
- **2025-09-13:** Observabilidad y debugging procedures

---

**Siguiente acción:** Documentar módulos ML y cola torch
**Responsable:** Equipo de platform/SRE