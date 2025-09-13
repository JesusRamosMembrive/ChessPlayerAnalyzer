# Utilidades y Helpers - ChessPlayerAnalyzer

**Audiencia:** Desarrolladores
**Última actualización:** 2025-09-13
**Estado:** Completo

Documentación completa de las utilidades, helpers y middleware que proporcionan funcionalidad de soporte al sistema ChessPlayerAnalyzer.

## <¯ Objetivo

Las utilidades proporcionan:
- **Conexión de base de datos** con pooling y reintentos
- **Integración Chess.com API** para descarga de partidas
- **Sistema de notificaciones** Redis/WebSocket en tiempo real
- **Progress tracking** para tareas Celery
- **Rate limiting** y middleware de seguridad
- **Sanitización de datos** y validación
- **Cache** y utilidades de desarrollo

## =à Módulos Principales

### 1. **utils.py** - Utilidades Core
Funciones principales para integración externa y comunicaciones.

#### Descarga de Partidas Chess.com
```python
def fetch_games(username: str, months: int = 12) -> List[Dict]:
    """
    Descarga partidas de Chess.com API con tiempos de movimiento.

    Args:
        username: Nombre de usuario de Chess.com
        months: Meses hacia atrás a descargar (default: 12)

    Returns:
        Lista de diccionarios con 'pgn', 'move_times', metadatos

    Raises:
        requests.RequestException: Error en API Chess.com
    """
```

**Características:**
- **User-Agent personalizado** para identificación
- **Extracción de tiempos** desde comentarios `[%clk]` en PGN
- **Timeout configurables** (10 segundos default)
- **Archivo local** de backup en `/archives`
- **Manejo de errores** robusto con logs detallados

**Ejemplo de output:**
```python
[
    {
        "pgn": "[Event \"Chess.com Game\"]\n[White \"hikaru\"]...",
        "move_times": [2500, 1200, 800, 1500, ...],  # milliseconds
        "white": "hikaru",
        "black": "opponent",
        "end_time": "2025-09-13T10:30:00+00:00"
    }
]
```

#### Sistema de Notificaciones
```python
def notify_ws(username: str, payload: dict) -> None:
    """
    Publica mensaje en canal Redis para notificaciones tiempo real.

    Args:
        username: Usuario destino
        payload: Datos JSON a enviar

    Canal: f"player:{username}"
    """

def update_progress(username: str, *, increment: int = 1) -> None:
    """
    Actualización atómica de progreso de análisis.

    - Actualiza Player.done_tasks, done_games, progress
    - Envía notificación WebSocket automática
    - Maneja concurrencia con SELECT FOR UPDATE
    """
```

#### Progress Tracking Celery
```python
def task_progress(task: Task, current: int, total: int, username: str = None) -> None:
    """
    Reporta progreso de tarea Celery en tiempo real.

    - Actualiza estado Celery con PROGRESS
    - Envía notificación WebSocket si username proporcionado
    - Calcula porcentaje automáticamente
    """
```

#### Cache y Utilidades
```python
def cache_get(task_name: str, args: list, kwargs: dict) -> Any:
    """Retrieve cached result for task with hash-based key"""

def cache_set(task_name: str, args: list, kwargs: dict, result: Any) -> None:
    """Store task result in Redis cache with TTL"""

def player_lock(username: str):
    """Context manager for player-level locking"""

@contextmanager
def player_lock(username: str):
    """
    Context manager para bloqueo por jugador usando Redis.
    Previene análisis concurrentes del mismo usuario.
    """
```

### 2. **database.py** - Gestión de Base de Datos
Conexión robusta a PostgreSQL con pool de conexiones y reintentos.

#### Configuración de Engine
```python
# Pool de conexiones
POOL_SIZE = 10              # Conexiones persistentes
MAX_OVERFLOW = 20           # Conexiones adicionales temporales
POOL_TIMEOUT = 30           # Segundos espera conexión libre
POOL_RECYCLE = 1800         # Reciclar después de 30 min

# Reintentos de conexión
MAX_RETRIES = 5             # Número máximo de reintentos
INITIAL_RETRY_DELAY = 2.0   # Delay inicial entre reintentos
```

#### Funciones principales
```python
def _create_engine_retry(url: str, **kwargs) -> Engine:
    """
    Crea Engine con reintentos exponenciales.
    - Backoff exponencial (2x delay)
    - Máximo 5 reintentos por defecto
    - Logs detallados de intentos
    """

def get_session() -> SQLModelSession:
    """
    Dependency para FastAPI que proporciona sesión SQLModel.
    Usado en endpoints con Depends(get_session)
    """

def init_db() -> None:
    """
    Inicializa tablas de base de datos.
    Ejecutado en startup de aplicación.
    """
```

#### Variables de Entorno
```bash
DATABASE_URL="postgresql+psycopg://chess:chess@postgres:5432/chessdb"
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20
DB_POOL_TIMEOUT=30
DB_POOL_RECYCLE=1800
DB_MAX_RETRIES=5
DB_RETRY_DELAY=2.0
```

### 3. **utils_sanitize.py** - Sanitización de Datos
Limpieza de datos para almacenamiento JSON seguro.

```python
def clean_json_numbers(obj):
    """
    Limpia objetos Python para almacenamiento JSON.

    Transformaciones:
    - NaN, ±Inf ’ None
    - numpy.floating ’ float nativo
    - numpy.integer ’ int nativo
    - Procesamiento recursivo de dict/list

    Uso: Antes de insertar JSON en PostgreSQL
    """
```

**Ejemplo:**
```python
import numpy as np

data = {
    'score': np.float64(85.5),
    'invalid': float('nan'),
    'nested': {
        'count': np.int32(42),
        'infinity': float('inf')
    }
}

clean_data = clean_json_numbers(data)
# {
#   'score': 85.5,
#   'invalid': None,
#   'nested': {
#     'count': 42,
#     'infinity': None
#   }
# }
```

## = Middleware de Seguridad

### 4. **rate_limiter.py** - Rate Limiting
Middleware de limitación de peticiones con algoritmo sliding window.

#### Configuración
```python
# Variables de entorno
RATE_LIMIT_MAX_REQUESTS = 100    # Peticiones por ventana
RATE_LIMIT_WINDOW_SECONDS = 60   # Ventana en segundos

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware que limita peticiones por IP.
    - Sliding window en memoria
    - Headers informativos en respuesta
    - JSON error response cuando se excede
    """
```

#### Algoritmo
- **Sliding window** con timestamps
- **Limpieza automática** de ventanas expiradas
- **Storage en memoria** (no requiere Redis)
- **Headers de respuesta:**
  ```
  X-RateLimit-Limit: 100
  X-RateLimit-Remaining: 85
  X-RateLimit-Reset: 1694606400
  ```

#### Response de Error (429)
```json
{
  "error": "rate_limit_exceeded",
  "message": "Too many requests. Try again in 45 seconds.",
  "retry_after": 45
}
```

### 5. **request_logger.py** - Request Logging
Middleware de logging estructurado de peticiones HTTP.

```python
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs structured information about HTTP requests:
    - Request method, path, headers
    - Response status, timing
    - Client IP, User-Agent
    - Request/response sizes
    """
```

**Log format (JSON):**
```json
{
  "timestamp": "2025-09-13T10:30:00Z",
  "method": "POST",
  "path": "/api/v1/players/hikaru",
  "status_code": 202,
  "duration_ms": 125,
  "client_ip": "192.168.1.100",
  "user_agent": "Mozilla/5.0...",
  "request_size": 0,
  "response_size": 156
}
```

### 6. **trace_context.py** - Distributed Tracing
Middleware para añadir headers de tracing distribuido.

```python
class TraceContextMiddleware(BaseHTTPMiddleware):
    """
    Adds distributed tracing headers:
    - X-Trace-Id: Unique request identifier
    - traceparent: W3C Trace Context standard
    - Integration with OpenTelemetry
    """
```

**Headers añadidos:**
```
X-Trace-Id: abc123-def456-789ghi
traceparent: 00-abc123def456789ghi-def456789ghiabc-01
```

## =' Utilidades de Desarrollo

### Redis Client
```python
# Configuración global
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

# Funciones helper
def cache_get(key: str) -> Any
def cache_set(key: str, value: Any, ttl: int = 3600) -> None
def pubsub_publish(channel: str, message: dict) -> None
```

### Utilidades de Archivos
```python
# Paths importantes
TB_PATH = Path(os.getenv("SYZYGY_PATH", "/data/syzygy"))
ARCHIVE_DIR = Path(os.getenv("FETCH_ARCHIVE_DIR", "archives"))

# Regex patterns
CLK_RGX = re.compile(r"\[%clk\s+([\d:.]+)]")  # Chess.com clock times
```

### SQLAlchemy Helpers
```python
def sa_to_dict(obj) -> dict:
    """Convert SQLAlchemy object to dictionary"""
    return {c.name: getattr(obj, c.name) for c in obj.__table__.columns}

def bulk_insert_or_update(session, model, data: List[dict]) -> None:
    """Efficient bulk operations with conflict resolution"""
```

## =€ Configuración y Variables de Entorno

### Redis/Cache
```bash
REDIS_URL="redis://redis:6379/0"
CACHE_TTL=3600                    # Default cache TTL
```

### Base de Datos
```bash
DATABASE_URL="postgresql+psycopg://chess:chess@postgres:5432/chessdb"
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20
```

### Rate Limiting
```bash
RATE_LIMIT_MAX_REQUESTS=100
RATE_LIMIT_WINDOW_SECONDS=60
```

### Chess.com API
```bash
FETCH_ARCHIVE_DIR="./archives"    # Local backup directory
USER_AGENT="chess-analyzer/0.2"
```

### Paths
```bash
SYZYGY_PATH="/data/syzygy"        # Tablebase files (opcional)
```

## =Ê Métricas y Monitoreo

### Request Metrics
- **Response times** por endpoint
- **Status codes** distribution
- **Rate limit hits** por IP
- **Error rates** y patrones

### Database Metrics
- **Connection pool** utilization
- **Query performance** y timeouts
- **Retry attempts** y failures

### Cache Metrics
- **Hit/miss ratios** por tipo de cache
- **TTL effectiveness**
- **Memory usage** Redis

## = Debugging y Logging

### Log Levels
```python
# Configuración en logging_config.py
LOGGING_CONFIG = {
    'version': 1,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'json'
        }
    },
    'formatters': {
        'json': {
            'class': 'pythonjsonlogger.jsonlogger.JsonFormatter'
        }
    }
}
```

### Structured Logging Examples
```python
import logging
logger = logging.getLogger(__name__)

# Progress tracking
logger.info("fetch_games: Starting", extra={
    "username": username,
    "months": months,
    "operation": "fetch_start"
})

# Error handling
logger.error("Database connection failed", extra={
    "error": str(e),
    "retry_count": attempt,
    "max_retries": MAX_RETRIES
})
```

## =Ú Referencias

- [Utils Core](../../app/utils.py) - Utilidades principales
- [Database Config](../../app/database.py) - Configuración BD
- [Rate Limiter](../../app/middleware/rate_limiter.py) - Middleware seguridad
- [Sanitization](../../app/utils_sanitize.py) - Limpieza de datos

## = Historial de Cambios

- **2025-09-13:** Documentación inicial de utilidades y helpers
- **2025-09-13:** Middleware de seguridad y configuraciones

---

**Siguiente acción:** Documentar módulos ML y configuración avanzada
**Responsable:** Equipo de infraestructura