# Plan de Refactor Unificado - ChessPlayerAnalyzer

## Filosofía Central

**Objetivo**: Transformar el codebase en un sistema debugeable, mantenible y comprensible mediante la eliminación de efectos secundarios ocultos, estados globales y dependencias circulares.

**Principios**:
- Efectos secundarios explícitos y controlados
- Dependencias inyectadas, no importadas globalmente
- Separación clara de responsabilidades por dominio
- API pública estable durante la transición

## Análisis de Problemas Críticos

### 1. **Inicialización Caótica**
- **Problema**: Cada módulo arranca logging, OpenTelemetry y Prometheus al importarse
- **Ubicación**: `app/main.py:67-107`, `app/celery_tasks.py:18-97`
- **Impacto**: Conexiones activadas al solo importar utilidades, debugging imposible

### 2. **Estado Global Distribuido**
- **Problema**: Tres variantes de locks Redis sin cleanup compartido
- **Ubicación**: `app/main.py:41-65`, `app/api/v1/endpoints/players.py:95-165`, `app/utils.py:210-220`
- **Impacto**: Race conditions ocultas, cleanup inconsistente

### 3. **Módulo God-Class**
- **Problema**: `app/utils.py` mezcla HTTP, Redis, caching, JSON y progreso
- **Ubicación**: `app/utils.py:1-210` (210 líneas de responsabilidades mezcladas)
- **Impacto**: Testing imposible sin dependencias completas

### 4. **Arquitectura Dual Confusa**
- **Problema**: V1/V2 coexisten con routing triplicado
- **Ubicación**: `app/main.py:109-116` (mismo router montado 3 veces)
- **Impacto**: Imposible determinar qué código se ejecuta

### 5. **JSON Blob Antipattern**
- **Problema**: Métricas como JSON gigante sin validación
- **Ubicación**: `app/models.py` AnalysisResult.metrics (305 líneas de documentación)
- **Impacto**: Queries imposibles, debugging opaco

## Plan de Implementación

### FASE 1: Estabilización Inmediata (3-5 días)

#### 1.1 Centralización de Inicialización
**Target**: `app/main.py`, `app/celery_tasks.py`

```python
# Nuevo: app/factories.py
def create_app(enable_telemetry: bool = True) -> FastAPI:
    app = FastAPI(...)
    if enable_telemetry:
        init_otel()
        instrument_fastapi(app)
    return app

def create_worker(enable_telemetry: bool = True) -> Celery:
    celery_app = Celery(...)
    if enable_telemetry:
        init_otel()
    return celery_app
```

**Criterio de éxito**: Poder importar módulos sin efectos secundarios

#### 1.2 Unificación de Locks Redis
**Target**: `app/main.py:41-65`, `app/api/v1/endpoints/players.py`, `app/utils.py`

```python
# Nuevo: app/services/analysis_lock.py
class AnalysisLockService:
    def acquire_player_lock(self, username: str) -> ContextManager
    def is_analysis_in_progress(self, username: str) -> bool
    def cleanup_stale_locks(self) -> int
```

**Criterio de éxito**: Un solo punto de control de concurrencia

#### 1.3 Descomposición de utils.py
**Target**: `app/utils.py:1-210`

```
app/
├── infrastructure/
│   ├── redis_service.py     # Redis ops + caching
│   └── http_client.py       # Chess.com API calls
├── analysis/
│   └── fetch_service.py     # Game fetching logic
└── serialization/
    └── json_utils.py        # JSON sanitization
```

**Criterio de éxito**: Cada módulo testeable independientemente

#### 1.4 Limpieza de Legacy
**Target**: `legacy/`, routing duplicado, imports circulares

- Eliminar directorio `legacy/` completo
- Remover routing V1 de `app/main.py:109-116`
- Resolver imports circulares en `app/utils.py:25`, `app/database.py:147`

**Criterio de éxito**: Solo una versión de cada endpoint activa

### FASE 2: Separación de Responsabilidades (5-7 días)

#### 2.1 Decomposición de AnalysisEngine
**Target**: `app/analysis/engine.py:1-679`

```python
# Separar en 4 componentes independientes:
class StockfishAnalyzer:     # Solo engine chess
class MetricsCalculator:     # Solo cálculos matemáticos
class ResultPersister:       # Solo operaciones DB
class AnalysisOrchestrator:  # Solo coordinación
```

**Criterio de éxito**: Testing unitario sin Stockfish ni DB

#### 2.2 Dependency Injection Container
**Target**: Dependencias circulares globales

```python
# Nuevo: app/container.py
@dataclass
class ServiceContainer:
    db: DatabaseService
    redis: RedisService
    lock_service: AnalysisLockService
    analyzer: StockfishAnalyzer

def create_container() -> ServiceContainer:
    # Factory pattern para testing
```

**Criterio de éxito**: Dependencies explícitas en cada componente

#### 2.3 Simplificación de Database Layer
**Target**: `app/database.py:33-148`

```python
# Modo simple para desarrollo (sin réplicas)
class SimpleDatabaseService:
    def get_session(self) -> Session

# Modo avanzado para producción
class ReplicatedDatabaseService:
    def get_read_session(self) -> Session
    def get_write_session(self) -> Session
```

**Criterio de éxito**: Setup de DB trivial para testing local

### FASE 3: Modelo de Datos Relacional (7-10 días)

#### 3.1 Diseño de Schema Relacional
**Target**: `app/models.py` JSON blob

```sql
-- Reemplazar AnalysisResult.metrics JSON con:
CREATE TABLE quality_metrics (
    analysis_result_id INT PRIMARY KEY,
    acpl FLOAT NOT NULL,
    match_rate FLOAT NOT NULL,
    blunder_rate FLOAT NOT NULL,
    wdl_loss FLOAT
);

CREATE TABLE timing_metrics (
    analysis_result_id INT PRIMARY KEY,
    mean_move_time FLOAT,
    time_variance FLOAT,
    lag_spike_count INT
);

CREATE TABLE move_analysis (
    id SERIAL PRIMARY KEY,
    analysis_result_id INT REFERENCES analysis_result(id),
    move_number INT NOT NULL,
    played VARCHAR(10) NOT NULL,
    best VARCHAR(10) NOT NULL,
    cp_loss INT,
    time_spent FLOAT
);
```

**Criterio de éxito**: Queries SQL normales funcionan

#### 3.2 Migración de Datos
**Target**: Conversión JSON → Relacional

```python
# Script de migración
def migrate_json_to_relational():
    # Extraer datos de JSON blobs existentes
    # Validar estructura
    # Insertar en nuevas tablas
    # Verificar integridad
```

**Criterio de éxito**: Zero data loss, queries 10x más rápidas

### FASE 4: Optimizaciones Finales (3-5 días)

#### 4.1 Simplificación de Celery
**Target**: `app/celery_tasks.py:1-565`

```python
# Una sola task con estados internos claros
@celery_app.task
def process_player_simple(username: str) -> PlayerAnalysisResult:
    with ServiceContainer() as container:
        # Estado explícito, sin callbacks
        # Progress tracking simplificado
        # Error handling localizado
```

#### 4.2 Configuration Centralization
**Target**: Variables scattered

```python
@dataclass
class AppConfig:
    database_url: str
    redis_url: str
    stockfish_path: str
    enable_telemetry: bool

    @classmethod
    def from_env(cls) -> 'AppConfig':
        # Toda la lógica de env vars centralizada
```

#### 4.3 Limpieza Final
- Eliminar helpers sin uso (`app/main.py:51-65`)
- Reducir verbosidad de logs (`app/analysis/quality.py:72-103`)
- Clarificar `app/application/` (usar o eliminar)
- Eliminar sys.path hacks (`app/analysis/quality.py:12-19`)

## Criterios de Éxito Global

### Debugging
- ✅ Debuggear análisis individual sin setup completo
- ✅ Stack traces apuntan al código real, no wrappers
- ✅ State visible en todo momento sin logging extra

### Testing
- ✅ Tests unitarios corren en <5 segundos
- ✅ Mocks simples para dependencias externas
- ✅ Testing sin Docker/Postgres/Redis

### Comprensión
- ✅ Desarrollador nuevo entiende flujo en <2 horas
- ✅ Modificar métrica no requiere entender todo el sistema
- ✅ Dependencies explícitas en cada módulo

### Mantenimiento
- ✅ Cambio en un componente no afecta otros
- ✅ Deploy sin sorpresas
- ✅ Logs útiles, no ruido

## Gestión de Riesgos

### Compatibilidad API
- **Riesgo**: Frontend React requiere cambios
- **Mitigación**: Mantener endpoints públicos exactos durante transición

### Tiempo de Desarrollo
- **Riesgo**: 3-4 semanas de refactor
- **Mitigación**: Releases incrementales, feature flags

### Migración de Datos
- **Riesgo**: Loss de datos JSON → Relacional
- **Mitigación**: Backup completo, validación exhaustiva, rollback plan

### Team Velocity
- **Riesgo**: Desarrollo nuevo features pausado
- **Mitigación**: Paralelizar refactor con bug fixes menores

## Entregables por Fase

**Fase 1**: Sistema sin efectos secundarios al importar
**Fase 2**: Components testeable independientemente
**Fase 3**: Queries SQL funcionan, JSON eliminado
**Fase 4**: Sistema production-ready simplificado

**Success Metric Final**: Time-to-debug de horas → minutos