# REFACTOR V2 - Plan de Simplificación y Reestructuración

## Objetivos del Refactor

### Principales Problemas Identificados
1. **Complejidad excesiva**: Archivos de 800+ líneas (main.py: 983, celery_app.py: 883)
2. **Duplicación de funcionalidades**: `player_analysis` vs `player_analysis_enhanced`
3. **Arquitectura monolítica**: Lógica de negocio mezclada con infraestructura
4. **Dificultad de mantenimiento**: Código difícil de expandir y testear
5. **Acoplamiento alto**: Dependencias circulares entre módulos

### Restricciones
- **NO cambiar** el output de la API hacia la UI (mantener `docs/results_expected_by_UI.json`)
- **Eliminar código legacy** sin mantener compatibilidad hacia atrás
- **Simplificar** por encima de optimizar prematuramente

## Estrategia de Refactorización

### Fase 1: Reestructuración de Arquitectura
**Objetivo**: Separar responsabilidades siguiendo Domain-Driven Design

#### 1.1 Nueva Estructura de Directorios
```
app/
├── core/                    # Configuración y utilidades centrales
│   ├── config.py           # Variables de entorno y configuración
│   ├── database.py         # Setup de BD y sessiones
│   ├── logging.py          # Configuración de logging
│   └── dependencies.py     # Inyección de dependencias
├── domain/                  # Lógica de negocio pura
│   ├── entities/           # Modelos de dominio
│   ├── repositories/       # Interfaces de repositorio
│   ├── services/           # Servicios de dominio
│   └── value_objects/      # Objetos de valor
├── infrastructure/         # Implementaciones concretas
│   ├── database/          # SQLModel implementations
│   ├── external/          # Chess.com API, Stockfish
│   ├── messaging/         # Redis, Celery
│   └── web/              # FastAPI routes
├── analysis/              # Módulos de análisis (mantener estructura actual)
│   ├── core/             # Engine y utilidades comunes
│   ├── metrics/          # Cálculo de métricas específicas
│   └── aggregation/      # Agregación de resultados
└── application/           # Casos de uso y orquestación
    ├── commands/         # Comandos (write operations)
    ├── queries/          # Consultas (read operations)
    └── handlers/         # Handlers de comandos/queries
```

#### 1.2 Separación de Responsabilidades

**Domain Layer (Pura)**:
- `Player`, `Game`, `Analysis` como entidades de dominio
- `AnalysisService` para lógica de negocio
- `PlayerRepository`, `GameRepository` como interfaces

**Infrastructure Layer**:
- `SQLPlayerRepository`, `SQLGameRepository` como implementaciones
- `ChessComClient`, `StockfishEngine` como servicios externos
- `CeleryTaskRunner` para procesamiento asíncrono

**Application Layer**:
- `AnalyzePlayerCommand`, `GetPlayerAnalysisQuery`
- Command/Query handlers para orquestación

### Fase 2: Simplificación de Modelos

#### 2.1 Unificación de Análisis
**Eliminar duplicación**: Unificar `PlayerAnalysisDetailed` como el único modelo de análisis

**Antes**:
```python
class Player:
    analysis: Optional["PlayerAnalysisDetailed"]
    # Campos duplicados de estado...

class PlayerAnalysisDetailed:
    # Múltiples campos JSON no tipados...
```

**Después**:
```python
class Player:
    username: str
    status: PlayerStatus
    analysis_result: Optional["PlayerAnalysis"]

class PlayerAnalysis:
    # Campos tipados y estructurados
    quality_metrics: QualityMetrics
    timing_metrics: TimingMetrics
    opening_metrics: OpeningMetrics
    risk_assessment: RiskAssessment
```

#### 2.2 Value Objects para Métricas
```python
@dataclass(frozen=True)
class QualityMetrics:
    avg_acpl: float
    avg_wdl_loss: float
    match_rate: float
    ipr: float

@dataclass(frozen=True)
class RiskAssessment:
    risk_score: int
    risk_factors: Dict[str, bool]
    confidence_level: int
```

### Fase 3: Simplificación de Celery Tasks

#### 3.1 Reducción de Complejidad
**Problema actual**: 3 tasks encadenados con lógica compleja en cada uno

**Nueva estrategia**: 2 tasks principales con responsabilidades claras
```python
@celery_app.task
def download_player_games(username: str) -> List[int]:
    """Solo descarga y almacena partidas. Retorna IDs."""

@celery_app.task
def analyze_player_complete(username: str, game_ids: List[int]) -> dict:
    """Analiza todas las partidas y genera resultado final."""
```

#### 3.2 Eliminación de Tasks Legacy
- **Eliminar**: `process_player_enhanced`, `analyze_game_task`, `analyze_game_detailed`
- **Simplificar**: Una sola cadena `download_player_games -> analyze_player_complete`

### Fase 4: Refactorización de API

#### 4.1 Separación de Endpoints
**Actual**: `main.py` con 983 líneas

**Nuevo**:
```
infrastructure/web/
├── routers/
│   ├── players.py      # Endpoints de jugadores
│   ├── games.py        # Endpoints de partidas
│   ├── analysis.py     # Endpoints de análisis
│   └── streaming.py    # SSE endpoints
└── dependencies/       # Dependencias específicas de web
```

#### 4.2 Handlers Dedicados
```python
# application/handlers/analyze_player_handler.py
class AnalyzePlayerHandler:
    def __init__(self,
                 player_repo: PlayerRepository,
                 task_runner: TaskRunner):
        self.player_repo = player_repo
        self.task_runner = task_runner

    async def handle(self, command: AnalyzePlayerCommand) -> PlayerAnalysisResult:
        # Lógica limpia y testeable
```

### Fase 5: Mejoras en Testing y Configuración

#### 5.1 Testabilidad
- **Inyección de dependencias** para facilitar mocking
- **Interfaces claras** entre capas
- **Casos de uso aislados** y testeables

#### 5.2 Configuración Simplificada
```python
# core/config.py
@dataclass
class Settings:
    database_url: str
    redis_url: str
    stockfish_path: str
    stockfish_depth: int

    @classmethod
    def from_env(cls) -> "Settings":
        # Carga desde variables de entorno
```

## Plan de Implementación

### Sprint 1: Foundation (Semana 1)
1. ✅ Crear nueva estructura de directorios
2. ✅ Mover configuración a `core/config.py`
3. ✅ Crear value objects para métricas
4. ✅ Definir interfaces de repositorio

### Sprint 2: Domain Refactor (Semana 2)
1. ✅ Refactorizar modelos de dominio
2. ✅ Implementar servicios de dominio
3. ✅ Crear repositorios SQL
4. ✅ Tests unitarios para domain layer

### Sprint 3: Application Layer (Semana 3)
1. ✅ Implementar command/query handlers
2. ✅ Crear casos de uso principales
3. ✅ Integrar con infrastructure layer
4. ✅ Tests de integración

### Sprint 4: Infrastructure Simplification (Semana 4)
1. ✅ Refactorizar Celery tasks
2. ✅ Simplificar API endpoints
3. ✅ Migrar análisis a nueva arquitectura
4. ✅ Eliminar código legacy

### Sprint 5: Cleanup & Performance (Semana 5)
1. ✅ Eliminar archivos no utilizados
2. ✅ Optimizar queries de BD
3. ✅ Revisar configuración de Celery
4. ✅ Tests end-to-end

## Beneficios Esperados

### Mantenibilidad
- **Archivos más pequeños**: Máximo 200-300 líneas por archivo
- **Responsabilidad única**: Cada módulo con propósito claro
- **Testeable**: Lógica de negocio aislada y sin dependencias externas

### Escalabilidad
- **Nuevas métricas**: Fácil agregar nuevos tipos de análisis
- **Nuevas fuentes**: Simple integrar otros sitios de ajedrez
- **Performance**: Optimizaciones puntuales sin afectar el resto

### Developer Experience
- **Onboarding**: Estructura clara y autodocumentada
- **Debugging**: Errores localizados y trazables
- **Features**: Desarrollo de nuevas funcionalidades más rápido

## Riesgos y Mitigaciones

### Riesgo: Breaking Changes en API
**Mitigación**:
- Tests de contrato para output JSON
- Endpoint de compatibilidad temporal si es necesario

### Riesgo: Performance Degradation
**Mitigación**:
- Benchmarks antes y después del refactor
- Profiling de endpoints críticos
- Optimización post-refactor si es necesario

### Riesgo: Bugs en Migración
**Mitigación**:
- Refactor incremental con tests en cada paso
- Feature flags para rollback rápido
- Comparación de resultados viejo vs nuevo

## Métricas de Éxito

1. **Complejidad**: Reducir archivos >500 líneas a <300
2. **Cobertura**: Alcanzar 80%+ test coverage
3. **Performance**: Mantener tiempos de respuesta actuales
4. **Mantenibilidad**: Nuevas features en <50% tiempo actual

## Estado del Refactor - Tracking de Interacciones

### Interacción 1 (2025-09-21)
**✅ COMPLETADO**: Plan inicial de refactorización
- Análisis del codebase actual
- Identificación de problemas principales
- Diseño de nueva arquitectura por capas
- Plan de 5 sprints definido

**🎯 SIGUIENTE**: Setup de testing local y debugging workflow

---

### Interacción 2 (Pendiente)
**En progreso**: Mejora del workflow de desarrollo
- [ ] Setup de testing local sin Docker
- [ ] Creación de test suite inicial
- [ ] Documentación de debugging workflow mejorado

---

## Testing Strategy & Local Development

### Problema Actual
- **Sin tests reales**: Solo pytest configurado pero sin implementar
- **Debugging con Docker**: Workflow lento (rebuild + restart containers)
- **Sin aislamiento**: Imposible testear componentes individuales

### Nueva Estrategia de Testing

#### 1. **Local Development sin Docker**
```bash
# Setup local rápido para desarrollo
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

# Base de datos local temporal para tests
export DATABASE_URL="sqlite:///test.db"
export REDIS_URL="redis://localhost:6379/1"  # DB diferente para tests
export STOCKFISH_PATH="/usr/games/stockfish"
export STOCKFISH_DEPTH="1"  # Rápido para tests
```

#### 2. **Test Suite Estructura**
```
tests/
├── conftest.py              # Fixtures pytest globales
├── unit/                    # Tests unitarios (sin DB, sin red)
│   ├── test_analysis/       # Tests de módulos de análisis
│   ├── test_domain/         # Tests de lógica de negocio
│   └── test_utils/          # Tests de utilidades
├── integration/             # Tests con BD local/Redis
│   ├── test_repositories/   # Tests de acceso a datos
│   ├── test_services/       # Tests de servicios
│   └── test_api/            # Tests de endpoints
└── e2e/                     # Tests end-to-end (con mocks)
    └── test_workflows/      # Flujos completos de usuario
```

#### 3. **Mocking Strategy**
- **Stockfish**: Mock del engine para tests rápidos
- **Chess.com API**: Responses pre-grabados
- **Redis/Celery**: In-memory durante tests
- **Database**: SQLite en memoria para tests unitarios

#### 4. **Debug Workflow Mejorado**
```bash
# Para development:
python -m app.main  # FastAPI local con hot reload
python -m pytest tests/unit/  # Tests rápidos (<5s)
python -m pytest tests/integration/  # Tests con BD (~30s)

# Para debugging específico:
python -c "from app.analysis import quality; quality.calculate_acpl([...])"
pytest tests/unit/test_analysis/test_quality.py::test_calculate_acpl -v
```

### Docker Solo Para Production-Like Testing
```bash
# Solo cuando necesites verificar integración completa
docker-compose up postgres redis  # Solo servicios necesarios
python -m app.main  # API local conectada a servicios Docker
```

## Conclusión

Este refactor V2 busca transformar el codebase de un monolito complejo a una arquitectura modular, mantenible y escalable. La prioridad es la simplicidad y claridad del código, facilitando el desarrollo futuro y reduciendo el tiempo de onboarding para nuevos desarrolladores.

El plan es agresivo pero factible, con un enfoque incremental que permite rollback en cualquier momento si surgen problemas críticos.