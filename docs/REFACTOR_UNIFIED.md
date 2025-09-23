# REFACTOR UNIFICADO - Hoja de Ruta

## Objetivo
Simplificar la arquitectura interna eliminando dependencias circulares entre tablas y módulos, manteniendo exactamente los mismos endpoints y respuestas JSON para la interfaz React.

## Principios del Refactor
- ✅ **Mantener endpoints actuales**: GET /api/players/{username} y GET /metrics/player/{username}
- ✅ **Conservar estructura JSON de respuesta** exacta para React
- ✅ **Eliminar dependencias circulares** entre módulos de análisis
- ✅ **Simplificar modelo de base de datos**: De 4 tablas principales a 2
- ✅ **Pipeline unificado** de análisis en orden determinístico
- ✅ **Nombres definitivos** (no prefijos como "Unified")

---

## FASE 1: Preparación y Nuevas Estructuras

### 1.1 Crear nuevos modelos de base de datos
- [x] Crear `app/models_v2.py` con nuevos modelos simplificados:
  ```python
  class Game(SQLModel, table=True):
      id: int = Field(primary_key=True)
      pgn: str
      white_username: str
      black_username: str
      created_at: datetime
      # Solo metadatos básicos

  class AnalysisResult(SQLModel, table=True):
      id: int = Field(primary_key=True)
      game_id: int = Field(foreign_key="game.id")
      player_username: str
      player_color: str  # 'white' | 'black'
      metrics: Dict = Field(sa_column=Column(JSON))  # TODAS las métricas
      analyzed_at: datetime
      engine_depth: int
  ```

### 1.2 Crear migration script
- [x] Crear `alembic/versions/f1a2b3c4d5e6_add_unified_tables_v2.py`
- [x] Migración que crea las nuevas tablas sin eliminar las actuales

### 1.3 Crear nuevo AnalysisEngine unificado
- [x] Crear `app/analysis/engine_v2.py` con clase `AnalysisEngine`:
  ```python
  class AnalysisEngine:
      def analyze_game(self, game: Game, username: str, color: str) -> AnalysisResult
      def analyze_player(self, username: str) -> Dict
      def _compute_all_metrics(self, moves_df: pd.DataFrame, game: Game) -> Dict
  ```

---

## FASE 2: Implementación del Pipeline Unificado

### 2.1 Refactorizar módulos de análisis
- [x] Modificar `app/analysis/quality.py`:
  - [x] Asegurar que `aggregate_quality_features()` sea independiente
  - [x] No dependa de tablas de BD, solo de DataFrames
  - [x] Entrada: moves_df, game_metadata → Salida: dict con métricas

- [x] Modificar `app/analysis/timing.py`:
  - [x] Asegurar que `aggregate_time_features()` sea independiente
  - [x] Entrada: moves_df → Salida: dict con métricas

- [x] Modificar `app/analysis/openings.py`:
  - [x] Asegurar que `aggregate_opening_features()` sea independiente
  - [x] Entrada: moves_df, game_metadata → Salida: dict con métricas

- [x] Modificar `app/analysis/longitudinal.py`:
  - [x] Asegurar que `aggregate_longitudinal_features()` sea independiente
  - [x] Entrada: lista de games_data → Salida: dict con métricas

### 2.2 Implementar pipeline determinístico
- [x] En `AnalysisEngine._compute_all_metrics()`:
  ```python
  def _compute_all_metrics(self, moves_df, game) -> Dict:
      # Orden determinístico sin dependencias circulares
      quality_metrics = aggregate_quality_features(moves_df, elo=None)
      timing_metrics = aggregate_time_features(moves_df)
      opening_metrics = aggregate_opening_features(moves_df, game)

      return {
          'quality': quality_metrics,
          'timing': timing_metrics,
          'opening': opening_metrics
      }
  ```

### 2.3 Implementar análisis a nivel jugador
- [x] Método `AnalysisEngine.analyze_player()`:
  - [x] Obtener todos los `AnalysisResult` del jugador
  - [x] Agregar métricas longitudinales de todas las partidas
  - [x] Generar la misma estructura JSON que espera React

---

## FASE 3: Migración de Celery Tasks

### 3.1 Crear nuevas tasks paralelas
- [x] Crear `app/celery_tasks_v2.py` con:
  - [x] `analyze_game_v2(game_id, username, color)` usando `AnalysisEngine`
  - [x] `analyze_player_v2(username)` usando agregación unificada
  - [x] `process_player_enhanced_v2(username)` pipeline completo
  - [x] Mantener misma interfaz de progreso/notificaciones

### 3.2 Adaptar endpoints para usar nuevas tasks
- [x] Crear `app/config_v2.py` con flags de configuración
- [x] Crear `app/adapters/analysis_adapter.py` para alternar V1/V2
- [x] Crear `app/api/v1/endpoints/players_v2.py` con endpoints adaptados
- [x] Crear `app/api/v1/endpoints/metrics_v2.py` con métricas adaptadas
- [x] Crear `app/main_v2.py` como punto de entrada alternativo
- [x] Verificar compatibilidad de respuestas JSON con `test_json_compatibility.py`

---

## FASE 4: Testing y Validación

### 4.1 Tests de compatibilidad
- [ ] Crear `tests/test_refactor_compatibility.py`:
  - [ ] Comparar respuestas JSON v1 vs v2
  - [ ] Validar que métricas calculadas sean equivalentes
  - [ ] Test de endpoints con ambas versiones

### 4.2 Tests de performance
- [ ] Benchmarking de tiempo de análisis v1 vs v2
- [ ] Verificar que no hay regresión de performance
- [ ] Validar uso de memoria mejorado

### 4.3 Test de migración de datos
- [ ] Script para migrar datos existentes de v1 a v2
- [ ] Validar integridad de datos migrados
- [ ] Rollback plan si es necesario

---

## FASE 5: Deployment y Limpieza

### 5.1 Activar versión v2 por defecto
- [ ] Cambiar endpoints para usar tasks_v2 por defecto
- [ ] Monitorear logs/errores en producción
- [ ] Verificar que React frontend sigue funcionando

### 5.2 Deprecar código v1
- [ ] Eliminar `app/models.py` antiguos (GameAnalysisDetailed, etc.)
- [ ] Eliminar `app/celery_tasks.py` v1
- [ ] Eliminar `app/analysis/engine.py` v1
- [ ] Renombrar `*_v2.py` → nombres definitivos

### 5.3 Limpieza de base de datos
- [ ] Migration para eliminar tablas v1 obsoletas
- [ ] Cleanup de código muerto
- [ ] Actualizar documentación

---

## FASE 6: Expansión (Post-Refactor)

### 6.1 Nuevos módulos de análisis
- [ ] Framework simple para agregar nuevos módulos:
  ```python
  def aggregate_new_feature(moves_df) -> Dict:
      return {"new_metric": value}

  # En AnalysisEngine._compute_all_metrics():
  # metrics['new_feature'] = aggregate_new_feature(moves_df)
  ```

### 6.2 Optimizaciones adicionales
- [ ] Caching de métricas calculadas
- [ ] Paralelización de cálculos independientes
- [ ] Optimización de queries SQL

---

## Estructura Final Objetivo

```
app/
├── models.py                 # Game, AnalysisResult (simplificado)
├── analysis/
│   ├── engine.py            # AnalysisEngine principal
│   ├── quality.py           # Métricas de calidad (independiente)
│   ├── timing.py            # Métricas de tiempo (independiente)
│   ├── openings.py          # Métricas de aperturas (independiente)
│   └── longitudinal.py      # Agregaciones longitudinales
├── celery_tasks.py          # Tasks unificadas
└── api/v1/endpoints/        # Endpoints (sin cambios externos)
```

## Criterios de Éxito

✅ **Endpoints mantienen exactamente la misma respuesta JSON**
✅ **Frontend React funciona sin cambios**
✅ **No hay dependencias circulares entre módulos**
✅ **Pipeline de análisis es determinístico y predecible**
✅ **Fácil agregar nuevos módulos de análisis**
✅ **Performance igual o mejor que v1**
✅ **Código más mantenible y simple**

---

## Notas de Implementación

- **Cada fase debe completarse antes de la siguiente**
- **Mantener v1 funcionando hasta que v2 esté 100% validado**
- **Usar feature flags para alternar entre versiones**
- **Testing exhaustivo de compatibilidad en cada paso**
- **Rollback plan preparado para cada fase**