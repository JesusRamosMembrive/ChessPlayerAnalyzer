# Enhanced Timing Patterns - Plan de Implementación

## Objetivo

Expandir el análisis de timing más allá del actual `move_times` básico para incluir análisis de ritmo sofisticado, correlación con complejidad posicional, y detección de patrones temporales avanzados.

## Arquitectura Actual

### Componentes Existentes
- **move_times**: Array JSON en Game model con tiempos por movimiento
- **sigma_total**: Desviación estándar de tiempos calculada en compute_game_metrics
- **constant_time**: Flag binario para sigma_total < 1.0
- **pause_spike**: Detección de mejora después de pausas (T_PAUSE = 10s)

### Código Base Relevante
```python
# models.py - Game.move_times
move_times: list[int] | None = Field(sa_column=Column(JSON))

# celery_app.py - Análisis actual
sigma_total = stdev(times) if len(times) > 1 else 0.0
constant_time = sigma_total < 1.0
pause_index = next((i for i, t in enumerate(times) if t > T_PAUSE), None)
```

## Nuevas Funcionalidades

### 1. Análisis de Ritmo Avanzado

#### 1.1 Métricas de Ritmo Base
**Subtarea**: Implementar cálculo de ritmo promedio por fase de juego
- Dividir partida en opening (moves 1-15), middlegame (16-40), endgame (40+)
- Calcular ritmo promedio para cada fase
- Detectar cambios significativos de ritmo entre fases

**Subtarea**: Implementar análisis de aceleración/desaceleración
- Calcular derivada de tiempos de movimiento
- Detectar patrones de aceleración consistente
- Identificar desaceleraciones súbitas

#### 1.2 Patrones de Ritmo Complejos
**Subtarea**: Detectar ritmo "metrónomo" (intervalos regulares)
- Análisis de Fourier para detectar periodicidad
- Calcular coeficiente de variación por ventanas deslizantes
- Flag para ritmo artificialmente regular

**Subtarea**: Implementar análisis de "burst patterns"
- Detectar secuencias de movimientos rápidos seguidos de pausas
- Calcular ratio burst/pause
- Correlacionar con calidad de movimientos

### 2. Correlación con Complejidad Posicional

#### 2.1 Métricas de Complejidad
**Subtarea**: Calcular complejidad posicional usando datos de Stockfish
- Usar multipv spread (diferencia entre mejor y peor evaluación)
- Contar número de movimientos candidatos viables
- Calcular "mobility score" basado en opciones disponibles

**Subtarea**: Correlacionar tiempo vs complejidad
- Ratio tiempo_usado/complejidad_posicion
- Detectar movimientos "demasiado rápidos" para posiciones complejas
- Identificar "overthinking" en posiciones simples

#### 2.2 Análisis de Decisión
**Subtarea**: Implementar "decision time analysis"
- Tiempo esperado vs tiempo real por complejidad
- Detectar patrones de tiempo inconsistentes con dificultad
- Calcular "efficiency score" de uso de tiempo

### 3. Patrones Temporales Avanzados

#### 3.1 Análisis de Ventanas Temporales
**Subtarea**: Implementar análisis de ventanas deslizantes
- Calcular métricas de timing en ventanas de 5-10 movimientos
- Detectar cambios súbitos en patrones de timing
- Identificar "timing shifts" significativos

**Subtarea**: Análisis de correlación temporal
- Correlación entre tiempos de movimientos consecutivos
- Detectar patrones de "mirroring" (copiar ritmo del oponente)
- Análisis de auto-correlación en series temporales

#### 3.2 Detección de Anomalías Temporales
**Subtarea**: Implementar detección de outliers temporales
- Z-score analysis para movimientos individuales
- Detección de clusters de movimientos anómalos
- Clasificación de tipos de anomalías (pause, rush, inconsistent)

**Subtarea**: Análisis de "timing fingerprint"
- Crear perfil único de timing por jugador
- Detectar desviaciones del perfil normal
- Comparar perfiles entre partidas

### 4. Integración con Sistema de Sospecha

#### 4.1 Métricas de Timing Sospechoso
**Subtarea**: Expandir sistema de flags de sospecha
- Timing demasiado consistente (low variance + high accuracy)
- Patrones de timing que no correlacionan con complejidad
- Cambios súbitos de patrón temporal mid-game

**Subtarea**: Scoring probabilístico de timing
- Reemplazar binary flags con probabilidad de sospecha
- Combinar múltiples métricas de timing en score único
- Calibrar thresholds basado en data histórica

## Migraciones de Base de Datos

### Nuevos Campos en GameMetrics
```sql
ALTER TABLE gamemetrics ADD COLUMN rhythm_opening FLOAT;
ALTER TABLE gamemetrics ADD COLUMN rhythm_middlegame FLOAT;
ALTER TABLE gamemetrics ADD COLUMN rhythm_endgame FLOAT;
ALTER TABLE gamemetrics ADD COLUMN timing_consistency_score FLOAT;
ALTER TABLE gamemetrics ADD COLUMN complexity_correlation FLOAT;
ALTER TABLE gamemetrics ADD COLUMN decision_efficiency FLOAT;
ALTER TABLE gamemetrics ADD COLUMN timing_anomaly_count INTEGER;
ALTER TABLE gamemetrics ADD COLUMN timing_suspicion_score FLOAT;
```

### Nueva Tabla: TimingPatterns
```sql
CREATE TABLE timingpatterns (
    id SERIAL PRIMARY KEY,
    game_id INTEGER REFERENCES game(id),
    move_number INTEGER,
    time_used INTEGER,
    position_complexity FLOAT,
    expected_time FLOAT,
    anomaly_type VARCHAR(50),
    anomaly_score FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

## Nuevos Celery Tasks

### compute_timing_patterns
**Subtarea**: Implementar task principal de análisis de timing
- Procesar move_times array completo
- Calcular todas las métricas de ritmo
- Detectar anomalías y patrones
- Guardar resultados en TimingPatterns table

### analyze_position_complexity
**Subtarea**: Task para calcular complejidad posicional
- Usar análisis de Stockfish existente
- Calcular métricas de complejidad por posición
- Correlacionar con tiempos de movimiento
- Integrar con timing analysis

### update_timing_suspicion
**Subtarea**: Task para actualizar scores de sospecha
- Combinar múltiples métricas de timing
- Calcular probabilidad de comportamiento sospechoso
- Actualizar GameMetrics.timing_suspicion_score
- Trigger alerts para scores altos

## API Extensions

### Nuevos Endpoints
**Subtarea**: Extender `/metrics/game/{game_id}`
- Incluir nuevas métricas de timing en response
- Agregar timing_patterns array con detalles por movimiento
- Incluir timing_suspicion_score y breakdown

**Subtarea**: Nuevo endpoint `/timing/analysis/{game_id}`
- Análisis detallado de patrones temporales
- Gráficos de timing vs complejidad
- Detección de anomalías con explicaciones

**Subtarea**: Endpoint `/timing/player/{username}`
- Perfil de timing del jugador across partidas
- Comparación con promedios de rating similar
- Evolución de patrones temporales over time

## Testing Strategy

### Unit Tests
**Subtarea**: Tests para cálculos de métricas de ritmo
- Test rhythm calculation por fase de juego
- Test detection de timing anomalies
- Test correlation calculations

**Subtarea**: Tests para análisis de complejidad
- Mock Stockfish analysis data
- Test complexity scoring algorithms
- Test time vs complexity correlation

### Integration Tests
**Subtarea**: Tests end-to-end de timing analysis
- Test complete timing analysis pipeline
- Test database storage y retrieval
- Test API response formatting

### Performance Tests
**Subtarea**: Tests de rendimiento para large datasets
- Benchmark timing analysis en partidas largas
- Test memory usage con múltiples análisis concurrentes
- Optimize algorithms para mejor performance

## Consideraciones de Implementación

### Rendimiento
- Usar numpy para cálculos vectorizados de timing metrics
- Cache position complexity calculations
- Batch processing para análisis de múltiples partidas

### Precisión
- Calibrar thresholds usando data de jugadores conocidos
- Validar métricas contra análisis manual de expertos
- Continuous learning para mejorar detection accuracy

### Escalabilidad
- Diseñar para procesar miles de partidas diariamente
- Implement incremental analysis para partidas nuevas
- Optimize database queries para timing pattern searches

## Cronograma de Implementación

**Semana 1**: Métricas de ritmo base y análisis por fases
**Semana 2**: Correlación con complejidad posicional
**Semana 3**: Patrones temporales avanzados y detección de anomalías
**Semana 4**: Integración con sistema de sospecha y API endpoints
**Semana 5**: Testing, optimización y documentación

## Dependencias

- Análisis de Stockfish existente (multipv data)
- GameMetrics y MoveAnalysis models
- Celery task infrastructure
- numpy/scipy para análisis estadístico avanzado
