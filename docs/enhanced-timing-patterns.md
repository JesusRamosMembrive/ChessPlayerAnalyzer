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
*Objetivo*: Detectar jugadores que muestran cambios anómalos en timing patterns entre fases de juego, lo cual puede indicar uso selectivo de asistencia en fases específicas.
*Enfoque*: Segmentar move_times por game phases usando move numbers, calcular statistical measures (mean, median, variance) para cada fase, y aplicar change point detection para identify significant rhythm shifts que excedan variabilidad natural.
- Dividir partida en opening (moves 1-15), middlegame (16-40), endgame (40+)
- Calcular ritmo promedio para cada fase
- Detectar cambios significativos de ritmo entre fases

**Subtarea**: Implementar análisis de aceleración/desaceleración
*Objetivo*: Identificar patterns artificiales de acceleration/deceleration que no correspondan a natural game flow, indicando posible automated timing control.
*Enfoque*: Calcular first y second derivatives de move_times series, apply smoothing filters para reduce noise, y detect sustained acceleration/deceleration patterns que excedan expected human timing variability.
- Calcular derivada de tiempos de movimiento
- Detectar patrones de aceleración consistente
- Identificar desaceleraciones súbitas

#### 1.2 Patrones de Ritmo Complejos
**Subtarea**: Detectar ritmo "metrónomo" (intervalos regulares)
*Objetivo*: Identificar timing patterns artificialmente regulares que indiquen automated move execution o programmed timing intervals.
*Enfoque*: Apply Fast Fourier Transform para detect periodic components en timing series, calculate rolling coefficient of variation, y compare con expected human timing irregularity para flag anomalously regular patterns.
- Análisis de Fourier para detectar periodicidad
- Calcular coeficiente de variación por ventanas deslizantes
- Flag para ritmo artificialmente regular

**Subtarea**: Implementar análisis de "burst patterns"
*Objetivo*: Detectar patterns de rapid move sequences followed by pauses que podrían indicate batch processing de moves o consultation periods con external assistance.
*Enfoque*: Identify burst sequences usando threshold-based detection, calculate burst/pause ratios, y correlate con move quality metrics para detect anomalous patterns que suggest non-human timing behavior.
- Detectar secuencias de movimientos rápidos seguidos de pausas
- Calcular ratio burst/pause
- Correlacionar con calidad de movimientos

### 2. Correlación con Complejidad Posicional

#### 2.1 Métricas de Complejidad
**Subtarea**: Calcular complejidad posicional usando datos de Stockfish
*Objetivo*: Cuantificar objective position complexity para correlacionar con timing patterns y detect anomalous time usage que no corresponda a position difficulty.
*Enfoque*: Use existing Stockfish multipv analysis para calculate evaluation spread, count viable candidate moves, y compute mobility-based complexity scores que permitan objective correlation con human timing patterns.
- Usar multipv spread (diferencia entre mejor y peor evaluación)
- Contar número de movimientos candidatos viables
- Calcular "mobility score" basado en opciones disponibles

**Subtarea**: Correlacionar tiempo vs complejidad
*Objetivo*: Detectar timing patterns que no correlacionen apropiadamente con position complexity, indicando possible external assistance o artificial timing control.
*Enfoque*: Calculate time/complexity ratios, establish expected correlation baselines from human players, y detect significant deviations que suggest inappropriate time allocation for position difficulty.
- Ratio tiempo_usado/complejidad_posicion
- Detectar movimientos "demasiado rápidos" para posiciones complejas
- Identificar "overthinking" en posiciones simples

#### 2.2 Análisis de Decisión
**Subtarea**: Implementar "decision time analysis"
*Objetivo*: Evaluate time allocation efficiency para detect players que show artificial optimization en time usage que no refleje natural human decision-making processes.
*Enfoque*: Build predictive models para expected decision time based on complexity, compare con actual time usage, y calculate efficiency scores que capture deviations from natural human timing patterns.
- Tiempo esperado vs tiempo real por complejidad
- Detectar patrones de tiempo inconsistentes con dificultad
- Calcular "efficiency score" de uso de tiempo

### 3. Patrones Temporales Avanzados

#### 3.1 Análisis de Ventanas Temporales
**Subtarea**: Implementar análisis de ventanas deslizantes
*Objetivo*: Detect sudden changes en timing behavior que podrían indicate activation/deactivation de external assistance during games.
*Enfoque*: Apply sliding window analysis con overlapping windows, calculate timing metrics para each window, y use change point detection algorithms para identify significant timing shifts que exceed natural variability.
- Calcular métricas de timing en ventanas de 5-10 movimientos
- Detectar cambios súbitos en patrones de timing
- Identificar "timing shifts" significativos

**Subtarea**: Análisis de correlación temporal
*Objetivo*: Identify artificial timing correlations que no refleje natural human timing behavior, including opponent mirroring o automated timing patterns.
*Enfoque*: Calculate autocorrelation functions para timing series, detect cross-correlation con opponent timing, y identify anomalous correlation patterns que suggest non-human timing coordination.
- Correlación entre tiempos de movimientos consecutivos
- Detectar patrones de "mirroring" (copiar ritmo del oponente)
- Análisis de auto-correlación en series temporales

#### 3.2 Detección de Anomalías Temporales
**Subtarea**: Implementar detección de outliers temporales
*Objetivo*: Identify individual moves con timing patterns que deviate significantly from player's normal behavior, indicating possible external assistance o technical issues.
*Enfoque*: Apply statistical outlier detection (Z-score, IQR methods), use clustering algorithms para group anomalous moves, y classify anomaly types para enable targeted investigation de suspicious patterns.
- Z-score analysis para movimientos individuales
- Detección de clusters de movimientos anómalos
- Clasificación de tipos de anomalías (pause, rush, inconsistent)

**Subtarea**: Análisis de "timing fingerprint"
*Objetivo*: Create unique timing profiles para each player que capture their natural timing characteristics y enable detection de deviations que suggest external assistance.
*Enfoque*: Extract timing features (distribution parameters, rhythm patterns, complexity correlations) para create player fingerprints, y use similarity metrics para detect significant profile deviations across games.
- Crear perfil único de timing por jugador
- Detectar desviaciones del perfil normal
- Comparar perfiles entre partidas

### 4. Integración con Sistema de Sospecha

#### 4.1 Métricas de Timing Sospechoso
**Subtarea**: Expandir sistema de flags de sospecha
*Objetivo*: Extend current binary suspicious flag system con timing-specific indicators que capture subtle timing anomalies indicative of external assistance.
*Enfoque*: Define timing-based suspicion criteria, implement threshold-based detection para each criterion, y integrate con existing suspicious behavior detection para comprehensive assessment.
- Timing demasiado consistente (low variance + high accuracy)
- Patrones de timing que no correlacionan con complejidad
- Cambios súbitos de patrón temporal mid-game

**Subtarea**: Scoring probabilístico de timing
*Objetivo*: Replace binary timing flags con probabilistic scoring system que provide nuanced assessment de timing-based suspicion con confidence intervals.
*Enfoque*: Develop weighted scoring model que combine multiple timing metrics, use historical data para calibrate probability thresholds, y provide interpretable suspicion scores con contributing factor breakdown.
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
*Objetivo*: Create comprehensive timing analysis pipeline que process all timing data y generate detailed timing metrics para suspicious behavior detection.
*Enfoque*: Implement Celery task que process complete move_times arrays, calculate all rhythm y pattern metrics, apply anomaly detection algorithms, y store detailed results para posterior analysis y correlation.
- Procesar move_times array completo
- Calcular todas las métricas de ritmo
- Detectar anomalías y patrones
- Guardar resultados en TimingPatterns table

### analyze_position_complexity
**Subtarea**: Task para calcular complejidad posicional
*Objetivo*: Provide objective position complexity metrics que enable correlation con timing patterns para detect anomalous time allocation.
*Enfoque*: Leverage existing Stockfish analysis para extract complexity indicators, implement complexity scoring algorithms, y integrate con timing analysis pipeline para enable comprehensive time/complexity correlation analysis.
- Usar análisis de Stockfish existente
- Calcular métricas de complejidad por posición
- Correlacionar con tiempos de movimiento
- Integrar con timing analysis

### update_timing_suspicion
**Subtarea**: Task para actualizar scores de sospecha
*Objetivo*: Integrate timing analysis results con overall suspicion scoring system para provide comprehensive assessment de player behavior.
*Enfoque*: Combine multiple timing metrics usando weighted scoring model, calculate probabilistic suspicion scores, update database con results, y implement alerting system para high-risk cases requiring investigation.
- Combinar múltiples métricas de timing
- Calcular probabilidad de comportamiento sospechoso
- Actualizar GameMetrics.timing_suspicion_score
- Trigger alerts para scores altos

## API Extensions

### Nuevos Endpoints
**Subtarea**: Extender `/metrics/game/{game_id}`
*Objetivo*: Enhance existing game metrics endpoint con comprehensive timing analysis results para enable detailed investigation de timing patterns.
*Enfoque*: Extend current API response structure para include new timing metrics, add detailed move-by-move timing analysis, y provide suspicion score breakdown con contributing factors para investigative purposes.
- Incluir nuevas métricas de timing en response
- Agregar timing_patterns array con detalles por movimiento
- Incluir timing_suspicion_score y breakdown

**Subtarea**: Nuevo endpoint `/timing/analysis/{game_id}`
*Objetivo*: Provide specialized timing analysis endpoint que offer detailed temporal pattern analysis para investigative y research purposes.
*Enfoque*: Create dedicated endpoint que return comprehensive timing analysis, visualization-ready data para timing/complexity correlations, y detailed anomaly reports con explanations para investigative analysis.
- Análisis detallado de patrones temporales
- Gráficos de timing vs complejidad
- Detección de anomalías con explicaciones

**Subtarea**: Endpoint `/timing/player/{username}`
*Objetivo*: Provide longitudinal timing analysis para individual players que enable detection de timing pattern evolution y comparison con peer groups.
*Enfoque*: Aggregate timing data across multiple games, implement comparison con rating-matched control groups, y track temporal evolution de timing patterns para detect suspicious changes over time.
- Perfil de timing del jugador across partidas
- Comparación con promedios de rating similar
- Evolución de patrones temporales over time

## Testing Strategy

### Unit Tests
**Subtarea**: Tests para cálculos de métricas de ritmo
*Objetivo*: Validate accuracy de rhythm calculation algorithms para ensure reliable timing pattern detection y analysis.
*Enfoque*: Create comprehensive test suite con synthetic timing data, validate rhythm calculations against known patterns, y test anomaly detection accuracy usando controlled test cases con known ground truth.
- Test rhythm calculation por fase de juego
- Test detection de timing anomalies
- Test correlation calculations

**Subtarea**: Tests para análisis de complejidad
*Objetivo*: Ensure accuracy de position complexity calculations y their correlation con timing patterns para reliable analysis results.
*Enfoque*: Create mock Stockfish data con known complexity characteristics, validate complexity scoring algorithms, y test correlation calculations para ensure reliable time/complexity relationship detection.
- Mock Stockfish analysis data
- Test complexity scoring algorithms
- Test time vs complexity correlation

### Integration Tests
**Subtarea**: Tests end-to-end de timing analysis
*Objetivo*: Validate complete timing analysis workflow para ensure reliable integration de all timing analysis components.
*Enfoque*: Create integration tests que validate entire timing analysis pipeline, test database operations accuracy, y verify API response correctness para ensure system reliability en production environment.
- Test complete timing analysis pipeline
- Test database storage y retrieval
- Test API response formatting

### Performance Tests
**Subtarea**: Tests de rendimiento para large datasets
*Objetivo*: Ensure timing analysis system can handle production-scale data volumes sin performance degradation o resource exhaustion.
*Enfoque*: Implement comprehensive performance testing con realistic data volumes, profile memory usage y CPU utilization, y optimize algorithms para maintain acceptable performance at scale.
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
