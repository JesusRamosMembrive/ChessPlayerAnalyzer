# Cross-Game Correlation - Plan de Implementación

## Objetivo

Extender el framework actual de PlayerMetrics para detectar patrones de consistencia across diferentes controles de tiempo, oponentes, y fases de juego, proporcionando análisis longitudinal del comportamiento del jugador.

## Arquitectura Actual

### Componentes Existentes
- **PlayerMetrics**: username, game_count, opening_entropy, most_played, low_entropy
- **Game Model**: eco_code, opening_key, move_times
- **GameMetrics**: pct_top1, pct_top3, acl, suspicious, timing metrics
- **Cross-Game Analysis**: compute_player_metrics task con opening entropy

### Código Base Relevante
```python
# celery_app.py - Análisis actual cross-game
rows = s.exec(
    select(models.Game.opening_key)
    .where(or_(models.Game.white == username, models.Game.black == username))
    .where(models.Game.opening_key.is_not(None))
).all()
entropy = -sum(p * log2(p) for p in probs)
```

## Nuevas Funcionalidades

### 1. Análisis de Consistencia Temporal

#### 1.1 Patrones de Rendimiento Over Time
**Subtarea**: Implementar tracking de evolución de métricas
- Calcular moving averages de pct_top3, acl por períodos
- Detectar trends de mejora/deterioro over time
- Identificar períodos de rendimiento anómalo

**Subtarea**: Análisis de estabilidad de rendimiento
- Calcular variance de métricas key across partidas
- Detectar jugadores con rendimiento "demasiado estable"
- Identificar cambios súbitos en nivel de juego

#### 1.2 Correlación con Factores Temporales
**Subtarea**: Análisis por hora del día/día de semana
- Correlacionar rendimiento con timestamp de partidas
- Detectar patrones de rendimiento por horario
- Identificar "optimal playing times" por jugador

**Subtarea**: Análisis de fatiga y sesiones de juego
- Detectar deterioro de rendimiento en sesiones largas
- Calcular "fatigue coefficient" basado en partidas consecutivas
- Identificar patrones de recuperación entre sesiones

### 2. Análisis por Control de Tiempo

#### 2.1 Rendimiento Across Time Controls
**Subtarea**: Implementar métricas específicas por time control
- Separar análisis por blitz, rapid, classical
- Calcular relative strength por control de tiempo
- Detectar inconsistencias sospechosas entre controles

**Subtarea**: Análisis de adaptación a time controls
- Medir eficiencia de uso de tiempo por control
- Detectar jugadores que no adaptan estilo a tiempo disponible
- Calcular "time management score" por control

#### 2.2 Correlación de Timing Patterns
**Subtarea**: Comparar patrones de timing across controles
- Analizar si timing patterns se mantienen consistentes
- Detectar cambios artificiales en timing behavior
- Calcular "timing consistency score" cross-control

### 3. Análisis por Tipo de Oponente

#### 3.1 Rendimiento vs Rating del Oponente
**Subtarea**: Implementar análisis de rendimiento relativo
- Calcular expected vs actual performance por rating gap
- Detectar over/under-performance sistemático
- Identificar "rating manipulation" patterns

**Subtarea**: Análisis de adaptación a nivel de oponente
- Medir cambio en estilo de juego vs oponentes fuertes/débiles
- Detectar "sandbagging" o performance artificial
- Calcular "opponent adaptation score"

#### 3.2 Patrones de Juego vs Estilos de Oponente
**Subtarea**: Clasificar estilos de oponentes
- Categorizar oponentes por métricas de juego
- Analizar adaptación del jugador a diferentes estilos
- Detectar rigidez o flexibilidad en approach

### 4. Análisis de Fases de Juego

#### 4.1 Consistencia por Fase de Partida
**Subtarea**: Implementar análisis por opening/middlegame/endgame
- Separar métricas por fase de juego
- Detectar fortalezas/debilidades específicas por fase
- Calcular consistency scores por fase

**Subtarea**: Análisis de transiciones entre fases
- Medir performance en transiciones críticas
- Detectar patterns de collapse en fases específicas
- Identificar "phase specialization" patterns

#### 4.2 Correlación de Errores por Fase
**Subtarea**: Análisis de error patterns por fase
- Correlacionar cp_loss con fase de juego
- Detectar systematic weaknesses por fase
- Calcular "phase reliability score"

### 5. Detección de Anomalías Cross-Game

#### 5.1 Identificación de Outlier Games
**Subtarea**: Implementar detección de partidas anómalas
- Usar statistical methods para detectar outliers
- Clasificar tipos de anomalías (performance, timing, style)
- Calcular "game anomaly score"

**Subtarea**: Clustering de patrones de juego
- Agrupar partidas por similarity en métricas
- Detectar cambios en clusters over time
- Identificar "behavioral shifts"

#### 5.2 Análisis de Streaks y Patterns
**Subtarea**: Detectar winning/losing streaks anómalos
- Analizar probabilidad de streaks observados
- Detectar patterns de alternating results
- Calcular "streak anomaly score"

**Subtarea**: Análisis de patterns cíclicos
- Detectar periodicidad en rendimiento
- Identificar patterns de "scheduled" good/bad games
- Usar time series analysis para pattern detection

## Migraciones de Base de Datos

### Extensión de PlayerMetrics
```sql
ALTER TABLE playermetrics ADD COLUMN avg_pct_top3 FLOAT;
ALTER TABLE playermetrics ADD COLUMN performance_variance FLOAT;
ALTER TABLE playermetrics ADD COLUMN time_control_consistency FLOAT;
ALTER TABLE playermetrics ADD COLUMN opponent_adaptation_score FLOAT;
ALTER TABLE playermetrics ADD COLUMN phase_consistency_opening FLOAT;
ALTER TABLE playermetrics ADD COLUMN phase_consistency_middlegame FLOAT;
ALTER TABLE playermetrics ADD COLUMN phase_consistency_endgame FLOAT;
ALTER TABLE playermetrics ADD COLUMN anomaly_game_count INTEGER;
ALTER TABLE playermetrics ADD COLUMN behavioral_stability_score FLOAT;
```

### Nueva Tabla: GameCorrelations
```sql
CREATE TABLE gamecorrelations (
    id SERIAL PRIMARY KEY,
    player_username VARCHAR(255) REFERENCES playermetrics(username),
    game_id INTEGER REFERENCES game(id),
    time_control VARCHAR(50),
    opponent_rating INTEGER,
    performance_deviation FLOAT, -- deviation from player average
    timing_consistency_score FLOAT,
    phase_performance_opening FLOAT,
    phase_performance_middlegame FLOAT,
    phase_performance_endgame FLOAT,
    anomaly_flags TEXT[], -- array of anomaly types detected
    correlation_score FLOAT, -- overall correlation with player profile
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Nueva Tabla: PlayerTrends
```sql
CREATE TABLE playertrends (
    id SERIAL PRIMARY KEY,
    player_username VARCHAR(255) REFERENCES playermetrics(username),
    analysis_period VARCHAR(50), -- 'weekly', 'monthly', 'quarterly'
    period_start DATE,
    period_end DATE,
    games_analyzed INTEGER,
    avg_performance FLOAT,
    performance_trend FLOAT, -- positive = improving, negative = declining
    consistency_score FLOAT,
    anomaly_rate FLOAT,
    significant_changes TEXT[], -- array of detected changes
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Nueva Tabla: CrossGamePatterns
```sql
CREATE TABLE crossgamepatterns (
    id SERIAL PRIMARY KEY,
    player_username VARCHAR(255) REFERENCES playermetrics(username),
    pattern_type VARCHAR(100), -- 'time_control_inconsistency', 'opponent_adaptation', etc.
    pattern_strength FLOAT, -- how pronounced the pattern is
    games_involved INTEGER[],
    statistical_significance FLOAT,
    description TEXT,
    first_detected DATE,
    last_updated TIMESTAMP DEFAULT NOW()
);
```

## Nuevos Celery Tasks

### compute_cross_game_correlations
**Subtarea**: Task principal para análisis cross-game
- Analizar todas las partidas del jugador
- Calcular correlaciones entre diferentes métricas
- Detectar patterns y anomalías
- Actualizar GameCorrelations table

### analyze_player_trends
**Subtarea**: Task para análisis de trends temporales
- Calcular moving averages y trends
- Detectar cambios significativos en rendimiento
- Actualizar PlayerTrends table
- Generate alerts para cambios súbitos

### detect_behavioral_patterns
**Subtarea**: Task para detección de patrones de comportamiento
- Usar machine learning para pattern recognition
- Clasificar tipos de patterns detectados
- Calcular statistical significance
- Actualizar CrossGamePatterns table

### update_player_profile
**Subtarea**: Task para actualizar perfil completo del jugador
- Integrar todos los análisis cross-game
- Calcular scores de consistencia y adaptación
- Actualizar PlayerMetrics con nuevas métricas
- Generate comprehensive player report

## API Extensions

### Nuevos Endpoints
**Subtarea**: Endpoint `/correlation/player/{username}`
- Análisis completo de correlaciones cross-game
- Breakdown por time control, opponent type, etc.
- Trends y patterns detectados

**Subtarea**: Endpoint `/trends/player/{username}`
- Historical trends de rendimiento
- Predicciones basadas en trends actuales
- Identificación de períodos anómalos

**Subtarea**: Endpoint `/patterns/behavioral/{username}`
- Patrones de comportamiento detectados
- Statistical significance de cada pattern
- Recommendations basadas en patterns

**Subtarea**: Endpoint `/comparison/players`
- Comparar profiles de múltiples jugadores
- Identificar similarities y differences
- Cluster analysis de tipos de jugadores

## Machine Learning Integration

### Pattern Recognition Models
**Subtarea**: Implementar clustering algorithms
- K-means clustering para agrupar partidas similares
- DBSCAN para detectar outliers
- Hierarchical clustering para player profiles

**Subtarea**: Time series analysis
- ARIMA models para trend prediction
- Seasonal decomposition para patterns cíclicos
- Change point detection para behavioral shifts

### Anomaly Detection Models
**Subtarea**: Implementar statistical anomaly detection
- Isolation Forest para outlier detection
- One-class SVM para normal behavior modeling
- Statistical process control para monitoring

**Subtarea**: Ensemble methods para robust detection
- Combinar múltiples algorithms para mejor accuracy
- Voting systems para anomaly classification
- Confidence scoring para detected anomalies

## Testing Strategy

### Statistical Validation
**Subtarea**: Validar statistical significance de correlaciones
- Bootstrap methods para confidence intervals
- Permutation tests para null hypothesis testing
- Multiple testing correction para false discovery rate

**Subtarea**: Cross-validation de pattern detection
- Time series cross-validation para trend analysis
- Stratified sampling para balanced testing
- Out-of-sample validation para generalization

### Performance Testing
**Subtarea**: Test scalability con large player datasets
- Benchmark analysis time para players con miles de partidas
- Memory usage optimization para large correlation matrices
- Parallel processing para multiple player analysis

## Consideraciones de Implementación

### Performance Optimization
- Usar vectorized operations para correlation calculations
- Implement incremental analysis para nuevas partidas
- Cache intermediate results para faster re-analysis
- Parallel processing para multiple player analysis

### Statistical Rigor
- Proper handling de multiple comparisons
- Robust statistical tests para small sample sizes
- Confidence intervals para all reported metrics
- Clear documentation de statistical assumptions

### Privacy y Ethics
- Anonymization options para sensitive analysis
- Clear consent para behavioral pattern analysis
- Transparent reporting de analysis methods
- Ethical guidelines para anomaly reporting

## Cronograma de Implementación

**Semana 1**: Análisis de consistencia temporal y time controls
**Semana 2**: Análisis por tipo de oponente y fases de juego
**Semana 3**: Detección de anomalías y pattern recognition
**Semana 4**: Machine learning integration y statistical validation
**Semana 5**: API endpoints y comprehensive testing
**Semana 6**: Performance optimization y documentation

## Dependencias

- PlayerMetrics model y compute_player_metrics task
- GameMetrics data para correlation analysis
- Statistical libraries (scipy, scikit-learn)
- Time series analysis libraries (statsmodels)
- Machine learning frameworks para pattern recognition
