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
*Objetivo*: Detectar cambios graduales o súbitos en el rendimiento del jugador que podrían indicar uso de asistencia externa o cambios en habilidad natural.
*Enfoque*: Calcular moving averages de pct_top3, acl por ventanas temporales (semanal, mensual), usar statistical process control para detectar trends significativos, y aplicar change point detection algorithms para identificar momentos específicos de cambio.
- Calcular moving averages de pct_top3, acl por períodos
- Detectar trends de mejora/deterioro over time
- Identificar períodos de rendimiento anómalo

**Subtarea**: Análisis de estabilidad de rendimiento
*Objetivo*: Identificar jugadores cuyo rendimiento es artificialmente consistente, lo cual puede indicar uso de software de asistencia que mantiene un nivel constante.
*Enfoque*: Calcular coeficiente de variación para métricas clave, comparar con distribuciones esperadas para jugadores de similar rating, y usar statistical tests para detectar consistency anómala que exceda variabilidad humana normal.
- Calcular variance de métricas key across partidas
- Detectar jugadores con rendimiento "demasiado estable"
- Identificar cambios súbitos en nivel de juego

#### 1.2 Correlación con Factores Temporales
**Subtarea**: Análisis por hora del día/día de semana
*Objetivo*: Detectar patrones de rendimiento que correlacionen con factores temporales, identificando posible uso de asistencia en horarios específicos o fatiga natural.
*Enfoque*: Extraer features temporales de game timestamps, aplicar análisis de correlación entre hora/día y métricas de rendimiento, y usar clustering para identificar "optimal windows" que podrían indicar uso programado de asistencia.
- Correlacionar rendimiento con timestamp de partidas
- Detectar patrones de rendimiento por horario
- Identificar "optimal playing times" por jugador

**Subtarea**: Análisis de fatiga y sesiones de juego
*Objetivo*: Identificar jugadores que no muestran fatiga natural esperada durante sesiones largas, lo cual puede indicar asistencia automatizada.
*Enfoque*: Agrupar partidas en sesiones basadas en timestamps, calcular decline rate de métricas durante sesiones, y comparar con curvas de fatiga esperadas para detectar ausencia anómala de deterioro por cansancio.
- Detectar deterioro de rendimiento en sesiones largas
- Calcular "fatigue coefficient" basado en partidas consecutivas
- Identificar patrones de recuperación entre sesiones

### 2. Análisis por Control de Tiempo

#### 2.1 Rendimiento Across Time Controls
**Subtarea**: Implementar métricas específicas por time control
*Objetivo*: Detectar jugadores que muestran rendimiento inconsistente entre diferentes time controls, especialmente aquellos que juegan mejor en controles más rápidos (indicativo de asistencia).
*Enfoque*: Segmentar análisis por categorías de time control, calcular rating performance relativo para cada categoría, y usar statistical tests para detectar diferencias significativas que excedan variabilidad esperada.
- Separar análisis por blitz, rapid, classical
- Calcular relative strength por control de tiempo
- Detectar inconsistencias sospechosas entre controles

**Subtarea**: Análisis de adaptación a time controls
*Objetivo*: Identificar jugadores que no adaptan su gestión de tiempo apropiadamente a diferentes controles, lo cual puede indicar dependencia de asistencia externa.
*Enfoque*: Analizar distribución de move_times por time control, calcular efficiency ratios (tiempo usado vs tiempo disponible), y detectar patrones de timing que no se adaptan naturalmente a las restricciones temporales.
- Medir eficiencia de uso de tiempo por control
- Detectar jugadores que no adaptan estilo a tiempo disponible
- Calcular "time management score" por control

#### 2.2 Correlación de Timing Patterns
**Subtarea**: Comparar patrones de timing across controles
*Objetivo*: Detectar jugadores cuyo timing signature se mantiene artificialmente similar across diferentes time controls, indicando posible automatización.
*Enfoque*: Extraer timing fingerprints por control de tiempo, calcular similarity scores entre fingerprints, y detectar consistency anómala que no refleje adaptación natural a diferentes presiones temporales.
- Analizar si timing patterns se mantienen consistentes
- Detectar cambios artificiales en timing behavior
- Calcular "timing consistency score" cross-control

### 3. Análisis por Tipo de Oponente

#### 3.1 Rendimiento vs Rating del Oponente
**Subtarea**: Implementar análisis de rendimiento relativo
*Objetivo*: Detectar jugadores que consistentemente over-perform contra oponentes más fuertes o under-perform contra más débiles, indicando posible manipulación de rating o uso selectivo de asistencia.
*Enfoque*: Usar modelos de expected performance basados en rating differences, calcular deviations sistemáticas, y aplicar statistical tests para detectar patterns que excedan variabilidad natural del rendimiento.
- Calcular expected vs actual performance por rating gap
- Detectar over/under-performance sistemático
- Identificar "rating manipulation" patterns

**Subtarea**: Análisis de adaptación a nivel de oponente
*Objetivo*: Identificar jugadores que no adaptan su estilo de juego apropiadamente al nivel del oponente, o que muestran patterns artificiales de sandbagging.
*Enfoque*: Analizar métricas de agresividad, risk-taking, y complexity por rating del oponente, detectar ausencia de adaptación natural, y calcular scores que capturen flexibility apropiada en approach táctico.
- Medir cambio en estilo de juego vs oponentes fuertes/débiles
- Detectar "sandbagging" o performance artificial
- Calcular "opponent adaptation score"

#### 3.2 Patrones de Juego vs Estilos de Oponente
**Subtarea**: Clasificar estilos de oponentes
*Objetivo*: Evaluar si el jugador adapta su estilo apropiadamente a diferentes tipos de oponentes, detectando rigidez que podría indicar dependencia de asistencia programática.
*Enfoque*: Usar clustering para categorizar oponentes por style metrics (agresivo, posicional, táctico), analizar adaptation del jugador a cada cluster, y detectar lack of flexibility que sugiera inability to adapt without external assistance.
- Categorizar oponentes por métricas de juego
- Analizar adaptación del jugador a diferentes estilos
- Detectar rigidez o flexibilidad en approach

### 4. Análisis de Fases de Juego

#### 4.1 Consistencia por Fase de Partida
**Subtarea**: Implementar análisis por opening/middlegame/endgame
*Objetivo*: Detectar jugadores que muestran strength inconsistente entre fases de juego, especialmente aquellos que mejoran dramáticamente en fases complejas donde engines proporcionan mayor ventaja.
*Enfoque*: Segmentar move analysis por game phase usando move numbers y material count, calcular phase-specific performance metrics, y detectar anomalous improvements en fases donde computer assistance es más valuable.
- Separar métricas por fase de juego
- Detectar fortalezas/debilidades específicas por fase
- Calcular consistency scores por fase

**Subtarea**: Análisis de transiciones entre fases
*Objetivo*: Identificar jugadores que muestran cambios súbitos en performance durante transiciones de fase, lo cual puede indicar activation/deactivation de asistencia externa.
*Enfoque*: Detectar transition points usando game state analysis, medir performance changes durante transitions, y identificar patterns de sudden improvement/decline que no correspondan a natural game flow.
- Medir performance en transiciones críticas
- Detectar patterns de collapse en fases específicas
- Identificar "phase specialization" patterns

#### 4.2 Correlación de Errores por Fase
**Subtarea**: Análisis de error patterns por fase
*Objetivo*: Detectar jugadores cuyo error rate no correlaciona naturalmente con la complejidad típica de cada fase de juego, indicando posible asistencia selectiva.
*Enfoque*: Analizar distribución de cp_loss por game phase, comparar con expected error patterns para el rating level, y detectar anomalous accuracy en fases donde human players típicamente cometen más errores.
- Correlacionar cp_loss con fase de juego
- Detectar systematic weaknesses por fase
- Calcular "phase reliability score"

### 5. Detección de Anomalías Cross-Game

#### 5.1 Identificación de Outlier Games
**Subtarea**: Implementar detección de partidas anómalas
*Objetivo*: Identificar partidas individuales que se desvían significativamente del perfil normal del jugador, indicando posible uso ocasional de asistencia.
*Enfoque*: Aplicar multivariate outlier detection usando métricas de performance, timing, y style, clasificar anomalies por type y severity, y calcular composite anomaly scores que consideren múltiples dimensions simultaneously.
- Usar statistical methods para detectar outliers
- Clasificar tipos de anomalías (performance, timing, style)
- Calcular "game anomaly score"

**Subtarea**: Clustering de patrones de juego
*Objetivo*: Detectar cambios fundamentales en el estilo de juego del jugador que podrían indicar adoption de asistencia externa o cambios en methodology.
*Enfoque*: Usar unsupervised clustering (K-means, DBSCAN) para agrupar partidas por similarity, track cluster membership over time, y detectar sudden shifts en behavioral patterns que no correspondan a natural skill evolution.
- Agrupar partidas por similarity en métricas
- Detectar cambios en clusters over time
- Identificar "behavioral shifts"

#### 5.2 Análisis de Streaks y Patterns
**Subtarea**: Detectar winning/losing streaks anómalos
*Objetivo*: Identificar sequences de resultados que son estadísticamente improbables dado el rating y historical performance del jugador, indicando posible manipulation.
*Enfoque*: Usar probability theory para calcular likelihood de observed streaks, aplicar runs tests para detectar non-random patterns, y calcular anomaly scores basados en deviation from expected win/loss distributions.
- Analizar probabilidad de streaks observados
- Detectar patterns de alternating results
- Calcular "streak anomaly score"

**Subtarea**: Análisis de patterns cíclicos
*Objetivo*: Detectar periodicidad artificial en el rendimiento que podría indicar uso programado de asistencia o manipulation deliberada de results.
*Enfoque*: Aplicar Fourier analysis y autocorrelation para detectar periodic patterns, usar time series decomposition para identificar cyclical components, y detectar regularity que exceda natural variability en human performance.
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
*Objetivo*: Implementar el pipeline central que procese todas las partidas de un jugador para detectar correlaciones y patterns sospechosos across múltiples games.
*Enfoque*: Crear Celery task que fetch all player games, compute correlation matrices entre métricas, apply statistical tests para significance, y store results en GameCorrelations table con proper error handling y progress tracking.
- Analizar todas las partidas del jugador
- Calcular correlaciones entre diferentes métricas
- Detectar patterns y anomalías
- Actualizar GameCorrelations table

### analyze_player_trends
**Subtarea**: Task para análisis de trends temporales
*Objetivo*: Detectar cambios temporales en el rendimiento del jugador que podrían indicar adoption o discontinuation de asistencia externa.
*Enfoque*: Implementar time series analysis con moving averages, trend detection usando regression analysis, change point detection algorithms, y automated alerting system para significant performance shifts.
- Calcular moving averages y trends
- Detectar cambios significativos en rendimiento
- Actualizar PlayerTrends table
- Generate alerts para cambios súbitos

### detect_behavioral_patterns
**Subtarea**: Task para detección de patrones de comportamiento
*Objetivo*: Aplicar machine learning para identificar patterns complejos de comportamiento que no son detectables con análisis estadístico simple.
*Enfoque*: Implementar ensemble de ML algorithms (clustering, anomaly detection, classification), train models en historical data con ground truth, y deploy models para real-time pattern detection con confidence scoring.
- Usar machine learning para pattern recognition
- Clasificar tipos de patterns detectados
- Calcular statistical significance
- Actualizar CrossGamePatterns table

### update_player_profile
**Subtarea**: Task para actualizar perfil completo del jugador
*Objetivo*: Consolidar todos los análisis cross-game en un perfil comprehensivo que proporcione assessment holístico del jugador.
*Enfoque*: Agregar results de todos los analysis tasks, compute weighted composite scores, update PlayerMetrics con nuevas métricas, y generate detailed reports con actionable insights y recommendations.
- Integrar todos los análisis cross-game
- Calcular scores de consistencia y adaptación
- Actualizar PlayerMetrics con nuevas métricas
- Generate comprehensive player report

## API Extensions

### Nuevos Endpoints
**Subtarea**: Endpoint `/correlation/player/{username}`
*Objetivo*: Proporcionar acceso programático a análisis comprehensivo de correlaciones cross-game para integration con frontend y external systems.
*Enfoque*: Implementar REST endpoint que return structured data con correlation matrices, statistical significance tests, breakdown por dimensions (time control, opponent), y visualization-ready data formats.
- Análisis completo de correlaciones cross-game
- Breakdown por time control, opponent type, etc.
- Trends y patterns detectados

**Subtarea**: Endpoint `/trends/player/{username}`
*Objetivo*: Exponer análisis temporal del jugador con predictive capabilities para anticipar future performance y detectar anomalies.
*Enfoque*: Crear endpoint que return time series data, trend analysis results, forecasting models output, y anomaly detection results con confidence intervals y statistical significance measures.
- Historical trends de rendimiento
- Predicciones basadas en trends actuales
- Identificación de períodos anómalos

**Subtarea**: Endpoint `/patterns/behavioral/{username}`
*Objetivo*: Proporcionar insights detallados sobre behavioral patterns detectados con actionable recommendations para investigation o monitoring.
*Enfoque*: Implementar endpoint que return pattern classification results, statistical significance scores, interpretability explanations, y automated recommendations basadas en pattern severity y type.
- Patrones de comportamiento detectados
- Statistical significance de cada pattern
- Recommendations basadas en patterns

**Subtarea**: Endpoint `/comparison/players`
*Objetivo*: Facilitar comparative analysis entre jugadores para identificar similar behavioral patterns y detect potential coordinated cheating or account sharing.
*Enfoque*: Crear endpoint que accept multiple usernames, compute similarity metrics entre player profiles, perform cluster analysis para group similar players, y return comparative visualizations y insights.
- Comparar profiles de múltiples jugadores
- Identificar similarities y differences
- Cluster analysis de tipos de jugadores

## Machine Learning Integration

### Pattern Recognition Models
**Subtarea**: Implementar clustering algorithms
*Objetivo*: Agrupar partidas y jugadores por similarity para detectar patterns anómalos y identify behavioral clusters que indiquen cheating coordination.
*Enfoque*: Implementar multiple clustering algorithms con feature engineering apropiado, optimize hyperparameters usando validation metrics, y create interpretable cluster profiles con statistical characterization de cada group.
- K-means clustering para agrupar partidas similares
- DBSCAN para detectar outliers
- Hierarchical clustering para player profiles

**Subtarea**: Time series analysis
*Objetivo*: Aplicar técnicas avanzadas de time series para detectar temporal patterns, predict future behavior, y identify sudden changes que indiquen cheating adoption.
*Enfoque*: Implementar ARIMA/SARIMA models con proper model selection, seasonal decomposition usando STL o X-13, y change point detection usando PELT o binary segmentation algorithms con statistical validation.
- ARIMA models para trend prediction
- Seasonal decomposition para patterns cíclicos
- Change point detection para behavioral shifts

### Anomaly Detection Models
**Subtarea**: Implementar statistical anomaly detection
*Objetivo*: Detectar partidas y behaviors que se desvían significativamente del perfil normal del jugador usando multiple statistical approaches.
*Enfoque*: Implementar ensemble de anomaly detection algorithms, tune parameters usando cross-validation con labeled data, y combine results usando voting o stacking para robust anomaly scoring.
- Isolation Forest para outlier detection
- One-class SVM para normal behavior modeling
- Statistical process control para monitoring

**Subtarea**: Ensemble methods para robust detection
*Objetivo*: Mejorar accuracy y robustness de anomaly detection combinando múltiples algorithms y proporcionando confidence measures.
*Enfoque*: Implementar ensemble methods (bagging, boosting, stacking), design voting systems con weighted contributions basados en algorithm performance, y develop confidence scoring que considere agreement entre multiple detectors.
- Combinar múltiples algorithms para mejor accuracy
- Voting systems para anomaly classification
- Confidence scoring para detected anomalies

## Testing Strategy

### Statistical Validation
**Subtarea**: Validar statistical significance de correlaciones
*Objetivo*: Asegurar que las correlaciones detectadas son estadísticamente significativas y no resultado de multiple testing o chance findings.
*Enfoque*: Implementar bootstrap resampling para robust confidence intervals, permutation tests para null hypothesis validation, y apply Benjamini-Hochberg correction para control false discovery rate en multiple comparisons.
- Bootstrap methods para confidence intervals
- Permutation tests para null hypothesis testing
- Multiple testing correction para false discovery rate

**Subtarea**: Cross-validation de pattern detection
*Objetivo*: Validar que los patterns detectados generalizan apropiadamente y no son resultado de overfitting a specific data characteristics.
*Enfoque*: Implementar time series cross-validation con proper temporal splits, stratified sampling para maintain class balance, y rigorous out-of-sample testing usando holdout datasets con known ground truth.
- Time series cross-validation para trend analysis
- Stratified sampling para balanced testing
- Out-of-sample validation para generalization

### Performance Testing
**Subtarea**: Test scalability con large player datasets
*Objetivo*: Asegurar que el sistema puede handle players con thousands de partidas sin performance degradation o memory issues.
*Enfoque*: Implement comprehensive benchmarking con realistic data volumes, profile memory usage y optimize data structures, y design parallel processing architecture que scale horizontally con player volume.
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
