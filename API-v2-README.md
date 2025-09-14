# Chess Analyzer API v2 - Advanced Integration

## 🚀 Overview

La API v2 de Chess Analyzer aprovecha completamente la nueva arquitectura modular y las optimizaciones de performance (4.6x speedup) mientras mantiene **compatibilidad completa** con la UI actual.

## ✨ Nuevas Funcionalidades

### 🎯 **Performance Optimizada**
- **4.6x speedup** en análisis utilizando optimizaciones NumPy
- **Batch processing** para análisis masivos eficientes
- **Adaptive rate limiting** que escala con las mejoras de performance

### 🔗 **API Avanzada**
- **GraphQL endpoint** para queries flexibles y eficientes
- **WebSocket streaming** para actualizaciones en tiempo real
- **Server-Sent Events** para streaming de progreso
- **Advanced aggregations** con estadísticas complejas

### 📊 **Monitoreo y Observabilidad**
- **Performance metrics** integrados en cada respuesta
- **Health checks** avanzados con estado de optimizaciones
- **Real-time monitoring** de efectividad de optimizaciones

## 🔧 Endpoints Principales

### Base URL
```
https://api.chessanalyzer.com/api/v2
```

### 👤 Players (Enhanced)
```http
# Análisis mejorado con configuraciones avanzadas
POST /api/v2/players/{username}/analyze
{
  "include_openings": true,
  "include_time_analysis": true,
  "include_longitudinal": true,
  "include_fairness": false,
  "priority": "high",
  "enable_streaming": true
}

# Status con métricas de performance
GET /api/v2/players/{username}/status
# Respuesta incluye: speedup_factor, performance_metrics, estimated_completion

# Insights avanzados
GET /api/v2/players/{username}/insights?include_peer_comparison=true&include_trends=true
```

### 🎮 Games (Enhanced)
```http
# Análisis PGN optimizado
POST /api/v2/games/analyze-pgn
{
  "pgn_content": "1. e4 e5 2. Nf3...",
  "depth": 20,
  "include_deep_insights": true
}
# Respuesta incluye: processing_time_ms, speedup_factor, pattern_recognition
```

### 📈 Analysis (Enhanced)
```http
# Métricas de juego con performance insights
GET /api/v2/analysis/metrics/game/{game_id}
# Incluye: critical_moments, improvement_suggestions, speedup_factor

# Métricas de jugador con análisis longitudinal
GET /api/v2/analysis/metrics/player/{username}?include_peer_benchmarking=true

# Análisis en lotes
POST /api/v2/analysis/batch
{
  "game_ids": [1, 2, 3, 4, 5],
  "parallel_workers": 8,
  "priority": "high"
}

# Resumen de performance del sistema
GET /api/v2/analysis/performance/summary
# Retorna: speedup_factors, optimization_status, performance_grade
```

### 🔄 Batch Operations
```http
# Análisis masivo de juegos
POST /api/v2/batch/games
{
  "game_ids": [1, 2, 3, ..., 1000],
  "parallel_workers": 16,
  "include_deep_analysis": false,
  "priority": "normal"
}

# Status de operación batch
GET /api/v2/batch/{batch_id}/status
# Incluye: performance_metrics, items_per_minute, estimated_remaining_time

# Resultados de batch
GET /api/v2/batch/{batch_id}/results
# Incluye: performance_summary, speedup_factors, success_rate
```

### 📡 Streaming & Real-time
```http
# WebSocket para actualizaciones en tiempo real
WS /api/v2/streaming/ws/{client_id}
# Mensajes: subscribe, unsubscribe, progress_updates

# Server-Sent Events para progreso de análisis
GET /api/v2/streaming/players/{username}
# Stream: progress_percentage, games_analyzed, speedup_factor, estimated_completion

# Performance metrics en vivo
GET /api/v2/streaming/performance/live
# Stream: current_analysis_rate, avg_speedup_factor, system_health
```

### 📊 Advanced Aggregations
```http
# Leaderboard de performance
POST /api/v2/aggregates/leaderboard
{
  "time_range": "30d",
  "rating_range": {"min": 1500, "max": 2500},
  "min_games": 10,
  "use_optimized_engine": true
}

# Estadísticas de aperturas
POST /api/v2/aggregates/opening-stats
{
  "time_range": "90d",
  "min_games_per_opening": 20,
  "use_optimized_engine": true
}

# Análisis de tendencias globales
GET /api/v2/aggregates/performance-trends?time_range=30d&granularity=daily

# Análisis de cohortes
POST /api/v2/aggregates/cohort-analysis
{
  "cohort_definition": "rating_range",
  "analysis_period": "90d"
}
```

### 🔍 GraphQL
```http
# GraphQL endpoint principal
POST /api/v2/graphql
# GraphiQL playground disponible en GET /api/v2/graphql

# Información del schema
GET /api/v2/graphql/schema
```

## 💡 Ejemplos de Uso

### 1. Análisis Rápido con Performance Tracking
```javascript
// Análisis optimizado de jugador
const response = await fetch('/api/v2/players/magnus_carlsen/analyze', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    include_openings: true,
    include_time_analysis: true,
    priority: 'high',
    enable_streaming: true
  })
});

const result = await response.json();
console.log(`Analysis started with ${result.speedup_factor}x speedup`);
console.log(`Estimated completion: ${result.estimated_completion}`);
console.log(`Stream URL: ${result.analysis_url}`);
```

### 2. Streaming en Tiempo Real
```javascript
// Server-Sent Events para progreso
const eventSource = new EventSource('/api/v2/streaming/players/magnus_carlsen');

eventSource.onmessage = function(event) {
  const data = JSON.parse(event.data);

  if (data.type === 'progress_update') {
    console.log(`Progress: ${data.data.progress_percentage}%`);
    console.log(`Speedup: ${data.data.speedup_factor}x`);
    console.log(`Rate: ${data.data.analysis_rate_per_minute} games/min`);
  }

  if (data.type === 'analysis_completed') {
    console.log('Analysis completed!');
    console.log(`Results: ${data.data.results_url}`);
  }
};
```

### 3. WebSocket para Interacción Avanzada
```javascript
// WebSocket connection
const ws = new WebSocket('ws://localhost:8000/api/v2/streaming/ws/client123');

ws.onopen = function() {
  // Subscribe to player analysis
  ws.send(JSON.stringify({
    type: 'subscribe',
    payload: {
      resource_type: 'players',
      resource_id: 'magnus_carlsen'
    }
  }));
};

ws.onmessage = function(event) {
  const message = JSON.parse(event.data);
  console.log('Real-time update:', message);
};
```

### 4. GraphQL Query Compleja
```graphql
query GetPlayerWithInsights($username: String!) {
  player(username: $username) {
    username
    status
    metrics {
      totalGames
      overallAccuracy
      averageAcpl
      consistencyScore
      performanceInfo {
        speedupFactor
        optimizationUsed
        processingTimeMs
      }
      strengthProgression
      openingRepertoire
    }
  }

  performanceLeaderboard(limit: 10, metric: "accuracy") {
    username
    overallAccuracy
    totalGames
    performanceInfo {
      speedupFactor
    }
  }
}
```

### 5. Batch Processing Avanzado
```javascript
// Análisis masivo con tracking
const batchResponse = await fetch('/api/v2/batch/games', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    game_ids: Array.from({length: 500}, (_, i) => i + 1),
    parallel_workers: 12,
    priority: 'high'
  })
});

const batch = await batchResponse.json();
console.log(`Batch started: ${batch.batch_id}`);
console.log(`Expected ${batch.performance_estimate.expected_speedup}x speedup`);

// Monitoreo de progreso
const statusInterval = setInterval(async () => {
  const statusResponse = await fetch(`/api/v2/batch/${batch.batch_id}/status`);
  const status = await statusResponse.json();

  console.log(`Progress: ${status.progress}%`);
  console.log(`Performance: ${status.performance_metrics.avg_speedup_factor}x`);
  console.log(`Rate: ${status.performance_metrics.items_per_minute} items/min`);

  if (status.status === 'completed') {
    clearInterval(statusInterval);
    // Get results
    const results = await fetch(`/api/v2/batch/${batch.batch_id}/results`);
    console.log('Batch completed!', await results.json());
  }
}, 5000);
```

## 🔒 Rate Limiting Inteligente

La API v2 incluye **adaptive rate limiting** que escala automáticamente basado en las optimizaciones de performance:

### Límites Base (multiplicados por performance gains)
- **General requests**: 60/min → 276/min (4.6x)
- **Analysis requests**: 10/min → 46/min (4.6x)
- **Batch requests**: 5/hour → 23/hour (4.6x)

### Headers de Rate Limiting
```http
X-RateLimit-Limit: 276
X-RateLimit-Remaining: 250
X-RateLimit-Reset: 1694722800
X-RateLimit-Type: adaptive
X-Performance-Multiplier: 4.6
```

### Factores Adaptativos
- **User tier**: free (1x), premium (3x), enterprise (10x)
- **System health**: Auto-reduce si CPU/memoria alta
- **Speedup bonus**: Bonus si sistema supera 4.6x baseline

## ⚡ Performance Metrics

Todas las respuestas incluyen métricas de performance:

```json
{
  "data": { ... },
  "performance_info": {
    "processing_time_ms": 45.2,
    "speedup_factor": 4.8,
    "optimization_used": true,
    "cache_hit": false,
    "numpy_operations_count": 1250
  }
}
```

## 📈 Health Checks Avanzados

```http
GET /api/v2/health
```

```json
{
  "status": "healthy",
  "timestamp": "2025-09-14T10:30:00Z",
  "version": "2.0.0",
  "api_version": "v2",
  "components": {
    "database": {"status": "healthy"},
    "redis": {"status": "healthy"},
    "optimizations": {
      "status": "active",
      "numpy_optimizations": {
        "quality_module": {"active": true, "speedup_factor": 6.7},
        "timing_module": {"active": true, "speedup_factor": 2.4},
        "longitudinal_module": {"active": true, "speedup_factor": 2.4}
      },
      "overall_speedup": 4.6,
      "performance_grade": "excellent"
    }
  }
}
```

## 🔄 Backward Compatibility

**Garantía**: La API v2 **NO** rompe la funcionalidad de la UI actual.

- **API v1** sigue disponible sin cambios en `/api/v1/*`
- **Endpoints legacy** mantenidos para compatibilidad
- **Schemas compatibles** donde sea aplicable
- **Wrappers de compatibilidad** para funciones críticas

## 📋 Migration Guide

### Para desarrolladores que quieran migrar de v1 a v2:

1. **URLs**: Cambiar `/api/v1/` por `/api/v2/`
2. **Enhanced responses**: Aprovechar nuevos campos como `performance_info`
3. **Streaming**: Implementar WebSocket/SSE para tiempo real
4. **GraphQL**: Considerar migrar queries complejas a GraphQL
5. **Batch processing**: Usar endpoints batch para operaciones masivas

### Ejemplo de migración:
```javascript
// v1 (sigue funcionando)
const player = await fetch('/api/v1/players/username').then(r => r.json());

// v2 (enhanced)
const player = await fetch('/api/v2/players/username/status').then(r => r.json());
// Ahora incluye: progress, performance_metrics, estimated_completion, etc.
```

## 🛠️ Development & Testing

### Local Development
```bash
# Instalar dependencias adicionales para v2
pip install strawberry-graphql[fastapi] redis websockets

# Ejecutar con v2 habilitado
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Endpoints de Testing
- **OpenAPI Docs**: `http://localhost:8000/docs` (incluye v2)
- **GraphQL Playground**: `http://localhost:8000/api/v2/graphql`
- **Health Check**: `http://localhost:8000/api/v2/health`

## 📚 Documentation

### Enlaces Útiles
- **OpenAPI Schema**: `/api/v2/openapi.json`
- **GraphQL Schema**: `/api/v2/graphql/schema`
- **Performance Docs**: Ver `PERFORMANCE_REPORT.md`
- **CI/CD Docs**: Ver `CI-CD-README.md`

## 🎯 Success Metrics

**Resultados Alcanzados:**
- ✅ **API v2 completa** con todas las funcionalidades avanzadas
- ✅ **4.6x performance baseline** aprovechado en todos los endpoints
- ✅ **Backward compatibility** mantenida al 100%
- ✅ **Real-time capabilities** con WebSocket y SSE
- ✅ **GraphQL integration** para queries eficientes
- ✅ **Batch processing** para análisis masivos
- ✅ **Adaptive rate limiting** que escala automáticamente
- ✅ **Comprehensive monitoring** integrado

---

**Status**: ✅ **PRODUCTION READY** - API v2 completamente implementada y operational con backward compatibility garantizada.