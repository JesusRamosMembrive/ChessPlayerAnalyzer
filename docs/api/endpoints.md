# API Endpoints - ChessPlayerAnalyzer

**Audiencia:** Desarrolladores, Integradores
**Última actualización:** 2025-09-13
**Estado:** Completo

Documentación completa de los endpoints REST de la API v1 de ChessPlayerAnalyzer.

## <¯ Objetivo

La API FastAPI proporciona endpoints para:
- **Análisis de jugadores** - Procesamiento completo de historial de partidas
- **Análisis de partidas** - Evaluación individual de archivos PGN
- **Consulta de métricas** - Acceso a resultados de análisis
- **Monitoreo de tareas** - Control de procesamiento asíncrono
- **Estado del sistema** - Health checks y diagnostics

## =Ë Base URL y Versionado

**Base URL:** `http://localhost:8000`
**API v1:** `/api/v1/`
**Documentación:** `/api/v1/docs` (Swagger UI)

## =€ Endpoints Principales

### 1. **Health Check**

#### `GET /health`
Verificación básica de estado del sistema.

**Response:**
```json
{
  "status": "healthy",
  "version": "v1",
  "timestamp": "2025-09-13T10:30:00Z"
}
```

#### `GET /api/v1/health`
Health check detallado con estado de servicios.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-09-13T10:30:00Z",
  "database": "connected",
  "redis": "connected",
  "celery": "active_workers_3",
  "system": {
    "cpu_percent": 15.4,
    "memory_percent": 42.1,
    "disk_percent": 68.5
  }
}
```

### 2. **Players - Análisis de Jugadores**

#### `GET /api/v1/players/{username}`
Obtiene el estado actual del análisis de un jugador.

**Parameters:**
- `username` (path) - Nombre de usuario en Chess.com

**Response:**
```json
{
  "username": "hikaru",
  "status": "ready",
  "progress": 100,
  "total_games": 150,
  "done_games": 150,
  "requested_at": "2025-09-13T08:00:00Z",
  "finished_at": "2025-09-13T10:15:00Z",
  "error": null,
  "last_task_id": "abc123-456def-789ghi"
}
```

**Status Values:**
- `not_analyzed` - Jugador no analizado aún
- `pending` - Análisis en progreso
- `ready` - Análisis completado
- `error` - Error en el procesamiento

#### `POST /api/v1/players/{username}`
Inicia el análisis completo de un jugador.

**Parameters:**
- `username` (path) - Nombre de usuario en Chess.com
- `months` (query, default=12) - Meses hacia atrás a analizar

**Response (202 Accepted):**
```json
{
  "status": "analysis_started",
  "username": "hikaru",
  "task_id": "abc123-456def-789ghi",
  "progress": 0,
  "estimated_games": 150,
  "message": "Analysis started. Use GET /players/{username} to check progress."
}
```

#### `POST /api/v1/players/{username}/refresh`
Fuerza un nuevo análisis completo del jugador.

**Response:**
```json
{
  "status": "refresh_started",
  "username": "hikaru",
  "task_id": "new123-456def-789ghi",
  "message": "Previous analysis cleared. Starting fresh analysis."
}
```

#### `DELETE /api/v1/players/{username}`
Elimina completamente un jugador y todos sus análisis.

**Response:**
```json
{
  "status": "deleted",
  "username": "hikaru",
  "deleted_games": 150,
  "deleted_analyses": 150,
  "message": "Player and all associated data deleted successfully."
}
```

### 3. **Games - Análisis de Partidas**

#### `GET /api/v1/games/{game_id}`
Obtiene los detalles de una partida analizada.

**Parameters:**
- `game_id` (path) - ID único de la partida

**Response:**
```json
{
  "id": 12345,
  "created_at": "2025-09-13T09:30:00Z",
  "pgn": "[Event \"Chess.com\"]\n[White \"player1\"]...",
  "white_username": "player1",
  "black_username": "player2",
  "eco_code": "B90",
  "opening_key": "sicilian_najdorf",
  "moves": [
    {
      "move_number": 1,
      "played": "e4",
      "best": "e4",
      "best_rank": 1,
      "cp_loss": 0,
      "eval_before": 25,
      "eval_after": 25
    }
  ]
}
```

#### `POST /api/v1/games/analyze`
Analiza una partida individual desde PGN.

**Request Body:**
```json
{
  "pgn": "[Event \"Test Game\"]\n[White \"Player1\"]...",
  "priority": 5
}
```

**Response (202 Accepted):**
```json
{
  "task_id": "game123-456def-789ghi",
  "status": "queued",
  "message": "Game analysis queued. Check status with GET /tasks/{task_id}",
  "estimated_time_minutes": 3
}
```

### 4. **Analysis - Métricas y Resultados**

#### `GET /api/v1/analysis/metrics/game/{game_id}`
Obtiene métricas detalladas de una partida.

**Response:**
```json
{
  "game_id": 12345,
  "quality": {
    "acpl": 23.5,
    "match_rate": 0.87,
    "weighted_match_rate": 0.82,
    "ipr": 2156,
    "ipr_z_score": 1.34,
    "precision_burst_count": 3
  },
  "timing": {
    "mean_move_time": 12.5,
    "time_variance": 28.1,
    "time_complexity_corr": 0.65,
    "lag_spike_count": 2,
    "uniformity_score": 0.89,
    "clutch_accuracy_diff": -15.4
  },
  "opening": {
    "opening_entropy": 2.34,
    "novelty_depth": 12,
    "second_choice_rate": 0.18,
    "opening_breadth": 5
  },
  "endgame": {
    "conversion_efficiency": 32,
    "tb_match_rate": 0.76,
    "dtz_deviation": 1.4
  },
  "advanced": {
    "anomaly_score": 0.12,
    "suspicion_score": 0.05,
    "bayesian_posterior": 0.03
  }
}
```

#### `GET /api/v1/analysis/metrics/player/{username}`
Obtiene métricas agregadas de un jugador.

**Response:**
```json
{
  "username": "hikaru",
  "analysis_date": "2025-09-13T10:15:00Z",
  "total_games": 150,
  "time_period_months": 12,
  "overall_metrics": {
    "avg_acpl": 18.2,
    "avg_match_rate": 0.91,
    "avg_ipr": 2680,
    "consistency_score": 0.94
  },
  "performance_trends": {
    "trend_acpl": -0.5,
    "trend_match_rate": 0.02,
    "improvement_rate": 0.15
  },
  "risk_assessment": {
    "risk_score": 0.03,
    "confidence_interval": [0.01, 0.08],
    "classification": "low_risk"
  },
  "time_patterns": {
    "mean_move_time": 15.3,
    "time_variance": 45.2,
    "uniformity_score": 0.82
  },
  "opening_repertoire": {
    "breadth_score": 42,
    "novelty_frequency": 0.23,
    "preparation_depth": 16.5
  }
}
```

### 5. **Tasks - Monitoreo de Celery**

#### `GET /api/v1/tasks/{task_id}`
Obtiene el estado de una tarea Celery.

**Response:**
```json
{
  "task_id": "abc123-456def-789ghi",
  "state": "SUCCESS",
  "status": "Analysis completed successfully",
  "progress": 100,
  "result": {
    "games_processed": 150,
    "analysis_time_seconds": 847,
    "final_status": "ready"
  },
  "started_at": "2025-09-13T08:00:00Z",
  "completed_at": "2025-09-13T10:15:00Z"
}
```

#### `POST /api/v1/tasks/{task_id}/cancel`
Cancela una tarea en ejecución.

**Response:**
```json
{
  "task_id": "abc123-456def-789ghi",
  "status": "cancelled",
  "message": "Task cancellation requested. May take a few seconds to take effect."
}
```

### 6. **Streaming - Actualizaciones en Tiempo Real**

#### `GET /api/v1/stream/{username}`
Stream SSE (Server-Sent Events) para actualizaciones de progreso.

**Headers Required:**
```
Accept: text/event-stream
Cache-Control: no-cache
```

**Stream Events:**
```
event: progress
data: {"username": "hikaru", "progress": 45, "games_done": 67, "total_games": 150}

event: status_change
data: {"username": "hikaru", "old_status": "pending", "new_status": "ready"}

event: error
data: {"username": "hikaru", "error": "Chess.com API rate limit exceeded"}
```

## =' Parámetros Globales

### Query Parameters Comunes
- `months` (int, 1-36) - Rango temporal para análisis de jugadores
- `priority` (int, 0-9) - Prioridad de cola Celery (0=alta, 9=baja)
- `force_refresh` (bool) - Forzar nuevo análisis ignorando cache

### Headers Recomendados
```
Content-Type: application/json
Accept: application/json
User-Agent: YourApp/1.0
X-Request-ID: unique-request-id
```

##   Rate Limiting

**Límites por defecto:**
- **100 requests/minuto** por IP
- **Endpoints de análisis:** 10 requests/minuto adicional
- **Headers de respuesta:**
  ```
  X-RateLimit-Limit: 100
  X-RateLimit-Remaining: 85
  X-RateLimit-Reset: 1694606400
  ```

## = Códigos de Error

### HTTP Status Codes
- `200` - Success
- `202` - Accepted (análisis encolado)
- `400` - Bad Request (PGN inválido, parámetros incorrectos)
- `404` - Not Found (jugador/partida no encontrada)
- `422` - Validation Error (schema inválido)
- `429` - Rate Limit Exceeded
- `500` - Internal Server Error
- `503` - Service Unavailable (workers offline)

### Error Response Format
```json
{
  "error": "validation_error",
  "message": "Invalid PGN format",
  "details": {
    "field": "pgn",
    "line": 3,
    "expected": "valid_pgn_header"
  },
  "request_id": "req-abc123",
  "timestamp": "2025-09-13T10:30:00Z"
}
```

## =Ú Referencias

- [Esquemas de Datos](./schemas.md) - Modelos de request/response
- [Ejemplos de Uso](./examples.md) - Casos de uso prácticos
- [Arquitectura](../architecture/overview.md) - Diseño del sistema
- [Flujo Celery](../architecture/celery-workflow.md) - Procesamiento asíncrono

## = Historial de Cambios

- **2025-09-13:** Documentación inicial de endpoints v1
- **2025-09-13:** Adición de ejemplos y códigos de error

---

**Siguiente acción:** Documentar esquemas de datos detallados
**Responsable:** Equipo de API