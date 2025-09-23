# Testing V2 - Guía de Pruebas

## ✅ Lo que hemos preparado

1. **Sistema V2 completo**:
   - Tablas simplificadas: `Game`, `AnalysisResult`, `Player`
   - Pipeline unificado sin dependencias circulares
   - **Misma interfaz JSON** que espera React

2. **Docker Compose V2**:
   - `docker-compose.v2.yml` configurado específicamente para V2
   - Variables de entorno para activar V2
   - Base de datos separada para testing

## 🧪 Cómo hacer la prueba

### Paso 1: Build y Start
```bash
# Build (en progreso ahora)
docker compose -f docker-compose.v2.yml build --no-cache

# Start todos los servicios
docker compose -f docker-compose.v2.yml up -d

# Ver logs del backend
docker compose -f docker-compose.v2.yml logs -f backend_v2

# Ver logs del worker
docker compose -f docker-compose.v2.yml logs -f celery_v2
```

### Paso 2: Verificar que V2 está activo
```bash
# Verificar configuración
curl http://localhost:8000/config/version

# Debería mostrar:
# {
#   "api_version": "2.0.0",
#   "engine_version": "v2",
#   "use_v2_engine": true,
#   ...
# }
```

### Paso 3: Probar endpoints como React los usa

#### 1. Analizar un jugador (POST)
```bash
curl -X POST "http://localhost:8000/players/TestUser123" \
  -H "Content-Type: application/json"

# Debería devolver:
# {
#   "message": "Analysis started for player TestUser123",
#   "task_id": "...",
#   "username": "TestUser123",
#   "status": "pending"
# }
```

#### 2. Monitorear progreso (GET - como hace React)
```bash
curl "http://localhost:8000/players/TestUser123"

# Debería devolver exactamente como React espera:
# {
#   "username": "TestUser123",
#   "status": "pending",  # o "ready" cuando termine
#   "progress": 45,
#   "total_games": 100,
#   "done_games": 45,
#   "requested_at": "2025-09-23T...",
#   "finished_at": null,
#   "error": null,
#   "last_task_id": "..."
# }
```

#### 3. Obtener métricas cuando termine (GET)
```bash
curl "http://localhost:8000/metrics/player/TestUser123"

# Debería devolver la misma estructura que React espera:
# {
#   "username": "TestUser123",
#   "games_analyzed": 100,
#   "avg_acpl": 68.5,
#   "opening_patterns": { ... },
#   "phase_quality": { ... },
#   "risk": { ... },
#   ...
# }
```

## 🔍 Qué verificar

### ✅ Funcionalidad V2
- [ ] Backend inicia correctamente con V2
- [ ] Worker Celery V2 funciona
- [ ] Se crean tablas V2 automáticamente
- [ ] POST /players/{username} inicia análisis
- [ ] GET /players/{username} muestra progreso
- [ ] GET /metrics/player/{username} devuelve métricas
- [ ] JSON response es **idéntico** al actual

### ✅ Performance
- [ ] Análisis es igual o más rápido que V1
- [ ] No hay memory leaks
- [ ] Logs muestran "DEBUG V2" para confirmar versión

### ✅ Compatibilidad
- [ ] React frontend funciona **sin cambios**
- [ ] WebSocket notifications funcionan
- [ ] SSE stream funciona
- [ ] Tracing funciona (Jaeger UI: http://localhost:16686)

## 🚨 Si algo falla

### Debugging
```bash
# Ver logs detallados
docker compose -f docker-compose.v2.yml logs backend_v2 | grep "DEBUG V2"
docker compose -f docker-compose.v2.yml logs celery_v2 | grep "ERROR"

# Verificar base de datos
docker compose -f docker-compose.v2.yml exec postgres psql -U chess -d chessdb -c "\dt"

# Verificar Redis
docker compose -f docker-compose.v2.yml exec redis redis-cli ping
```

### Rollback a V1
```bash
# Parar V2
docker compose -f docker-compose.v2.yml down

# Volver a V1
docker compose up -d
```

### Logs importantes
- `"DEBUG V2: Starting player analysis for X using v2"`
- `"V2 tables created successfully!"`
- `"AnalysisAdapter initialized with version: v2"`

## 💡 Modo Híbrido (alternativo)

Si quieres probar gradualmente:

```bash
# Usar V1 por defecto, V2 solo para usuarios específicos
docker compose -f docker-compose.v2.yml down
docker compose -f docker-compose.v2.yml up -e USE_V2_ENGINE=false -e HYBRID_MODE=true
```

## 🎯 Éxito esperado

**La prueba es exitosa si:**
1. ✅ React frontend funciona **exactamente igual**
2. ✅ Respuestas JSON son **idénticas**
3. ✅ Performance es igual o mejor
4. ✅ No hay errores en logs
5. ✅ Pipeline V2 procesa jugadores correctamente

**El objetivo es que React no note la diferencia!**