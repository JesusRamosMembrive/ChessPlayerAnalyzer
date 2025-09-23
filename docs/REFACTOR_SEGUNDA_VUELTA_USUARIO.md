# Segunda vuelta de simplificacion Codex

## Meta
- Reducir efectos secundarios y estados globales para facilitar debugging.
- Mantener la API publica intacta mientras se racionaliza el nucleo.

## Prioridad alta
- **Centralizar inicializacion y telemetria** (`app/main.py:67-107`, `app/celery_tasks.py:18-97`): Cada modulo arranca logging, OpenTelemetry y Prometheus al importarse, lo que dispara conexiones al solo abrir utilidades y hace mas dificil reproducir errores en pruebas. Proponer factories `create_app()` y `create_worker()` que apliquen configuracion bajo demanda y banderas de entorno para activar telemetria solo cuando corresponda.
- **Unificar control de locks Redis** (`app/main.py:41-65`, `app/api/v1/endpoints/players.py:95-165`, `app/utils.py:210-220`): Hay tres variantes de control de concurrencia que no comparten cleanup; algunas funciones incluso quedaron sin uso. Extraer un servicio `AnalysisLockService` o reutilizar el context manager `player_lock` para todas las rutas y workers, y eliminar helpers duplicados.
- **Partir `app/utils.py` en componentes acotados** (`app/utils.py:1-210`, `app/utils.py:165-208`, `app/utils_sanitize.py:1-18`): El modulo mezcla HTTP contra chess.com, manejo de progreso, locks Redis, cacheo de tareas y sanitizado JSON, ademas de duplicar `clean_json_numbers`. Separar por dominio (p.ej. `infra/redis.py`, `analysis/fetch.py`, `serialization/json.py`) permite probar y depurar cada pieza sin arrastrar dependencias ajenas.

## Prioridad media
- **Eliminar hacks de sys.path en modulos de analisis** (`app/analysis/quality.py:12-19`): Inyectar el repo en `sys.path` oculta problemas de empaquetado y rompe tests aislados. Ajustar imports relativos dentro de `app.analysis` y mover decoradores compartidos a un paquete accesible evita efectos secundarios al importar.
- **Racionalizar exposicion de routers** (`app/main.py:109-116`): Montar el mismo router tres veces genera rutas duplicadas, metricas y trazas repetidas. Mantener `/api/v1` y proveer alias explicitos via redireccion o routers con `include_in_schema=False` reduce ruido operativo.
- **Simplificar capa de base de datos para desarrollo** (`app/database.py:33-148`): El `RoutingSession` con replicas, reintentos agresivos y `dispose()` inmediato complica depuracion local. Ofrecer un modo sencillo sin replicas cuando `READ_REPLICA_URLS` esta vacio y encapsular la logica avanzada en una clase activable por flag aclara el flujo y evita sorpresas en tests.

## Prioridad baja
- **Bajar verbosidad de logs de metrica** (`app/analysis/quality.py:72-103`, `app/celery_tasks.py:148-198`): Mensajes `logger.info` por movimiento y tarea saturan Jaeger y stdout; moverlos a nivel debug o resumirlos en contadores facilita seguir errores reales.
- **Clarificar rol de `app/application`** (`app/application/`): El arbol existe pero practicamente esta vacio; quien depura pierde tiempo pensando que contiene la capa de aplicacion. Fijar si sera hogar de comandos/queries reales o retirarlo para evitar rutas muertas.
- **Revisar helpers sin uso** (`app/main.py:51-65`): Las funciones `set_*_in_progress` quedaron huerfanas tras el refactor; retirarlas o cubrirlas con pruebas evita suposiciones incorrectas durante debugging.
