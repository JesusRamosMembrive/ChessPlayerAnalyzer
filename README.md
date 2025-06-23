# ChessPlayerAnalyzer

ChessPlayerAnalyzer es una aplicación que permite analizar partidas y jugadores de Chess.com usando Stockfish y un conjunto de métricas avanzadas. Incluye una API basada en **FastAPI**, tareas de análisis con **Celery**, base de datos PostgreSQL y comunicación por **Redis**. Está pensada para ejecutarse en contenedores Docker mediante `docker-compose`.

## Arquitectura

- **backend**: servicio FastAPI que expone la API REST.
- **celery**: worker que ejecuta las tareas de análisis.
- **postgres**: base de datos donde se almacenan partidas, métricas y estados.
- **redis**: usado tanto como broker y backend de Celery como para notificaciones en tiempo real.
- **migrate**: contenedor que aplica las migraciones de la base de datos al iniciarse.

El archivo `docker-compose.yml` define estos servicios y las variables de entorno necesarias.

## Puesta en marcha rápida

1. Asegúrate de tener Docker y docker-compose instalados.
2. Ejecuta:
   ```bash
   docker-compose up --build
   ```
3. La API quedará disponible en `http://localhost:8000` y el worker Celery se pondrá a procesar tareas.

## Flujo de análisis (workflow)

1. **Inicio del análisis**
   - El cliente realiza `POST /players/{username}` para solicitar el análisis completo de un jugador de Chess.com.
   - El endpoint crea (o reutiliza) un registro `Player` en la base de datos y publica una tarea única `process_player_enhanced` en Celery.
   - Se responde inmediatamente con un estado `pending` y el `task_id` asociado.

2. **Descarga de partidas** (`process_player_enhanced`)
   - La tarea descarga los archivos de partidas del jugador mediante la API pública de Chess.com.
   - Cada partida se guarda en la tabla `Game` junto con los tiempos de movimiento si están disponibles.
   - Para cada partida se construye una `chain` de Celery con dos pasos:
     1. `analyze_game_task`: invoca Stockfish para evaluar jugada por jugada y almacena `MoveAnalysis`.
     2. `analyze_game_detailed`: a partir de esas evaluaciones calcula métricas avanzadas (`GameAnalysisDetailed`).
   - Todas las cadenas se agrupan con `group` y al completarse ejecutan un `chord` final que llama a `analyze_player_detailed`.

3. **Análisis agregado del jugador** (`analyze_player_detailed`)
   - Una vez procesadas todas las partidas, se calculan métricas longitudinales y un `risk_score` global del jugador, quedando registrado en `PlayerAnalysisDetailed`.
   - Se actualiza el progreso del jugador en la tabla `Player` (del 0 al 100 %) y se envían notificaciones por Redis para el endpoint de streaming.

4. **Consulta de resultados**
   - `GET /players/{username}` devuelve el estado actual del análisis (not_analyzed, pending, ready o error) y el progreso en porcentaje.
   - `GET /metrics/player/{username}` devuelve las métricas agregadas del jugador una vez finalizado.
   - `GET /games/{game_id}` o `GET /metrics/game/{game_id}` permiten obtener los detalles de una partida concreta.
   - `GET /stream/{username}` expone un stream SSE con actualizaciones en tiempo real para clientes que deseen mostrar una barra de progreso.

Este flujo garantiza que varias peticiones simultáneas al mismo jugador no generen trabajos duplicados y permite reiniciar o refrescar análisis de forma segura.

## Endpoints principales

| Método | Ruta                               | Descripción                                   |
|------- |------------------------------------|-----------------------------------------------|
| `GET`  | `/`                                | Información general de la API.                |
| `POST` | `/analyze`                         | Analiza una partida individual.               |
| `GET`  | `/games/{game_id}`                 | Obtiene la partida almacenada.                |
| `GET`  | `/metrics/game/{game_id}`          | Métricas detalladas de una partida.           |
| `GET`  | `/players/{username}`              | Estado de análisis de un jugador.             |
| `POST` | `/players/{username}`              | Lanza el análisis completo del jugador.       |
| `POST` | `/players/{username}/refresh`      | Fuerza un nuevo análisis del jugador.         |
| `POST` | `/players/{username}/stop`         | Detiene un análisis en curso.                 |
| `DELETE` | `/players/{username}`            | Elimina al jugador y sus análisis asociados.  |
| `GET`  | `/metrics/player/{username}`       | Métricas agregadas del jugador.               |
| `GET`  | `/stream/{username}`               | Stream SSE de progreso.                       |

## CLI y utilidades

- `backend/player_cli.py` ofrece una interfaz básica por línea de comandos para solicitar el análisis de un jugador y mostrar el progreso.
- `backend/player_analyze_cli.py` incorpora una visualización más rica con colores y tablas.
- `bulk_upload.py` permite encolar múltiples partidas desde un archivo JSON a través del endpoint `/analyze`.

## Estructura del repositorio

```
backend/
├── app/                 Código de la API y lógica de análisis
│   ├── analysis/        Módulos de métricas (quality, timing, openings…)
│   ├── celery_app.py    Definición de tareas Celery
│   ├── main.py          Entrypoint FastAPI
│   ├── models.py        Modelos SQLModel
│   └── utils.py         Utilidades y helpers
├── migrations/          Archivos de migración (alembic)
├── player_cli.py        CLI sencillo de ejemplo
└── player_analyze_cli.py CLI avanzado con colores y tablas
```

## Licencia

Este proyecto se distribuye bajo los términos de la licencia MIT incluida en `LICENSE`.

