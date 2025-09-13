# ChessPlayerAnalyzer

Aplicación de análisis avanzado de partidas y jugadores de Chess.com usando Stockfish, métricas estadísticas y machine learning.

## 🚀 Inicio Rápido

```bash
# Clonar y ejecutar
git clone <repository>
cd ChessPlayerAnalyzer
docker-compose up --build
```

**API disponible en:** http://localhost:8000
**Interfaz de trazas:** http://localhost:16686

## 📚 Documentación Completa

La documentación técnica está organizada en [`/docs`](./docs/):

- **[🏗️ Arquitectura](./docs/architecture/)** - Diseño del sistema
- **[🔌 API](./docs/api/)** - Endpoints y esquemas
- **[🧩 Módulos](./docs/modules/)** - Componentes internos
- **[📖 Guías](./docs/guides/)** - Setup, deployment, troubleshooting
- **[🧮 Algoritmos](./docs/algorithms/)** - Modelos estadísticos y métricas

Ver [**Índice Principal de Documentación**](./docs/README.md) para navegación completa.

## ⚡ Comandos Principales

```bash
# Desarrollo
docker-compose --profile dev up --build

# Con soporte ML/PyTorch
docker-compose --profile ml up --build

# Tests
pytest

# Análisis de jugador (CLI)
python player_analyze_cli.py <username>
```

## 🏛️ Arquitectura

- **FastAPI** - API REST asíncrona
- **Celery** - Procesamiento distribuido de análisis
- **PostgreSQL** - Base de datos de partidas y métricas
- **Redis** - Message broker y notificaciones tiempo real
- **Stockfish** - Motor de análisis de ajedrez
- **Docker** - Containerización y orquestación

---

📖 **Documentación técnica completa:** [`/docs`](./docs/)
🐛 **Troubleshooting:** [`/docs/guides/troubleshooting.md`](./docs/guides/troubleshooting.md)
⚙️ **Setup avanzado:** [`/docs/guides/setup/`](./docs/guides/setup/)