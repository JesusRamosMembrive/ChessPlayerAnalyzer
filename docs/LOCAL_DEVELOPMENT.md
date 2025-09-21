# Local Development & Testing Guide

## Nuevo Workflow de Desarrollo (Sin Docker)

### ❌ Problema Anterior
```bash
# Workflow lento anterior
docker-compose down
docker-compose up --build  # 2-3 minutos
# Probar cambio
# Repetir...
```

### ✅ Nuevo Workflow Rápido
```bash
# Una sola vez: setup inicial
pip install -r requirements.txt -r requirements-dev.txt

# Para cada cambio:
python3 -m pytest tests/unit/  # < 5 segundos
python3 -m pytest tests/integration/  # < 30 segundos

# Debug específico:
python3 -c "from app.analysis.quality import calculate_acpl; print(calculate_acpl([{'cp_loss': 10}]))"
```

## Setup Inicial

### 1. Instalar Dependencias Locales
```bash
pip install -r requirements.txt -r requirements-dev.txt
```

### 2. Instalar Stockfish (si no está instalado)
```bash
# Ubuntu/Debian
sudo apt-get install stockfish

# macOS
brew install stockfish

# Verificar instalación
which stockfish  # Debe mostrar path
```

### 3. Redis Local (Opcional para tests de integración)
```bash
# Ubuntu/Debian
sudo apt-get install redis-server
sudo systemctl start redis-server

# macOS
brew install redis
brew services start redis

# Verificar
redis-cli ping  # Debe responder "PONG"
```

## Comandos de Testing

### Tests Rápidos (Sin Dependencias Externas)
```bash
# Solo tests unitarios
python3 -m pytest tests/unit/ -v

# Test específico
python3 -m pytest tests/unit/test_analysis/test_quality.py::TestCalculateACPL -v

# Con coverage
python3 -m pytest tests/unit/ --cov=app --cov-report=html
```

### Tests de Integración (Con BD/Redis)
```bash
# Tests de integración completos
python3 -m pytest tests/integration/ -v

# Solo tests de API
python3 -m pytest tests/integration/test_api/ -v
```

### Script Todo-en-Uno
```bash
# Ejecutar script completo
./scripts/test_local.sh
```

## Debugging Interactivo

### Debug de Módulos Específicos
```python
# En Python REPL o script
from app.analysis.quality import calculate_acpl, calculate_match_rate

# Test data rápido
moves = [{"cp_loss": 10}, {"cp_loss": 20}]
print(f"ACPL: {calculate_acpl(moves)}")

# Test match rate
moves_with_best = [
    {"played": "e4", "best": "e4"},
    {"played": "Nf3", "best": "Nc3"}
]
print(f"Match rate: {calculate_match_rate(moves_with_best)}")
```

### Debug de API Local
```bash
# Terminal 1: Levantar API local
export DATABASE_URL="sqlite:///test.db"
export REDIS_URL="redis://localhost:6379/1"
export STOCKFISH_PATH="/usr/games/stockfish"
export STOCKFISH_DEPTH="1"
uvicorn app.main:app --reload --port 8000

# Terminal 2: Test endpoints
curl http://localhost:8000/health
curl -X POST http://localhost:8000/players/testuser
```

## Variables de Entorno para Development

### Testing Local
```bash
export DATABASE_URL="sqlite:///test.db"  # BD en memoria/archivo
export REDIS_URL="redis://localhost:6379/1"  # DB diferente para tests
export STOCKFISH_PATH="/usr/games/stockfish"
export STOCKFISH_DEPTH="1"  # Rápido para tests
export ENABLE_TRACING="false"  # Sin overhead de tracing
```

### Development Local
```bash
export DATABASE_URL="sqlite:///dev.db"  # BD persistente para dev
export REDIS_URL="redis://localhost:6379/0"  # DB default
export STOCKFISH_PATH="/usr/games/stockfish"
export STOCKFISH_DEPTH="12"  # Análisis completo
export ENABLE_TRACING="true"  # Con tracing si tienes Jaeger local
```

## Cuándo Usar Docker

### Solo Para Production-Like Testing
```bash
# Levantar solo servicios necesarios
docker-compose up postgres redis

# API local conectada a servicios Docker
export DATABASE_URL="postgresql+psycopg://chess:chess@localhost:5432/chessdb"
export REDIS_URL="redis://localhost:6379/0"
uvicorn app.main:app --reload
```

### Para CI/CD o Testing Completo
```bash
# Solo cuando necesites verificar integración completa
docker-compose up --build
```

## Estructura de Tests Creada

```
tests/
├── conftest.py              # Fixtures globales (DB, client, mocks)
├── unit/                    # Tests rápidos sin dependencias
│   ├── test_analysis/       # Tests de módulos de análisis
│   └── test_utils/          # Tests de utilidades
├── integration/             # Tests con BD/Redis local
│   ├── test_api/            # Tests de endpoints
│   ├── test_repositories/   # Tests de acceso a datos
│   └── test_services/       # Tests de servicios
└── e2e/                     # Tests end-to-end (futuro)
```

## Ejemplos de Testing

### Test Unitario Típico
```python
def test_calculate_acpl():
    moves_data = [{"cp_loss": 10}, {"cp_loss": 20}]
    result = calculate_acpl(moves_data)
    assert result == 15.0
```

### Test de Integración Típico
```python
def test_create_player(session):
    player = Player(username="test", status=PlayerStatus.pending)
    session.add(player)
    session.commit()
    assert player.username == "test"
```

### Test de API Típico
```python
def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
```

## Troubleshooting

### Error: "No module named app"
```bash
# Asegúrate de estar en el directorio raíz del proyecto
cd /home/jesusramos/Git/ChessPlayerAnalyzer
export PYTHONPATH=$PWD:$PYTHONPATH
```

### Error: "stockfish not found"
```bash
# Verificar instalación
which stockfish
# Si no está instalado:
sudo apt-get install stockfish
```

### Error: "Connection refused" (Redis)
```bash
# Verificar que Redis esté corriendo
redis-cli ping
# Si no responde:
sudo systemctl start redis-server
```

## Beneficios de Este Workflow

1. **Velocidad**: Tests unitarios en < 5 segundos vs 2-3 minutos con Docker
2. **Iteración rápida**: Cambio → Test → Debug sin overhead
3. **Debugging granular**: Puedes testear funciones individuales
4. **Menos recursos**: No consume CPU/memoria de Docker
5. **IDE friendly**: Mejor integración con editores y debuggers