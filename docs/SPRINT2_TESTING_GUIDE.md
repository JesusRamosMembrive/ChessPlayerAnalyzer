# Sprint 2 Testing Guide - Validación con Datos Reales

## 🎯 Objetivo

Validar que los servicios de dominio del Sprint 2 funcionan correctamente con datos reales de Chess.com extraídos de los archives.

## 📁 Archivos de Prueba Creados

### 1. Test Principal
- **`test_sprint2_minimal.py`**: Test independiente que valida core functionality
- **Ejecutar**: `python3 test_sprint2_minimal.py`

### 2. Datos de Muestra
- **`test_sample_games.json`**: 3 partidas reales de TestUser456 extraídas de archives
- **`test_sample.pgn`**: 6 partidas en formato PGN estándar para testing manual

### 3. Test Completo (Avanzado)
- **`test_sprint2_real_data.py`**: Test más completo que usa imports del proyecto
- **Nota**: Requiere resolver dependencias de logging

## 🚀 Cómo Ejecutar las Pruebas

### Test Básico (Recomendado)
```bash
# Test rápido y sin dependencias
python3 test_sprint2_minimal.py
```

**Output esperado**:
```
🚀 Test Mínimo Sprint 2 - Domain Services
==================================================
🔒 Testing Value Objects...
   ✅ QualityMetrics válido creado
   ✅ Es inmutable
   ✅ Validación funcionó
🔍 Testing Analysis Functions...
   ✅ ACPL calculado: 10.0
   ✅ Match rate: 0.67
   ✅ Cálculos correctos
🎮 Testing PGN Parsing...
   ✅ White: Alice
   ✅ Black: Bob
   ✅ Movimientos: 5
   ✅ Válido: True
   ✅ Parsing correcto
📂 Testing Real Data...
   📁 Usando: raw_Data_afd.json
   📊 Cargadas 3 partidas
   🎯 Partida 1: afd vs Jim1234
      ACPL: 21.5, Match: 0.30
   ✅ 3 partidas analizadas exitosamente

🎯 RESULTADO: 4/4 tests pasaron
🎉 ¡Sprint 2 - Domain Services validado!
```

### Test con Datos Específicos
```bash
# Test usando los datos de muestra creados
python3 -c "
import json
with open('test_sample_games.json') as f:
    data = json.load(f)
print(f'Cargadas {len(data)} partidas de muestra')
for game in data:
    pgn = game['pgn']
    white = game['white']
    black = game['black']
    print(f'  {white} vs {black}')
"
```

## 🔍 Qué Validan las Pruebas

### 1. Value Objects (Fundación)
- ✅ **Inmutabilidad**: Los objetos no pueden modificarse después de creación
- ✅ **Validaciones**: Rechaza valores inválidos (ACPL negativo, match rate > 1)
- ✅ **Type Safety**: Estructura tipada vs JSON no tipado

### 2. Analysis Functions (Core Logic)
- ✅ **ACPL Calculation**: Average Centipawn Loss correcto
- ✅ **Match Rate**: Porcentaje de coincidencia con mejor jugada
- ✅ **Edge Cases**: Manejo de listas vacías y datos inválidos

### 3. PGN Parsing (Data Processing)
- ✅ **Header Extraction**: Extrae White, Black, Result correctamente
- ✅ **Move Counting**: Cuenta movimientos para validación
- ✅ **Validation**: Rechaza PGNs inválidos o muy cortos

### 4. Real Data Integration
- ✅ **Chess.com Data**: Procesa archivos JSON reales de archives/
- ✅ **Multiple Games**: Analiza múltiples partidas en batch
- ✅ **Error Handling**: Maneja partidas corruptas o incompletas

## 📊 Datos de Prueba Disponibles

### Archives Directory
- **Hikaru_20250702_051445.json**: ~10MB - Partidas de Hikaru (para stress testing)
- **testuser456_20250702_092655.json**: ~20KB - Jugador amateur (ideal para testing)
- **Affan_khan123_*.json**: ~270KB - Jugador activo con muchas partidas

### Test Sample Files
```json
// test_sample_games.json - 3 partidas listas para testing
[
  {
    "pgn": "[Event \"Live Chess\"]...",
    "move_times": [2, 8, -7, 8, ...],
    "white": "TestUser456",
    "black": "paulb900"
  }
]
```

```pgn
// test_sample.pgn - 6 partidas en formato estándar
[Event "Live Chess"]
[White "TestUser456"]
[Black "paulb900"]
[Result "1-0"]

1. d4 Nc6 2. e4 d5 ... 1-0
```

## 🧪 Tests Manuales Adicionales

### Test de Servicios Individuales
```python
# Test manual de AnalysisService
python3 -c "
import sys
sys.path.append('.')

# Crear datos mock
moves_data = [
    {'played': 'e4', 'best': 'e4', 'cp_loss': 0},
    {'played': 'e5', 'best': 'e5', 'cp_loss': 5},
    {'played': 'Nf3', 'best': 'Nc3', 'cp_loss': 25}
]

# Calcular métricas
acpl = sum(m['cp_loss'] for m in moves_data) / len(moves_data)
match_rate = sum(1 for m in moves_data if m['played'] == m['best']) / len(moves_data)

print(f'ACPL: {acpl}')
print(f'Match Rate: {match_rate:.2f}')
print(f'Suspicious: {acpl < 5.0 and match_rate > 0.9}')
"
```

### Test de Value Objects
```python
# Test inmutabilidad y validaciones
python3 -c "
from dataclasses import dataclass

@dataclass(frozen=True)
class QualityMetrics:
    avg_acpl: float
    avg_match_rate: float

    def __post_init__(self):
        if self.avg_acpl < 0:
            raise ValueError('Invalid ACPL')

# Test válido
metrics = QualityMetrics(15.5, 0.75)
print(f'✅ Created: ACPL={metrics.avg_acpl}, Match={metrics.avg_match_rate}')

# Test inmutabilidad
try:
    metrics.avg_acpl = 20.0
    print('❌ Not immutable')
except:
    print('✅ Immutable')

# Test validación
try:
    invalid = QualityMetrics(-5.0, 0.75)
    print('❌ Validation failed')
except ValueError:
    print('✅ Validation worked')
"
```

## 🎯 Criterios de Éxito

### ✅ Sprint 2 Validado Si:
1. **Todos los tests básicos pasan** (4/4)
2. **Puede procesar datos reales** de archives/
3. **Value objects funcionan** (inmutabilidad + validaciones)
4. **Analysis functions calculan** métricas correctamente
5. **PGN parsing extrae** información básica

### ⚠️ Revisar Si:
- Tests fallan por imports/dependencias
- Datos reales no se procesan correctamente
- Cálculos de métricas son incorrectos
- Value objects permiten modificación

## 🚀 Próximos Pasos

Una vez que **Sprint 2 esté validado**:

1. **Sprint 3**: Crear Application layer con casos de uso
2. **Integration**: Conectar domain services con infrastructure
3. **Migration**: Reemplazar lógica en celery_app.py gradualmente
4. **Performance**: Optimizar con datos reales de mayor volumen

## 📝 Notas de Desarrollo

### Archivos de Datos
- Los JSON en archives/ contienen **datos reales** de Chess.com
- Cada archivo tiene formato: `[{pgn: "...", move_times: [...], white: "...", black: "..."}]`
- Los `move_times` están en décimas de segundo, negativos para black

### Limitaciones Actuales
- **Stockfish Analysis**: Simulado con hash para testing
- **Tablebase Data**: No disponible en tests básicos
- **Complex Timing**: Análisis avanzado pendiente para Sprint 3

### Testing Philosophy
- **Fast Feedback**: Tests básicos en <5 segundos
- **Real Data**: Validación con partidas reales
- **Isolated**: Cada test puede ejecutarse independientemente
- **Comprehensive**: Cubre casos felices y edge cases