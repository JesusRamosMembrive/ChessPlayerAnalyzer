# Guía de tests locales (sin BD/Celery) para ChessPlayerAnalyzer

Este documento explica cómo ejecutar análisis locales sobre datasets pequeños, validar métricas y replicar las evaluaciones de motor de la aplicación sin levantar contenedores.

## Objetivos
- Dividir datasets grandes en partidas individuales o lotes pequeños.
- Ejecutar los cálculos locales (timing, calidad, aperturas, longitudinal) con máxima flexibilidad.
- Replicar evaluaciones de Stockfish para obtener ACPL/IPR/quality_score no-NaN.
- Exportar resultados a JSON y CSV para verificación rápida.

## Requisitos
- Python 3.10+ (recomendado 3.12).
- Paquetes ya presentes en el repo (pandas, numpy, python-chess, etc.).
- Stockfish instalado (para calidad no-NaN):
  - Linux: /usr/games/stockfish o /usr/local/bin/stockfish
  - macOS: /opt/homebrew/bin/stockfish
  - Windows: "C:\Program Files\Stockfish\stockfish.exe"
  - Alternativa: exporta STOCKFISH_PATH o usa --engine-path.

Archivos clave:
- Runner local: test/run_local_analysis.py
- Divisor de JSON: test/split_json.py
- Documentación y ejemplos: test/README.md

## Paso 1: Preparar datos (dividir JSON)
Entrada esperada: una lista de partidas (campos típicos: pgn, move_times, white, black, end_time).

- Per-game (un archivo por partida):
  - python3 test/split_json.py --input archives/Affan_khan123_20250708_081840.json --mode per-game --out-dir test/out
  - Salida: test/out/per_game/game_00001.json, …

- Chunks (lotes de N partidas):
  - python3 test/split_json.py --input archives/Affan_khan123_20250708_081840.json --mode chunks --chunk-size 10 --out-dir test/out
  - Salida: test/out/chunks_10/chunk_00001.json, …

Consejo: Trabaja con 10–20 partidas para ciclos rápidos.

## Paso 2: Ejecutar análisis local
Analiza un archivo (partida única o chunk) o una carpeta completa. Guarda JSON y (opcional) CSV por partida.

- Archivo individual:
  - python3 test/run_local_analysis.py --input test/out/per_game/game_00001.json --out-dir test/out/results

- Carpeta completa:
  - python3 test/run_local_analysis.py --input-dir test/out/per_game --pattern "*.json" --out-dir test/out/results

- Perspectiva de usuario (white/black según PGN):
  - añade --username TuUsuarioChessCom

- Reconstrucción de reloj (si el PGN tiene TimeControl):
  - añade --reconstruct-clock

### Activar Stockfish (recomendado)
- Linux:
  - python3 test/run_local_analysis.py --input test/out/chunks_10/chunk_00001.json --out-dir test/out/results --username TuUsuario --engine-enable --engine-path /usr/games/stockfish
- macOS:
  - --engine-enable --engine-path /opt/homebrew/bin/stockfish
- Windows:
  - python test\run_local_analysis.py --input test\out\chunks_10\chunk_00001.json --out-dir test\out\results --username TuUsuario --engine-enable --engine-path "C:\Program Files\Stockfish\stockfish.exe"

Flags de motor:
- --engine-enable
- --engine-path PATH (auto-detección si es posible)
- --engine-depth 12
- --engine-multipv 3

Exportación CSV:
- Añade --csv (opcionalmente --csv-path ruta\custom.csv)

Silenciar warnings de NaN:
- Añade --suppress-warnings

## Salidas
- JSON por archivo analizado:
  - test/out/results/&lt;nombre&gt;.results.json
  - Incluye per_game y aggregates (longitudinal, táctico/clutch, top 3 ECO y focus %).

- CSV por archivo analizado:
  - test/out/results/&lt;nombre&gt;.per_game.csv
  - Una fila por partida con columnas de: metadatos, timing, calidad por color (white_*, black_*, user_*), aperturas.

Con motor activado verás:
- ACPL, match_rate, weighted_match_rate, IPR, quality_score, precision_burst_count, best_rank, cp_loss, is_engine_best.

## Validación rápida (checklist)
- Calidad no-NaN cuando usas --engine-enable y Stockfish accesible.
- Perspectiva de color:
  - Campos white_* y black_* siempre.
  - Campos user_* cuando --username coincide con White/Black del PGN.
- Timing:
  - mean_move_time, time_complexity_corr, lag_spike_count, clutch_accuracy_diff si hay reloj.
- Aperturas:
  - second_choice_rate y opening_entropy por partida.
  - aggregates.opening_top3 y opening_focus_top3_pct por archivo.

## Problemas comunes
- ACPL/IPR/quality como NaN:
  - Activa --engine-enable y verifica --engine-path.
- Warning “Mean of empty slice”:
  - Es normal si faltan datos; usa --suppress-warnings.
- Rutas en Windows:
  - Cuida comillas y backslashes en --engine-path y --csv-path.

## Plan de sesiones pequeñas (recomendado)
1) Preparar fixtures:
- Generar per_game para 30–60 partidas (3–6 chunks de 10).

2) Validar sin motor (5–10 min):
- Enfoque en timing y aperturas; revisar CSVs.

3) Validar con motor (15–30 min):
- Re-ejecutar 2–3 chunks con --engine-enable.
- Comparar: ACPL/IPR/quality_score y match rates entre chunks y colores.

4) Ajustes/afinado:
- Probar depth/multipv si buscas estabilidad vs. rapidez.

5) Revisión final:
- Consolidar resultados y abrir issues con hallazgos.

## Referencias de código
- Runner y flags: test/run_local_analysis.py
- Ejemplos de uso: test/README.md
- Módulos de análisis: app/analysis/{timing.py, quality.py, openings.py, longitudinal.py}
