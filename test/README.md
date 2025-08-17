# Herramientas de pruebas offline

Scripts para trabajar localmente con JSON de partidas sin levantar contenedores ni descargar datos.

## Estructura
- test/data/: coloca aquí tu JSON de entrada (lista de partidas). Ejemplo de origen: ../archives/Affan_khan123_20250708_081840.json
- test/out/: se generarán aquí los archivos de salida
- test/split_json.py: script de división
- test/run_local_analysis.py: ejecuta los cálculos localmente sin BD/Celery

## Formato de entrada
Se espera un JSON que sea:
- una lista de objetos, cada uno representando una partida, o
- un único objeto de partida

Cada objeto puede contener:
- pgn
- move_times
- white
- black
- end_time

Ejemplo real: consulta el archivo en:
- archives/Affan_khan123_20250708_081840.json

## Uso: división del JSON
Primero, copia un archivo real a test/data/:
- cp archives/Affan_khan123_20250708_081840.json test/data/input.json

Dividir en un archivo por partida:
- python3 test/split_json.py --mode per-game

Dividir en chunks de 10 partidas:
- python3 test/split_json.py --mode chunks --chunk-size 10

Dividir en chunks de 20 partidas:
- python3 test/split_json.py --mode chunks --chunk-size 20

Personalizar rutas:
- python3 test/split_json.py --input path/a/tu.json --mode chunks --chunk-size 15 --out-dir test/out

## Uso: análisis local sin BD/Celery
Analizar un archivo de partida individual:
- python3 test/run_local_analysis.py --input test/out/per_game/game_00001.json

Analizar un archivo con varias partidas (chunk):
- python3 test/run_local_analysis.py --input test/out/chunks_10/chunk_00001.json

Procesar en lote todos los JSON de una carpeta:
- python3 test/run_local_analysis.py --input-dir test/out/per_game --pattern "*.json"

Exportación a CSV de resultados por partida:
- Un archivo: python3 test/run_local_analysis.py --input test/out/per_game/game_00001.json --out-dir test/out/results --csv
  - CSV por defecto: test/out/results/game_00001.per_game.csv
- Un chunk: python3 test/run_local_analysis.py --input test/out/chunks_10/chunk_00001.json --out-dir test/out/results --csv
  - CSV por defecto: test/out/results/chunk_00001.per_game.csv
- Carpeta completa: python3 test/run_local_analysis.py --input-dir test/out/per_game --out-dir test/out/results --csv
- Ruta personalizada: añade --csv-path path/a/archivo.csv

Activar evaluaciones por jugada con motor (Stockfish):
- Requisitos: tener Stockfish instalado o indicar la ruta con --engine-path (o variable de entorno STOCKFISH_PATH).
- Flags:
  - --engine-enable: activa el análisis local con motor
  - --engine-path: ruta al binario UCI (por defecto "stockfish" o $STOCKFISH_PATH)
  - --engine-depth: profundidad de análisis (por defecto $STOCKFISH_DEPTH o 12)
  - --engine-multipv: número de PVs a considerar (por defecto 3)
- Ejemplos:
  - python3 test/run_local_analysis.py --input test/out/chunks_10/chunk_00001.json --out-dir test/out/results --username TuUsuario --engine-enable
  - python3 test/run_local_analysis.py --input test/out/per_game/game_00001.json --out-dir test/out/results --engine-enable --engine-depth 12 --engine-multipv 3
- Notas:
  - Con --engine-enable, se calculan eval_cp_before/after, best_rank, cp_loss y se habilitan métricas ACPL/IPR/quality_score y match_rate sin NaN.
  - La perspectiva de color se maneja automáticamente (white_*, black_* y user_* si pasas --username).

Activar evaluaciones por jugada con motor (Stockfish):
- Requisitos: tener Stockfish instalado o indicar la ruta con --engine-path (o variable de entorno STOCKFISH_PATH).
- Flags:
  - --engine-enable: activa el análisis local con motor
  - --engine-path: ruta al binario UCI (por defecto "stockfish" o $STOCKFISH_PATH)
  - --engine-depth: profundidad de análisis (por defecto $STOCKFISH_DEPTH o 12)
  - --engine-multipv: número de PVs a considerar (por defecto 3)
- Ejemplos:
  - python3 test/run_local_analysis.py --input test/out/chunks_10/chunk_00001.json --out-dir test/out/results --username TuUsuario --engine-enable
  - python3 test/run_local_analysis.py --input test/out/per_game/game_00001.json --out-dir test/out/results --engine-enable --engine-depth 12 --engine-multipv 3
- Notas:
  - Con --engine-enable, se calculan eval_cp_before/after, best_rank, cp_loss y se habilitan métricas ACPL/IPR/quality_score y match_rate sin NaN.
  - La perspectiva de color se maneja automáticamente (white_*, black_* y user_* si pasas --username).

Usar perspectiva del usuario (si coincide con White/Black del PGN):
- python3 test/run_local_analysis.py --input test/out/per_game/game_00001.json --username tuUsuarioChessCom

Reconstruir reloj del jugador (si el PGN tiene TimeControl):
- python3 test/run_local_analysis.py --input test/out/per_game/game_00001.json --reconstruct-clock

Directorio de resultados:
- test/out/results/<nombre_entrada>.results.json

Desactivar resumen por color en consola:
- python3 test/run_local_analysis.py --input test/out/chunks_10/chunk_00001.json --no-color-summary

Silenciar warnings (útil si no hay evals de motor y aparecen NaN):
- python3 test/run_local_analysis.py --input test/out/chunks_10/chunk_00001.json --suppress-warnings

Notas:
- Se usa “best effort”: si faltan campos opcionales (por ejemplo evaluaciones de motor), las métricas correspondientes aparecerán como NaN y el resto se calcularán.
- No se requiere Docker, base de datos ni Celery.

## Salida
- Modo per-game: test/out/per_game/game_00001.json, game_00002.json, ...
- Modo chunks: test/out/chunks_{N}/chunk_00001.json, chunk_00002.json, ...
- Resultados de análisis: test/out/results/<archivo>.results.json

Los objetos se preservan tal cual, sin modificaciones.
## Perspectiva de color y métricas ampliadas

- El script calcula métricas de calidad desde la perspectiva de:
  - blancas: claves white_*
  - negras: claves black_*
  - usuario: claves user_* si proporcionas --username y coincide con White/Black del PGN
- Esto asegura que las métricas sensibles al signo (por ejemplo, evaluaciones del motor) se interpreten correctamente para cada color.

Métricas incluidas por partida:
- Timing:
  - mean_move_time, time_variance
  - time_complexity_corr
  - lag_spike_count
  - uniformity_score
  - clutch_accuracy_diff (si hay player_clock_before reconstruido o presente)
  - timing_score
- Quality por color (white_*, black_* y user_* si aplica):
  - acpl
  - match_rate
  - weighted_match_rate
  - ipr
  - ipr_z_score (0.0 si no hay ELO)
  - quality_score
  - precision_burst_count
- Openings:
  - second_choice_rate y otros campos que devuelva openings.aggregate_opening_features
- Longitudinal y agregados por archivo:
  - aggregate_longitudinal_features(games_df)
  - aggregate_tactical_trends(games_df)
  - aggregate_clutch_accuracy(games_df)

Resumen por consola:
- Imprime medias de: mean_move_time, acpl, weighted_match_rate, ipr, quality_score, time_complexity_corr y el total de lag_spikes.
- Por defecto prioriza la perspectiva del usuario si has pasado --username; si no, usa la de blancas.

Agregados de salida (aggregates) por archivo:
- Longitudinal: aggregate_longitudinal_features(games_df)
- Calidad: aggregate_tactical_trends(games_df), aggregate_clutch_accuracy(games_df)
- Aperturas: opening_top3 (top 3 ECO con conteo) y opening_focus_top3_pct (porcentaje de partidas cubiertas por las 3 aperturas más frecuentes)
