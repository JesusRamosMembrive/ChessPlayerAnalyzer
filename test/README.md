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

Reconstruir reloj del jugador (si el PGN tiene TimeControl):
- python3 test/run_local_analysis.py --input test/out/per_game/game_00001.json --reconstruct-clock

Directorio de resultados:
- test/out/results/<nombre_entrada>.results.json

Notas:
- Se usa “best effort”: si faltan campos opcionales (por ejemplo evaluaciones de motor), las métricas correspondientes aparecerán como NaN y el resto se calcularán.
- No se requiere Docker, base de datos ni Celery.

## Salida
- Modo per-game: test/out/per_game/game_00001.json, game_00002.json, ...
- Modo chunks: test/out/chunks_{N}/chunk_00001.json, chunk_00002.json, ...
- Resultados de análisis: test/out/results/<archivo>.results.json

Los objetos se preservan tal cual, sin modificaciones.
