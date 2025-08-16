# Herramientas de pruebas offline

Scripts para trabajar localmente con JSON de partidas sin levantar contenedores ni descargar datos.

## Estructura
- test/data/: coloca aquí tu JSON de entrada (lista de partidas). Ejemplo de origen: ../archives/Affan_khan123_20250708_081840.json
- test/out/: se generarán aquí los archivos de salida
- test/split_json.py: script de división

## Formato de entrada
Se espera un JSON que sea una lista de objetos, cada uno representando una partida, con campos como:
- pgn
- move_times
- white
- black
- end_time

Ejemplo real: consulta el archivo en:
- archives/Affan_khan123_20250708_081840.json

## Uso
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

## Salida
- Modo per-game: test/out/per_game/game_00001.json, game_00002.json, ...
- Modo chunks: test/out/chunks_{N}/chunk_00001.json, chunk_00002.json, ...

Los objetos se preservan tal cual, sin modificaciones.
