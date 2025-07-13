# #!/usr/bin/env python3
# """
# Encola todas las partidas de un .json (descargado de chess.com) en la API.
#
# Uso:
#     python bulk_upload.py Bryden04_games_cut.json
# """
# import json
# import logging
# import sys
# import requests
# import pathlib
#
# API = "http://localhost:8000/analyze"
#
# logger = logging.getLogger(__name__)
#
# def main(path: str):
#     data = json.loads(pathlib.Path(path).read_text(encoding="utf‑8"))
#     for i, game in enumerate(data, 1):
#         pgn = game["pgn"]
#         game = {"pgn": pgn, "move_times": game.get("move_times_seconds")}
#         resp = requests.post(API, json=game, timeout=10)
#         resp.raise_for_status()
#         out = resp.json()
#         logger.info(f"{i:>3}. game_id={out['game_id']}  task_id={out['task_id']}")
#
# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         logger.error("Usage: bulk_upload.py <file.json>")
#         sys.exit(1)
#     main(sys.argv[1])


import csv, json, urllib.request, pathlib

url = "https://raw.githubusercontent.com/lichess-org/chess-openings/main/eco.csv"
csv_text = urllib.request.urlopen(url).read().decode()
reader = csv.DictReader(csv_text.splitlines())

eco_names = {row["eco"]: row["name"] for row in reader}
path = pathlib.Path("eco_table.json")
path.write_text(json.dumps(eco_names, indent=2, ensure_ascii=False))
print(f"Wrote {len(eco_names)} ECO entries to {path}")