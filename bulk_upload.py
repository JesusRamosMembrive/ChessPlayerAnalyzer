#!/usr/bin/env python3
"""
Encola todas las partidas de un .json (descargado de chess.com) en la API.

Uso:
    python bulk_upload.py Bryden04_games_cut.json
"""
import json, sys, requests, pathlib

API = "http://localhost:8000/analyze"

def main(path: str):
    data = json.loads(pathlib.Path(path).read_text(encoding="utf‑8"))
    for i, game in enumerate(data, 1):
        pgn = game["pgn"]
        game = {"pgn": pgn, "move_times": game.get("move_times_seconds")}
        resp = requests.post(API, json=game, timeout=10)
        resp.raise_for_status()
        out = resp.json()
        print(f"{i:>3}. game_id={out['game_id']}  task_id={out['task_id']}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: bulk_upload.py <file.json>")
        sys.exit(1)
    main(sys.argv[1])
