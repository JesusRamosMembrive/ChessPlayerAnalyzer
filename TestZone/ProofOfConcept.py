#!/usr/bin/env python3
"""
Descarga todas las partidas públicas de un jugador y guarda
un JSON con el PGN y los tiempos entre movimientos.
Requiere: requests, python-chess (opcional para análisis extra)
"""
from __future__ import annotations
import json, re, requests, pathlib
from datetime import UTC, datetime

USER = "anynoname"                 # ← cambia al usuario que quieras vigilar
OUT  = pathlib.Path(f"{USER}_games.json")
CLK_RGX = re.compile(r"\[%clk\s+([\d:.]+)]")

s = requests.Session()
s.headers["User-Agent"] = (
    "chess-analyzer/0.1 (contact: tu_correo@example.com)"
)

def fetch(url: str) -> dict:
    resp = s.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()

def sec(t: str) -> int:
    """Converte hh:mm:ss.f o m:ss.f a segundos (int)."""
    parts = list(map(float, t.split(":")))
    if len(parts) == 3:
        h, m, s_ = parts
    else:
        h, m, s_ = 0, *parts
    return int(h * 3600 + m * 60 + s_)

archives = fetch(f"https://api.chess.com/pub/player/{USER}/games/archives")["archives"]

all_games: list[dict] = []
for archive_url in archives:
    # print(f"Descargando partidas de {archive_url}...")

        for g in fetch(archive_url)["games"]:
            try:
                pgn = g["pgn"]
                clocks = CLK_RGX.findall(pgn)
                # diferencias entre relojes consecutivos
                deltas = [clocks[i-1] for i in range(1, len(clocks))] if clocks else []
                move_times = [sec(clocks[i-1]) - sec(clocks[i]) for i in range(1, len(clocks))]
                all_games.append({
                    "url": g["url"],
                    "end_time": datetime.fromtimestamp(g["end_time"], UTC).isoformat(),
                    "time_class": g["time_class"],
                    "time_control": g["time_control"],
                    "white": g["white"],
                    "black": g["black"],
                    "pgn": pgn,
                    "move_times_seconds": move_times
                })
            except KeyError as e:
                print(f"Error al descargar {archive_url}: {e}")
                continue

OUT.write_text(json.dumps(all_games, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Guardado en {OUT}")
