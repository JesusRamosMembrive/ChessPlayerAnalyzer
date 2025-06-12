#!/usr/bin/env python3
"""CLI: analiza todas las partidas públicas de un jugador con la API local.

Uso:
    python scripts/player_analyze_cli.py  # y responde al prompt

Requisitos:
    pip install requests tqdm

Pasos:
1. Pregunta usuario Chess.com.
2. Descarga todas sus partidas vía API pública.
3. Envía cada PGN a /analyze (API local en localhost:8000).
4. Espera a que Celery calcule métricas por partida.
5. Muestra resumen por partida y PlayerMetrics global.
"""
from __future__ import annotations

from typing import Dict, List
import sys
import json
import requests
import time
from tqdm import tqdm

API_BASE = "http://localhost:8000"

# Sesión global con UA
S = requests.Session()
S.headers["User-Agent"] = "chess-analyzer/0.1 (+https://github.com/tu-repo)"


# ───────────────────────── helpers ──────────────────────────

# helper con reintentos
def safe_get(url: str, tries: int = 3, backoff: float = 2.0):
    for n in range(tries):
        r = S.get(url, timeout=10)
        if r.status_code in (403, 429):
            time.sleep(backoff ** n)
            continue
        return r
    r.raise_for_status()

def get_archives(username: str):
    url = f"https://api.chess.com/pub/player/{username}/games/archives"
    r = safe_get(url)
    if r.status_code == 404:
        sys.exit(f"⚠️  El usuario '{username}' no existe.")
    return r.json()["archives"]

def fetch_month(url: str):
    return safe_get(url).json()["games"]


def post_game(pgn: str) -> int:
    r = requests.post(f"{API_BASE}/analyze", json={"pgn": pgn}, timeout=15)
    r.raise_for_status()
    return r.json()["game_id"]


def wait_game_metrics(game_id: int, timeout: int = 60):
    t0 = time.time()
    while True:
        r = requests.get(f"{API_BASE}/metrics/game/{game_id}", timeout=5)
        if r.status_code == 200:
            return r.json()
        if time.time() - t0 > timeout:
            raise TimeoutError(f"metrics for game {game_id} timeout")
        time.sleep(1)


def get_player_metrics(username: str):
    r = requests.get(f"{API_BASE}/metrics/player/{username}", timeout=5)
    return r.json() if r.status_code == 200 else None

# ───────────────────────── main ──────────────────────────

def main():
    username = input("Jugador Chess.com a analizar: ").strip().lower()
    print(f"Descargando partidas de {username}…")

    archives = get_archives(username)[-6:]  # últimos 6 meses para ir rápido
    games = []
    for month_url in tqdm(archives, desc="meses"):
        games.extend(fetch_month(month_url))

    if not games:
        print("No se encontraron partidas.")
        return

    print(f"→ {len(games)} partidas descargadas. Enviando al backend…")
    game_map: Dict[int, str] = {}
    for g in tqdm(games, desc="subiendo"):
        try:
            game_id = post_game(g["pgn"])
            game_map[game_id] = g["url"]
        except Exception as e:
            print("Error al subir partida:", e)

    print("Esperando métricas…")
    results = []
    for gid in tqdm(game_map, desc="metrics"):
        try:
            m = wait_game_metrics(gid)
            results.append((gid, m))
        except Exception as e:
            print("Timeout en game", gid, e)

    # ── mostrar resumen ───────────────────────────────────
    print("\nResumen por partida:")
    for gid, m in sorted(results, key=lambda x: x[1]["pct_top3"], reverse=True):
        flag = "⚠️" if m["suspicious"] else "  "
        print(f"{flag} pct_top3={m['pct_top3']:.1f}%  ACL={m['acl']:.1f}  → {game_map[gid]}")

    pm = get_player_metrics(username)
    if pm:
        print("\nPlayerMetrics:")
        print(json.dumps(pm, indent=2))
    else:
        print("\nPlayerMetrics aún no disponible (necesita ≥5 partidas).")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAbortado por el usuario.")
