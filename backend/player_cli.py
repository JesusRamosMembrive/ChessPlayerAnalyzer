#!/usr/bin/env python3
"""
CLI para usar la Chess-Analyzer API:
1. Pregunta el username (o usa --user).
2. GET /players/{username}
   • Si status == "ready" → imprime resultado y sale.
   • Si status == "pending" o "not_analyzed" → muestra una barra de
     progreso leyendo del stream SSE /stream/{username} hasta terminar.
Requisitos: requests, sseclient-py  (pip install requests sseclient-py)
"""

from __future__ import annotations

import json

import argparse
import requests
import sys
import time

API = "http://localhost:8000"

# --- sesión HTTP reutilizable ---------------------------------
s = requests.Session()
s.headers["User-Agent"] = "chess-cli/0.1 (+https://tu-url)"


def get_player(username: str) -> dict | None:
    r = s.get(f"{API}/players/{username}", timeout=5)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()

def wait_player(username: str):
    bar = lambda p: f"[{'#'*(p//5):<20}] {p:3d}%"
    while True:
        time.sleep(1)
        info = get_player(username)
        if not info:                # aún no creado
            print("\r Creando registro…      ", end="", flush=True)
            continue

        if info["status"] == "ready":
            print(f"\n✅ Análisis completo ({info['progress']} %).")
            return
        if info["status"] == "error":
            print("\n❌ Error:", info["error"]); sys.exit(1)

        prog = info.get("progress", 0)
        print("\r"+bar(prog), end="", flush=True)

def pretty(data: dict) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)

def stream_progress(username: str):
    while True:
        time.sleep(0.2)
        player = get_player(username)
        prog   = player.get("progress", 0)
        bar = f"[{'#' * (prog//5):<20}] {prog:3d}%"
        print(f"\r{bar}", end="", flush=True)
        if player.get("status") == "ready":
            print("\n✅ Análisis completo.")
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-u", "--user", help="usuario de Chess.com")
    args = ap.parse_args()

    username = args.user or input("Jugador Chess.com a analizar: ").strip()
    if not username:
        print("→ username vacío.")
        return

    p = get_player(username)

    if not p:
        print("⏳ Primera vez. Lanzando análisis…")
        s.post(f"{API}/players/{username}")  # endpoint que llama process_player
    wait_player(username)
    status = p["status"]

    if status == "ready":
        print("✅ Ya estaba analizado — datos disponibles:")
        print(pretty(p))
        return

    # status == pending  (o acaba de crearse not_analyzed→pending)
    stream_progress(username)

    # When ready, fetch final data
    print("\n📊 Datos finales:")
    player = get_player(username)
    print(pretty(player))


if __name__ == "__main__":
    main()
