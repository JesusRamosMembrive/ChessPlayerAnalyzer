# app/infrastructure/http_client.py
"""
HttpClient para extraer las llamadas HTTP a Chess.com API.

Siguiendo el patrón establecido en fases 1A-1C:
- Funciones simples, no clases complejas
- Testeable independientemente
- Sin side effects en imports
"""
from __future__ import annotations

import json
import logging
import pathlib
import re
from datetime import datetime, UTC
from typing import List, Dict
import os
import requests


# Configuration
CLK_RGX = re.compile(r"\[%clk\s+([\d:.]+)]")
UA = "chess-analyzer/0.2 (+https://github.com/tu_usuario)"


def _sec(t: str) -> int:
    """Convierte «hh:mm:ss.f» o «m:ss.f»  → segundos (int)."""
    parts = list(map(float, t.split(":")))
    parts = [0] * (3 - len(parts)) + parts
    if len(parts) == 3:
        h, m, s = parts
    else:
        h, m, s = 0, *parts
    return int(h * 3600 + m * 60 + s)


def fetch_games_from_chesscom(username: str, months: int = 12) -> List[Dict]:
    """
    Descarga partidas de Chess.com API.

    Returns:
        Lista de dicts con pgn, move_times, white, black, end_time
    """
    logging.info(f"fetch_games_from_chesscom: Starting for {username}, months={months}")

    s = requests.Session()
    s.headers["User-Agent"] = UA

    arch_url = f"https://api.chess.com/pub/player/{username}/games/archives"
    logging.info(f"fetch_games_from_chesscom: Fetching archives from {arch_url}")

    try:
        resp = s.get(arch_url, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error(f"fetch_games_from_chesscom: Error fetching archives: {e}")
        raise

    archives = resp.json()["archives"][-months:]  # los más recientes
    logging.info(f"fetch_games_from_chesscom: Found {len(archives)} archives to process")

    games: list[dict] = []

    for idx, url in enumerate(archives):
        logging.info(f"fetch_games_from_chesscom: Processing archive {idx + 1}/{len(archives)}: {url}")
        try:
            data = s.get(url, timeout=10).json()
            logging.info(f"fetch_games_from_chesscom: Found {len(data.get('games', []))} games in archive")

            for g in data["games"]:
                pgn = g["pgn"]
                clocks = CLK_RGX.findall(pgn)
                move_times = [
                    _sec(clocks[i - 1]) - _sec(clocks[i])
                    for i in range(1, len(clocks))
                ] if clocks else []

                games.append({
                    "pgn": pgn,
                    "move_times": move_times,
                    "white": g["white"]["username"],
                    "black": g["black"]["username"],
                    "end_time": datetime.fromtimestamp(g["end_time"], UTC).isoformat(),
                })
        except Exception as e:
            logging.error(f"fetch_games_from_chesscom: Error processing archive {url}: {e}")
            continue

    logging.info(f"fetch_games_from_chesscom: {username} → {len(games)} partidas")
    return games


def save_games_archive(games: List[Dict], username: str) -> None:
    """
    Guarda copia local de las partidas descargadas.
    Separado para mejor testabilidad.
    """
    try:
        archive_dir = pathlib.Path(
            os.getenv("FETCH_ARCHIVE_DIR", "archives")
        ).expanduser().resolve()

        archive_dir.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = archive_dir / f"{username}_{stamp}.json"

        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(games, fh, ensure_ascii=False, indent=2)

        logging.info(f"save_games_archive: Saved {len(games)} games to {out_path}")
    except Exception as exc:
        # No queremos que un fallo de disco interrumpa el análisis
        logging.warning(f"save_games_archive: could not archive games → {exc}")


# Factory function following established pattern
def get_http_client():
    """
    Factory function para HttpClient.
    Por ahora retorna las funciones directamente.
    En el futuro podría retornar una clase si se necesita estado.
    """
    return {
        'fetch_games_from_chesscom': fetch_games_from_chesscom,
        'save_games_archive': save_games_archive,
    }