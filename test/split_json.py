#!/usr/bin/env python3
import argparse
import json
import math
import sys
from pathlib import Path


def read_games(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("El JSON esperado debe ser una lista de partidas.")
    return data


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def split_per_game(games, out_dir: Path):
    out = out_dir / "per_game"
    count = 0
    for i, g in enumerate(games, start=1):
        name = f"game_{i:05d}.json"
        write_json(out / name, g)
        count += 1
    print(f"Per-game: {count} archivos en {out}")


def split_chunks(games, out_dir: Path, chunk_size: int):
    out = out_dir / f"chunks_{chunk_size}"
    total = len(games)
    chunks = math.ceil(total / chunk_size)
    for ci in range(chunks):
        start = ci * chunk_size
        end = min(start + chunk_size, total)
        part = games[start:end]
        name = f"chunk_{ci+1:05d}.json"
        write_json(out / name, part)
    print(f"Chunks: {chunks} archivos en {out} (size={chunk_size}, total_games={total})")


def main():
    p = argparse.ArgumentParser(description="Split Chess.com games JSON")
    p.add_argument("--input", "-i", default="test/data/input.json", help="Ruta del JSON de entrada")
    p.add_argument("--mode", "-m", choices=["per-game", "chunks"], required=True, help="Modo de salida")
    p.add_argument("--chunk-size", "-s", type=int, default=10, help="Tamaño del chunk (para mode=chunks)")
    p.add_argument("--out-dir", "-o", default="test/out", help="Directorio de salida")
    args = p.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)

    if not input_path.exists():
        print(f"No existe el archivo de entrada: {input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        games = read_games(input_path)
    except Exception as e:
        print(f"Error leyendo JSON: {e}", file=sys.stderr)
        sys.exit(1)

    if args.mode == "per-game":
        split_per_game(games, out_dir)
    else:
        if args.chunk_size <= 0:
            print("--chunk-size debe ser > 0", file=sys.stderr)
            sys.exit(2)
        split_chunks(games, out_dir, args.chunk_size)

    print("OK")


if __name__ == "__main__":
    main()
