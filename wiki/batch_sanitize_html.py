#!/usr/bin/env python3
"""
Batch sanitizer para HTML usando el script TypeScript `sanitize-html.ts`.

Ejemplos:
  # Sanitizar todos los .html en el directorio actual
  python batch_sanitize_html.py .

  # Sanitizar otra carpeta indicando la ruta del TS
  python batch_sanitize_html.py ./docs --ts-script ./sanitize-html.ts

  # Incluir también .htm y sobrescribir salidas si existen
  python batch_sanitize_html.py ./docs --include-htm --overwrite
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Iterable, List, Tuple


def find_html_files(root: Path, include_htm: bool) -> Iterable[Path]:
    """Devuelve los archivos .html (y opcionalmente .htm) dentro de root (recursivo)."""
    exts = {".html"}
    if include_htm:
        exts.add(".htm")
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def is_already_sanitized(path: Path, suffix: str) -> bool:
    """Evita re-procesar salidas ya 'sanitazed'."""
    expected_tail = f"{suffix}{path.suffix.lower()}"
    return path.name.lower().endswith(expected_tail)


def output_for(path: Path, suffix: str) -> Path:
    """Genera la ruta de salida insertando el sufijo antes de la extensión."""
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def run_ts_sanitizer(
    ts_script: Path, src: Path, dst: Path, dry_run: bool
) -> Tuple[bool, str]:
    """
    Lanza: npx -y ts-node <ts_script> <src> --out <dst>
    Devuelve (ok, mensaje).
    """
    cmd: List[str] = [
        "npx",
        "-y",
        "ts-node",
        str(ts_script),
        str(src),
        "--out",
        str(dst),
    ]

    if dry_run:
        return True, f"[dry-run] {' '.join(cmd)}"

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        return False, "Error: no se encontró 'npx'. ¿Está Node.js instalado y en PATH?"
    except Exception as exc:
        return False, f"Error inesperado al ejecutar el sanitizer: {exc}"

    if proc.returncode != 0:
        return (
            False,
            f"Fallo sanitizando '{src.name}':\n"
            f"  comando: {' '.join(cmd)}\n"
            f"  stdout: {proc.stdout.strip()}\n"
            f"  stderr: {proc.stderr.strip()}",
        )

    return True, f"OK  → {src.name}  ⟶  {dst.name}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sanitiza en lote archivos HTML con sanitize-html.ts"
    )
    parser.add_argument(
        "directory", nargs="?", default=".", help="Directorio raíz a escanear (por defecto '.')"
    )
    parser.add_argument(
        "--ts-script",
        default="sanitize-html.ts",
        help="Ruta al script TypeScript sanitizer (por defecto 'sanitize-html.ts')",
    )
    parser.add_argument(
        "--suffix",
        default="-sanitazed",
        help="Sufijo a añadir antes de la extensión (por defecto '-sanitazed')",
    )
    parser.add_argument(
        "--include-htm",
        action="store_true",
        help="Incluir también archivos .htm además de .html",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobrescribir archivos de salida si ya existen",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="No ejecuta nada, solo muestra qué haría",
    )

    args = parser.parse_args()

    root = Path(args.directory).resolve()
    ts_script = Path(args.ts_script).resolve()
    suffix: str = str(args.suffix)

    if not root.exists():
        print(f"Error: el directorio '{root}' no existe.")
        raise SystemExit(1)

    # No forzamos la existencia de ts_script: puede resolverse por cwd.
    if not ts_script.exists():
        print(
            f"Aviso: no se encontró '{ts_script}'. Se intentará igualmente con npx/ts-node."
        )

    files = list(find_html_files(root, args.include_htm))
    if not files:
        print("No se encontraron archivos .html (ni .htm).")
        return

    total = len(files)
    processed = 0
    skipped = 0
    failed = 0

    print(f"Encontrados {total} archivo(s). Comenzando...\n")

    for src in files:
        # Evita re-procesar si ya tienen el sufijo
        if is_already_sanitized(src, suffix):
            print(f"SKIP (ya sanitazed) → {src.relative_to(root)}")
            skipped += 1
            continue

        dst = output_for(src, suffix)

        # Evita sobrescribir si no se pidió --overwrite
        if dst.exists() and not args.overwrite:
            print(f"SKIP (existe salida) → {dst.relative_to(root)}  (usa --overwrite para forzar)")
            skipped += 1
            continue

        ok, msg = run_ts_sanitizer(ts_script, src, dst, args.dry_run)
        if ok:
            processed += 1
        else:
            failed += 1
        print(msg)

    print("\nResumen:")
    print(f"  Total      : {total}")
    print(f"  Procesados : {processed}")
    print(f"  Omitidos   : {skipped}")
    print(f"  Fallidos   : {failed}")

    if failed > 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
