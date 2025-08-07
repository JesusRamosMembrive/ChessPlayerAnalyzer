#!/usr/bin/env python3
"""Utilidad CLI para crear copias de seguridad de la base de datos PostgreSQL
(archivos .sql.gz) y restaurarlas posteriormente.

Uso rápido:

    python -m app.db_backup backup                # crea backup en ./db_backups/
    python -m app.db_backup backup --out /tmp     # backup en directorio custom

    python -m app.db_backup restore ./db_backups/chessdb_20250703_153012.sql.gz

El comando se basa en las herramientas cliente de PostgreSQL `pg_dump` y
`psql`. Ambos deben estar disponibles en el *PATH* dentro del contenedor
(docker) o el entorno donde se ejecute el script.

Variables de entorno compatibles (todas opcionales):
    DATABASE_URL  → Cadena SQLAlchemy/PostgreSQL (tiene prioridad)
    PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE → se usan como fallback

El script detecta automáticamente credenciales a partir de `DATABASE_URL`
(si existe) y las exporta como variables de entorno para los procesos hijos
para evitar prompts interactivos de contraseña.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple
from urllib.parse import urlparse
import shutil  # Agregado para _ensure_tools

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_conn() -> Tuple[str, str, str, int, str]:
    """Devuelve (user, password, host, port, db) a partir de DATABASE_URL o
    variables PG*. Prefiere DATABASE_URL si está definida.
    """
    url = os.getenv("DATABASE_URL")
    if url:
        # Aceptamos esquemas como postgresql+psycopg o postgresql
        url = url.replace("postgresql+psycopg://", "postgresql://")
        parsed = urlparse(url)
        user = parsed.username or os.getenv("PGUSER", "chess")
        password = parsed.password or os.getenv("PGPASSWORD", "chess")
        host = parsed.hostname or os.getenv("PGHOST", "localhost")
        port = parsed.port or int(os.getenv("PGPORT", "5432"))
        db = parsed.path.lstrip("/") or os.getenv("PGDATABASE", "chessdb")
    else:
        user = os.getenv("PGUSER", "chess")
        password = os.getenv("PGPASSWORD", "chess")
        host = os.getenv("PGHOST", "localhost")
        port = int(os.getenv("PGPORT", "5432"))
        db = os.getenv("PGDATABASE", "chessdb")
    return user, password, host, port, db


def _ensure_tools(*tools: str) -> None:
    """Comprueba que las herramientas indicadas existen en PATH."""
    for t in tools:
        if not shutil.which(t):
            sys.exit(f"❌ La herramienta '{t}' no está disponible en PATH. Instala el cliente de PostgreSQL.")


# ---------------------------------------------------------------------------
# Acciones
# ---------------------------------------------------------------------------

def backup(out_dir: Path, compress: bool = True) -> Path:
    """Crea una copia de seguridad y devuelve la ruta al archivo generado."""
    import shutil  # import interno para no requerir en restore

    user, password, host, port, db = _parse_conn()

    _ensure_tools("pg_dump", "gzip" if compress else "pg_dump")

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{db}_{ts}.sql"
    if compress:
        filename += ".gz"
    dest = out_dir / filename

    env = os.environ.copy()
    env["PGPASSWORD"] = password

    dump_cmd = [
        "pg_dump",
        "-h",
        host,
        "-p",
        str(port),
        "-U",
        user,
        "-d",
        db,
        "-F",
        "p",  # formato plain SQL
    ]

    if compress:
        # pg_dump | gzip > file.sql.gz
        with open(dest, "wb") as fout:
            proc_dump = subprocess.Popen(dump_cmd, env=env, stdout=subprocess.PIPE)
            proc_gzip = subprocess.Popen(["gzip", "-c"], stdin=proc_dump.stdout, stdout=fout)
            proc_gzip.communicate()
            ret = proc_gzip.returncode
    else:
        with open(dest, "wb") as fout:
            ret = subprocess.call(dump_cmd, env=env, stdout=fout)

    if ret != 0:
        dest.unlink(missing_ok=True)
        sys.exit("❌ Error creando la copia de seguridad. Revisa credenciales y conectividad.")

    print(f"✅ Copia de seguridad creada: {dest}")
    return dest


def restore(backup_file: Path) -> None:
    """Restaura la base de datos a partir de un archivo .sql o .sql.gz."""
    import shutil

    if not backup_file.exists():
        sys.exit(f"❌ El archivo {backup_file} no existe")

    user, password, host, port, db = _parse_conn()
    _ensure_tools("psql")

    env = os.environ.copy()
    env["PGPASSWORD"] = password

    # 1. Creación de DB si no existe
    create_cmd = [
        "psql",
        "-h",
        host,
        "-p",
        str(port),
        "-U",
        user,
        "-tc",
        f"SELECT 1 FROM pg_database WHERE datname='{db}'",
        "postgres",
    ]
    exists = subprocess.check_output(create_cmd, env=env).strip()
    if not exists:
        subprocess.check_call(
            [
                "psql",
                "-h",
                host,
                "-p",
                str(port),
                "-U",
                user,
                "-c",
                f"CREATE DATABASE {db};",
                "postgres",
            ],
            env=env,
        )
        print(f"ℹ️  Base de datos '{db}' creada.")

    # 2. Restaurar
    if backup_file.suffix == ".gz":
        decompress = subprocess.Popen(["gunzip", "-c", str(backup_file)], stdout=subprocess.PIPE)
        ret = subprocess.call(
            [
                "psql",
                "-h",
                host,
                "-p",
                str(port),
                "-U",
                user,
                "-d",
                db,
            ],
            env=env,
            stdin=decompress.stdout,
        )
    else:
        ret = subprocess.call(
            [
                "psql",
                "-h",
                host,
                "-p",
                str(port),
                "-U",
                user,
                "-d",
                db,
                "-f",
                str(backup_file),
            ],
            env=env,
        )

    if ret != 0:
        sys.exit("❌ Error restaurando la base de datos.")
    print("✅ Restauración completada con éxito.")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Copias de seguridad PostgreSQL")
    sub = p.add_subparsers(dest="cmd", required=True)

    # backup
    bkp = sub.add_parser("backup", help="Crear copia de seguridad")
    bkp.add_argument("--out", dest="out", default="./db_backups", help="Directorio destino (default: ./db_backups)")
    bkp.add_argument("--no-compress", action="store_true", help="No comprimir con gzip")

    # restore
    rst = sub.add_parser("restore", help="Restaurar desde backup")
    rst.add_argument("file", help="Ruta al archivo .sql[.gz]")

    return p


def _main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    match args.cmd:
        case "backup":
            backup(Path(args.out), compress=not args.no_compress)
        case "restore":
            restore(Path(args.file))
        case _:
            raise SystemExit("Comando no reconocido")


if __name__ == "__main__":
    _main() 