#!/usr/bin/env python3
"""Herramientas de línea de comandos (y funciones) para manejar migraciones Alembic.

Uso rápido en CLI:

    python -m app.db_migrations upgrade        # aplica todas las migraciones
    python -m app.db_migrations downgrade -1   # revierte la última migración
    python -m app.db_migrations downgrade base # revierte hasta la versión 'base'

Las funciones internas se pueden importar desde otros módulos/tests
para automatizar despliegues.

Variables de entorno:
    DATABASE_URL    → URL de conexión a la base de datos.
    ALEMBIC_CONFIG  → Ruta a un alembic.ini personalizado (opcional).

El script genera un objeto ``alembic.config.Config`` en memoria para
no depender de un fichero .ini si no existe, apuntando a la carpeta
``backend/alembic`` del proyecto.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

from alembic import command
from alembic.config import Config

# ---------------------------------------------------------------------------
# Configuración Alembic dinámica
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # → <repo>/backend
_SCRIPT_LOCATION = _PROJECT_ROOT / "alembic"  # carpeta con env.py y versions/


def _make_alembic_cfg(sql_url: Optional[str] = None) -> Config:
    """Crea y devuelve un objeto Config listo para usar con ``alembic.command``.

    Args:
        sql_url: Cadena de conexión. Si es ``None`` se leerá de la variable
                  ``DATABASE_URL`` o se usará un valor por defecto.
    """
    cfg = Config()  # no necesitamos un .ini físico; configuramos en memoria

    # Ubicación de scripts de migración
    cfg.set_main_option("script_location", str(_SCRIPT_LOCATION))

    # SQLAlchemy URL
    db_url = sql_url or os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://chess:chess@localhost:5432/chessdb",
    )
    cfg.set_main_option("sqlalchemy.url", db_url)

    # Otras opciones útiles
    cfg.set_main_option("sqlalchemy.echo", "False")
    cfg.set_main_option("timezone", "utc")

    return cfg


# ---------------------------------------------------------------------------
# Funciones públicas
# ---------------------------------------------------------------------------

def upgrade(revision: str = "head") -> None:
    """Aplica migraciones hasta ``revision`` (default: head)."""
    cfg = _make_alembic_cfg()
    command.upgrade(cfg, revision)


def downgrade(revision: str = "-1") -> None:
    """Revierte migraciones hasta ``revision`` (default: un paso).

    Ejemplos:
        downgrade("-1")   → deshacer la última migración
        downgrade("base") → deshacer TODO y dejar la base limpia
        downgrade("9a2b9628cf02") → ir a un revision id específico
    """
    cfg = _make_alembic_cfg()
    command.downgrade(cfg, revision)


def current() -> None:
    """Muestra la versión actual de la base de datos."""
    cfg = _make_alembic_cfg()
    command.current(cfg)


def history() -> None:
    """Muestra el historial de migraciones disponibles."""
    cfg = _make_alembic_cfg()
    command.history(cfg)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Gestión de migraciones Alembic")
    sub = p.add_subparsers(dest="cmd", required=True)

    # upgrade
    up = sub.add_parser("upgrade", help="Aplicar migraciones (default: head)")
    up.add_argument("revision", nargs="?", default="head", help="Revision destino (ej.: head, +1, 9a2b9628cf02)")

    # downgrade
    down = sub.add_parser("downgrade", help="Revertir migraciones (default: -1)")
    down.add_argument("revision", nargs="?", default="-1", help="Revision a la que volver (ej.: -1, base, 9a2b9628cf02)")

    # current
    sub.add_parser("current", help="Mostrar versión actual de la BD")

    # history
    sub.add_parser("history", help="Mostrar historial de migraciones")

    return p


def _main(argv: Optional[list[str]] = None) -> None:
    args = _build_parser().parse_args(argv)

    match args.cmd:
        case "upgrade":
            upgrade(args.revision)
        case "downgrade":
            downgrade(args.revision)
        case "current":
            current()
        case "history":
            history()
        case _:
            raise SystemExit("Comando no reconocido")


if __name__ == "__main__":
    _main() 