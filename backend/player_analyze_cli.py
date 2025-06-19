#!/usr/bin/env python3
"""
player_analyze_cli.py

CLI "sin‑IU" para Chess Player Pro.
Se conecta al backend FastAPI (o Flask) mediante HTTP; no importa dónde
esté desplegado mientras la API respete los endpoints que se indican abajo.

Endpoints esperados (métodos HTTP):
  GET    /players                    -> lista de jugadores almacenados
  GET    /players/{username}         -> datos completos de un jugador
  POST   /players/{username}         -> lanzar nuevo análisis (CORREGIDO)
  DELETE /players/{username}         -> borrar jugador y sus datos

Si tu API usa rutas distintas, basta con cambiar las constantes ENDPOINT_*.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import httpx  # pip install httpx
from tabulate import tabulate  # pip install tabulate
from tqdm import tqdm  # pip install tqdm
import time

# --------------------------- Configuración ---------------------------

DEFAULT_HOST = os.getenv("CPP_BACKEND_HOST", "http://localhost:8000")
DEFAULT_TIMEOUT = float(os.getenv("CPP_TIMEOUT", "30"))  # segundos para HTTP
DEFAULT_ANALYSIS_TIMEOUT = float(os.getenv("CPP_ANALYSIS_TIMEOUT", "10800"))  # 3 horas para análisis

ENDPOINT_PLAYERS = "/players"  # lista
ENDPOINT_PLAYER = "/players/{username}"  # show / delete / analyze (POST)


# --------------------------- Utilidades HTTP ------------------------

def make_client(base_url: str, timeout: float) -> httpx.Client:
    return httpx.Client(base_url=base_url, timeout=timeout)


def handle_response(resp: httpx.Response) -> Any:
    """
    Devuelve resp.json() si el status es 2xx.
    Si no, imprime el mensaje de error del backend (si lo hay)
    y finaliza con código !=0.
    """
    if resp.is_success:
        if resp.text:
            try:
                return resp.json()
            except ValueError:
                return resp.text  # p. ej. DELETE devuelve texto plano
        return None
    # --- error ---
    try:
        detail = resp.json().get("detail")
    except ValueError:
        detail = resp.text
    print(f"❌ Error {resp.status_code}: {detail}", file=sys.stderr)
    sys.exit(1)


# --------------------------- Comandos -------------------------------

def wait_for_analysis(client: httpx.Client, username: str,
                      timeout: int = None, analysis_timeout: int = None) -> Dict[str, Any]:
    """
    Espera a que el análisis termine, mostrando una barra de progreso.
    Devuelve el estado final del jugador.

    Args:
        client: Cliente HTTP
        username: Nombre de usuario a analizar
        timeout: Timeout para peticiones HTTP (heredado del cliente)
        analysis_timeout: Timeout total para esperar el análisis (default: 3 horas)
    """
    if analysis_timeout is None:
        analysis_timeout = DEFAULT_ANALYSIS_TIMEOUT

    print(f"⏳ Esperando a que termine el análisis (timeout: {analysis_timeout / 3600:.1f} horas)...")

    with tqdm(total=100, desc="Analizando",
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
        last_progress = 0
        start_time = time.time()

        while True:
            try:
                resp = client.get(ENDPOINT_PLAYER.format(username=username))
                if resp.status_code == 404:
                    time.sleep(1)
                    continue

                data = handle_response(resp)
                current_progress = data.get('progress', 0)

                # Actualizar barra de progreso
                if current_progress > last_progress:
                    pbar.update(current_progress - last_progress)
                    last_progress = current_progress

                # Verificar estado
                status = data.get('status')
                if status == 'ready':
                    pbar.update(100 - last_progress)
                    print("\n✅ Análisis completado exitosamente")
                    return data

                elif status == 'error':
                    print(f"\n❌ Error durante el análisis: {data.get('error', 'Error desconocido')}")
                    sys.exit(1)

                # Verificar timeout
                elapsed = time.time() - start_time
                if elapsed > analysis_timeout:
                    print(f"\n❌ Timeout esperando el análisis después de {elapsed / 3600:.1f} horas")
                    sys.exit(1)

                time.sleep(1)

            except Exception as e:
                print(f"\n❌ Error al verificar estado: {e}")
                sys.exit(1)


def cmd_list(args: argparse.Namespace) -> None:
    with make_client(args.host, args.timeout) as client:
        data = handle_response(client.get(ENDPOINT_PLAYERS))

    if not data:
        print("No hay jugadores almacenados todavía.")
        return

    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        rows = [[p["username"], p.get("last_analysis"), p.get("country"),
                 p.get("rapid_rating"), p.get("blitz_rating")] for p in data]
        print(tabulate(
            rows,
            headers=["Usuario", "Últ. análisis", "País", "Rápido", "Blitz"],
            tablefmt="github",
            numalign="right"))


def cmd_show(args: argparse.Namespace) -> None:
    with make_client(args.host, args.timeout) as client:
        data = handle_response(
            client.get(ENDPOINT_PLAYER.format(username=args.username)))

    print(json.dumps(data, indent=2, ensure_ascii=False)
          if args.json else tabulate_dict(data))


def cmd_analyze(args: argparse.Namespace) -> None:
    with make_client(args.host, args.timeout) as client:
        # Primero verificar si ya existe
        resp = client.get(ENDPOINT_PLAYER.format(username=args.username))

        if resp.status_code == 200:
            existing_data = resp.json()
            status = existing_data.get('status')

            if status == 'ready':
                print(f"ℹ️  El jugador ya está analizado.")
                if not args.force:
                    print("Usa --force para forzar un nuevo análisis.")
                    print(json.dumps(existing_data, indent=2, ensure_ascii=False))
                    return
                print("Forzando nuevo análisis...")

            elif status == 'pending':
                print("ℹ️  Ya hay un análisis en progreso.")
                if args.wait:
                    final_data = wait_for_analysis(client, args.username,
                                                   analysis_timeout=args.analysis_timeout)
                    if args.json:
                        print(json.dumps(final_data, indent=2, ensure_ascii=False))
                    return
                else:
                    print("Usa --wait para esperar a que termine.")
                    return

        # Lanzar nuevo análisis
        data = handle_response(
            client.post(ENDPOINT_PLAYER.format(username=args.username)))

        print("✅ Análisis lanzado correctamente.")
        if data and args.json and not args.wait:
            print(json.dumps(data, indent=2, ensure_ascii=False))

        # Esperar si se solicita
        if args.wait:
            final_data = wait_for_analysis(client, args.username,
                                           analysis_timeout=args.analysis_timeout)
            if args.json:
                print(json.dumps(final_data, indent=2, ensure_ascii=False))
            else:
                # Mostrar resumen del análisis
                print("\n📊 Resumen del análisis:")
                print(f"Usuario: {final_data.get('username')}")
                print(f"País: {final_data.get('country', 'N/A')}")
                print(f"Rating Rápido: {final_data.get('rapid_rating', 'N/A')}")
                print(f"Rating Blitz: {final_data.get('blitz_rating', 'N/A')}")
                if 'game_count' in final_data:
                    print(f"Partidas analizadas: {final_data.get('game_count')}")
                if 'opening_entropy' in final_data:
                    print(f"Entropía de aperturas: {final_data.get('opening_entropy', 0):.2f}")
                if 'low_entropy' in final_data:
                    print(f"Baja entropía: {'⚠️ Sí' if final_data.get('low_entropy') else '✓ No'}")


def cmd_delete(args: argparse.Namespace) -> None:
    with make_client(args.host, args.timeout) as client:
        handle_response(
            client.delete(ENDPOINT_PLAYER.format(username=args.username)))

    print(f"🗑️ Jugador «{args.username}» eliminado.")


# --------------------------- Helpers de formato ---------------------

def tabulate_dict(d: Dict[str, Any]) -> str:
    """Representa un dict anidado en forma de tabla clave/valor."""
    rows: List[List[str]] = []

    def rec(prefix: str, val: Any):
        if isinstance(val, dict):
            for k, v in val.items():
                rec(f"{prefix}{k}.", v)
        else:
            rows.append([prefix.rstrip("."), val])

    rec("", d)
    return tabulate(rows, headers=["Campo", "Valor"], tablefmt="github")


# --------------------------- CLI parser -----------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cpp",
        description="Chess Player Pro – CLI temporal sin interfaz gráfica")
    p.add_argument("--host", default=DEFAULT_HOST,
                   help=f"URL base del backend (defecto: {DEFAULT_HOST})")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                   help=f"Timeout en segundos para peticiones HTTP (defecto: {DEFAULT_TIMEOUT})")
    p.add_argument("--analysis-timeout", type=float, default=DEFAULT_ANALYSIS_TIMEOUT,
                   help=f"Timeout en segundos para esperar análisis completo (defecto: {DEFAULT_ANALYSIS_TIMEOUT / 3600:.1f} horas)")
    p.add_argument("--json", action="store_true",
                   help="Mostrar la salida en JSON puro (sin tabular)")

    sub = p.add_subparsers(dest="cmd", required=True)

    # list
    sub.add_parser("list", help="Listar todos los jugadores").set_defaults(
        func=cmd_list)

    # show
    sp_show = sub.add_parser("show", help="Mostrar detalle de un jugador")
    sp_show.add_argument("username", help="Nombre de usuario en chess.com")
    sp_show.set_defaults(func=cmd_show)

    # analyze
    sp_analyze = sub.add_parser("analyze", help="Analizar nuevo jugador")
    sp_analyze.add_argument("username", help="Nombre de usuario en chess.com")
    sp_analyze.add_argument("--wait", "-w", action="store_true",
                            help="Esperar a que termine el análisis")
    sp_analyze.add_argument("--force", "-f", action="store_true",
                            help="Forzar nuevo análisis aunque ya exista")
    sp_analyze.set_defaults(func=cmd_analyze)

    # delete
    sp_del = sub.add_parser("delete", help="Eliminar un jugador")
    sp_del.add_argument("username", help="Nombre de usuario en chess.com")
    sp_del.set_defaults(func=cmd_delete)

    return p


# --------------------------- main -----------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()