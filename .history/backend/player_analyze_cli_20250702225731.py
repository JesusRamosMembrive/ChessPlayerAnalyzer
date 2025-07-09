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
from datetime import datetime, timezone
import logging

import httpx  # pip install httpx
from tabulate import tabulate  # pip install tabulate
from tqdm import tqdm  # pip install tqdm
import time

# Logger
logging.basicConfig(
    level=logging.INFO,          # muestra info, warning y error
    format="%(message)s"         # sin adornos de fecha/módulo
)

logger = logging.getLogger(__name__)

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
    logger.error(f"❌ Error {resp.status_code}: {detail}")
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

    logger.info(f"⏳ Esperando a que termine el análisis (timeout: {analysis_timeout / 3600:.1f} horas)...")

    with tqdm(total=100, desc="Analizando",
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
        last_progress = 0
        start_time = time.time()
        poll_count = 0

        while True:
            try:
                poll_count += 1
                resp = client.get(ENDPOINT_PLAYER.format(username=username))
                if resp.status_code == 404:
                    time.sleep(1)
                    continue

                data = handle_response(resp)
                current_progress = data.get('progress', 0)

                # Debug: mostrar estado cada 10 polls
                if poll_count % 10 == 0:
                    pbar.set_postfix_str(f"Estado: {data.get('status', 'unknown')}")

                # Actualizar barra de progreso
                if current_progress > last_progress:
                    pbar.update(current_progress - last_progress)
                    last_progress = current_progress

                # Verificar estado
                status = data.get('status')
                if status == 'ready':
                    pbar.update(100 - last_progress)
                    logger.info("\n✅ Análisis completado exitosamente")
                    return data

                elif status == 'error':
                    logger.error(f"\n❌ Error durante el análisis: {data.get('error', 'Error desconocido')}")
                    sys.exit(1)

                elif status not in ['pending', 'processing']:
                    # Estado inesperado - salir del bucle
                    logger.warning(f"\n⚠️ Estado inesperado: '{status}' - finalizando espera")
                    return data

                # Verificar timeout
                elapsed = time.time() - start_time
                if elapsed > analysis_timeout:
                    logger.error(f"\n❌ Timeout esperando el análisis después de {elapsed / 3600:.1f} horas")
                    sys.exit(1)

                # Esperar antes del siguiente poll
                # Aumentar el intervalo si llevamos mucho tiempo esperando
                if elapsed > 300:  # Más de 5 minutos
                    time.sleep(2)  # Poll cada 2 segundos
                else:
                    time.sleep(1)  # Poll cada 1 segundo

            except KeyboardInterrupt:
                logger.warning("\n⚠️ Interrumpido por el usuario")
                raise
            except Exception as e:
                logger.error(f"\n❌ Error al verificar estado: {e}")
                sys.exit(1)


def cmd_list(args: argparse.Namespace) -> None:
    with make_client(args.host, args.timeout) as client:
        data = handle_response(client.get(ENDPOINT_PLAYERS))

    if not data:
        logger.info("No hay jugadores almacenados todavía.")
        return

    logger.info(f"\n<UNK> Jugadores almacenados: {len(data)}")
    # Mostrar en formato JSON o tabular
    # Si hay pocos jugadores, mostrar en JSON; si son muchos, tabular
    if len(data) > 100:
        logger.info("Demasiados jugadores para mostrar en JSON. Usando formato tabular.")
        args.json = False

    if args.json:
        logger.info(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        rows = [[p["username"], p.get("last_analysis"), p.get("country"),
                 p.get("rapid_rating"), p.get("blitz_rating")] for p in data]
        logger.info(tabulate(
            rows,
            headers=["Usuario", "Últ. análisis", "País", "Rápido", "Blitz"],
            tablefmt="github",
            numalign="right"))


def cmd_show(args: argparse.Namespace) -> None:
    with make_client(args.host, args.timeout) as client:
        data = handle_response(
            client.get(ENDPOINT_PLAYER.format(username=args.username)))

    logger.info(json.dumps(data, indent=2, ensure_ascii=False)
          if args.json else tabulate_dict(data))


def cmd_analyze(args: argparse.Namespace) -> None:
    logger.info("DEBUG CLI: Starting analysis command")
    logger.info(f"DEBUG CLI: Input parameters - username: {args.username}, host: {args.host}, timeout: {args.timeout}")
    logger.info(f"DEBUG CLI: Analysis timeout: {args.analysis_timeout}, wait: {args.wait}, force: {args.force}")

    with make_client(args.host, args.timeout) as client:
        # Primero verificar si ya existe
        logger.info(f"DEBUG CLI: Checking existing player status for {args.username}")
        resp = client.get(ENDPOINT_PLAYER.format(username=args.username))
        logger.info(f"DEBUG CLI: Player status response code: {resp.status_code}")

        if resp.status_code == 200:
            existing_data = resp.json()
            logger.info(f"DEBUG CLI: Existing player data: {json.dumps(existing_data, indent=2, ensure_ascii=False)}")
            status = existing_data.get('status')

            if status == 'ready':
                logger.info("ℹ️  El jugador ya está analizado.")
                if not args.force:
                    logger.info("Usa --force para forzar un nuevo análisis.")
                    logger.info(json.dumps(existing_data, indent=2, ensure_ascii=False))
                    return
                logger.info("Forzando nuevo análisis...")

            elif status == 'pending':
                logger.info("ℹ️  Ya hay un análisis en progreso.")
                if args.wait:
                    final_data = wait_for_analysis(client, args.username,
                                                   analysis_timeout=args.analysis_timeout)
                    if args.json:
                        logger.info(json.dumps(final_data, indent=2, ensure_ascii=False))
                    return
                else:
                    logger.info("Usa --wait para esperar a que termine.")
                    return

        # Lanzar nuevo análisis
        logger.info(f"DEBUG CLI: Launching new analysis for {args.username}")
        data = handle_response(
            client.post(ENDPOINT_PLAYER.format(username=args.username)))
        logger.info(f"DEBUG CLI: Analysis launch response: {json.dumps(data, indent=2, ensure_ascii=False) if data else 'None'}")

        logger.info("✅ Análisis lanzado correctamente.")
        if data and args.json and not args.wait:
            logger.info(json.dumps(data, indent=2, ensure_ascii=False))

        # Esperar si se solicita
        if args.wait:
            logger.info("DEBUG CLI: Waiting for analysis completion...")
            final_data = wait_for_analysis(client, args.username,
                                           analysis_timeout=args.analysis_timeout)
            logger.info(f"DEBUG CLI: Final analysis data: {json.dumps(final_data, indent=2, ensure_ascii=False)}")
            if args.json:
                logger.info(json.dumps(final_data, indent=2, ensure_ascii=False))
            else:
                # Mostrar resumen del análisis
                logger.info("\n📊 Resumen del análisis:")
                logger.info(f"Usuario: {final_data.get('username')}")
                logger.info(f"País: {final_data.get('country', 'N/A')}")
                logger.info(f"Rating Rápido: {final_data.get('rapid_rating', 'N/A')}")
                logger.info(f"Rating Blitz: {final_data.get('blitz_rating', 'N/A')}")
                if 'game_count' in final_data:
                    logger.info(f"Partidas analizadas: {final_data.get('game_count')}")
                if 'opening_entropy' in final_data:
                    logger.info(f"Entropía de aperturas: {final_data.get('opening_entropy', 0):.2f}")
                if 'low_entropy' in final_data:
                    logger.info(f"Baja entropía: {'⚠️ Sí' if final_data.get('low_entropy') else '✓ No'}")


def cmd_cancel(args: argparse.Namespace) -> None:
    """Cancela el análisis en curso de un jugador."""
    with make_client(args.host, args.timeout) as client:
        # Primero verificar el estado actual
        player_url = ENDPOINT_PLAYER.format(username=args.username)
        logger.info(f"DEBUG CANCEL: Checking player status at {args.host}{player_url}")
        resp = client.get(player_url)
        logger.info(f"DEBUG CANCEL: Response status code: {resp.status_code}")

        if resp.status_code == 404:
            logger.error(f"❌ Jugador '{args.username}' no encontrado")
            return

        player_data = resp.json()
        logger.info(f"DEBUG CANCEL: Player data: {json.dumps(player_data, indent=2, ensure_ascii=False)}")

        # Verificar si hay un análisis en progreso
        # Consideramos que hay un análisis en progreso si:
        # 1. El estado es "pending" O
        # 2. Hay progreso > 0 O
        # 3. Hay un task_id
        status = player_data.get('status')
        progress = player_data.get('progress', 0)
        task_id = player_data.get('last_task_id')

        logger.info(f"DEBUG CANCEL: Analysis status check - status: {status}, progress: {progress}, task_id: {task_id}")

        # Un análisis está en progreso si el estado es "pending" O si hay progreso > 0
        analysis_in_progress = (status == 'pending') or (progress > 0) or (task_id is not None)

        if not analysis_in_progress:
            if args.force:
                logger.warning(f"⚠️  No se detecta análisis en curso, pero se forzará la cancelación (--force)")
            else:
                logger.warning(f"⚠️  No hay análisis en progreso. Estado actual: {status}")
                logger.info("Usa --force para forzar la cancelación si crees que hay un análisis en curso.")
                return

        # Intentar cancelar
        stop_url = f"{player_url}/stop"
        logger.info(f"DEBUG CANCEL: Sending stop request to {args.host}{stop_url}")
        logger.info(f"🛑 Cancelando análisis de {args.username}...")
        cancel_resp = client.post(stop_url)
        logger.info(f"DEBUG CANCEL: Stop response status code: {cancel_resp.status_code}")

        try:
            response_text = cancel_resp.text
            logger.info(f"DEBUG CANCEL: Raw response: {response_text}")

            if cancel_resp.status_code == 200:
                data = cancel_resp.json()
                logger.info(f"DEBUG CANCEL: Response data: {json.dumps(data, indent=2, ensure_ascii=False)}")

                if data.get('status') == 'stopped':
                    logger.info("✅ Análisis cancelado exitosamente")
                    logger.info(f"   Task ID: {data.get('task_id')}")
                else:
                    logger.warning(f"⚠️  {data.get('message')}")
            else:
                try:
                    error_detail = cancel_resp.json().get('detail', 'Error desconocido')
                except:
                    error_detail = response_text

                # Handle specific error cases
                if "Player not found" in str(error_detail):
                    logger.error(f"❌ Error canceling: Not Found")
                else:
                    logger.error(f"❌ Error canceling: {error_detail}")
        except Exception as e:
            logger.error(f"DEBUG CANCEL: Error processing response: {str(e)}")


def cmd_active(args: argparse.Namespace) -> None:
    """Lista todos los análisis activos."""
    with make_client(args.host, args.timeout) as client:
        resp = client.get("/players/active")
        if resp.status_code != 200:
            logger.error(f"❌ Error al obtener análisis activos: {resp.text}")
            return

        data = resp.json()

        if data['active_count'] == 0:
            logger.info("No hay análisis activos en este momento.")
            return

        logger.info(f"\n📊 Análisis activos: {data['active_count']}")

        if args.json:
            logger.info(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            rows = []
            for analysis in data['analyses']:
                progress = analysis.get('progress', 0)
                done = analysis.get('done_games', 0)
                total = analysis.get('total_games', '?')

                # Calcular tiempo transcurrido
                elapsed = ""
                if analysis.get('started_at'):
                    try:
                        start_time = datetime.fromisoformat(analysis['started_at'].replace('Z', '+00:00'))
                        delta = datetime.now(timezone.utc) - start_time
                        hours = delta.total_seconds() / 3600
                        if hours < 1:
                            elapsed = f"{int(delta.total_seconds() / 60)}m"
                        else:
                            elapsed = f"{hours:.1f}h"
                    except:
                        pass

                rows.append([
                    analysis['username'],
                    f"{progress}%",
                    f"{done}/{total}",
                    analysis.get('task_state', 'UNKNOWN'),
                    elapsed,
                    analysis.get('task_id', 'N/A')[:8] + "..." if analysis.get('task_id') else 'N/A'
                ])

            logger.info(tabulate(
                rows,
                headers=["Usuario", "Progreso", "Partidas", "Estado", "Tiempo", "Task ID"],
                tablefmt="github",
                numalign="right"
            ))


def cmd_delete(args: argparse.Namespace) -> None:
    with make_client(args.host, args.timeout) as client:
        handle_response(
            client.delete(ENDPOINT_PLAYER.format(username=args.username)))

    logger.info(f"🗑️ Jugador «{args.username}» eliminado.")


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

    # cancel
    sp_cancel = sub.add_parser("cancel", help="Cancelar análisis en curso")
    sp_cancel.add_argument("username", help="Nombre de usuario en chess.com")
    sp_cancel.add_argument("--force", "-f", action="store_true",
                          help="Forzar cancelación incluso si no se detecta análisis en curso")
    sp_cancel.set_defaults(func=cmd_cancel)

    # active
    sp_active = sub.add_parser("active", help="Listar análisis activos")
    sp_active.set_defaults(func=cmd_active)

    return p


# --------------------------- main -----------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
