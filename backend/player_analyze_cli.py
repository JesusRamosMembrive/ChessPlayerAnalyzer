#!/usr/bin/env python3
"""
CLI mejorado: analiza jugador con métricas detalladas.

Muestra:
- Análisis básico (existente)
- Análisis detallado por partida (nuevo)
- Análisis longitudinal del jugador (nuevo)
- Visualización mejorada con colores

Requisitos:
    pip install requests tqdm colorama tabulate
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from typing import List, Optional, Tuple

import requests
from colorama import init, Fore, Style
from tabulate import tabulate
from tqdm import tqdm

# Inicializar colorama para Windows
init()

# Configuración
API_BASE = "http://localhost:8000"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sesión global
S = requests.Session()
S.headers["User-Agent"] = "chess-analyzer-cli/0.2"

# ============================================================
# HELPERS
# ============================================================

def safe_get(url: str, tries: int = 3, backoff: float = 2.0):
    """GET con reintentos."""
    for n in range(tries):
        try:
            r = S.get(url, timeout=10)
            if r.status_code in (403, 429):
                time.sleep(backoff ** n)
                continue
            return r
        except requests.exceptions.RequestException as e:
            if n == tries - 1:
                raise
            time.sleep(backoff ** n)
    
def print_header(text: str, color=Fore.CYAN):
    """Imprime header con estilo."""
    print(f"\n{color}{'='*60}")
    print(f"{text.center(60)}")
    print(f"{'='*60}{Style.RESET_ALL}")

def print_section(title: str, color=Fore.YELLOW):
    """Imprime título de sección."""
    print(f"\n{color}▶ {title}{Style.RESET_ALL}")

def format_suspicious(value: bool) -> str:
    """Formatea indicador de sospecha."""
    if value:
        return f"{Fore.RED}⚠️ SOSPECHOSO{Style.RESET_ALL}"
    return f"{Fore.GREEN}✓ Normal{Style.RESET_ALL}"

def format_score(score: float, thresholds: List[float] = [30, 50, 70]) -> str:
    """Colorea score según umbrales."""
    if score < thresholds[0]:
        return f"{Fore.GREEN}{score:.1f}{Style.RESET_ALL}"
    elif score < thresholds[1]:
        return f"{Fore.YELLOW}{score:.1f}{Style.RESET_ALL}"
    elif score < thresholds[2]:
        return f"{Fore.MAGENTA}{score:.1f}{Style.RESET_ALL}"
    else:
        return f"{Fore.RED}{score:.1f}{Style.RESET_ALL}"

# ============================================================
# API CALLS
# ============================================================

def get_player_status(username: str) -> dict:
    """Obtiene estado del jugador."""
    r = S.get(f"{API_BASE}/players/{username}")
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()

def start_player_analysis(username: str) -> dict:
    """Inicia análisis del jugador."""
    r = S.post(f"{API_BASE}/players/{username}")
    r.raise_for_status()
    return r.json()

def wait_for_player_ready(username: str, timeout: int = 600):
    """Espera hasta que el jugador esté analizado."""
    print_section("Progreso del análisis")
    
    with tqdm(total=100, desc="Analizando", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
        last_progress = 0
        start_time = time.time()
        
        while True:
            status = get_player_status(username)
            if not status:
                time.sleep(1)
                continue
                
            current_progress = status.get('progress', 0)
            if current_progress > last_progress:
                pbar.update(current_progress - last_progress)
                last_progress = current_progress
            
            if status['status'] == 'ready':
                pbar.update(100 - last_progress)
                print(f"\n{Fore.GREEN}✅ Análisis completado{Style.RESET_ALL}")
                return status
                
            if status['status'] == 'error':
                print(f"\n{Fore.RED}❌ Error: {status.get('error', 'Unknown error')}{Style.RESET_ALL}")
                sys.exit(1)
                
            if time.time() - start_time > timeout:
                print(f"\n{Fore.RED}❌ Timeout esperando análisis{Style.RESET_ALL}")
                sys.exit(1)
                
            time.sleep(1)

def get_player_games(username: str) -> List[dict]:
    """Obtiene partidas analizadas del jugador."""
    # TODO: Implementar endpoint para obtener lista de partidas
    # Por ahora usamos el método existente
    r = S.get(f"{API_BASE}/games", params={"player": username, "limit": 100})
    if r.status_code == 200:
        return r.json()
    return []

def get_game_detailed_analysis(game_id: int) -> Optional[dict]:
    """Obtiene análisis detallado de una partida."""
    # TODO: Implementar endpoint GET /games/{game_id}/analysis
    r = S.get(f"{API_BASE}/games/{game_id}/analysis")
    if r.status_code == 200:
        return r.json()
    return None

def get_player_detailed_analysis(username: str) -> Optional[dict]:
    """Obtiene análisis detallado del jugador."""
    # TODO: Implementar endpoint GET /players/{username}/analysis
    r = S.get(f"{API_BASE}/players/{username}/analysis")
    if r.status_code == 200:
        return r.json()
    return None

def get_player_metrics(username: str) -> Optional[dict]:
    """Obtiene métricas básicas del jugador."""
    r = S.get(f"{API_BASE}/metrics/player/{username}")
    if r.status_code == 200:
        return r.json()
    return None

# ============================================================
# DISPLAY FUNCTIONS
# ============================================================

def display_game_analysis(game_id: int, analysis: dict, url: str = ""):
    """Muestra análisis de una partida."""
    # Métricas básicas
    basic = {
        "ACPL": analysis.get('acl', 0),
        "Top-1 %": analysis.get('pct_top1', 0),
        "Top-3 %": analysis.get('pct_top3', 0),
        "Sospechoso": analysis.get('suspicious', False)
    }
    
    # Si hay análisis detallado
    detailed = get_game_detailed_analysis(game_id)
    if detailed:
        basic.update({
            "IPR": detailed.get('ipr', 0),
            "IPR Z-Score": detailed.get('ipr_z_score', 0),
            "Tiempo promedio": f"{detailed.get('mean_move_time', 0):.1f}s",
            "Correlación T-C": detailed.get('time_complexity_corr', 0),
            "Lag spikes": detailed.get('lag_spike_count', 0)
        })
    
    # Formatear para tabla
    row = [game_id]
    row.append(format_score(basic['ACPL'], [20, 40, 60]))
    row.append(format_score(basic['Top-3 %'], [80, 70, 60]))  # Invertido
    row.append(format_suspicious(basic['Sospechoso']))
    
    if detailed:
        row.extend([
            f"{basic['IPR']:.0f}",
            format_score(basic['IPR Z-Score'], [1, 2, 3]),
            basic['Tiempo promedio'],
            f"{basic['Correlación T-C']:.2f}"
        ])
    
    if url:
        row.append(f"{Fore.BLUE}{url[-20:]}{Style.RESET_ALL}")  # Últimos 20 chars
    
    return row

def display_player_summary(username: str, metrics: dict, detailed: Optional[dict] = None):
    """Muestra resumen del jugador."""
    print_header(f"ANÁLISIS DE {username.upper()}")
    
    # Métricas básicas
    print_section("Métricas Básicas", Fore.CYAN)
    basic_data = [
        ["Partidas analizadas", metrics.get('game_count', 0)],
        ["Entropía de aperturas", f"{metrics.get('opening_entropy', 0):.2f} bits"],
        ["Apertura más jugada", metrics.get('most_played', 'N/A')],
        ["Baja entropía", format_suspicious(metrics.get('low_entropy', False))]
    ]
    print(tabulate(basic_data, tablefmt="simple"))
    
    # Análisis detallado si está disponible
    if detailed:
        print_section("Análisis Detallado", Fore.MAGENTA)
        
        # Métricas de calidad
        quality_data = [
            ["ACPL promedio", format_score(detailed.get('avg_acpl', 0), [25, 40, 60])],
            ["Match rate promedio", f"{detailed.get('avg_match_rate', 0)*100:.1f}%"],
            ["IPR promedio", f"{detailed.get('avg_ipr', 0):.0f}"]
        ]
        print("\n📊 Calidad de juego:")
        print(tabulate(quality_data, tablefmt="simple"))
        
        # Métricas longitudinales
        long_data = [
            ["ROI medio", format_score(detailed.get('roi_mean', 0), [1, 2, 3])],
            ["ROI máximo", format_score(detailed.get('roi_max', 0), [2, 3, 4])],
            ["Racha más larga", f"{detailed.get('longest_streak', 0)} partidas"],
            ["Step function", format_suspicious(detailed.get('step_function_detected', False))],
            ["Delta vs peers (ACPL)", f"{detailed.get('peer_delta_acpl', 0):.1f} cp"]
        ]
        print("\n📈 Análisis longitudinal:")
        print(tabulate(long_data, tablefmt="simple"))
        
        # Score de riesgo
        risk_score = detailed.get('risk_score', 0)
        risk_factors = detailed.get('risk_factors', {})
        
        print_section("Evaluación de Riesgo", Fore.RED if risk_score > 50 else Fore.YELLOW)
        print(f"Score de riesgo: {format_score(risk_score, [30, 50, 70])}/100")
        
        if risk_factors:
            print("\nFactores de riesgo detectados:")
            for factor, present in risk_factors.items():
                if present:
                    print(f"  • {Fore.RED}{factor.replace('_', ' ').title()}{Style.RESET_ALL}")

def display_top_suspicious_games(games_analysis: List[Tuple[int, dict]], limit: int = 10):
    """Muestra las partidas más sospechosas."""
    print_section(f"Top {limit} Partidas Más Sospechosas", Fore.RED)
    
    # Ordenar por sospecha
    suspicious_games = [(gid, analysis) for gid, analysis in games_analysis if analysis.get('suspicious', False)]
    suspicious_games.sort(key=lambda x: (x[1].get('pct_top3', 0), -x[1].get('acl', 100)), reverse=True)
    
    if not suspicious_games:
        print("No se encontraron partidas sospechosas ✨")
        return
    
    headers = ["ID", "ACPL", "Top-3%", "Estado", "IPR", "Z-Score", "Tiempo", "T-C Corr", "URL"]
    rows = []
    
    for gid, analysis in suspicious_games[:limit]:
        url = f"game/{gid}"  # URL placeholder
        rows.append(display_game_analysis(gid, analysis, url))
    
    print(tabulate(rows, headers=headers, tablefmt="grid"))

# ============================================================
# MAIN
# ============================================================

def main():
    print_header("CHESS ANALYZER CLI v0.2", Fore.CYAN)
    
    # Obtener username
    username = input(f"\n{Fore.YELLOW}Jugador Chess.com a analizar: {Style.RESET_ALL}").strip().lower()
    if not username:
        print(f"{Fore.RED}❌ Username vacío{Style.RESET_ALL}")
        return
    
    print(f"\n🔍 Analizando a {Fore.CYAN}{username}{Style.RESET_ALL}...")
    
    try:
        # Verificar estado del jugador
        status = get_player_status(username)

        if not status or status.get('status') in {'not_analyzed', 'error'}:
            # Primera vez (o análisis fallido) → lanzamos POST
            print("Iniciando nuevo análisis…")
            start_result = start_player_analysis(username)
            print(f"Task ID: {start_result.get('task_id', 'N/A')}")
            status = wait_for_player_ready(username)
        
        elif status['status'] == 'pending':
            print("Análisis en progreso. Esperando...")
            status = wait_for_player_ready(username=username, timeout=3600)
        
        elif status['status'] == 'ready':
            print(f"{Fore.GREEN}✅ Jugador ya analizado{Style.RESET_ALL}")
            
            # Preguntar si re-analizar
            refresh = input(f"\n¿Deseas refrescar el análisis? (s/N): ").lower()
            if refresh == 's':
                r = S.post(f"{API_BASE}/players/{username}/refresh")
                if r.status_code == 200:
                    print("Análisis reiniciado. Esperando...")
                    status = wait_for_player_ready(username)
        
        # Obtener métricas
        print_section("Obteniendo métricas...")
        metrics = get_player_metrics(username)
        detailed_analysis = get_player_detailed_analysis(username)
        
        if not metrics:
            print(f"{Fore.YELLOW}⚠️ No se pudieron obtener métricas{Style.RESET_ALL}")
            return
        
        # Mostrar resumen del jugador
        display_player_summary(username, metrics, detailed_analysis)
        
        # Obtener y mostrar partidas
        # TODO: Cuando esté el endpoint, obtener partidas analizadas
        # games = get_player_games(username)
        # if games:
        #     display_top_suspicious_games(games)
        
        # Guardar reporte si se desea
        save = input(f"\n¿Guardar reporte completo? (s/N): ").lower()
        if save == 's':
            filename = f"{username}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            report = {
                "username": username,
                "timestamp": datetime.now().isoformat(),
                "basic_metrics": metrics,
                "detailed_analysis": detailed_analysis,
                "status": status
            }
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"{Fore.GREEN}✅ Reporte guardado en: {filename}{Style.RESET_ALL}")
        
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}⚠️ Abortado por el usuario{Style.RESET_ALL}")
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"{Fore.RED}❌ Error: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()