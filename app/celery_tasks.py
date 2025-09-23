# app/celery_tasks_v2.py
"""
Celery tasks V2 - Refactor unificado
Usa AnalysisEngine v2 y modelos simplificados
"""
from __future__ import annotations

import io
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import chess.engine
import chess.pgn

# Configurar logging estructurado JSON
from app.logging_config import setup_logging
setup_logging()

from app.analysis.engine import AnalysisEngine
from app.database import engine

# Application Performance Monitoring (APM)
from app.otel import init_otel

logger = logging.getLogger(__name__)

# Inicializar OpenTelemetry (solo una vez en worker)
init_otel()

from app.utils import (
    fetch_games,
    notify_ws,
    task_progress,
    sa_to_dict,
    redis_client,
    cache_get,
    cache_set,
)
from celery import Celery
from celery import chain, group, chord
from celery import current_task
from celery.signals import task_failure, task_revoked
from sqlmodel import Session, select
from sqlalchemy import func

# Modelos V2
from app.models import Game, AnalysisResult, Player, PlayerStatus

from kombu import Queue  # Añadido para configurar colas con prioridad
from celery.exceptions import SoftTimeLimitExceeded

# Configuración de prioridades (0 = más alta)
HIGH_PRIORITY   = 0
DEFAULT_PRIORITY = 5
LOW_PRIORITY    = 9

# Configuración del motor de análisis
ENGINE_PATH = os.getenv("STOCKFISH_PATH", "stockfish")
MAX_DEPTH = int(os.getenv("STOCKFISH_DEPTH", "12"))
TB_PATH = os.getenv("TABLEBASE_PATH", None)

# ──────────────────────────────────────────────────────────────
#  Task timeout & retry configuration (env-driven)
# ──────────────────────────────────────────────────────────────
TASK_SOFT_TIME_LIMIT = int(os.getenv("TASK_SOFT_TIME_LIMIT", "1800"))  # 30 min default
TASK_TIME_LIMIT      = int(os.getenv("TASK_TIME_LIMIT", "1860"))      # hard limit (soft + 1 min)
TASK_MAX_RETRIES     = int(os.getenv("TASK_MAX_RETRIES", "3"))         # default max retries

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
celery_app = Celery("chess_tasks", broker=REDIS_URL, backend=REDIS_URL)

# Declarar la cola por defecto con soporte de prioridad (máx. 10 en Redis)
celery_app.conf.task_default_queue = "default"
# El tuple final debe contener únicamente el objeto Queue
celery_app.conf.task_queues = (
    Queue("default", max_priority=10),
)

celery_app.conf.update(
    # When a worker is lost (OOM/timeout) we want the broker to re-queue the task
    task_reject_on_worker_lost=True,
    # Force ACK *after* the task finishes so it can be retried on crash
    task_acks_late=True,
    # Apply global time limits – individual tasks can override these
    task_soft_time_limit=TASK_SOFT_TIME_LIMIT,
    task_time_limit=TASK_TIME_LIMIT,
    # Global retry defaults (used by autoretry_for)
    task_default_retry_delay=60,  # seconds between automatic retries
    task_max_retries=TASK_MAX_RETRIES,
    # Result expiration
    result_expires=3600,  # 1 hour
    # Worker settings
    worker_prefetch_multiplier=1,
)

# ──────────────────────────────────────────────────────────────
#  Utility functions for V2
# ──────────────────────────────────────────────────────────────

def update_progress(username: str, progress: int, message: str = "") -> None:
    """Update player progress for V2 using simplified models."""
    try:
        with Session(engine) as session:
            player = session.exec(
                select(Player).where(Player.username == username)
            ).first()

            if player:
                player.progress = progress
                session.add(player)
                session.commit()
                logger.debug(f"Updated progress for {username}: {progress}% - {message}")
            else:
                logger.warning(f"Player {username} not found for progress update")
    except Exception as e:
        logger.error(f"Error updating progress for {username}: {e}")

# Crear instancia del motor de análisis
analysis_engine = AnalysisEngine(
    stockfish_path=ENGINE_PATH,
    depth=MAX_DEPTH,
    tablebase_path=TB_PATH
)


@celery_app.task(
    bind=True,
    autoretry_for=(Exception,),
    retry_kwargs={'max_retries': 3, 'countdown': 60},
    soft_time_limit=1800,  # 30 minutos
    time_limit=1860,       # 31 minutos
)
def analyze_game(self, game_id: int, username: str, color: str) -> dict:
    """
    Analiza una partida específica para un jugador usando AnalysisEngine V2.

    Args:
        game_id: ID de la partida a analizar
        username: Usuario a analizar
        color: 'white' o 'black'

    Returns:
        Dict con información del resultado del análisis
    """
    task_id = self.request.id
    logger.info(f"Starting game analysis V2: game_id={game_id}, user={username}, color={color}, task_id={task_id}")

    try:
        with Session(engine) as session:
            # Obtener la partida
            game = session.exec(
                select(Game).where(Game.id == game_id)
            ).first()

            if not game:
                raise ValueError(f"Game {game_id} not found")

            # Verificar si ya existe análisis para esta combinación
            existing = session.exec(
                select(AnalysisResult)
                .where(AnalysisResult.game_id == game_id)
                .where(AnalysisResult.player_username == username)
                .where(AnalysisResult.player_color == color)
            ).first()

            if existing:
                logger.info(f"Analysis already exists for game {game_id}, user {username} ({color})")
                return {
                    'task_id': task_id,
                    'game_id': game_id,
                    'username': username,
                    'result_id': existing.id,
                    'status': 'completed_existing'
                }

            # Realizar análisis
            logger.info(f"Starting analysis for game {game_id}")
            result = analysis_engine.analyze_game(game, username, color)

            logger.info(f"Game analysis completed: result_id={result.id}")
            return {
                'task_id': task_id,
                'game_id': game_id,
                'username': username,
                'result_id': result.id,
                'status': 'completed_new',
                'moves_analyzed': result.moves_analyzed
            }

    except SoftTimeLimitExceeded:
        logger.error(f"Task {task_id} exceeded soft time limit")
        raise
    except Exception as e:
        logger.error(f"Error analyzing game {game_id}: {e}")
        raise


@celery_app.task(
    bind=True,
    autoretry_for=(Exception,),
    retry_kwargs={'max_retries': 2, 'countdown': 120},
    soft_time_limit=3600,  # 1 hora
    time_limit=3660,       # 61 minutos
)
def analyze_player(self, username: str) -> dict:
    """
    Analiza todas las partidas de un jugador y genera métricas agregadas V2.

    Args:
        username: Usuario a analizar

    Returns:
        Dict con métricas agregadas del jugador
    """
    task_id = self.request.id
    logger.info(f"Starting player analysis V2: username={username}, task_id={task_id}")

    try:
        with Session(engine) as session:
            # Actualizar estado del jugador
            player = session.exec(
                select(Player).where(Player.username == username)
            ).first()

            if not player:
                # Crear registro de jugador si no existe
                player = Player(
                    username=username,
                    status=PlayerStatus.pending,
                    requested_at=datetime.now(timezone.utc),
                    last_task_id=task_id
                )
                session.add(player)
                session.commit()
                session.refresh(player)
            else:
                player.status = PlayerStatus.pending
                player.last_task_id = task_id
                session.add(player)
                session.commit()

            # Realizar análisis agregado
            logger.info(f"Starting aggregated analysis for player {username}")
            aggregated_metrics = analysis_engine.analyze_player(username)

            # Actualizar estado final del jugador
            player.status = PlayerStatus.ready
            player.finished_at = datetime.now(timezone.utc)
            player.progress = 100
            session.add(player)
            session.commit()

            # Notificar a través de WebSocket si está disponible
            try:
                notify_ws(username, {
                    'type': 'player_analysis_completed',
                    'username': username,
                    'status': 'ready'
                })
            except Exception as e:
                logger.warning(f"Failed to send WebSocket notification: {e}")

            logger.info(f"Player analysis completed for {username}")
            return {
                'task_id': task_id,
                'username': username,
                'status': 'completed',
                'games_analyzed': aggregated_metrics.get('games_analyzed', 0)
            }

    except SoftTimeLimitExceeded:
        logger.error(f"Player analysis task {task_id} exceeded soft time limit")
        # Marcar jugador como error
        with Session(engine) as session:
            player = session.exec(
                select(Player).where(Player.username == username)
            ).first()
            if player:
                player.status = PlayerStatus.error
                player.error = "Analysis timeout"
                session.add(player)
                session.commit()
        raise
    except Exception as e:
        logger.error(f"Error analyzing player {username}: {e}")
        # Marcar jugador como error
        with Session(engine) as session:
            player = session.exec(
                select(Player).where(Player.username == username)
            ).first()
            if player:
                player.status = PlayerStatus.error
                player.error = str(e)
                session.add(player)
                session.commit()
        raise


@celery_app.task(
    bind=True,
    autoretry_for=(Exception,),
    retry_kwargs={'max_retries': 3, 'countdown': 60},
    soft_time_limit=7200,  # 2 horas
    time_limit=7260,       # 2 horas 1 minuto
)
def process_player_enhanced(self, username: str, force_reanalysis: bool = False) -> dict:
    """
    Procesa completamente un jugador: descarga partidas + análisis V2.

    Args:
        username: Usuario a procesar
        force_reanalysis: Si True, reanaliza partidas existentes

    Returns:
        Dict con resultado del procesamiento
    """
    task_id = self.request.id
    logger.info(f"Starting enhanced player processing V2: username={username}, task_id={task_id}")

    try:
        with Session(engine) as session:
            # Crear/actualizar jugador
            player = session.exec(
                select(Player).where(Player.username == username)
            ).first()

            if not player:
                # Player debería existir ya (creado en endpoint), pero por si acaso
                player = Player(
                    username=username,
                    status=PlayerStatus.pending,
                    requested_at=datetime.now(timezone.utc),
                    last_task_id=task_id,
                    progress=0
                )
                session.add(player)
                session.commit()
                session.refresh(player)
            else:
                # Actualizar player existente con task_id actual
                player.status = PlayerStatus.pending
                player.last_task_id = task_id
                player.progress = 0
                session.add(player)
                session.commit()

            # Paso 1: Descargar partidas
            logger.info(f"Fetching games for {username}")
            update_progress(username, 5, "Downloading games...")

            games_data = fetch_games(username)
            if not games_data:
                raise ValueError(f"No games found for player {username}")

            player.total_games = len(games_data)
            session.add(player)
            session.commit()

            update_progress(username, 10, f"Found {len(games_data)} games")

            # Paso 2: Procesar partidas y crear registros Game
            logger.info(f"Processing {len(games_data)} games")
            game_ids = []

            for i, game_data in enumerate(games_data):
                # Crear registro Game V2
                game = Game(
                    pgn=game_data['pgn'],
                    white_username=game_data.get('white'),  # Corregido: usar 'white' en lugar de 'white_username'
                    black_username=game_data.get('black'),  # Corregido: usar 'black' en lugar de 'black_username'
                    white_elo=game_data.get('white_elo'),
                    black_elo=game_data.get('black_elo'),
                    time_control=game_data.get('time_control'),
                    termination=game_data.get('termination'),
                    eco_code=game_data.get('eco_code'),
                    opening_key=game_data.get('opening_key'),
                    move_times=game_data.get('move_times'),
                    created_at=game_data.get('created_at', datetime.now(timezone.utc))
                )
                session.add(game)
                session.flush()  # Para obtener el ID
                game_ids.append(game.id)

                # Actualizar progreso
                progress = 10 + (i * 30 / len(games_data))
                update_progress(username, int(progress), f"Processing game {i+1}/{len(games_data)}")

            session.commit()
            update_progress(username, 40, f"Games processed, starting analysis...")

            # Paso 3: Analizar partidas
            analyzed_count = 0
            username_ci = username.lower()
            for i, game_id in enumerate(game_ids):
                # Determinar color del jugador en esta partida (case-insensitive)
                game = session.get(Game, game_id)
                color = None
                white_user = (game.white_username or "").lower()
                black_user = (game.black_username or "").lower()

                if white_user == username_ci:
                    color = 'white'
                elif black_user == username_ci:
                    color = 'black'

                if color is None:
                    logger.warning(
                        "Player %s not found in game %s participants (white=%s, black=%s)",
                        username,
                        game_id,
                        game.white_username,
                        game.black_username,
                    )
                    # Guardar análisis vacío para mantener el conteo consistente
                    fallback_metrics = analysis_engine._empty_game_metrics(
                        game,
                        'white',
                        error="player_not_found",
                    )
                    fallback_result = AnalysisResult(
                        game_id=game_id,
                        player_username=username,
                        player_color='white',
                        analyzed_at=datetime.now(timezone.utc),
                        engine_depth=MAX_DEPTH,
                        moves_analyzed=0,
                        metrics=fallback_metrics,
                    )
                    session.add(fallback_result)
                    analyzed_count += 1
                    progress = 40 + (analyzed_count * 50 / len(game_ids)) if game_ids else 100
                    player.done_games = analyzed_count
                    player.progress = int(progress)
                    session.add(player)
                    session.commit()
                    update_progress(
                        username,
                        int(progress),
                        f"Analyzed {analyzed_count}/{len(game_ids)} games (fallback)",
                    )
                    continue

                if color:
                    try:
                        # Lanzar análisis de partida individual
                        logger.info(f"Analyzing game {game_id} for {username} ({color})")
                        result = analysis_engine.analyze_game(game, username, color)
                        analyzed_count += 1

                        # Actualizar progreso
                        progress = 40 + (analyzed_count * 50 / len(game_ids))
                        player.done_games = analyzed_count
                        player.progress = int(progress)
                        session.add(player)
                        session.commit()

                        update_progress(username, int(progress),
                                      f"Analyzed {analyzed_count}/{len(game_ids)} games")

                    except Exception as e:
                        logger.warning(f"Failed to analyze game {game_id}: {e}")
                        continue

            update_progress(username, 90, "Computing player metrics...")

            # Paso 4: Análisis agregado del jugador
            logger.info(f"Computing aggregated metrics for {username}")
            aggregated_metrics = analysis_engine.analyze_player(username)

            # Paso 5: Finalizar
            player.status = PlayerStatus.ready
            player.finished_at = datetime.now(timezone.utc)
            player.progress = 100
            player.done_games = analyzed_count
            session.add(player)
            session.commit()

            update_progress(username, 100, "Analysis completed!")

            # Notificación final
            try:
                notify_ws(username, {
                    'type': 'player_processing_completed',
                    'username': username,
                    'status': 'ready',
                    'games_analyzed': analyzed_count
                })
            except Exception as e:
                logger.warning(f"Failed to send final WebSocket notification: {e}")

            logger.info(f"Enhanced player processing completed for {username}: {analyzed_count} games analyzed")
            return {
                'task_id': task_id,
                'username': username,
                'status': 'completed',
                'games_processed': len(game_ids),
                'games_analyzed': analyzed_count
            }

    except SoftTimeLimitExceeded:
        logger.error(f"Enhanced processing task {task_id} exceeded soft time limit")
        with Session(engine) as session:
            player = session.exec(
                select(Player).where(Player.username == username)
            ).first()
            if player:
                player.status = PlayerStatus.error
                player.error = "Processing timeout"
                session.add(player)
                session.commit()
        raise
    except Exception as e:
        logger.error(f"Error in enhanced processing for {username}: {e}")
        with Session(engine) as session:
            player = session.exec(
                select(Player).where(Player.username == username)
            ).first()
            if player:
                player.status = PlayerStatus.error
                player.error = str(e)
                session.add(player)
                session.commit()
        raise


# ──────────────────────────────────────────────────────────────
# Signal handlers
# ──────────────────────────────────────────────────────────────

@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, traceback=None, einfo=None, **kwargs):
    """Maneja fallos de tasks V2"""
    logger.error(f"Task V2 failed: {task_id}, Exception: {exception}")

@task_revoked.connect
def task_revoked_handler(sender=None, task_id=None, reason=None, **kwargs):
    """Maneja revocación de tasks V2"""
    logger.warning(f"Task V2 revoked: {task_id}, Reason: {reason}")


# ──────────────────────────────────────────────────────────────
# Utilidades de compatibilidad
# ──────────────────────────────────────────────────────────────

def get_task_result(task_id: str) -> dict:
    """Obtiene el resultado de una task V2"""
    try:
        result = celery_app.AsyncResult(task_id)
        return {
            'task_id': task_id,
            'status': result.status,
            'result': result.result,
            'ready': result.ready()
        }
    except Exception as e:
        logger.error(f"Error getting task result V2 {task_id}: {e}")
        return {
            'task_id': task_id,
            'status': 'ERROR',
            'result': str(e),
            'ready': True
        }
