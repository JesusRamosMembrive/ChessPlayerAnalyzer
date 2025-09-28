# app/celery_tasks.py - BULLDOZER TOTAL
"""
BULLDOZER TOTAL: Celery tasks ultra-simplificados.

FILOSOFÍA:
- Task simple: recibe username → analiza todos los juegos → guarda en BULLDOZER
- Sin complex workflows, sin chains, sin chords
- Error handling directo
- Progress tracking simple
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

# Application factories
from app.factories import create_worker, get_logger

# BULLDOZER imports
from app.models import GameAnalysis, PlayerProgress
from app.analysis.bulldozer_engine import analyze_game_complete, save_analysis_to_db
from app.database import engine
from app.utils import fetch_games, notify_ws, task_progress

from celery import current_task
from celery.exceptions import SoftTimeLimitExceeded
from celery.signals import task_failure
from sqlmodel import Session, select

# Configuración del worker
celery_app = create_worker()
logger = get_logger(__name__)

# Configuración de prioridades (0 = más alta)
HIGH_PRIORITY = 0
DEFAULT_PRIORITY = 5
LOW_PRIORITY = 9


@celery_app.task(bind=True, soft_time_limit=3600)  # 1 hour limit
def process_player_bulldozer(self, username: str, months: int = 12) -> dict:
    """
    BULLDOZER TOTAL: Procesa un jugador completo.

    Simple: fetch games → analyze each → save to BULLDOZER table → done
    """
    try:
        logger.info(f"BULLDOZER: Starting analysis for {username}")

        # Update progress tracking
        with Session(engine) as session:
            stmt = select(PlayerProgress).where(PlayerProgress.username == username)
            progress = session.exec(stmt).first()

            if not progress:
                progress = PlayerProgress(
                    username=username,
                    status="pending",
                    progress=0,
                    requested_at=datetime.now(timezone.utc),
                    last_task_id=self.request.id
                )
                session.add(progress)
            else:
                progress.status = "pending"
                progress.progress = 0
                progress.last_task_id = self.request.id
                progress.requested_at = datetime.now(timezone.utc)

            session.commit()

        # Fetch games from Chess.com
        logger.info(f"BULLDOZER: Fetching games for {username}")
        games = fetch_games(username, months)

        if not games:
            logger.warning(f"BULLDOZER: No games found for {username}")
            _update_player_status(username, "error", "No games found")
            return {"error": "No games found", "username": username}

        total_games = len(games)
        logger.info(f"BULLDOZER: Found {total_games} games for {username}")

        # Update total games count
        with Session(engine) as session:
            stmt = select(PlayerProgress).where(PlayerProgress.username == username)
            progress = session.exec(stmt).first()
            if progress:
                progress.total_games = total_games
                session.commit()

        # Process each game
        games_processed = 0
        games_failed = 0

        for i, game_data in enumerate(games):
            try:
                # Check for task revocation
                if self.request.called_directly or current_task.request.id != self.request.id:
                    logger.info(f"BULLDOZER: Task revoked for {username}")
                    break

                pgn = game_data.get("pgn")
                if not pgn:
                    logger.warning(f"BULLDOZER: No PGN for game {i+1}")
                    games_failed += 1
                    continue

                # Determine player color
                color = _determine_player_color(pgn, username)
                if not color:
                    logger.warning(f"BULLDOZER: Could not determine color for {username} in game {i+1}")
                    games_failed += 1
                    continue

                # BULLDOZER analysis
                logger.debug(f"BULLDOZER: Analyzing game {i+1}/{total_games} for {username} ({color})")
                analysis = analyze_game_complete(pgn, username, color)

                if analysis and "error" not in analysis:
                    # Save to BULLDOZER database
                    with Session(engine) as session:
                        analysis_id = save_analysis_to_db(pgn, username, color, analysis, session)
                        session.commit()

                        if analysis_id:
                            games_processed += 1
                            logger.debug(f"BULLDOZER: Game {i+1} saved with ID {analysis_id}")
                        else:
                            games_failed += 1
                            logger.warning(f"BULLDOZER: Failed to save game {i+1}")
                else:
                    games_failed += 1
                    error_msg = analysis.get("error", "Unknown error") if analysis else "Analysis returned None"
                    logger.warning(f"BULLDOZER: Analysis failed for game {i+1}: {error_msg}")

                # Update progress
                progress_pct = int((i + 1) / total_games * 100)
                task_progress(self, i + 1, total_games, username)

                with Session(engine) as session:
                    stmt = select(PlayerProgress).where(PlayerProgress.username == username)
                    progress = session.exec(stmt).first()
                    if progress:
                        progress.done_games = games_processed
                        progress.progress = progress_pct
                        session.commit()

            except SoftTimeLimitExceeded:
                logger.error(f"BULLDOZER: Soft time limit exceeded for {username}")
                _update_player_status(username, "error", "Analysis timeout")
                return {"error": "Analysis timeout", "username": username, "games_processed": games_processed}

            except Exception as e:
                logger.error(f"BULLDOZER: Error processing game {i+1} for {username}: {e}")
                games_failed += 1
                continue

        # Final status update
        if games_processed > 0:
            _update_player_status(username, "ready", None, games_processed, total_games)
            logger.info(f"BULLDOZER: Completed analysis for {username}: {games_processed}/{total_games} games processed")

            # Notify completion
            notify_ws(username, {
                "type": "analysis_complete",
                "games_processed": games_processed,
                "games_failed": games_failed,
                "total_games": total_games
            })

            return {
                "username": username,
                "games_processed": games_processed,
                "games_failed": games_failed,
                "total_games": total_games,
                "status": "completed"
            }
        else:
            _update_player_status(username, "error", "No games could be processed")
            return {"error": "No games could be processed", "username": username}

    except Exception as e:
        logger.error(f"BULLDOZER: Fatal error processing {username}: {e}")
        _update_player_status(username, "error", str(e))
        return {"error": str(e), "username": username}


def _determine_player_color(pgn: str, username: str) -> str | None:
    """Determine if username is white or black in the game."""
    try:
        import chess.pgn
        import io

        game = chess.pgn.read_game(io.StringIO(pgn))
        if not game:
            return None

        white = game.headers.get("White", "").lower()
        black = game.headers.get("Black", "").lower()
        username_lower = username.lower()

        if username_lower in white:
            return "white"
        elif username_lower in black:
            return "black"
        else:
            return None

    except Exception as e:
        logger.error(f"Color determination failed: {e}")
        return None


def _update_player_status(username: str, status: str, error_message: str = None,
                         done_games: int = None, total_games: int = None):
    """Update player progress status in database."""
    try:
        with Session(engine) as session:
            stmt = select(PlayerProgress).where(PlayerProgress.username == username)
            progress = session.exec(stmt).first()

            if progress:
                progress.status = status
                if error_message:
                    progress.error_message = error_message
                if done_games is not None:
                    progress.done_games = done_games
                if total_games is not None:
                    progress.total_games = total_games
                if status == "ready":
                    progress.finished_at = datetime.now(timezone.utc)
                    progress.progress = 100

                session.commit()
                logger.debug(f"Updated status for {username}: {status}")

    except Exception as e:
        logger.error(f"Failed to update player status: {e}")


# Backward compatibility aliases
process_player_enhanced = process_player_bulldozer  # For existing API calls


# Celery signal handlers
@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, traceback=None, einfo=None, **kwargs):
    """Handle task failures."""
    logger.error(f"BULLDOZER: Task {task_id} failed: {exception}")


# Health check task
@celery_app.task
def health_check():
    """Simple health check for Celery worker."""
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


if __name__ == "__main__":
    celery_app.start()