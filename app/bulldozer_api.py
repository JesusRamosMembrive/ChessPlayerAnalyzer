# app/bulldozer_api.py
"""
BULLDOZER TOTAL: API ultra-simplificado.

FILOSOFÍA:
- Funciones simples que manejan requests HTTP
- Sin capas complejas, sin routers anidados, sin over-abstraction
- Backward compatibility con endpoints existentes
- Error handling explícito y simple
"""
import logging
from datetime import datetime, timezone
from typing import Dict, Optional

from fastapi import HTTPException
from sqlmodel import Session, select

from app.models import GameAnalysis, PlayerProgress
from app.analysis.bulldozer_engine import analyze_game_complete, save_analysis_to_db, get_player_analysis_summary
from app.database import engine as db_engine
from app.utils import fetch_games

logger = logging.getLogger(__name__)


def start_player_analysis_bulldozer(username: str) -> Dict:
    """
    BULLDOZER: Start player analysis - ultra simple version.

    Returns: {"status": "started", "message": "...", "task_id": "..."}
    """
    try:
        logger.info(f"BULLDOZER: Starting analysis for {username}")

        with Session(db_engine) as session:
            # Check if already exists
            stmt = select(PlayerProgress).where(PlayerProgress.username == username)
            existing = session.exec(stmt).first()

            if existing and existing.status == "ready":
                return {
                    "status": "already_done",
                    "message": f"Player {username} already analyzed",
                    "progress": 100
                }

            # Create or update progress record
            if existing:
                existing.status = "pending"
                existing.progress = 0
                existing.requested_at = datetime.now(timezone.utc)
            else:
                progress = PlayerProgress(
                    username=username,
                    status="pending",
                    progress=0,
                    requested_at=datetime.now(timezone.utc)
                )
                session.add(progress)

            session.commit()

        # Start async analysis (simplified - no Celery for now)
        # TODO: Add Celery integration if needed
        result = _analyze_player_sync(username)

        return {
            "status": "completed" if result else "error",
            "message": f"Analysis {'completed' if result else 'failed'} for {username}",
            "games_processed": result.get("games_processed", 0) if result else 0
        }

    except Exception as e:
        logger.error(f"BULLDOZER analysis start failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


def _analyze_player_sync(username: str) -> Optional[Dict]:
    """
    BULLDOZER: Synchronous player analysis (simplified for now).
    Later can be moved to Celery if needed.
    """
    try:
        logger.info(f"BULLDOZER: Running sync analysis for {username}")

        # 1. Fetch games from Chess.com
        games = fetch_games(username, months=12)
        if not games:
            logger.warning(f"No games found for {username}")
            return None

        logger.info(f"Found {len(games)} games for {username}")

        games_processed = 0
        with Session(db_engine) as session:
            # Update progress
            stmt = select(PlayerProgress).where(PlayerProgress.username == username)
            progress = session.exec(stmt).first()
            if progress:
                progress.total_games = len(games)
                session.commit()

            for i, game_data in enumerate(games):
                pgn = game_data.get("pgn")
                if not pgn:
                    continue

                # Determine player color
                color = _determine_player_color(pgn, username)
                if not color:
                    continue

                # Run BULLDOZER COMPLETE analysis
                analysis = analyze_game_complete(pgn, username, color)
                if analysis and "error" not in analysis:
                    # Save to database
                    analysis_id = save_analysis_to_db(pgn, username, color, analysis, session)
                    if analysis_id:
                        games_processed += 1

                # Update progress
                if progress:
                    progress.done_games = games_processed
                    progress.progress = int((i + 1) / len(games) * 100)
                    session.commit()

            # Mark as completed
            if progress:
                progress.status = "ready"
                progress.finished_at = datetime.now(timezone.utc)
                session.commit()

        logger.info(f"BULLDOZER analysis completed: {games_processed}/{len(games)} games processed")
        return {"games_processed": games_processed, "total_games": len(games)}

    except Exception as e:
        logger.error(f"BULLDOZER sync analysis failed: {e}")
        # Mark as error
        try:
            with Session(db_engine) as session:
                stmt = select(PlayerProgress).where(PlayerProgress.username == username)
                progress = session.exec(stmt).first()
                if progress:
                    progress.status = "error"
                    progress.error_message = str(e)
                    session.commit()
        except:
            pass
        return None


def _determine_player_color(pgn: str, username: str) -> Optional[str]:
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


def get_player_status_bulldozer(username: str) -> Dict:
    """Get player analysis status - BULLDOZER simple version."""
    try:
        with Session(db_engine) as session:
            stmt = select(PlayerProgress).where(PlayerProgress.username == username)
            progress = session.exec(stmt).first()

            if not progress:
                return {
                    "username": username,
                    "status": "not_analyzed",
                    "progress": 0,
                    "message": "Player not analyzed yet"
                }

            return {
                "username": username,
                "status": progress.status,
                "progress": progress.progress,
                "total_games": progress.total_games,
                "done_games": progress.done_games,
                "requested_at": progress.requested_at.isoformat() if progress.requested_at else None,
                "finished_at": progress.finished_at.isoformat() if progress.finished_at else None,
                "error": progress.error_message
            }

    except Exception as e:
        logger.error(f"Get player status failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


def get_player_metrics_bulldozer(username: str) -> Dict:
    """Get player aggregated metrics - BULLDOZER simple version."""
    try:
        with Session(db_engine) as session:
            summary = get_player_analysis_summary(username, session)

            if "error" in summary:
                raise HTTPException(status_code=404, detail=summary["error"])

            return summary

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get player metrics failed: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")


def get_game_analysis_bulldozer(username: str, limit: int = 10) -> Dict:
    """Get individual game analyses for a player - BULLDOZER simple version."""
    try:
        with Session(db_engine) as session:
            stmt = (select(GameAnalysis)
                   .where(GameAnalysis.analyzed_username == username)
                   .order_by(GameAnalysis.analyzed_at.desc())
                   .limit(limit))

            analyses = session.exec(stmt).all()

            games = []
            for analysis in analyses:
                games.append({
                    "id": analysis.id,
                    "white_username": analysis.white_username,
                    "black_username": analysis.black_username,
                    "analyzed_color": analysis.analyzed_color,
                    "analyzed_at": analysis.analyzed_at.isoformat(),
                    "moves_analyzed": analysis.moves_analyzed,
                    "quality": analysis.analysis.get("quality", {}),
                    "timing": analysis.analysis.get("timing", {}),
                    "pgn": analysis.pgn
                })

            return {
                "username": username,
                "games": games,
                "total_found": len(games)
            }

    except Exception as e:
        logger.error(f"Get game analyses failed: {e}")
        raise HTTPException(status_code=500, detail=f"Game retrieval failed: {str(e)}")