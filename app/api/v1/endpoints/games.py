from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select
from celery import chain
import io
import chess.pgn


from app.models import GameAnalysis
from app.database import get_session
from app.celery_tasks import celery_app
from celery.result import AsyncResult
from app.schemas import (
    AnalyzeGameIn,
    TaskQueuedOut,
    GameOut,
    GameAnalysisStatusOut,
    GameAnalysisCancelOut,
)
import hashlib
from app.utils import redis_client

router = APIRouter()

@router.get(
    "/{game_id}",
    response_model=GameOut,
    summary="Obtener los detalles de una partida analizada",
    description="Devuelve la información almacenada y las jugadas evaluadas de la partida identificada por **game_id**.",
    responses={404: {"description": "Partida no encontrada"}},
)
async def get_game(game_id: int, session: Session = Depends(get_session)):
    """Get details of an analyzed game - BULLDOZER version."""
    analysis = session.get(GameAnalysis, game_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Game analysis not found")

    # Extract player info from PGN headers
    try:
        game = chess.pgn.read_game(io.StringIO(analysis.pgn))
        white_username = game.headers.get("White", "") if game else ""
        black_username = game.headers.get("Black", "") if game else ""
        eco_code = game.headers.get("ECO", "") if game else ""
    except:
        white_username = ""
        black_username = ""
        eco_code = ""

    return {
        "id": analysis.id,
        "created_at": analysis.analyzed_at.isoformat() if analysis.analyzed_at else None,
        "pgn": analysis.pgn,
        "white_username": white_username,
        "black_username": black_username,
        "analyzed_username": analysis.analyzed_username,
        "eco_code": eco_code,
        "analysis": analysis.analysis
    }

@router.post(
    "/analyze",
    response_model=TaskQueuedOut,
    summary="Encolar análisis de una partida",
    description="Analiza directamente el PGN proporcionado usando BULLDOZER engine.",
    status_code=status.HTTP_202_ACCEPTED,
)
async def analyze_game_endpoint(
    req: AnalyzeGameIn,
    session: Session = Depends(get_session)
):
    """Analyze a single game with BULLDOZER engine."""
    try:
        from app.analysis.bulldozer_engine import analyze_game_complete, save_analysis_to_db

        # Extract username from PGN
        game_pgn_obj = chess.pgn.read_game(io.StringIO(req.pgn))
        if not game_pgn_obj:
            raise HTTPException(status_code=400, detail="Invalid PGN format")

        white = game_pgn_obj.headers.get("White", "").lower()
        black = game_pgn_obj.headers.get("Black", "").lower()

        # For single game analysis, analyze for both colors
        results = []

        for color, username in [("white", white), ("black", black)]:
            if username:
                # Analyze game
                analysis = analyze_game_complete(req.pgn, username, color)

                if analysis and "error" not in analysis:
                    # Save to database
                    analysis_id = save_analysis_to_db(req.pgn, username, color, analysis, session)
                    session.commit()

                    if analysis_id:
                        results.append({
                            "username": username,
                            "color": color,
                            "analysis_id": analysis_id
                        })

        if results:
            return {
                "game_id": results[0]["analysis_id"],  # Return first analysis ID
                "task_id": f"direct_analysis_{results[0]['analysis_id']}",
                "status": "completed",
                "results": results
            }
        else:
            raise HTTPException(status_code=500, detail="Analysis failed for all players")

    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

@router.get(
    "/{game_id}/status/{task_id}",
    response_model=GameAnalysisStatusOut,
    summary="Consultar estado del análisis de partida",
    description="Devuelve el estado actual del análisis - BULLDOZER version.",
    responses={404: {"description": "Análisis no encontrado"}},
)
async def get_game_analysis_status(
    game_id: int,
    task_id: str,
    session: Session = Depends(get_session)
):
    """Get the status of a game analysis - BULLDOZER version."""
    # Check if analysis exists
    analysis = session.get(GameAnalysis, game_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Game analysis not found")

    # For BULLDOZER, analysis is usually complete when it exists
    return {
        "game_id": game_id,
        "task_id": task_id,
        "status": "SUCCESS",
        "result": {"analysis_complete": True, "analysis_id": analysis.id}
    }

@router.post(
    "/{game_id}/cancel/{task_id}",
    response_model=GameAnalysisCancelOut,
    summary="Cancelar el análisis de una partida",
    description="Para BULLDOZER, el análisis es directo (no cancelable).",
    responses={404: {"description": "Análisis no encontrado"}},
)
async def cancel_game_analysis(
    game_id: int,
    task_id: str,
    session: Session = Depends(get_session)
):
    """Cancel analysis - BULLDOZER version (direct analysis, not cancellable)."""
    # Check if analysis exists
    analysis = session.get(GameAnalysis, game_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Game analysis not found")

    # For BULLDOZER, analysis is direct and immediate, so "cancellation" is not applicable
    return {
        "status": "not_cancellable",
        "game_id": game_id,
        "task_id": task_id,
        "message": "BULLDOZER analysis is direct and cannot be cancelled"
    }
