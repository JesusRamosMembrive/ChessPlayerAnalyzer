"""
Handlers para endpoints relacionados con partidas.
Actúan como bridge entre FastAPI y los use cases.
"""
from typing import List, Optional
from dataclasses import asdict
from datetime import datetime
from fastapi import HTTPException

from ..container import get_container
from ..queries.game_queries import (
    GetGameAnalysisQuery,
    GetPlayerGamesQuery,
    GetSuspiciousGamesQuery
)


class GameHandlers:
    """Handlers para operaciones de partidas."""

    def __init__(self):
        self._container = get_container()

    async def get_game_analysis(
        self,
        game_id: int,
        include_moves: bool = True,
        include_detailed_metrics: bool = True
    ) -> dict:
        """Obtiene análisis de una partida específica."""
        query = GetGameAnalysisQuery(
            game_id=game_id,
            include_moves=include_moves,
            include_detailed_metrics=include_detailed_metrics
        )

        use_case = self._container.get_get_game_analysis_use_case()
        result = await use_case.execute(query)

        if not result:
            raise HTTPException(status_code=404, detail="Game not found")

        response = {
            "game_id": result.game.id,
            "username": result.game.username,
            "game_url": result.game.game_url,
            "time_control": result.game.time_control,
            "result": result.game.result,
            "played_at": result.game.played_at.isoformat() if result.game.played_at else None,
        }

        if result.analysis and include_detailed_metrics:
            response["analysis"] = {
                "quality_metrics": {
                    "avg_acpl": result.analysis.quality_metrics.avg_acpl,
                    "avg_wdl_loss": result.analysis.quality_metrics.avg_wdl_loss,
                    "avg_match_rate": result.analysis.quality_metrics.avg_match_rate
                },
                "timing_metrics": {
                    "avg_move_time": result.analysis.timing_metrics.avg_move_time,
                    "time_variance": result.analysis.timing_metrics.time_variance,
                    "quick_moves_rate": result.analysis.timing_metrics.quick_moves_rate
                },
                "risk_assessment": {
                    "cheat_probability": result.analysis.risk_assessment.cheat_probability,
                    "risk_level": result.analysis.risk_assessment.risk_level,
                    "suspicion_flags": result.analysis.risk_assessment.suspicion_flags
                },
                "analyzed_at": result.analysis.analyzed_at.isoformat()
            }

        if result.moves_data and include_moves:
            response["moves"] = [asdict(move) for move in result.moves_data]

        return response

    async def get_player_games(
        self,
        username: str,
        limit: int = 20,
        offset: int = 0,
        include_analysis: bool = False,
        time_control_filter: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        only_analyzed: bool = False
    ) -> dict:
        """Obtiene partidas de un jugador con filtros."""
        query = GetPlayerGamesQuery(
            username=username,
            limit=limit,
            offset=offset,
            include_analysis=include_analysis,
            time_control_filter=time_control_filter,
            date_from=date_from,
            date_to=date_to,
            only_analyzed=only_analyzed
        )

        use_case = self._container.get_get_player_games_use_case()
        result = await use_case.execute(query)

        if not result:
            raise HTTPException(status_code=404, detail="Player not found")

        games_data = []
        for game in result.games:
            game_data = {
                "id": game.id,
                "username": game.username,
                "game_url": game.game_url,
                "time_control": game.time_control,
                "result": game.result,
                "played_at": game.played_at.isoformat() if game.played_at else None,
                "is_analyzed": game.moves_data is not None
            }
            games_data.append(game_data)

        return {
            "username": username,
            "games": games_data,
            "pagination": {
                "total_count": result.total_count,
                "limit": limit,
                "offset": offset,
                "has_more": result.has_more
            }
        }

    async def get_suspicious_games(
        self,
        username: Optional[str] = None,
        risk_threshold: int = 70,
        limit: int = 50,
        include_analysis: bool = True
    ) -> dict:
        """Obtiene partidas con alta probabilidad de trampa."""
        query = GetSuspiciousGamesQuery(
            username=username,
            risk_threshold=risk_threshold,
            limit=limit,
            include_analysis=include_analysis
        )

        use_case = self._container.get_get_suspicious_games_use_case()
        result = await use_case.execute(query)

        suspicious_games = []
        for game_result in result.games:
            game_data = {
                "id": game_result.game.id,
                "username": game_result.game.username,
                "game_url": game_result.game.game_url,
                "time_control": game_result.game.time_control,
                "result": game_result.game.result,
                "played_at": game_result.game.played_at.isoformat() if game_result.game.played_at else None,
            }

            if game_result.analysis and include_analysis:
                game_data["risk_assessment"] = {
                    "cheat_probability": game_result.analysis.risk_assessment.cheat_probability,
                    "risk_level": game_result.analysis.risk_assessment.risk_level,
                    "suspicion_flags": game_result.analysis.risk_assessment.suspicion_flags
                }

            suspicious_games.append(game_data)

        return {
            "suspicious_games": suspicious_games,
            "filters": {
                "username": username,
                "risk_threshold": risk_threshold,
                "limit": limit
            },
            "total_count": result.total_count
        }

    async def analyze_game(
        self,
        game_id: int,
        force_reanalysis: bool = False
    ) -> dict:
        """Analiza una partida específica."""
        use_case = self._container.get_analyze_game_use_case()
        result = await use_case.execute(game_id, force_reanalysis)

        if not result:
            raise HTTPException(status_code=404, detail="Game not found")

        response = {
            "game_id": result.game.id,
            "analyzed": result.analysis is not None,
            "message": "Game analyzed successfully" if result.analysis else "Game could not be analyzed"
        }

        if result.analysis:
            response["analysis"] = {
                "quality_metrics": {
                    "avg_acpl": result.analysis.quality_metrics.avg_acpl,
                    "avg_wdl_loss": result.analysis.quality_metrics.avg_wdl_loss,
                    "avg_match_rate": result.analysis.quality_metrics.avg_match_rate
                },
                "risk_assessment": {
                    "cheat_probability": result.analysis.risk_assessment.cheat_probability,
                    "risk_level": result.analysis.risk_assessment.risk_level
                }
            }

        return response
