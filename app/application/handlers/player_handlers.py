"""
Handlers para endpoints relacionados con jugadores.
Actúan como bridge entre FastAPI y los use cases.
"""
from typing import List, Optional
from fastapi import HTTPException

from ..container import get_container
from ..commands.player_commands import (
    AnalyzePlayerCommand,
    RefreshPlayerAnalysisCommand,
    DeletePlayerCommand,
    UpdatePlayerProgressCommand
)
from ..queries.player_queries import (
    GetPlayerAnalysisQuery,
    GetPlayerStatusQuery,
    ListPlayersQuery,
    GetPlayerStatisticsQuery
)


class PlayerHandlers:
    """Handlers para operaciones de jugadores."""

    def __init__(self):
        self._container = get_container()

    async def analyze_player(
        self,
        username: str,
        force_refresh: bool = False,
        priority: int = 5,
        months_to_analyze: int = 12
    ) -> dict:
        """Inicia análisis de un jugador."""
        command = AnalyzePlayerCommand(
            username=username,
            force_refresh=force_refresh,
            priority=priority,
            months_to_analyze=months_to_analyze
        )

        use_case = self._container.get_analyze_player_use_case()
        result = await use_case.execute(command)

        if not result.success:
            raise HTTPException(status_code=400, detail=result.error_message)

        return {
            "success": True,
            "player_id": result.player_id,
            "task_id": result.task_id,
            "message": f"Analysis started for player {username}"
        }

    async def get_player_analysis(
        self,
        username: str,
        include_games: bool = False,
        include_suspicious_games: bool = False
    ) -> dict:
        """Obtiene análisis completo de un jugador."""
        query = GetPlayerAnalysisQuery(
            username=username,
            include_games=include_games,
            include_suspicious_games=include_suspicious_games
        )

        use_case = self._container.get_get_player_analysis_use_case()
        analysis = await use_case.execute(query)

        if not analysis:
            raise HTTPException(status_code=404, detail="Player analysis not found")

        # Convertir análisis domain a dict para respuesta
        return {
            "username": username,
            "analysis": {
                "overall_metrics": {
                    "avg_acpl": analysis.overall_metrics.avg_acpl,
                    "avg_wdl_loss": analysis.overall_metrics.avg_wdl_loss,
                    "avg_match_rate": analysis.overall_metrics.avg_match_rate
                },
                "timing_metrics": {
                    "avg_move_time": analysis.timing_metrics.avg_move_time,
                    "time_variance": analysis.timing_metrics.time_variance,
                    "quick_moves_rate": analysis.timing_metrics.quick_moves_rate
                },
                "risk_assessment": {
                    "cheat_probability": analysis.risk_assessment.cheat_probability,
                    "risk_level": analysis.risk_assessment.risk_level,
                    "suspicion_flags": analysis.risk_assessment.suspicion_flags
                },
                "games_analyzed": analysis.games_analyzed,
                "total_games": analysis.total_games,
                "analysis_date": analysis.analysis_date.isoformat()
            }
        }

    async def get_player_status(
        self,
        username: str,
        include_progress_details: bool = True
    ) -> dict:
        """Obtiene estado actual de un jugador."""
        query = GetPlayerStatusQuery(
            username=username,
            include_progress_details=include_progress_details
        )

        use_case = self._container.get_get_player_status_use_case()
        status = await use_case.execute(query)

        if not status:
            raise HTTPException(status_code=404, detail="Player not found")

        return {
            "username": status.username,
            "status": status.status,
            "progress": {
                "percentage": status.progress_percentage,
                "done_games": status.done_games,
                "total_games": status.total_games
            },
            "task_id": status.task_id,
            "error_message": status.error_message
        }

    async def refresh_player_analysis(
        self,
        username: str,
        delete_existing_data: bool = True
    ) -> dict:
        """Refresca análisis de un jugador existente."""
        command = RefreshPlayerAnalysisCommand(
            username=username,
            delete_existing_data=delete_existing_data
        )

        use_case = self._container.get_refresh_player_analysis_use_case()
        result = await use_case.execute(command)

        if not result.success:
            raise HTTPException(status_code=400, detail=result.error_message)

        return {
            "success": True,
            "player_id": result.player_id,
            "message": f"Analysis refresh started for player {username}"
        }

    async def delete_player(
        self,
        username: str,
        confirm_deletion: bool = False
    ) -> dict:
        """Elimina un jugador y todos sus datos."""
        if not confirm_deletion:
            raise HTTPException(
                status_code=400,
                detail="Must confirm deletion by setting confirm_deletion=true"
            )

        command = DeletePlayerCommand(
            username=username,
            confirm_deletion=confirm_deletion
        )

        use_case = self._container.get_delete_player_use_case()
        success = await use_case.execute(command)

        if not success:
            raise HTTPException(status_code=404, detail="Player not found")

        return {
            "success": True,
            "message": f"Player {username} and all associated data deleted"
        }

    async def update_player_progress(
        self,
        username: str,
        done_games: int,
        total_games: int,
        task_id: Optional[str] = None
    ) -> dict:
        """Actualiza progreso de análisis de un jugador."""
        command = UpdatePlayerProgressCommand(
            username=username,
            done_games=done_games,
            total_games=total_games,
            task_id=task_id
        )

        use_case = self._container.get_update_player_progress_use_case()
        success = await use_case.execute(command)

        if not success:
            raise HTTPException(status_code=404, detail="Player not found")

        return {
            "success": True,
            "message": f"Progress updated for player {username}"
        }