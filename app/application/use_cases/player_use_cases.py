"""
Use cases relacionados con jugadores.
Orquestan domain services y repositories.
"""
from typing import List, Optional
from dataclasses import dataclass

from ...domain.entities.player import Player, PlayerStatus
from ...domain.entities.game import Game
from ...domain.entities.analysis import PlayerAnalysis
from ...domain.repositories.player_repository import PlayerRepository
from ...domain.repositories.game_repository import GameRepository
from ...domain.repositories.analysis_repository import AnalysisRepository
from ...domain.services.player_service import PlayerService
from ...domain.services.analysis_service import AnalysisService
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


@dataclass
class AnalyzePlayerResult:
    """Resultado del análisis de jugador."""
    success: bool
    player_id: Optional[int] = None
    task_id: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class PlayerStatusResult:
    """Estado del jugador."""
    username: str
    status: str
    progress_percentage: float
    done_games: int
    total_games: int
    task_id: Optional[str] = None
    error_message: Optional[str] = None


class AnalyzePlayerUseCase:
    """Use case para iniciar análisis de jugador."""

    def __init__(
        self,
        player_repository: PlayerRepository,
        game_repository: GameRepository,
        analysis_repository: AnalysisRepository,
        player_service: PlayerService,
        analysis_service: AnalysisService
    ):
        self._player_repository = player_repository
        self._game_repository = game_repository
        self._analysis_repository = analysis_repository
        self._player_service = player_service
        self._analysis_service = analysis_service

    async def execute(self, command: AnalyzePlayerCommand) -> AnalyzePlayerResult:
        """Ejecuta el análisis de jugador."""
        try:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Starting analyze player use case for {command.username}")

            # Verificar si el jugador ya existe
            existing_player = await self._player_repository.get_by_username(command.username)
            logger.info(f"Retrieved existing player: {existing_player is not None}")

            if existing_player and not command.force_refresh:
                # Si ya tiene análisis completo, no hacer nada
                if existing_player.status == PlayerStatus.READY:
                    return AnalyzePlayerResult(
                        success=True,
                        player_id=existing_player.id,
                        error_message="Player already analyzed"
                    )

            # Solicitar análisis del jugador (crea nuevo o actualiza existente)
            logger.info("About to call player service request_analysis")
            player = await self._player_service.request_analysis(
                username=command.username,
                force_refresh=command.force_refresh
            )
            logger.info(f"Player service returned player with ID: {player.id}")

            # Guardar en repositorio
            logger.info("About to save player to repository")
            saved_player = await self._player_repository.save(player)
            logger.info(f"Repository returned saved player with ID: {saved_player.id}")

            # TODO: Aquí se enviaría tarea a Celery para análisis asíncrono
            # Por ahora solo marcamos como pending

            return AnalyzePlayerResult(
                success=True,
                player_id=saved_player.id,
                task_id=None  # Se asignaría cuando se envíe a Celery
            )

        except Exception as e:
            return AnalyzePlayerResult(
                success=False,
                error_message=str(e)
            )


class GetPlayerAnalysisUseCase:
    """Use case para obtener análisis de jugador."""

    def __init__(
        self,
        player_repository: PlayerRepository,
        analysis_repository: AnalysisRepository,
        game_repository: GameRepository
    ):
        self._player_repository = player_repository
        self._analysis_repository = analysis_repository
        self._game_repository = game_repository

    async def execute(self, query: GetPlayerAnalysisQuery) -> Optional[PlayerAnalysis]:
        """Obtiene el análisis completo del jugador."""
        # Verificar que el jugador existe
        player = await self._player_repository.get_by_username(query.username)
        if not player:
            return None

        # Obtener análisis principal
        analysis = await self._analysis_repository.get_by_player_id(player.id)
        if not analysis:
            return None

        # Si se requieren partidas, cargarlas
        if query.include_games:
            games = await self._game_repository.get_by_player_id(player.id)
            # TODO: Agregar partidas al análisis si es necesario

        # Si se requieren partidas sospechosas, filtrarlas
        if query.include_suspicious_games:
            suspicious_games = await self._game_repository.get_suspicious_by_player_id(
                player.id,
                risk_threshold=70
            )
            # TODO: Agregar partidas sospechosas al análisis

        return analysis


class GetPlayerStatusUseCase:
    """Use case para obtener estado del jugador."""

    def __init__(self, player_repository: PlayerRepository):
        self._player_repository = player_repository

    async def execute(self, query: GetPlayerStatusQuery) -> Optional[PlayerStatusResult]:
        """Obtiene el estado actual del jugador."""
        player = await self._player_repository.get_by_username(query.username)
        if not player:
            return None

        return PlayerStatusResult(
            username=player.username,
            status=player.status,
            progress_percentage=player.progress_percentage,
            done_games=player.done_games,
            total_games=player.total_games,
            task_id=player.task_id,
            error_message=player.error_message
        )


class RefreshPlayerAnalysisUseCase:
    """Use case para refrescar análisis de jugador."""

    def __init__(
        self,
        player_repository: PlayerRepository,
        game_repository: GameRepository,
        analysis_repository: AnalysisRepository,
        player_service: PlayerService
    ):
        self._player_repository = player_repository
        self._game_repository = game_repository
        self._analysis_repository = analysis_repository
        self._player_service = player_service

    async def execute(self, command: RefreshPlayerAnalysisCommand) -> AnalyzePlayerResult:
        """Refresca el análisis de un jugador existente."""
        try:
            player = await self._player_repository.get_by_username(command.username)
            if not player:
                return AnalyzePlayerResult(
                    success=False,
                    error_message="Player not found"
                )

            if command.delete_existing_data:
                # Eliminar datos existentes
                await self._game_repository.delete_by_player_id(player.id)
                await self._analysis_repository.delete_by_player_id(player.id)

            # Reset del jugador para nuevo análisis
            refreshed_player = self._player_service.reset_for_reanalysis(player)
            saved_player = await self._player_repository.save(refreshed_player)

            # TODO: Enviar nueva tarea a Celery

            return AnalyzePlayerResult(
                success=True,
                player_id=saved_player.id
            )

        except Exception as e:
            return AnalyzePlayerResult(
                success=False,
                error_message=str(e)
            )


class DeletePlayerUseCase:
    """Use case para eliminar jugador y todos sus datos."""

    def __init__(
        self,
        player_repository: PlayerRepository,
        game_repository: GameRepository,
        analysis_repository: AnalysisRepository
    ):
        self._player_repository = player_repository
        self._game_repository = game_repository
        self._analysis_repository = analysis_repository

    async def execute(self, command: DeletePlayerCommand) -> bool:
        """Elimina completamente un jugador y sus datos."""
        try:
            player = await self._player_repository.get_by_username(command.username)
            if not player:
                return False

            # Eliminar en orden: análisis -> partidas -> jugador
            await self._analysis_repository.delete_by_player_id(player.id)
            await self._game_repository.delete_by_player_id(player.id)
            await self._player_repository.delete(player.id)

            return True

        except Exception:
            return False


class UpdatePlayerProgressUseCase:
    """Use case para actualizar progreso de análisis."""

    def __init__(
        self,
        player_repository: PlayerRepository,
        player_service: PlayerService
    ):
        self._player_repository = player_repository
        self._player_service = player_service

    async def execute(self, command: UpdatePlayerProgressCommand) -> bool:
        """Actualiza el progreso del análisis de un jugador."""
        try:
            player = await self._player_repository.get_by_username(command.username)
            if not player:
                return False

            updated_player = self._player_service.update_progress(
                player=player,
                done_games=command.done_games,
                total_games=command.total_games,
                task_id=command.task_id
            )

            await self._player_repository.save(updated_player)
            return True

        except Exception:
            return False