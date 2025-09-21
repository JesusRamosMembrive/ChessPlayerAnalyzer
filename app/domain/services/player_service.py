"""
Servicio de dominio para gestión de jugadores.
"""
from typing import Optional, List
from datetime import datetime

from ..entities.player import Player, PlayerStatus
from ..entities.analysis import PlayerAnalysis
from ..repositories.player_repository import PlayerRepository
from ..repositories.analysis_repository import AnalysisRepository


class PlayerService:
    """
    Servicio de dominio para lógica de negocio de jugadores.
    Orquesta operaciones complejas sin depender de infraestructura específica.
    """

    def __init__(self, player_repo: PlayerRepository, analysis_repo: AnalysisRepository):
        self.player_repo = player_repo
        self.analysis_repo = analysis_repo

    async def request_analysis(self, username: str, force_refresh: bool = False) -> Player:
        """
        Solicita análisis de un jugador.

        Args:
            username: Nombre del jugador
            force_refresh: Si forzar análisis aunque ya exista

        Returns:
            Player con estado actualizado

        Raises:
            ValueError: Si el jugador no puede ser analizado
        """
        # Obtener o crear jugador
        player = await self.player_repo.get_by_username(username)
        if not player:
            player = Player(username=username)

        # Verificar si se puede analizar
        if not force_refresh and not player.is_ready_for_analysis():
            if player.status == PlayerStatus.PENDING:
                raise ValueError(f"Player {username} is already being analyzed")
            if player.status == PlayerStatus.READY and not player.can_be_refreshed():
                raise ValueError(f"Player {username} has recent analysis, use force_refresh=True")

        # Marcar como pendiente
        task_id = f"analyze_{username}_{int(datetime.utcnow().timestamp())}"
        player.mark_as_pending(task_id)

        # Guardar estado
        await self.player_repo.save(player)

        return player

    async def update_analysis_progress(self, username: str, done_games: int,
                                     total_games: int) -> Player:
        """
        Actualiza el progreso de análisis de un jugador.

        Args:
            username: Nombre del jugador
            done_games: Partidas analizadas
            total_games: Total de partidas

        Returns:
            Player con progreso actualizado
        """
        player = await self.player_repo.get_by_username(username)
        if not player:
            raise ValueError(f"Player {username} not found")

        if player.status != PlayerStatus.PENDING:
            raise ValueError(f"Player {username} is not being analyzed")

        player.update_progress(done_games, total_games)
        await self.player_repo.save(player)

        return player

    async def complete_analysis(self, username: str, analysis: PlayerAnalysis) -> Player:
        """
        Completa el análisis de un jugador.

        Args:
            username: Nombre del jugador
            analysis: Resultado del análisis

        Returns:
            Player marcado como completado
        """
        player = await self.player_repo.get_by_username(username)
        if not player:
            raise ValueError(f"Player {username} not found")

        # Guardar análisis
        await self.analysis_repo.save_player_analysis(analysis)

        # Marcar jugador como completado
        player.mark_as_ready()
        await self.player_repo.save(player)

        return player

    async def mark_analysis_error(self, username: str, error_message: str) -> Player:
        """
        Marca análisis de jugador con error.

        Args:
            username: Nombre del jugador
            error_message: Descripción del error

        Returns:
            Player marcado con error
        """
        player = await self.player_repo.get_by_username(username)
        if not player:
            raise ValueError(f"Player {username} not found")

        player.mark_as_error(error_message)
        await self.player_repo.save(player)

        return player

    async def get_player_with_analysis(self, username: str) -> tuple[Optional[Player], Optional[PlayerAnalysis]]:
        """
        Obtiene jugador junto con su análisis.

        Args:
            username: Nombre del jugador

        Returns:
            Tupla (Player, PlayerAnalysis) o (None, None) si no existe
        """
        player = await self.player_repo.get_by_username(username)
        if not player:
            return None, None

        analysis = None
        if player.status == PlayerStatus.READY:
            analysis = await self.analysis_repo.get_player_analysis(username)

        return player, analysis

    async def delete_player_completely(self, username: str) -> bool:
        """
        Elimina completamente un jugador y todos sus datos.

        Args:
            username: Nombre del jugador

        Returns:
            True si se eliminó, False si no existía
        """
        player = await self.player_repo.get_by_username(username)
        if not player:
            return False

        # Eliminar análisis asociados
        await self.analysis_repo.delete_player_analysis(username)

        # Eliminar jugador
        await self.player_repo.delete(username)

        return True

    async def refresh_analysis(self, username: str) -> Player:
        """
        Refresca el análisis de un jugador existente.

        Args:
            username: Nombre del jugador

        Returns:
            Player con análisis refrescado

        Raises:
            ValueError: Si el jugador no puede ser refrescado
        """
        player = await self.player_repo.get_by_username(username)
        if not player:
            raise ValueError(f"Player {username} not found")

        if not player.can_be_refreshed():
            raise ValueError(f"Player {username} cannot be refreshed (status: {player.status})")

        # Eliminar análisis anterior
        await self.analysis_repo.delete_player_analysis(username)

        # Solicitar nuevo análisis
        return await self.request_analysis(username, force_refresh=True)

    def validate_analysis_request(self, username: str, force_refresh: bool = False) -> dict:
        """
        Valida si se puede solicitar análisis de un jugador.

        Args:
            username: Nombre del jugador
            force_refresh: Si forzar análisis

        Returns:
            Dict con resultado de validación {valid: bool, reason: str}
        """
        if not username or not username.strip():
            return {"valid": False, "reason": "Username cannot be empty"}

        if len(username) > 50:  # Límite típico de Chess.com
            return {"valid": False, "reason": "Username too long"}

        # Validar caracteres permitidos (alfanuméricos, guiones, guiones bajos)
        if not username.replace('-', '').replace('_', '').isalnum():
            return {"valid": False, "reason": "Username contains invalid characters"}

        return {"valid": True, "reason": "Valid request"}

    async def get_analysis_statistics(self) -> dict:
        """
        Obtiene estadísticas generales de análisis.

        Returns:
            Dict con estadísticas
        """
        pending_players = await self.player_repo.list_pending()

        return {
            "pending_analyses": len(pending_players),
            "total_players_in_queue": len(pending_players),
            "average_queue_time": None,  # Requiere implementación más avanzada
            "system_status": "operational" if len(pending_players) < 100 else "busy"
        }