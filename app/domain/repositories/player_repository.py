"""
Interface del repositorio de jugadores.
"""
from abc import ABC, abstractmethod
from typing import Optional, List
from ..entities.player import Player


class PlayerRepository(ABC):
    """Interface para persistencia de jugadores."""

    @abstractmethod
    async def get_by_username(self, username: str) -> Optional[Player]:
        """Obtiene un jugador por username."""
        pass

    @abstractmethod
    async def save(self, player: Player) -> Player:
        """Guarda un jugador."""
        pass

    @abstractmethod
    async def delete(self, username: str) -> bool:
        """Elimina un jugador."""
        pass

    @abstractmethod
    async def list_pending(self) -> List[Player]:
        """Lista jugadores con análisis pendiente."""
        pass

    @abstractmethod
    async def update_progress(self, username: str, progress: int,
                            done_games: int, total_games: int) -> None:
        """Actualiza el progreso de análisis."""
        pass

    @abstractmethod
    async def mark_as_ready(self, username: str) -> None:
        """Marca jugador como listo."""
        pass

    @abstractmethod
    async def mark_as_error(self, username: str, error: str) -> None:
        """Marca jugador con error."""
        pass