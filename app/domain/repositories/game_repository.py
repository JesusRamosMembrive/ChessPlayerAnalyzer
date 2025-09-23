"""
Interface del repositorio de partidas.
"""
from abc import ABC, abstractmethod
from typing import Optional, List
from ..entities.game import Game


class GameRepository(ABC):
    """Interface para persistencia de partidas."""

    @abstractmethod
    async def get_by_id(self, game_id: int) -> Optional[Game]:
        """Obtiene una partida por ID."""
        pass

    @abstractmethod
    async def save(self, game: Game) -> Game:
        """Guarda una partida."""
        pass

    @abstractmethod
    async def get_by_player(self, username: str) -> List[Game]:
        """Obtiene todas las partidas de un jugador."""
        pass

    @abstractmethod
    async def get_by_player_id(self, player_id: str) -> List[Game]:
        """Alias que acepta identificador lógico del jugador."""
        pass

    @abstractmethod
    async def get_analyzed_count(self, username: str) -> int:
        """Cuenta partidas analizadas de un jugador."""
        pass

    @abstractmethod
    async def delete_by_player(self, username: str) -> int:
        """Elimina todas las partidas de un jugador."""
        pass

    @abstractmethod
    async def bulk_save(self, games: List[Game]) -> List[Game]:
        """Guarda múltiples partidas de forma eficiente."""
        pass
