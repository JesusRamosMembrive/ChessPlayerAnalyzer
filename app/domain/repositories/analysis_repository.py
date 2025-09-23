"""
Interface del repositorio de análisis.
"""
from abc import ABC, abstractmethod
from typing import Optional, List
from ..entities.analysis import PlayerAnalysis, GameAnalysis


class AnalysisRepository(ABC):
    """Interface para persistencia de análisis."""

    @abstractmethod
    async def get_player_analysis(self, username: str) -> Optional[PlayerAnalysis]:
        """Obtiene el análisis de un jugador."""
        pass

    @abstractmethod
    async def save_player_analysis(self, analysis: PlayerAnalysis) -> PlayerAnalysis:
        """Guarda análisis de jugador."""
        pass

    @abstractmethod
    async def get_game_analysis(self, game_id: int) -> Optional[GameAnalysis]:
        """Obtiene el análisis de una partida."""
        pass

    @abstractmethod
    async def save_game_analysis(self, analysis: GameAnalysis) -> GameAnalysis:
        """Guarda análisis de partida."""
        pass

    async def get_game_analysis_by_id(self, game_id: int) -> Optional[GameAnalysis]:
        """Alias temporal para compatibilidad."""
        return await self.get_game_analysis(game_id)

    @abstractmethod
    async def get_game_analyses_by_player(self, username: str) -> List[GameAnalysis]:
        """Obtiene todos los análisis de partidas de un jugador."""
        pass

    @abstractmethod
    async def delete_player_analysis(self, username: str) -> bool:
        """Elimina análisis de un jugador."""
        pass
