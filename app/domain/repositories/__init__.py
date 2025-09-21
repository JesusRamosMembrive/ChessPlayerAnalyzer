"""
Interfaces de repositorio del dominio.
Definen contratos sin implementación específica.
"""
from .player_repository import PlayerRepository
from .game_repository import GameRepository
from .analysis_repository import AnalysisRepository

__all__ = [
    "PlayerRepository",
    "GameRepository",
    "AnalysisRepository"
]