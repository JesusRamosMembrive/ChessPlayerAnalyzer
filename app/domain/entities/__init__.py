"""
Entidades de dominio.
Representan los conceptos centrales del negocio.
"""
from .player import Player, PlayerStatus
from .game import Game
from .analysis import PlayerAnalysis, GameAnalysis

__all__ = [
    "Player",
    "PlayerStatus",
    "Game",
    "PlayerAnalysis",
    "GameAnalysis"
]