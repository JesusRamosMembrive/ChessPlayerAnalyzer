"""
Handlers - Bridge entre FastAPI endpoints y use cases.
"""
from .player_handlers import PlayerHandlers
from .game_handlers import GameHandlers

__all__ = [
    "PlayerHandlers",
    "GameHandlers"
]