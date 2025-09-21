"""
Queries - Read operations que no cambian el estado del sistema.
"""
from .player_queries import (
    GetPlayerAnalysisQuery,
    GetPlayerStatusQuery,
    ListPlayersQuery
)
from .game_queries import (
    GetGameAnalysisQuery,
    GetPlayerGamesQuery,
    GetSuspiciousGamesQuery
)

__all__ = [
    "GetPlayerAnalysisQuery",
    "GetPlayerStatusQuery",
    "ListPlayersQuery",
    "GetGameAnalysisQuery",
    "GetPlayerGamesQuery",
    "GetSuspiciousGamesQuery"
]