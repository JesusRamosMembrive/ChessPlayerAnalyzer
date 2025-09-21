"""
Use Cases - Application layer orchestration.
Coordinan domain services y repositories para implementar casos de uso de negocio.
"""
from .player_use_cases import (
    AnalyzePlayerUseCase,
    GetPlayerAnalysisUseCase,
    GetPlayerStatusUseCase,
    RefreshPlayerAnalysisUseCase,
    DeletePlayerUseCase,
    UpdatePlayerProgressUseCase,
    AnalyzePlayerResult,
    PlayerStatusResult
)
from .game_use_cases import (
    GetGameAnalysisUseCase,
    GetPlayerGamesUseCase,
    GetSuspiciousGamesUseCase,
    AnalyzeGameUseCase,
    GameAnalysisResult,
    PlayerGamesResult,
    SuspiciousGamesResult
)

__all__ = [
    # Player use cases
    "AnalyzePlayerUseCase",
    "GetPlayerAnalysisUseCase",
    "GetPlayerStatusUseCase",
    "RefreshPlayerAnalysisUseCase",
    "DeletePlayerUseCase",
    "UpdatePlayerProgressUseCase",

    # Game use cases
    "GetGameAnalysisUseCase",
    "GetPlayerGamesUseCase",
    "GetSuspiciousGamesUseCase",
    "AnalyzeGameUseCase",

    # Result objects
    "AnalyzePlayerResult",
    "PlayerStatusResult",
    "GameAnalysisResult",
    "PlayerGamesResult",
    "SuspiciousGamesResult"
]