"""
Queries relacionadas con partidas.
"""
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime


@dataclass(frozen=True)
class GetGameAnalysisQuery:
    """Query para obtener análisis de una partida específica."""
    game_id: int
    include_moves: bool = True
    include_detailed_metrics: bool = True

    def __post_init__(self):
        """Validaciones de la query."""
        if self.game_id <= 0:
            raise ValueError("Game ID must be positive")


@dataclass(frozen=True)
class GetPlayerGamesQuery:
    """Query para obtener partidas de un jugador."""
    username: str
    limit: int = 20
    offset: int = 0
    include_analysis: bool = False
    time_control_filter: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    only_analyzed: bool = False

    def __post_init__(self):
        """Validaciones de la query."""
        if not self.username or not self.username.strip():
            raise ValueError("Username cannot be empty")
        if self.limit < 1 or self.limit > 100:
            raise ValueError("Limit must be between 1 and 100")
        if self.offset < 0:
            raise ValueError("Offset cannot be negative")
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("Date from cannot be after date to")


@dataclass(frozen=True)
class GetSuspiciousGamesQuery:
    """Query para obtener partidas sospechosas."""
    username: Optional[str] = None
    risk_threshold: int = 70
    limit: int = 50
    include_analysis: bool = True

    def __post_init__(self):
        """Validaciones de la query."""
        if self.risk_threshold < 0 or self.risk_threshold > 100:
            raise ValueError("Risk threshold must be between 0 and 100")
        if self.limit < 1 or self.limit > 100:
            raise ValueError("Limit must be between 1 and 100")