"""
Queries relacionadas con jugadores.
Queries representan solicitudes de lectura de datos.
"""
from dataclasses import dataclass
from typing import Optional, List
from enum import Enum


class PlayerStatusFilter(str, Enum):
    """Filtros de estado para listado de jugadores."""
    ALL = "all"
    PENDING = "pending"
    READY = "ready"
    ERROR = "error"
    NOT_ANALYZED = "not_analyzed"


@dataclass(frozen=True)
class GetPlayerAnalysisQuery:
    """Query para obtener análisis completo de un jugador."""
    username: str
    include_games: bool = False
    include_suspicious_games: bool = False

    def __post_init__(self):
        """Validaciones de la query."""
        if not self.username or not self.username.strip():
            raise ValueError("Username cannot be empty")


@dataclass(frozen=True)
class GetPlayerStatusQuery:
    """Query para obtener estado actual de un jugador."""
    username: str
    include_progress_details: bool = True

    def __post_init__(self):
        """Validaciones de la query."""
        if not self.username or not self.username.strip():
            raise ValueError("Username cannot be empty")


@dataclass(frozen=True)
class ListPlayersQuery:
    """Query para listar jugadores con filtros."""
    status_filter: PlayerStatusFilter = PlayerStatusFilter.ALL
    limit: int = 50
    offset: int = 0
    order_by: str = "requested_at"  # requested_at, finished_at, username
    order_desc: bool = True

    def __post_init__(self):
        """Validaciones de la query."""
        if self.limit < 1 or self.limit > 100:
            raise ValueError("Limit must be between 1 and 100")
        if self.offset < 0:
            raise ValueError("Offset cannot be negative")
        if self.order_by not in ["requested_at", "finished_at", "username", "progress"]:
            raise ValueError("Invalid order_by field")


@dataclass(frozen=True)
class GetPlayerStatisticsQuery:
    """Query para obtener estadísticas generales de jugadores."""
    include_risk_breakdown: bool = True
    include_recent_activity: bool = True
    days_back: int = 30

    def __post_init__(self):
        """Validaciones de la query."""
        if self.days_back < 1 or self.days_back > 365:
            raise ValueError("Days back must be between 1 and 365")