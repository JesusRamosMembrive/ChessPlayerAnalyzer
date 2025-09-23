"""
Entidad Player del dominio.
"""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class PlayerStatus(str, Enum):
    """Estado del análisis de un jugador."""
    NOT_ANALYZED = "not_analyzed"
    PENDING = "pending"
    READY = "ready"
    ERROR = "error"


@dataclass
class Player:
    """Jugador de ajedrez."""
    username: str  # Primary key, no need for separate id
    status: PlayerStatus = PlayerStatus.NOT_ANALYZED
    requested_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    progress: int = 0
    total_games: int = 0
    done_games: int = 0
    error: Optional[str] = None
    last_task_id: Optional[str] = None

    @property
    def id(self) -> str:
        """Return username as id for backward compatibility."""
        return self.username

    def mark_as_pending(self, task_id: str) -> None:
        """Marca el jugador como en proceso."""
        self.status = PlayerStatus.PENDING
        self.requested_at = datetime.utcnow()
        self.last_task_id = task_id
        self.progress = 0
        self.error = None

    def update_progress(self, done_games: int, total_games: int) -> None:
        """Actualiza el progreso del análisis."""
        self.done_games = done_games
        self.total_games = total_games
        if total_games > 0:
            self.progress = int((done_games / total_games) * 100)

    def mark_as_ready(self) -> None:
        """Marca el jugador como completado."""
        self.status = PlayerStatus.READY
        self.finished_at = datetime.utcnow()
        self.progress = 100
        self.error = None

    def mark_as_error(self, error_message: str) -> None:
        """Marca el jugador con error."""
        self.status = PlayerStatus.ERROR
        self.error = error_message
        self.finished_at = datetime.utcnow()

    def is_ready_for_analysis(self) -> bool:
        """Verifica si el jugador está listo para análisis."""
        return self.status in [PlayerStatus.NOT_ANALYZED, PlayerStatus.ERROR]

    def can_be_refreshed(self) -> bool:
        """Verifica si se puede refrescar el análisis."""
        return self.status in [PlayerStatus.READY, PlayerStatus.ERROR]