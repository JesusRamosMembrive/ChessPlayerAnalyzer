"""
Commands relacionados con jugadores.
Commands representan intenciones de cambio de estado.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AnalyzePlayerCommand:
    """Comando para iniciar análisis de un jugador."""
    username: str
    force_refresh: bool = False
    priority: int = 5  # 0 = alta, 9 = baja
    months_to_analyze: int = 12

    def __post_init__(self):
        """Validaciones del comando."""
        if not self.username or not self.username.strip():
            raise ValueError("Username cannot be empty")
        if len(self.username) > 50:
            raise ValueError("Username too long")
        if self.months_to_analyze < 1 or self.months_to_analyze > 24:
            raise ValueError("Months to analyze must be between 1 and 24")


@dataclass(frozen=True)
class RefreshPlayerAnalysisCommand:
    """Comando para refrescar análisis existente."""
    username: str
    delete_existing_data: bool = True

    def __post_init__(self):
        """Validaciones del comando."""
        if not self.username or not self.username.strip():
            raise ValueError("Username cannot be empty")


@dataclass(frozen=True)
class DeletePlayerCommand:
    """Comando para eliminar jugador y todos sus datos."""
    username: str
    confirm_deletion: bool = False

    def __post_init__(self):
        """Validaciones del comando."""
        if not self.username or not self.username.strip():
            raise ValueError("Username cannot be empty")
        if not self.confirm_deletion:
            raise ValueError("Must confirm deletion")


@dataclass(frozen=True)
class UpdatePlayerProgressCommand:
    """Comando para actualizar progreso de análisis."""
    username: str
    done_games: int
    total_games: int
    task_id: Optional[str] = None

    def __post_init__(self):
        """Validaciones del comando."""
        if not self.username:
            raise ValueError("Username cannot be empty")
        if self.done_games < 0:
            raise ValueError("Done games cannot be negative")
        if self.total_games < 0:
            raise ValueError("Total games cannot be negative")
        if self.done_games > self.total_games:
            raise ValueError("Done games cannot exceed total games")