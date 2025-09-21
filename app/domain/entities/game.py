"""
Entidad Game del dominio.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


@dataclass
class MoveData:
    """Datos de un movimiento."""
    move_number: int
    played: str
    best: str
    cp_loss: int
    eval_before: Optional[int] = None
    eval_after: Optional[int] = None
    time_spent: Optional[float] = None


@dataclass
class Game:
    """Partida de ajedrez."""
    id: Optional[int]
    pgn: str
    white_username: Optional[str] = None
    black_username: Optional[str] = None
    white_elo: Optional[int] = None
    black_elo: Optional[int] = None
    time_control: Optional[str] = None
    termination: Optional[str] = None
    eco_code: Optional[str] = None
    opening_key: Optional[str] = None
    move_times: Optional[List[int]] = None
    moves: Optional[List[MoveData]] = None
    created_at: Optional[datetime] = None

    def get_player_color(self, username: str) -> Optional[str]:
        """Obtiene el color del jugador especificado."""
        if self.white_username == username:
            return "white"
        elif self.black_username == username:
            return "black"
        return None

    def get_player_elo(self, username: str) -> Optional[int]:
        """Obtiene el ELO del jugador especificado."""
        color = self.get_player_color(username)
        if color == "white":
            return self.white_elo
        elif color == "black":
            return self.black_elo
        return None

    def is_analyzed(self) -> bool:
        """Verifica si la partida tiene análisis de movimientos."""
        return self.moves is not None and len(self.moves) > 0

    def get_total_moves(self) -> int:
        """Obtiene el número total de movimientos."""
        return len(self.moves) if self.moves else 0