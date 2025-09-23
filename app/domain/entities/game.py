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
    best: Optional[str] = None
    cp_loss: Optional[int] = None
    eval_before: Optional[int] = None
    eval_after: Optional[int] = None
    time_spent: Optional[float] = None
    best_rank: Optional[int] = None
    match_rate: Optional[float] = None


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
    username: Optional[str] = None
    player_id: Optional[str] = None
    game_url: Optional[str] = None
    result: Optional[str] = None
    played_at: Optional[datetime] = None
    pgn_data: Optional[str] = None
    moves_data: Optional[List[MoveData]] = None

    def __post_init__(self):
        """Backfill derived attributes for compatibility with new services."""
        if self.pgn_data is None:
            self.pgn_data = self.pgn

        if self.played_at is None and self.created_at is not None:
            self.played_at = self.created_at
        if self.created_at is None and self.played_at is not None:
            self.created_at = self.played_at

        if self.moves_data is None and self.moves:
            self.moves_data = self.moves
        if self.moves is None and self.moves_data:
            self.moves = self.moves_data

        if self.username is None and self.player_id is not None:
            self.username = str(self.player_id)
        if self.player_id is None and self.username is not None:
            self.player_id = self.username

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

    def with_moves_data(self, moves_data: List[MoveData]) -> "Game":
        """Devuelve copia con datos de movimientos adjuntos."""
        return Game(
            id=self.id,
            pgn=self.pgn,
            white_username=self.white_username,
            black_username=self.black_username,
            white_elo=self.white_elo,
            black_elo=self.black_elo,
            time_control=self.time_control,
            termination=self.termination,
            eco_code=self.eco_code,
            opening_key=self.opening_key,
            move_times=self.move_times,
            moves=moves_data,
            created_at=self.created_at,
            username=self.username,
            player_id=self.player_id,
            game_url=self.game_url,
            result=self.result,
            played_at=self.played_at,
            pgn_data=self.pgn_data,
            moves_data=moves_data
        )
