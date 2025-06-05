from datetime import datetime, UTC
from typing import Optional, List
from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy import Column, JSON   # ←  faltaba

class Game(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    pgn: str

    moves: List["MoveAnalysis"] = Relationship(back_populates="game")
    metrics: Optional["GameMetrics"] = Relationship(back_populates="game")
    move_times: list[int] | None = Field(sa_column=Column(JSON))
    eco_code: str | None = None          # «C23», «B12»…
    opening_key: str | None = None       # SAN de los 1-8 plies (“e4 e5 Nf3 …”)

class MoveAnalysis(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    game_id: int = Field(foreign_key="game.id")

    move_number: int
    played: str
    best: str
    best_rank: int       # 0 = mejor, 1 = 2.º, 2 = 3.º, 3 = fuera del top‑3
    cp_loss: int         # centipawns perdidos

    game: Game = Relationship(back_populates="moves")

class GameMetrics(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    game_id: int = Field(foreign_key="game.id", unique=True)

    sigma_total: float | None = None
    constant_time: bool = False
    pause_spike: bool = False

    pct_top1: float      # % jugadas rank 0
    pct_top3: float      # % jugadas rank ≤2
    acl: float           # Average Centipawn Loss
    suspicious: bool = False
    computed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    game: Game = Relationship(back_populates="metrics")


class PlayerMetrics(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(unique=True)
    game_count: int
    opening_entropy: float               # H bits
    most_played: str | None = None       # opening_key más frecuente
    low_entropy: bool = False            # bandera (< 1.0 bits)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
