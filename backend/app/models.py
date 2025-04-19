from datetime import datetime, UTC
from typing import Optional, List
from sqlmodel import SQLModel, Field, Relationship

class Game(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    pgn: str

    moves: List["MoveAnalysis"] = Relationship(back_populates="game")
    metrics: Optional["GameMetrics"] = Relationship(back_populates="game")

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

    pct_top1: float      # % jugadas rank 0
    pct_top3: float      # % jugadas rank ≤2
    acl: float           # Average Centipawn Loss
    suspicious: bool = False
    computed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    game: Game = Relationship(back_populates="metrics")