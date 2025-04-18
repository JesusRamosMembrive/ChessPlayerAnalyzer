# -----------------------------------------------------------------------------
# File: backend/app/models.py
# -----------------------------------------------------------------------------
from datetime import datetime, UTC
from typing import Optional, List
from sqlmodel import SQLModel, Field, Relationship


class Game(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    pgn: str

    moves: List["MoveAnalysis"] = Relationship(back_populates="game")


class MoveAnalysis(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    game_id: int = Field(foreign_key="game.id")

    move_number: int
    played: str
    best: str
    best_rank: int
    cp_loss: int

    game: Game = Relationship(back_populates="moves")