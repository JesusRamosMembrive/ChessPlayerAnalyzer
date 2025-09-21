"""
Implementación SQL del repositorio de jugadores.
"""
from typing import Optional, List
from sqlmodel import Session, select

from app.domain.entities.player import Player as DomainPlayer
from app.domain.repositories.player_repository import PlayerRepository
from app.models import Player as SQLPlayer, PlayerStatus as SQLPlayerStatus
from .mappers import player_to_domain, domain_to_player


class SQLPlayerRepository(PlayerRepository):
    """
    Implementación SQL del repositorio de jugadores.
    """

    def __init__(self, session: Session):
        self.session = session

    async def get_by_username(self, username: str) -> Optional[DomainPlayer]:
        """Obtiene un jugador por username."""
        sql_player = self.session.get(SQLPlayer, username)
        if sql_player:
            return player_to_domain(sql_player)
        return None

    async def save(self, player: DomainPlayer) -> DomainPlayer:
        """Guarda un jugador."""
        # Verificar si ya existe
        existing = self.session.get(SQLPlayer, player.username)

        if existing:
            # Actualizar campos
            existing.status = SQLPlayerStatus(player.status.value)
            existing.requested_at = player.requested_at
            existing.finished_at = player.finished_at
            existing.progress = player.progress
            existing.total_games = player.total_games
            existing.done_games = player.done_games
            existing.error = player.error
            existing.last_task_id = player.last_task_id
            sql_player = existing
        else:
            # Crear nuevo
            sql_player = domain_to_player(player)
            self.session.add(sql_player)

        self.session.commit()
        self.session.refresh(sql_player)

        return player_to_domain(sql_player)

    async def delete(self, username: str) -> bool:
        """Elimina un jugador."""
        sql_player = self.session.get(SQLPlayer, username)
        if sql_player:
            self.session.delete(sql_player)
            self.session.commit()
            return True
        return False

    async def list_pending(self) -> List[DomainPlayer]:
        """Lista jugadores con análisis pendiente."""
        statement = select(SQLPlayer).where(
            SQLPlayer.status == SQLPlayerStatus.pending
        )
        sql_players = self.session.exec(statement).all()

        return [player_to_domain(player) for player in sql_players]

    async def update_progress(self, username: str, progress: int,
                            done_games: int, total_games: int) -> None:
        """Actualiza el progreso de análisis."""
        sql_player = self.session.get(SQLPlayer, username)
        if sql_player:
            sql_player.progress = progress
            sql_player.done_games = done_games
            sql_player.total_games = total_games
            self.session.commit()

    async def mark_as_ready(self, username: str) -> None:
        """Marca jugador como listo."""
        sql_player = self.session.get(SQLPlayer, username)
        if sql_player:
            sql_player.status = SQLPlayerStatus.ready
            sql_player.progress = 100
            sql_player.error = None
            self.session.commit()

    async def mark_as_error(self, username: str, error: str) -> None:
        """Marca jugador con error."""
        sql_player = self.session.get(SQLPlayer, username)
        if sql_player:
            sql_player.status = SQLPlayerStatus.error
            sql_player.error = error
            self.session.commit()

    # Métodos adicionales específicos de SQL

    def get_by_username_sync(self, username: str) -> Optional[DomainPlayer]:
        """Versión síncrona para compatibilidad."""
        sql_player = self.session.get(SQLPlayer, username)
        if sql_player:
            return player_to_domain(sql_player)
        return None

    def save_sync(self, player: DomainPlayer) -> DomainPlayer:
        """Versión síncrona para compatibilidad."""
        # Reutilizar lógica async (sin await)
        existing = self.session.get(SQLPlayer, player.username)

        if existing:
            existing.status = SQLPlayerStatus(player.status.value)
            existing.requested_at = player.requested_at
            existing.finished_at = player.finished_at
            existing.progress = player.progress
            existing.total_games = player.total_games
            existing.done_games = player.done_games
            existing.error = player.error
            existing.last_task_id = player.last_task_id
            sql_player = existing
        else:
            sql_player = domain_to_player(player)
            self.session.add(sql_player)

        self.session.commit()
        self.session.refresh(sql_player)

        return player_to_domain(sql_player)

    def count_by_status(self, status: str) -> int:
        """Cuenta jugadores por estado."""
        statement = select(SQLPlayer).where(
            SQLPlayer.status == SQLPlayerStatus(status)
        )
        return len(self.session.exec(statement).all())

    def get_recent_players(self, limit: int = 10) -> List[DomainPlayer]:
        """Obtiene jugadores recientes."""
        statement = select(SQLPlayer).order_by(
            SQLPlayer.requested_at.desc()
        ).limit(limit)
        sql_players = self.session.exec(statement).all()

        return [player_to_domain(player) for player in sql_players]