"""
Implementación SQL del repositorio de partidas.
"""
from typing import Optional, List, Union
from sqlmodel import Session, select, or_

from app.domain.entities.game import Game as DomainGame
from app.domain.repositories.game_repository import GameRepository
from app.models import Game as SQLGame
from .mappers import game_to_domain, domain_to_game


class SQLGameRepository(GameRepository):
    """
    Implementación SQL del repositorio de partidas.
    """

    def __init__(self, session: Session):
        self.session = session

    async def get_by_id(self, game_id: int) -> Optional[DomainGame]:
        """Obtiene una partida por ID."""
        sql_game = self.session.get(SQLGame, game_id)
        if sql_game:
            return game_to_domain(sql_game, include_moves=True)
        return None

    async def save(self, game: DomainGame) -> DomainGame:
        """Guarda una partida."""
        if game.id:
            # Actualizar existente
            sql_game = self.session.get(SQLGame, game.id)
            if sql_game:
                # Actualizar campos
                sql_game.pgn = game.pgn
                sql_game.white_username = game.white_username
                sql_game.black_username = game.black_username
                sql_game.white_elo = game.white_elo
                sql_game.black_elo = game.black_elo
                sql_game.time_control = game.time_control
                sql_game.termination = game.termination
                sql_game.eco_code = game.eco_code
                sql_game.opening_key = game.opening_key
                sql_game.move_times = game.move_times
            else:
                raise ValueError(f"Game with ID {game.id} not found")
        else:
            # Crear nueva
            sql_game = domain_to_game(game)
            self.session.add(sql_game)

        self.session.commit()
        self.session.refresh(sql_game)

        return game_to_domain(sql_game)

    async def get_by_player(self, username: str) -> List[DomainGame]:
        """Obtiene todas las partidas de un jugador."""
        statement = select(SQLGame).where(
            or_(
                SQLGame.white_username == username,
                SQLGame.black_username == username
            )
        ).order_by(SQLGame.created_at.desc())

        sql_games = self.session.exec(statement).all()
        return [game_to_domain(game) for game in sql_games]

    async def get_by_player_id(self, player_id: Union[str, int]) -> List[DomainGame]:
        """Compat wrapper para obtener partidas usando player_id lógico."""
        if player_id is None:
            return []

        username = str(player_id)
        return await self.get_by_player(username)

    async def get_analyzed_count(self, username: str) -> int:
        """Cuenta partidas analizadas de un jugador."""
        # Una partida está analizada si tiene análisis detallado
        from app.models import GameAnalysisDetailed

        statement = select(SQLGame).join(GameAnalysisDetailed).where(
            or_(
                SQLGame.white_username == username,
                SQLGame.black_username == username
            )
        )

        sql_games = self.session.exec(statement).all()
        return len(sql_games)

    async def delete_by_player(self, username: str) -> int:
        """Elimina todas las partidas de un jugador."""
        statement = select(SQLGame).where(
            or_(
                SQLGame.white_username == username,
                SQLGame.black_username == username
            )
        )

        sql_games = self.session.exec(statement).all()
        count = len(sql_games)

        for game in sql_games:
            self.session.delete(game)

        self.session.commit()
        return count

    async def bulk_save(self, games: List[DomainGame]) -> List[DomainGame]:
        """Guarda múltiples partidas de forma eficiente."""
        sql_games = []

        for game in games:
            if game.id is None:  # Solo nuevas partidas
                sql_game = domain_to_game(game)
                sql_games.append(sql_game)
                self.session.add(sql_game)

        if sql_games:
            self.session.commit()

            # Refresh para obtener IDs asignados
            for sql_game in sql_games:
                self.session.refresh(sql_game)

        # Convertir de vuelta a domain
        return [game_to_domain(sql_game) for sql_game in sql_games]

    # Métodos adicionales específicos de SQL

    def get_by_id_sync(self, game_id: int) -> Optional[DomainGame]:
        """Versión síncrona para compatibilidad."""
        sql_game = self.session.get(SQLGame, game_id)
        if sql_game:
            return game_to_domain(sql_game, include_moves=True)
        return None

    def save_sync(self, game: DomainGame) -> DomainGame:
        """Versión síncrona para compatibilidad."""
        if game.id:
            sql_game = self.session.get(SQLGame, game.id)
            if sql_game:
                sql_game.pgn = game.pgn
                sql_game.white_username = game.white_username
                sql_game.black_username = game.black_username
                sql_game.white_elo = game.white_elo
                sql_game.black_elo = game.black_elo
                sql_game.time_control = game.time_control
                sql_game.termination = game.termination
                sql_game.eco_code = game.eco_code
                sql_game.opening_key = game.opening_key
                sql_game.move_times = game.move_times
            else:
                raise ValueError(f"Game with ID {game.id} not found")
        else:
            sql_game = domain_to_game(game)
            self.session.add(sql_game)

        self.session.commit()
        self.session.refresh(sql_game)

        return game_to_domain(sql_game)

    def get_games_by_date_range(self, username: str, start_date, end_date) -> List[DomainGame]:
        """Obtiene partidas por rango de fechas."""
        statement = select(SQLGame).where(
            or_(
                SQLGame.white_username == username,
                SQLGame.black_username == username
            ),
            SQLGame.created_at >= start_date,
            SQLGame.created_at <= end_date
        ).order_by(SQLGame.created_at.desc())

        sql_games = self.session.exec(statement).all()
        return [game_to_domain(game) for game in sql_games]

    def get_games_by_time_control(self, username: str, time_control: str) -> List[DomainGame]:
        """Obtiene partidas por control de tiempo."""
        statement = select(SQLGame).where(
            or_(
                SQLGame.white_username == username,
                SQLGame.black_username == username
            ),
            SQLGame.time_control == time_control
        ).order_by(SQLGame.created_at.desc())

        sql_games = self.session.exec(statement).all()
        return [game_to_domain(game) for game in sql_games]

    def count_total_games(self, username: str) -> int:
        """Cuenta total de partidas de un jugador."""
        statement = select(SQLGame).where(
            or_(
                SQLGame.white_username == username,
                SQLGame.black_username == username
            )
        )

        sql_games = self.session.exec(statement).all()
        return len(sql_games)

    def get_recent_games(self, username: str, limit: int = 10) -> List[DomainGame]:
        """Obtiene partidas recientes de un jugador."""
        statement = select(SQLGame).where(
            or_(
                SQLGame.white_username == username,
                SQLGame.black_username == username
            )
        ).order_by(SQLGame.created_at.desc()).limit(limit)

        sql_games = self.session.exec(statement).all()
        return [game_to_domain(game) for game in sql_games]
