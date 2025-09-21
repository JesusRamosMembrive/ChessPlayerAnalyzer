"""
Use cases relacionados con partidas.
Orquestan domain services y repositories.
"""
from typing import List, Optional
from dataclasses import dataclass

from ...domain.entities.game import Game
from ...domain.entities.analysis import GameAnalysis
from ...domain.repositories.game_repository import GameRepository
from ...domain.repositories.analysis_repository import AnalysisRepository
from ...domain.repositories.player_repository import PlayerRepository
from ...domain.services.analysis_service import AnalysisService
from ...domain.services.game_service import GameService
from ..queries.game_queries import (
    GetGameAnalysisQuery,
    GetPlayerGamesQuery,
    GetSuspiciousGamesQuery
)


@dataclass
class GameAnalysisResult:
    """Resultado del análisis de partida."""
    game: Game
    analysis: Optional[GameAnalysis]
    moves_data: Optional[List[dict]] = None


@dataclass
class PlayerGamesResult:
    """Resultado de búsqueda de partidas de jugador."""
    games: List[Game]
    total_count: int
    has_more: bool


@dataclass
class SuspiciousGamesResult:
    """Resultado de búsqueda de partidas sospechosas."""
    games: List[GameAnalysisResult]
    total_count: int


class GetGameAnalysisUseCase:
    """Use case para obtener análisis de una partida específica."""

    def __init__(
        self,
        game_repository: GameRepository,
        analysis_repository: AnalysisRepository
    ):
        self._game_repository = game_repository
        self._analysis_repository = analysis_repository

    async def execute(self, query: GetGameAnalysisQuery) -> Optional[GameAnalysisResult]:
        """Obtiene el análisis completo de una partida."""
        # Obtener la partida
        game = await self._game_repository.get_by_id(query.game_id)
        if not game:
            return None

        # Obtener análisis si existe
        analysis = None
        if query.include_detailed_metrics:
            analysis = await self._analysis_repository.get_game_analysis_by_id(query.game_id)

        # Obtener datos de movimientos si se requieren
        moves_data = None
        if query.include_moves and game.moves_data:
            moves_data = game.moves_data

        return GameAnalysisResult(
            game=game,
            analysis=analysis,
            moves_data=moves_data
        )


class GetPlayerGamesUseCase:
    """Use case para obtener partidas de un jugador."""

    def __init__(
        self,
        game_repository: GameRepository,
        player_repository: PlayerRepository,
        analysis_repository: AnalysisRepository
    ):
        self._game_repository = game_repository
        self._player_repository = player_repository
        self._analysis_repository = analysis_repository

    async def execute(self, query: GetPlayerGamesQuery) -> Optional[PlayerGamesResult]:
        """Obtiene las partidas de un jugador con filtros."""
        # Verificar que el jugador existe
        player = await self._player_repository.get_by_username(query.username)
        if not player:
            return None

        # Construir filtros
        filters = {
            'player_id': player.id,
            'limit': query.limit,
            'offset': query.offset,
            'time_control_filter': query.time_control_filter,
            'date_from': query.date_from,
            'date_to': query.date_to,
            'only_analyzed': query.only_analyzed
        }

        # Obtener partidas
        games = await self._game_repository.get_by_filters(**filters)

        # Obtener total para paginación
        total_count = await self._game_repository.count_by_filters(**filters)

        # Si se incluye análisis, cargar análisis de cada partida
        if query.include_analysis:
            for game in games:
                game_analysis = await self._analysis_repository.get_game_analysis_by_id(game.id)
                # TODO: Asociar análisis con partida si es necesario

        return PlayerGamesResult(
            games=games,
            total_count=total_count,
            has_more=(query.offset + len(games)) < total_count
        )


class GetSuspiciousGamesUseCase:
    """Use case para obtener partidas sospechosas."""

    def __init__(
        self,
        game_repository: GameRepository,
        analysis_repository: AnalysisRepository,
        player_repository: PlayerRepository
    ):
        self._game_repository = game_repository
        self._analysis_repository = analysis_repository
        self._player_repository = player_repository

    async def execute(self, query: GetSuspiciousGamesQuery) -> SuspiciousGamesResult:
        """Obtiene partidas con alta probabilidad de trampa."""
        filters = {
            'risk_threshold': query.risk_threshold,
            'limit': query.limit
        }

        # Si se especifica usuario, filtrar por él
        if query.username:
            player = await self._player_repository.get_by_username(query.username)
            if player:
                filters['player_id'] = player.id
            else:
                return SuspiciousGamesResult(games=[], total_count=0)

        # Obtener partidas sospechosas
        suspicious_games = await self._game_repository.get_suspicious_games(**filters)

        # Construir resultados con análisis
        results = []
        for game in suspicious_games:
            analysis = None
            if query.include_analysis:
                analysis = await self._analysis_repository.get_game_analysis_by_id(game.id)

            results.append(GameAnalysisResult(
                game=game,
                analysis=analysis
            ))

        # Total count para estadísticas
        total_count = await self._game_repository.count_suspicious_games(**filters)

        return SuspiciousGamesResult(
            games=results,
            total_count=total_count
        )


class AnalyzeGameUseCase:
    """Use case para analizar una partida específica."""

    def __init__(
        self,
        game_repository: GameRepository,
        analysis_repository: AnalysisRepository,
        analysis_service: AnalysisService,
        game_service: GameService
    ):
        self._game_repository = game_repository
        self._analysis_repository = analysis_repository
        self._analysis_service = analysis_service
        self._game_service = game_service

    async def execute(self, game_id: int, force_reanalysis: bool = False) -> Optional[GameAnalysisResult]:
        """Analiza una partida y guarda el resultado."""
        try:
            # Obtener la partida
            game = await self._game_repository.get_by_id(game_id)
            if not game:
                return None

            # Verificar si ya tiene análisis
            existing_analysis = await self._analysis_repository.get_game_analysis_by_id(game_id)
            if existing_analysis and not force_reanalysis:
                return GameAnalysisResult(
                    game=game,
                    analysis=existing_analysis,
                    moves_data=game.moves_data
                )

            # Validar que la partida tenga datos para analizar
            if not game.pgn_data:
                return GameAnalysisResult(game=game, analysis=None)

            # Extraer movimientos del PGN
            moves_data = self._game_service.extract_moves_from_pgn(game.pgn_data)
            if not moves_data:
                return GameAnalysisResult(game=game, analysis=None)

            # Realizar análisis con Stockfish
            analysis = self._analysis_service.analyze_game(game, moves_data)

            # Guardar análisis
            saved_analysis = await self._analysis_repository.save_game_analysis(analysis)

            # Actualizar partida con datos de movimientos si no los tenía
            if not game.moves_data:
                updated_game = game.with_moves_data(moves_data)
                await self._game_repository.save(updated_game)
                game = updated_game

            return GameAnalysisResult(
                game=game,
                analysis=saved_analysis,
                moves_data=moves_data
            )

        except Exception as e:
            # Log error pero no fallar completamente
            print(f"Error analyzing game {game_id}: {e}")
            return GameAnalysisResult(game=game, analysis=None) if game else None