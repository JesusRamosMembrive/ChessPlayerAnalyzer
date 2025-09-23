"""
Dependency Injection Container.
Configuración central de dependencias para la aplicación.
"""
from typing import Protocol, runtime_checkable, Optional
import asyncio

from ..core.config import get_config
from ..domain.repositories.player_repository import PlayerRepository
from ..domain.repositories.game_repository import GameRepository
from ..domain.repositories.analysis_repository import AnalysisRepository
from ..domain.services.player_service import PlayerService
from ..domain.services.game_service import GameService
from ..domain.services.analysis_service import AnalysisService
from ..infrastructure.database.sql_player_repository import SQLPlayerRepository
from ..infrastructure.database.sql_game_repository import SQLGameRepository
from ..infrastructure.database.sql_analysis_repository import SQLAnalysisRepository
from ..infrastructure.external.stockfish_engine import StockfishEngine

from .use_cases.player_use_cases import (
    AnalyzePlayerUseCase,
    GetPlayerAnalysisUseCase,
    GetPlayerStatusUseCase,
    RefreshPlayerAnalysisUseCase,
    DeletePlayerUseCase,
    UpdatePlayerProgressUseCase
)
from .use_cases.game_use_cases import (
    GetGameAnalysisUseCase,
    GetPlayerGamesUseCase,
    GetSuspiciousGamesUseCase,
    AnalyzeGameUseCase
)


@runtime_checkable
class Container(Protocol):
    """Protocol que define el contrato del container DI."""

    def get_analyze_player_use_case(self) -> AnalyzePlayerUseCase: ...
    def get_get_player_analysis_use_case(self) -> GetPlayerAnalysisUseCase: ...
    def get_get_player_status_use_case(self) -> GetPlayerStatusUseCase: ...
    def get_refresh_player_analysis_use_case(self) -> RefreshPlayerAnalysisUseCase: ...
    def get_delete_player_use_case(self) -> DeletePlayerUseCase: ...
    def get_update_player_progress_use_case(self) -> UpdatePlayerProgressUseCase: ...

    def get_get_game_analysis_use_case(self) -> GetGameAnalysisUseCase: ...
    def get_get_player_games_use_case(self) -> GetPlayerGamesUseCase: ...
    def get_get_suspicious_games_use_case(self) -> GetSuspiciousGamesUseCase: ...
    def get_analyze_game_use_case(self) -> AnalyzeGameUseCase: ...


class DIContainer:
    """Container de dependencias simple y explícito."""

    def __init__(self):
        self._config = get_config()
        self._instances = {}

    # === Repositories ===

    def _get_player_repository(self) -> PlayerRepository:
        """Obtiene repository de jugadores."""
        if 'player_repo' not in self._instances:
            from ..database import SessionLocal
            session = SessionLocal()
            self._instances['player_repo'] = SQLPlayerRepository(session)
        return self._instances['player_repo']

    def _get_game_repository(self) -> GameRepository:
        """Obtiene repository de partidas."""
        if 'game_repo' not in self._instances:
            from ..database import SessionLocal
            session = SessionLocal()
            self._instances['game_repo'] = SQLGameRepository(session)
        return self._instances['game_repo']

    def _get_analysis_repository(self) -> AnalysisRepository:
        """Obtiene repository de análisis."""
        if 'analysis_repo' not in self._instances:
            from ..database import SessionLocal
            session = SessionLocal()
            self._instances['analysis_repo'] = SQLAnalysisRepository(session)
        return self._instances['analysis_repo']

    # === Domain Services ===

    def _get_player_service(self) -> PlayerService:
        """Obtiene servicio de jugadores."""
        if 'player_service' not in self._instances:
            self._instances['player_service'] = PlayerService(
                player_repo=self._get_player_repository(),
                analysis_repo=self._get_analysis_repository()
            )
        return self._instances['player_service']

    def _get_game_service(self) -> GameService:
        """Obtiene servicio de partidas."""
        if 'game_service' not in self._instances:
            self._instances['game_service'] = GameService(
                game_repo=self._get_game_repository()
            )
        return self._instances['game_service']

    def _get_analysis_service(self) -> AnalysisService:
        """Obtiene servicio de análisis."""
        if 'analysis_service' not in self._instances:
            from ..infrastructure.external.stockfish_adapter import StockfishEngineAdapter
            stockfish_adapter = StockfishEngineAdapter(self._config.stockfish)
            self._instances['analysis_service'] = AnalysisService(stockfish_adapter)
        return self._instances['analysis_service']

    # === Player Use Cases ===

    def get_analyze_player_use_case(self) -> AnalyzePlayerUseCase:
        """Obtiene use case para analizar jugador."""
        return AnalyzePlayerUseCase(
            player_repository=self._get_player_repository(),
            game_repository=self._get_game_repository(),
            analysis_repository=self._get_analysis_repository(),
            player_service=self._get_player_service(),
            analysis_service=self._get_analysis_service()
        )

    def get_get_player_analysis_use_case(self) -> GetPlayerAnalysisUseCase:
        """Obtiene use case para obtener análisis de jugador."""
        return GetPlayerAnalysisUseCase(
            player_repository=self._get_player_repository(),
            analysis_repository=self._get_analysis_repository(),
            game_repository=self._get_game_repository()
        )

    def get_get_player_status_use_case(self) -> GetPlayerStatusUseCase:
        """Obtiene use case para obtener estado de jugador."""
        return GetPlayerStatusUseCase(
            player_repository=self._get_player_repository()
        )

    def get_refresh_player_analysis_use_case(self) -> RefreshPlayerAnalysisUseCase:
        """Obtiene use case para refrescar análisis de jugador."""
        return RefreshPlayerAnalysisUseCase(
            player_repository=self._get_player_repository(),
            game_repository=self._get_game_repository(),
            analysis_repository=self._get_analysis_repository(),
            player_service=self._get_player_service()
        )

    def get_delete_player_use_case(self) -> DeletePlayerUseCase:
        """Obtiene use case para eliminar jugador."""
        return DeletePlayerUseCase(
            player_repository=self._get_player_repository(),
            game_repository=self._get_game_repository(),
            analysis_repository=self._get_analysis_repository()
        )

    def get_update_player_progress_use_case(self) -> UpdatePlayerProgressUseCase:
        """Obtiene use case para actualizar progreso de jugador."""
        return UpdatePlayerProgressUseCase(
            player_repository=self._get_player_repository(),
            player_service=self._get_player_service()
        )

    # === Game Use Cases ===

    def get_get_game_analysis_use_case(self) -> GetGameAnalysisUseCase:
        """Obtiene use case para obtener análisis de partida."""
        return GetGameAnalysisUseCase(
            game_repository=self._get_game_repository(),
            analysis_repository=self._get_analysis_repository()
        )

    def get_get_player_games_use_case(self) -> GetPlayerGamesUseCase:
        """Obtiene use case para obtener partidas de jugador."""
        return GetPlayerGamesUseCase(
            game_repository=self._get_game_repository(),
            player_repository=self._get_player_repository(),
            analysis_repository=self._get_analysis_repository()
        )

    def get_get_suspicious_games_use_case(self) -> GetSuspiciousGamesUseCase:
        """Obtiene use case para obtener partidas sospechosas."""
        return GetSuspiciousGamesUseCase(
            game_repository=self._get_game_repository(),
            analysis_repository=self._get_analysis_repository(),
            player_repository=self._get_player_repository()
        )

    def get_analyze_game_use_case(self) -> AnalyzeGameUseCase:
        """Obtiene use case para analizar partida."""
        return AnalyzeGameUseCase(
            game_repository=self._get_game_repository(),
            analysis_repository=self._get_analysis_repository(),
            analysis_service=self._get_analysis_service(),
            game_service=self._get_game_service()
        )

    async def cleanup(self):
        """Limpia recursos del container."""
        # Cerrar conexiones de base de datos si es necesario
        for instance in self._instances.values():
            if hasattr(instance, 'close'):
                if asyncio.iscoroutinefunction(instance.close):
                    await instance.close()
                else:
                    instance.close()


# Instancia global del container
_container: Optional[DIContainer] = None


def get_container() -> DIContainer:
    """Obtiene la instancia global del container."""
    global _container
    if _container is None:
        _container = DIContainer()
    return _container


async def cleanup_container():
    """Limpia el container global."""
    global _container
    if _container is not None:
        await _container.cleanup()
        _container = None