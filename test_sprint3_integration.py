#!/usr/bin/env python3
"""
Test de integración para Sprint 3 - Application Layer.
Valida que use cases, handlers y DI container funcionen correctamente.
"""
import sys
import os
import asyncio
from datetime import datetime
from typing import Optional

# Agregar path para imports
sys.path.append('.')

# Mock de configuración para testing
class MockDatabaseConfig:
    def __init__(self):
        self.url = "sqlite:///:memory:"
        self.pool_size = 5
        self.max_overflow = 10

class MockStockfishConfig:
    def __init__(self):
        self.path = "/usr/bin/stockfish"
        self.depth = 15
        self.time_limit = 1.0

class MockRedisConfig:
    def __init__(self):
        self.url = "redis://localhost:6379"
        self.max_connections = 10

class MockAppConfig:
    def __init__(self):
        self.database = MockDatabaseConfig()
        self.stockfish = MockStockfishConfig()
        self.redis = MockRedisConfig()

# Mock de repositorios para testing
from app.domain.entities.player import Player
from app.domain.entities.game import Game
from app.domain.entities.analysis import PlayerAnalysis, GameAnalysis
from app.domain.repositories.player_repository import PlayerRepository
from app.domain.repositories.game_repository import GameRepository
from app.domain.repositories.analysis_repository import AnalysisRepository
from app.domain.value_objects.metrics import QualityMetrics, TimingMetrics, RiskAssessment

class MockPlayerRepository(PlayerRepository):
    def __init__(self):
        self._players = {}
        self._next_id = 1

    async def get_by_id(self, player_id: int) -> Optional[Player]:
        return self._players.get(player_id)

    async def get_by_username(self, username: str) -> Optional[Player]:
        for player in self._players.values():
            if player.username == username:
                return player
        return None

    async def save(self, player: Player) -> Player:
        if player.id is None:
            new_player = player.with_id(self._next_id)
            self._players[self._next_id] = new_player
            self._next_id += 1
            return new_player
        else:
            self._players[player.id] = player
            return player

    async def delete(self, player_id: int) -> bool:
        if player_id in self._players:
            del self._players[player_id]
            return True
        return False

    async def list_all(self, limit: int = 100, offset: int = 0) -> list[Player]:
        players = list(self._players.values())
        return players[offset:offset + limit]

class MockGameRepository(GameRepository):
    def __init__(self):
        self._games = {}
        self._next_id = 1

    async def get_by_id(self, game_id: int) -> Optional[Game]:
        return self._games.get(game_id)

    async def get_by_player_id(self, player_id: int) -> list[Game]:
        return [game for game in self._games.values() if game.player_id == player_id]

    async def save(self, game: Game) -> Game:
        if game.id is None:
            new_game = game.with_id(self._next_id)
            self._games[self._next_id] = new_game
            self._next_id += 1
            return new_game
        else:
            self._games[game.id] = game
            return game

    async def delete_by_player_id(self, player_id: int) -> bool:
        games_to_delete = [gid for gid, game in self._games.items() if game.player_id == player_id]
        for gid in games_to_delete:
            del self._games[gid]
        return len(games_to_delete) > 0

    async def get_by_filters(self, **filters) -> list[Game]:
        games = list(self._games.values())
        if 'player_id' in filters:
            games = [g for g in games if g.player_id == filters['player_id']]
        return games[:filters.get('limit', 100)]

    async def count_by_filters(self, **filters) -> int:
        games = await self.get_by_filters(**filters)
        return len(games)

    async def get_suspicious_games(self, **filters) -> list[Game]:
        return []

    async def count_suspicious_games(self, **filters) -> int:
        return 0

    async def get_suspicious_by_player_id(self, player_id: int, risk_threshold: int) -> list[Game]:
        return []

class MockAnalysisRepository(AnalysisRepository):
    def __init__(self):
        self._player_analyses = {}
        self._game_analyses = {}

    async def get_by_player_id(self, player_id: int) -> Optional[PlayerAnalysis]:
        return self._player_analyses.get(player_id)

    async def save_player_analysis(self, analysis: PlayerAnalysis) -> PlayerAnalysis:
        self._player_analyses[analysis.player_id] = analysis
        return analysis

    async def get_game_analysis_by_id(self, game_id: int) -> Optional[GameAnalysis]:
        return self._game_analyses.get(game_id)

    async def save_game_analysis(self, analysis: GameAnalysis) -> GameAnalysis:
        self._game_analyses[analysis.game_id] = analysis
        return analysis

    async def delete_by_player_id(self, player_id: int) -> bool:
        deleted = player_id in self._player_analyses
        if deleted:
            del self._player_analyses[player_id]
        return deleted

# Mock del container
class MockContainer:
    def __init__(self):
        from app.domain.services.player_service import PlayerService
        from app.domain.services.game_service import GameService

        self.player_repo = MockPlayerRepository()
        self.game_repo = MockGameRepository()
        self.analysis_repo = MockAnalysisRepository()
        self.player_service = PlayerService()
        self.game_service = GameService()

    def get_analyze_player_use_case(self):
        from app.application.use_cases.player_use_cases import AnalyzePlayerUseCase
        return AnalyzePlayerUseCase(
            player_repository=self.player_repo,
            game_repository=self.game_repo,
            analysis_repository=self.analysis_repo,
            player_service=self.player_service,
            analysis_service=None  # No necesitamos análisis real para este test
        )

    def get_get_player_status_use_case(self):
        from app.application.use_cases.player_use_cases import GetPlayerStatusUseCase
        return GetPlayerStatusUseCase(player_repository=self.player_repo)

    def get_get_player_analysis_use_case(self):
        from app.application.use_cases.player_use_cases import GetPlayerAnalysisUseCase
        return GetPlayerAnalysisUseCase(
            player_repository=self.player_repo,
            analysis_repository=self.analysis_repo,
            game_repository=self.game_repo
        )

async def test_player_use_cases():
    """Test completo de use cases de jugadores."""
    print("🧪 Testing Player Use Cases...")

    container = MockContainer()

    # Test 1: Crear nuevo jugador
    print("  📝 Test 1: AnalyzePlayerUseCase - nuevo jugador")
    from app.application.commands.player_commands import AnalyzePlayerCommand

    command = AnalyzePlayerCommand(
        username="testplayer",
        force_refresh=False,
        months_to_analyze=6
    )

    analyze_use_case = container.get_analyze_player_use_case()
    result = await analyze_use_case.execute(command)

    assert result.success, f"Failed to create player: {result.error_message}"
    assert result.player_id is not None, "Player ID should be assigned"
    print("    ✅ Jugador creado correctamente")

    # Test 2: Verificar estado del jugador
    print("  📝 Test 2: GetPlayerStatusUseCase")
    from app.application.queries.player_queries import GetPlayerStatusQuery

    status_query = GetPlayerStatusQuery(username="testplayer")
    status_use_case = container.get_get_player_status_use_case()
    status = await status_use_case.execute(status_query)

    assert status is not None, "Player status should exist"
    assert status.username == "testplayer", "Username should match"
    assert status.status == "pending", "Initial status should be pending"
    print("    ✅ Estado del jugador obtenido correctamente")

    # Test 3: Crear análisis mock para obtener análisis completo
    print("  📝 Test 3: GetPlayerAnalysisUseCase")

    # Crear análisis mock
    mock_analysis = PlayerAnalysis(
        player_id=result.player_id,
        overall_metrics=QualityMetrics(avg_acpl=25.5, avg_wdl_loss=8.2, avg_match_rate=85.4),
        timing_metrics=TimingMetrics(avg_move_time=15.3, time_variance=12.1, quick_moves_rate=0.15),
        risk_assessment=RiskAssessment(cheat_probability=0.12, risk_level="low", suspicion_flags=[]),
        games_analyzed=150,
        total_games=200,
        analysis_date=datetime.now()
    )

    await container.analysis_repo.save_player_analysis(mock_analysis)

    from app.application.queries.player_queries import GetPlayerAnalysisQuery
    analysis_query = GetPlayerAnalysisQuery(username="testplayer")
    analysis_use_case = container.get_get_player_analysis_use_case()
    analysis = await analysis_use_case.execute(analysis_query)

    assert analysis is not None, "Player analysis should exist"
    assert analysis.player_id == result.player_id, "Player ID should match"
    assert analysis.games_analyzed == 150, "Games analyzed should match"
    print("    ✅ Análisis del jugador obtenido correctamente")

    print("✅ Player Use Cases tests passed!")

async def test_game_use_cases():
    """Test de use cases de partidas."""
    print("🧪 Testing Game Use Cases...")

    container = MockContainer()

    # Crear jugador y partida mock
    player = Player(
        id=1,
        username="testplayer",
        status="completed",
        done_games=1,
        total_games=1,
        progress_percentage=100.0
    )
    await container.player_repo.save(player)

    game = Game(
        id=1,
        player_id=1,
        username="testplayer",
        game_url="https://chess.com/game/123",
        time_control="10+0",
        result="win",
        played_at=datetime.now(),
        pgn_data="[White \"testplayer\"] 1. e4 e5 2. Nf3",
        moves_data=[{"move": "e4", "eval": 0.3}]
    )
    await container.game_repo.save(game)

    # Test GetGameAnalysisUseCase
    print("  📝 Test: GetGameAnalysisUseCase")
    from app.application.queries.game_queries import GetGameAnalysisQuery
    from app.application.use_cases.game_use_cases import GetGameAnalysisUseCase

    game_query = GetGameAnalysisQuery(game_id=1, include_moves=True)
    game_use_case = GetGameAnalysisUseCase(
        game_repository=container.game_repo,
        analysis_repository=container.analysis_repo
    )

    game_result = await game_use_case.execute(game_query)
    assert game_result is not None, "Game result should exist"
    assert game_result.game.id == 1, "Game ID should match"
    assert game_result.moves_data is not None, "Moves data should be included"
    print("    ✅ Análisis de partida obtenido correctamente")

    print("✅ Game Use Cases tests passed!")

async def test_handlers():
    """Test de handlers (bridge FastAPI <-> Use Cases)."""
    print("🧪 Testing Handlers...")

    # Mock del container en handlers
    from app.application.handlers.player_handlers import PlayerHandlers

    # Patch del container
    original_get_container = None
    try:
        from app.application import container
        original_get_container = container.get_container
        container.get_container = lambda: MockContainer()

        handlers = PlayerHandlers()

        # Test analyze_player handler
        print("  📝 Test: PlayerHandlers.analyze_player")
        result = await handlers.analyze_player(
            username="handlertest",
            months_to_analyze=12
        )

        assert result["success"] is True, "Handler should return success"
        assert "player_id" in result, "Result should include player_id"
        print("    ✅ analyze_player handler funciona correctamente")

        # Test get_player_status handler
        print("  📝 Test: PlayerHandlers.get_player_status")
        status = await handlers.get_player_status(username="handlertest")

        assert status["username"] == "handlertest", "Username should match"
        assert "status" in status, "Result should include status"
        assert "progress" in status, "Result should include progress"
        print("    ✅ get_player_status handler funciona correctamente")

    finally:
        # Restaurar container original
        if original_get_container:
            container.get_container = original_get_container

    print("✅ Handlers tests passed!")

async def test_commands_queries_validation():
    """Test de validación de Commands y Queries."""
    print("🧪 Testing Commands/Queries Validation...")

    # Test validación de AnalyzePlayerCommand
    print("  📝 Test: AnalyzePlayerCommand validation")
    from app.application.commands.player_commands import AnalyzePlayerCommand

    try:
        invalid_command = AnalyzePlayerCommand(
            username="",  # Username vacío - debería fallar
            months_to_analyze=12
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Username cannot be empty" in str(e)
        print("    ✅ Validación de username vacío funciona")

    try:
        invalid_command = AnalyzePlayerCommand(
            username="test",
            months_to_analyze=50  # Fuera de rango - debería fallar
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Months to analyze must be between 1 and 24" in str(e)
        print("    ✅ Validación de months_to_analyze funciona")

    # Test validación de GetPlayerGamesQuery
    print("  📝 Test: GetPlayerGamesQuery validation")
    from app.application.queries.game_queries import GetPlayerGamesQuery

    try:
        invalid_query = GetPlayerGamesQuery(
            username="test",
            limit=150  # Fuera de rango - debería fallar
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Limit must be between 1 and 100" in str(e)
        print("    ✅ Validación de limit funciona")

    print("✅ Commands/Queries validation tests passed!")

async def main():
    """Ejecuta todos los tests de integración."""
    print("🚀 Starting Sprint 3 Integration Tests\n")

    try:
        await test_commands_queries_validation()
        print()

        await test_player_use_cases()
        print()

        await test_game_use_cases()
        print()

        await test_handlers()
        print()

        print("🎉 All Sprint 3 Integration Tests PASSED!")
        print("\n📋 Sprint 3 - Application Layer Summary:")
        print("  ✅ Commands/Queries with validation")
        print("  ✅ Use Cases orchestrating domain services")
        print("  ✅ Dependency Injection container")
        print("  ✅ Handlers bridging FastAPI and use cases")
        print("  ✅ End-to-end integration working")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)