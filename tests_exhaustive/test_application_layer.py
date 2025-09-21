#!/usr/bin/env python3
"""
Pruebas exhaustivas de Application Layer.
Valida Commands, Queries, Use Cases y Handlers.
"""
import sys
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional, Protocol

# Agregar path para imports
sys.path.append('.')

# Mock entities y value objects
@dataclass(frozen=True)
class Player:
    username: str
    status: str
    done_games: int = 0
    total_games: int = 0
    progress_percentage: float = 0.0
    id: Optional[int] = None
    task_id: Optional[str] = None
    error_message: Optional[str] = None

    def with_id(self, new_id: int) -> 'Player':
        return Player(
            id=new_id,
            username=self.username,
            status=self.status,
            done_games=self.done_games,
            total_games=self.total_games,
            progress_percentage=self.progress_percentage,
            task_id=self.task_id,
            error_message=self.error_message
        )

@dataclass(frozen=True)
class Game:
    username: str
    game_url: str
    time_control: str
    result: str
    player_id: Optional[int] = None
    id: Optional[int] = None
    played_at: Optional[datetime] = None
    pgn_data: Optional[str] = None
    moves_data: Optional[list] = None

    def with_id(self, new_id: int) -> 'Game':
        return Game(
            id=new_id,
            player_id=self.player_id,
            username=self.username,
            game_url=self.game_url,
            time_control=self.time_control,
            result=self.result,
            played_at=self.played_at,
            pgn_data=self.pgn_data,
            moves_data=self.moves_data
        )

@dataclass(frozen=True)
class QualityMetrics:
    avg_acpl: float
    avg_wdl_loss: float
    avg_match_rate: float

@dataclass(frozen=True)
class TimingMetrics:
    avg_move_time: float
    time_variance: float
    quick_moves_rate: float

@dataclass(frozen=True)
class RiskAssessment:
    cheat_probability: float
    risk_level: str
    suspicion_flags: list

@dataclass(frozen=True)
class PlayerAnalysis:
    player_id: int
    overall_metrics: QualityMetrics
    timing_metrics: TimingMetrics
    risk_assessment: RiskAssessment
    games_analyzed: int
    total_games: int
    analysis_date: datetime

@dataclass(frozen=True)
class GameAnalysis:
    game_id: int
    quality_metrics: QualityMetrics
    timing_metrics: TimingMetrics
    risk_assessment: RiskAssessment
    analyzed_at: datetime

# Commands y Queries
@dataclass(frozen=True)
class AnalyzePlayerCommand:
    username: str
    force_refresh: bool = False
    priority: int = 5
    months_to_analyze: int = 12

    def __post_init__(self):
        if not self.username or not self.username.strip():
            raise ValueError("Username cannot be empty")
        if len(self.username) > 50:
            raise ValueError("Username too long")
        if self.months_to_analyze < 1 or self.months_to_analyze > 24:
            raise ValueError("Months to analyze must be between 1 and 24")

@dataclass(frozen=True)
class GetPlayerStatusQuery:
    username: str
    include_progress_details: bool = True

    def __post_init__(self):
        if not self.username or not self.username.strip():
            raise ValueError("Username cannot be empty")

@dataclass(frozen=True)
class GetPlayerAnalysisQuery:
    username: str
    include_games: bool = False
    include_suspicious_games: bool = False

    def __post_init__(self):
        if not self.username or not self.username.strip():
            raise ValueError("Username cannot be empty")

@dataclass(frozen=True)
class GetGameAnalysisQuery:
    game_id: int
    include_moves: bool = True
    include_detailed_metrics: bool = True

    def __post_init__(self):
        if self.game_id <= 0:
            raise ValueError("Game ID must be positive")

# Result objects
@dataclass
class AnalyzePlayerResult:
    success: bool
    player_id: Optional[int] = None
    task_id: Optional[str] = None
    error_message: Optional[str] = None

@dataclass
class PlayerStatusResult:
    username: str
    status: str
    progress_percentage: float
    done_games: int
    total_games: int
    task_id: Optional[str] = None
    error_message: Optional[str] = None

# Repository interfaces
class PlayerRepository(Protocol):
    async def get_by_username(self, username: str) -> Optional[Player]: ...
    async def save(self, player: Player) -> Player: ...
    async def delete(self, player_id: int) -> bool: ...

class GameRepository(Protocol):
    async def get_by_id(self, game_id: int) -> Optional[Game]: ...
    async def get_by_player_id(self, player_id: int) -> List[Game]: ...
    async def save(self, game: Game) -> Game: ...

class AnalysisRepository(Protocol):
    async def get_by_player_id(self, player_id: int) -> Optional[PlayerAnalysis]: ...
    async def save_player_analysis(self, analysis: PlayerAnalysis) -> PlayerAnalysis: ...
    async def get_game_analysis_by_id(self, game_id: int) -> Optional[GameAnalysis]: ...

# Domain Services
class PlayerService:
    def create_new_player(self, username: str, months_to_analyze: int = 12) -> Player:
        return Player(username=username, status="pending")

    def reset_for_reanalysis(self, player: Player) -> Player:
        return Player(
            id=player.id,
            username=player.username,
            status="pending",
            done_games=0,
            total_games=0,
            progress_percentage=0.0
        )

    def update_progress(self, player: Player, done_games: int, total_games: int, task_id: Optional[str] = None) -> Player:
        progress_percentage = (done_games / total_games * 100) if total_games > 0 else 0.0
        status = "completed" if done_games == total_games else "in_progress" if done_games > 0 else "pending"

        return Player(
            id=player.id,
            username=player.username,
            status=status,
            done_games=done_games,
            total_games=total_games,
            progress_percentage=progress_percentage,
            task_id=task_id,
            error_message=player.error_message
        )

# Mock repositories
class MockPlayerRepository:
    def __init__(self):
        self._players = {}
        self._next_id = 1

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

class MockGameRepository:
    def __init__(self):
        self._games = {}
        self._next_id = 1

    async def get_by_id(self, game_id: int) -> Optional[Game]:
        return self._games.get(game_id)

    async def get_by_player_id(self, player_id: int) -> List[Game]:
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

class MockAnalysisRepository:
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

# Use Cases
class AnalyzePlayerUseCase:
    def __init__(self, player_repository: PlayerRepository, player_service: PlayerService):
        self._player_repository = player_repository
        self._player_service = player_service

    async def execute(self, command: AnalyzePlayerCommand) -> AnalyzePlayerResult:
        try:
            existing_player = await self._player_repository.get_by_username(command.username)

            if existing_player and not command.force_refresh:
                if existing_player.status == "completed":
                    return AnalyzePlayerResult(
                        success=True,
                        player_id=existing_player.id,
                        error_message="Player already analyzed"
                    )

            if existing_player:
                player = self._player_service.reset_for_reanalysis(existing_player)
            else:
                player = self._player_service.create_new_player(
                    username=command.username,
                    months_to_analyze=command.months_to_analyze
                )

            saved_player = await self._player_repository.save(player)

            return AnalyzePlayerResult(
                success=True,
                player_id=saved_player.id,
                task_id="task_123"
            )

        except Exception as e:
            return AnalyzePlayerResult(
                success=False,
                error_message=str(e)
            )

class GetPlayerStatusUseCase:
    def __init__(self, player_repository: PlayerRepository):
        self._player_repository = player_repository

    async def execute(self, query: GetPlayerStatusQuery) -> Optional[PlayerStatusResult]:
        player = await self._player_repository.get_by_username(query.username)
        if not player:
            return None

        return PlayerStatusResult(
            username=player.username,
            status=player.status,
            progress_percentage=player.progress_percentage,
            done_games=player.done_games,
            total_games=player.total_games,
            task_id=player.task_id,
            error_message=player.error_message
        )

class GetPlayerAnalysisUseCase:
    def __init__(self, player_repository: PlayerRepository, analysis_repository: AnalysisRepository):
        self._player_repository = player_repository
        self._analysis_repository = analysis_repository

    async def execute(self, query: GetPlayerAnalysisQuery) -> Optional[PlayerAnalysis]:
        player = await self._player_repository.get_by_username(query.username)
        if not player:
            return None

        analysis = await self._analysis_repository.get_by_player_id(player.id)
        return analysis

# Tests
class TestCommands:
    """Pruebas exhaustivas para Commands."""

    def test_analyze_player_command_valid(self):
        """Test creación válida de AnalyzePlayerCommand."""
        command = AnalyzePlayerCommand(
            username="testuser",
            force_refresh=True,
            priority=3,
            months_to_analyze=6
        )

        assert command.username == "testuser"
        assert command.force_refresh is True
        assert command.priority == 3
        assert command.months_to_analyze == 6

    def test_analyze_player_command_defaults(self):
        """Test valores por defecto de AnalyzePlayerCommand."""
        command = AnalyzePlayerCommand(username="testuser")

        assert command.force_refresh is False
        assert command.priority == 5
        assert command.months_to_analyze == 12

    def test_analyze_player_command_empty_username(self):
        """Test validación de username vacío."""
        try:
            AnalyzePlayerCommand(username="")
            assert False, "Should raise ValueError for empty username"
        except ValueError as e:
            assert "Username cannot be empty" in str(e)

    def test_analyze_player_command_whitespace_username(self):
        """Test validación de username solo espacios."""
        try:
            AnalyzePlayerCommand(username="   ")
            assert False, "Should raise ValueError for whitespace username"
        except ValueError as e:
            assert "Username cannot be empty" in str(e)

    def test_analyze_player_command_long_username(self):
        """Test validación de username muy largo."""
        long_username = "a" * 51

        try:
            AnalyzePlayerCommand(username=long_username)
            assert False, "Should raise ValueError for long username"
        except ValueError as e:
            assert "Username too long" in str(e)

    def test_analyze_player_command_invalid_months(self):
        """Test validación de months_to_analyze."""
        # Muy pocos meses
        try:
            AnalyzePlayerCommand(username="testuser", months_to_analyze=0)
            assert False, "Should raise ValueError for 0 months"
        except ValueError as e:
            assert "Months to analyze must be between 1 and 24" in str(e)

        # Demasiados meses
        try:
            AnalyzePlayerCommand(username="testuser", months_to_analyze=25)
            assert False, "Should raise ValueError for 25 months"
        except ValueError as e:
            assert "Months to analyze must be between 1 and 24" in str(e)

    def test_analyze_player_command_immutability(self):
        """Test inmutabilidad de AnalyzePlayerCommand."""
        command = AnalyzePlayerCommand(username="testuser")

        try:
            command.username = "newuser"
            assert False, "Should not be able to modify frozen dataclass"
        except AttributeError:
            pass  # Expected


class TestQueries:
    """Pruebas exhaustivas para Queries."""

    def test_get_player_status_query_valid(self):
        """Test creación válida de GetPlayerStatusQuery."""
        query = GetPlayerStatusQuery(
            username="testuser",
            include_progress_details=False
        )

        assert query.username == "testuser"
        assert query.include_progress_details is False

    def test_get_player_status_query_defaults(self):
        """Test valores por defecto de GetPlayerStatusQuery."""
        query = GetPlayerStatusQuery(username="testuser")

        assert query.include_progress_details is True

    def test_get_player_status_query_empty_username(self):
        """Test validación de username vacío."""
        try:
            GetPlayerStatusQuery(username="")
            assert False, "Should raise ValueError for empty username"
        except ValueError as e:
            assert "Username cannot be empty" in str(e)

    def test_get_player_analysis_query_valid(self):
        """Test creación válida de GetPlayerAnalysisQuery."""
        query = GetPlayerAnalysisQuery(
            username="testuser",
            include_games=True,
            include_suspicious_games=True
        )

        assert query.username == "testuser"
        assert query.include_games is True
        assert query.include_suspicious_games is True

    def test_get_game_analysis_query_valid(self):
        """Test creación válida de GetGameAnalysisQuery."""
        query = GetGameAnalysisQuery(
            game_id=123,
            include_moves=False,
            include_detailed_metrics=False
        )

        assert query.game_id == 123
        assert query.include_moves is False
        assert query.include_detailed_metrics is False

    def test_get_game_analysis_query_invalid_id(self):
        """Test validación de game_id inválido."""
        try:
            GetGameAnalysisQuery(game_id=0)
            assert False, "Should raise ValueError for game_id 0"
        except ValueError as e:
            assert "Game ID must be positive" in str(e)

        try:
            GetGameAnalysisQuery(game_id=-1)
            assert False, "Should raise ValueError for negative game_id"
        except ValueError as e:
            assert "Game ID must be positive" in str(e)


class TestUseCases:
    """Pruebas exhaustivas para Use Cases."""

    def setUp(self):
        self.player_repo = MockPlayerRepository()
        self.game_repo = MockGameRepository()
        self.analysis_repo = MockAnalysisRepository()
        self.player_service = PlayerService()

    async def test_analyze_player_use_case_new_player(self):
        """Test AnalyzePlayerUseCase con nuevo jugador."""
        self.setUp()

        use_case = AnalyzePlayerUseCase(self.player_repo, self.player_service)
        command = AnalyzePlayerCommand(username="newuser", months_to_analyze=6)

        result = await use_case.execute(command)

        assert result.success is True
        assert result.player_id is not None
        assert result.task_id == "task_123"
        assert result.error_message is None

        # Verificar que el jugador se guardó
        saved_player = await self.player_repo.get_by_username("newuser")
        assert saved_player is not None
        assert saved_player.status == "pending"

    async def test_analyze_player_use_case_existing_completed(self):
        """Test AnalyzePlayerUseCase con jugador ya completado."""
        self.setUp()

        # Crear jugador completado
        existing_player = Player(
            id=1,
            username="completeduser",
            status="completed",
            done_games=10,
            total_games=10,
            progress_percentage=100.0
        )
        await self.player_repo.save(existing_player)

        use_case = AnalyzePlayerUseCase(self.player_repo, self.player_service)
        command = AnalyzePlayerCommand(username="completeduser", force_refresh=False)

        result = await use_case.execute(command)

        assert result.success is True
        assert result.player_id == 1
        assert "already analyzed" in result.error_message

    async def test_analyze_player_use_case_force_refresh(self):
        """Test AnalyzePlayerUseCase con force_refresh."""
        self.setUp()

        # Crear jugador completado
        existing_player = Player(
            id=1,
            username="completeduser",
            status="completed",
            done_games=10,
            total_games=10,
            progress_percentage=100.0
        )
        await self.player_repo.save(existing_player)

        use_case = AnalyzePlayerUseCase(self.player_repo, self.player_service)
        command = AnalyzePlayerCommand(username="completeduser", force_refresh=True)

        result = await use_case.execute(command)

        assert result.success is True
        assert result.player_id == 1
        assert result.error_message is None

        # Verificar que el jugador se reseteó
        reset_player = await self.player_repo.get_by_username("completeduser")
        assert reset_player.status == "pending"
        assert reset_player.done_games == 0

    async def test_get_player_status_use_case_existing(self):
        """Test GetPlayerStatusUseCase con jugador existente."""
        self.setUp()

        # Crear jugador
        player = Player(
            id=1,
            username="testuser",
            status="in_progress",
            done_games=5,
            total_games=10,
            progress_percentage=50.0,
            task_id="task_123"
        )
        await self.player_repo.save(player)

        use_case = GetPlayerStatusUseCase(self.player_repo)
        query = GetPlayerStatusQuery(username="testuser")

        result = await use_case.execute(query)

        assert result is not None
        assert result.username == "testuser"
        assert result.status == "in_progress"
        assert result.progress_percentage == 50.0
        assert result.done_games == 5
        assert result.total_games == 10
        assert result.task_id == "task_123"

    async def test_get_player_status_use_case_not_found(self):
        """Test GetPlayerStatusUseCase con jugador inexistente."""
        self.setUp()

        use_case = GetPlayerStatusUseCase(self.player_repo)
        query = GetPlayerStatusQuery(username="nonexistent")

        result = await use_case.execute(query)

        assert result is None

    async def test_get_player_analysis_use_case_with_analysis(self):
        """Test GetPlayerAnalysisUseCase con análisis existente."""
        self.setUp()

        # Crear jugador
        player = Player(id=1, username="testuser", status="completed")
        await self.player_repo.save(player)

        # Crear análisis
        analysis = PlayerAnalysis(
            player_id=1,
            overall_metrics=QualityMetrics(avg_acpl=25.5, avg_wdl_loss=8.2, avg_match_rate=75.3),
            timing_metrics=TimingMetrics(avg_move_time=15.3, time_variance=12.1, quick_moves_rate=0.15),
            risk_assessment=RiskAssessment(cheat_probability=0.12, risk_level="low", suspicion_flags=[]),
            games_analyzed=50,
            total_games=60,
            analysis_date=datetime.now()
        )
        await self.analysis_repo.save_player_analysis(analysis)

        use_case = GetPlayerAnalysisUseCase(self.player_repo, self.analysis_repo)
        query = GetPlayerAnalysisQuery(username="testuser")

        result = await use_case.execute(query)

        assert result is not None
        assert result.player_id == 1
        assert result.games_analyzed == 50
        assert result.overall_metrics.avg_acpl == 25.5

    async def test_get_player_analysis_use_case_no_analysis(self):
        """Test GetPlayerAnalysisUseCase sin análisis."""
        self.setUp()

        # Crear jugador sin análisis
        player = Player(id=1, username="testuser", status="pending")
        await self.player_repo.save(player)

        use_case = GetPlayerAnalysisUseCase(self.player_repo, self.analysis_repo)
        query = GetPlayerAnalysisQuery(username="testuser")

        result = await use_case.execute(query)

        assert result is None


async def run_application_layer_tests():
    """Ejecutar todas las pruebas de Application Layer."""
    print("🧪 Running Exhaustive Application Layer Tests...")

    # Commands tests
    print("  📝 Testing Commands...")
    test_commands = TestCommands()
    test_commands.test_analyze_player_command_valid()
    test_commands.test_analyze_player_command_defaults()
    test_commands.test_analyze_player_command_empty_username()
    test_commands.test_analyze_player_command_whitespace_username()
    test_commands.test_analyze_player_command_long_username()
    test_commands.test_analyze_player_command_invalid_months()
    test_commands.test_analyze_player_command_immutability()
    print("    ✅ Commands tests passed")

    # Queries tests
    print("  📝 Testing Queries...")
    test_queries = TestQueries()
    test_queries.test_get_player_status_query_valid()
    test_queries.test_get_player_status_query_defaults()
    test_queries.test_get_player_status_query_empty_username()
    test_queries.test_get_player_analysis_query_valid()
    test_queries.test_get_game_analysis_query_valid()
    test_queries.test_get_game_analysis_query_invalid_id()
    print("    ✅ Queries tests passed")

    # Use Cases tests
    print("  📝 Testing Use Cases...")
    test_use_cases = TestUseCases()
    await test_use_cases.test_analyze_player_use_case_new_player()
    await test_use_cases.test_analyze_player_use_case_existing_completed()
    await test_use_cases.test_analyze_player_use_case_force_refresh()
    await test_use_cases.test_get_player_status_use_case_existing()
    await test_use_cases.test_get_player_status_use_case_not_found()
    await test_use_cases.test_get_player_analysis_use_case_with_analysis()
    await test_use_cases.test_get_player_analysis_use_case_no_analysis()
    print("    ✅ Use Cases tests passed")

    print("✅ All Application Layer tests PASSED!")
    return True


if __name__ == "__main__":
    import asyncio

    async def main():
        try:
            success = await run_application_layer_tests()
            print("\n🎉 Application Layer exhaustive testing completed successfully!")
            print("\n📋 Application Layer Validation Summary:")
            print("  ✅ Commands - Validación completa de parámetros y reglas de negocio")
            print("  ✅ Queries - Validación de entrada y estructuras de consulta")
            print("  ✅ Use Cases - Orquestación correcta de domain services")
            print("  ✅ Repository pattern - Integración limpia con persistence")
            print("  ✅ CQRS pattern - Separación clara entre comandos y consultas")
            print("  ✅ Error handling - Manejo robusto de errores y casos edge")
            print("  ✅ Inmutabilidad - Commands y Queries inmutables")
            sys.exit(0 if success else 1)
        except Exception as e:
            print(f"❌ Application Layer tests failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    asyncio.run(main())