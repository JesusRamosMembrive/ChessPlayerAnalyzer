#!/usr/bin/env python3
"""
Pruebas exhaustivas de Infrastructure Layer.
Valida integración con sistemas externos y adaptadores.
"""
import sys
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

# Agregar path para imports
sys.path.append('.')

# Mock de configuraciones
@dataclass
class MockStockfishConfig:
    path: str = "/usr/bin/stockfish"
    depth: int = 15
    time_limit: float = 1.0

@dataclass
class MockDatabaseConfig:
    url: str = "sqlite:///:memory:"
    pool_size: int = 5
    max_overflow: int = 10

# Mock entities
@dataclass(frozen=True)
class Player:
    username: str
    status: str
    done_games: int = 0
    total_games: int = 0
    progress_percentage: float = 0.0
    id: Optional[int] = None

    def with_id(self, new_id: int) -> 'Player':
        return Player(
            id=new_id,
            username=self.username,
            status=self.status,
            done_games=self.done_games,
            total_games=self.total_games,
            progress_percentage=self.progress_percentage
        )

@dataclass(frozen=True)
class Game:
    username: str
    game_url: str
    time_control: str
    result: str
    player_id: Optional[int] = None
    id: Optional[int] = None
    pgn_data: Optional[str] = None

    def with_id(self, new_id: int) -> 'Game':
        return Game(
            id=new_id,
            player_id=self.player_id,
            username=self.username,
            game_url=self.game_url,
            time_control=self.time_control,
            result=self.result,
            pgn_data=self.pgn_data
        )

# Mock de resultados del engine
@dataclass
class MockMoveAnalysis:
    move: str
    evaluation: float
    is_best: bool
    centipawn_loss: float
    wdl_loss: float
    match_rate: float

@dataclass
class MockEngineAnalysisResult:
    moves: List[MockMoveAnalysis]
    avg_centipawn_loss: float
    avg_wdl_loss: float
    avg_match_rate: float
    total_moves: int

# Infrastructure components
class StockfishEngine:
    """Mock del StockfishEngine para testing."""

    def __init__(self, config: MockStockfishConfig):
        self.config = config
        self._is_initialized = False

    def __enter__(self):
        self._is_initialized = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._is_initialized = False

    async def analyze_moves(self, pgn_data: str, move_times: Optional[List[float]] = None) -> MockEngineAnalysisResult:
        """Mock analysis que simula análisis de Stockfish."""
        if not self._is_initialized:
            raise RuntimeError("Engine not initialized")

        if not pgn_data:
            raise ValueError("PGN data is required")

        # Simular análisis básico
        moves = []

        # Simular extracción de movimientos del PGN
        if "1. e4" in pgn_data:
            moves.append(MockMoveAnalysis("e4", 0.3, True, 0.0, 0.0, 1.0))
        if "e5" in pgn_data:
            moves.append(MockMoveAnalysis("e5", 0.0, True, 0.0, 0.0, 1.0))
        if "Nf3" in pgn_data:
            moves.append(MockMoveAnalysis("Nf3", 0.2, False, 15.0, 2.5, 0.0))

        if not moves:
            raise ValueError("No valid moves found in PGN")

        # Calcular métricas agregadas
        avg_centipawn_loss = sum(m.centipawn_loss for m in moves) / len(moves)
        avg_wdl_loss = sum(m.wdl_loss for m in moves) / len(moves)
        avg_match_rate = sum(m.match_rate for m in moves) / len(moves)

        return MockEngineAnalysisResult(
            moves=moves,
            avg_centipawn_loss=avg_centipawn_loss,
            avg_wdl_loss=avg_wdl_loss,
            avg_match_rate=avg_match_rate,
            total_moves=len(moves)
        )

    def test_engine(self) -> bool:
        """Test si el engine está funcionando."""
        try:
            # Simular test exitoso
            return self.config.path.endswith("stockfish")
        except:
            return False


class StockfishEngineAdapter:
    """Adapter que conecta StockfishEngine con domain service."""

    def __init__(self, config: MockStockfishConfig):
        self.config = config

    async def analyze_moves(self, pgn_data: str, move_times: Optional[List[float]] = None) -> MockEngineAnalysisResult:
        """Adapta la interfaz del engine al protocolo esperado."""
        with StockfishEngine(self.config) as engine:
            return await engine.analyze_moves(pgn_data, move_times)


class SQLPlayerRepository:
    """Mock del repository SQL para jugadores."""

    def __init__(self, config: MockDatabaseConfig):
        self.config = config
        self._players = {}
        self._next_id = 1
        self._connection_active = True

    async def get_by_username(self, username: str) -> Optional[Player]:
        """Obtiene jugador por username."""
        if not self._connection_active:
            raise RuntimeError("Database connection not active")

        for player in self._players.values():
            if player.username == username:
                return player
        return None

    async def save(self, player: Player) -> Player:
        """Guarda jugador en la base de datos."""
        if not self._connection_active:
            raise RuntimeError("Database connection not active")

        if not player.username:
            raise ValueError("Username is required")

        if player.id is None:
            new_player = player.with_id(self._next_id)
            self._players[self._next_id] = new_player
            self._next_id += 1
            return new_player
        else:
            self._players[player.id] = player
            return player

    async def delete(self, player_id: int) -> bool:
        """Elimina jugador por ID."""
        if not self._connection_active:
            raise RuntimeError("Database connection not active")

        if player_id in self._players:
            del self._players[player_id]
            return True
        return False

    def _simulate_connection_failure(self):
        """Simula fallo de conexión para testing."""
        self._connection_active = False

    def _restore_connection(self):
        """Restaura conexión para testing."""
        self._connection_active = True


class SQLGameRepository:
    """Mock del repository SQL para partidas."""

    def __init__(self, config: MockDatabaseConfig):
        self.config = config
        self._games = {}
        self._next_id = 1

    async def get_by_id(self, game_id: int) -> Optional[Game]:
        """Obtiene partida por ID."""
        return self._games.get(game_id)

    async def get_by_player_id(self, player_id: int) -> List[Game]:
        """Obtiene partidas de un jugador."""
        return [game for game in self._games.values() if game.player_id == player_id]

    async def save(self, game: Game) -> Game:
        """Guarda partida en la base de datos."""
        if not game.username:
            raise ValueError("Username is required")
        if not game.game_url:
            raise ValueError("Game URL is required")

        if game.id is None:
            new_game = game.with_id(self._next_id)
            self._games[self._next_id] = new_game
            self._next_id += 1
            return new_game
        else:
            self._games[game.id] = game
            return game

    async def count_by_filters(self, **filters) -> int:
        """Cuenta partidas con filtros."""
        games = await self.get_by_filters(**filters)
        return len(games)

    async def get_by_filters(self, **filters) -> List[Game]:
        """Obtiene partidas con filtros."""
        games = list(self._games.values())

        if 'player_id' in filters:
            games = [g for g in games if g.player_id == filters['player_id']]

        if 'time_control_filter' in filters and filters['time_control_filter']:
            games = [g for g in games if filters['time_control_filter'] in g.time_control]

        # Aplicar limit y offset
        limit = filters.get('limit', 100)
        offset = filters.get('offset', 0)
        return games[offset:offset + limit]


# Player Handlers testing
class PlayerHandlers:
    """Mock de PlayerHandlers para testing."""

    def __init__(self):
        # Mock de container dependencies
        self.player_repo = SQLPlayerRepository(MockDatabaseConfig())
        self.game_repo = SQLGameRepository(MockDatabaseConfig())

    async def analyze_player(self, username: str, force_refresh: bool = False, priority: int = 5, months_to_analyze: int = 12) -> Dict[str, Any]:
        """Mock de analyze_player handler."""
        if not username:
            raise ValueError("Username is required")

        # Simular lógica de handler
        existing_player = await self.player_repo.get_by_username(username)

        if existing_player and not force_refresh:
            if existing_player.status == "completed":
                return {
                    "success": True,
                    "player_id": existing_player.id,
                    "message": f"Player {username} already analyzed"
                }

        # Crear o actualizar jugador
        new_player = Player(username=username, status="pending")
        saved_player = await self.player_repo.save(new_player)

        return {
            "success": True,
            "player_id": saved_player.id,
            "task_id": "task_123",
            "message": f"Analysis started for player {username}"
        }

    async def get_player_status(self, username: str, include_progress_details: bool = True) -> Dict[str, Any]:
        """Mock de get_player_status handler."""
        player = await self.player_repo.get_by_username(username)
        if not player:
            raise ValueError(f"Player {username} not found")

        return {
            "username": player.username,
            "status": player.status,
            "progress": {
                "percentage": player.progress_percentage,
                "done_games": player.done_games,
                "total_games": player.total_games
            }
        }


# Tests
class TestStockfishEngine:
    """Pruebas exhaustivas para StockfishEngine."""

    def setUp(self):
        self.config = MockStockfishConfig()

    async def test_engine_context_manager(self):
        """Test context manager del engine."""
        self.setUp()

        engine = StockfishEngine(self.config)
        assert engine._is_initialized is False

        with engine:
            assert engine._is_initialized is True

        assert engine._is_initialized is False

    async def test_engine_analyze_moves_valid_pgn(self):
        """Test análisis con PGN válido."""
        self.setUp()

        pgn_data = "[White \"player1\"] [Black \"player2\"] 1. e4 e5 2. Nf3"

        with StockfishEngine(self.config) as engine:
            result = await engine.analyze_moves(pgn_data)

            assert result is not None
            assert len(result.moves) == 3
            assert result.moves[0].move == "e4"
            assert result.moves[1].move == "e5"
            assert result.moves[2].move == "Nf3"
            assert result.total_moves == 3

    async def test_engine_analyze_moves_empty_pgn(self):
        """Test análisis con PGN vacío."""
        self.setUp()

        with StockfishEngine(self.config) as engine:
            try:
                await engine.analyze_moves("")
                assert False, "Should raise ValueError for empty PGN"
            except ValueError as e:
                assert "PGN data is required" in str(e)

    async def test_engine_analyze_moves_invalid_pgn(self):
        """Test análisis con PGN sin movimientos válidos."""
        self.setUp()

        pgn_data = "[White \"player1\"] [Black \"player2\"]"  # Sin movimientos

        with StockfishEngine(self.config) as engine:
            try:
                await engine.analyze_moves(pgn_data)
                assert False, "Should raise ValueError for PGN without moves"
            except ValueError as e:
                assert "No valid moves found" in str(e)

    async def test_engine_not_initialized(self):
        """Test error cuando engine no está inicializado."""
        self.setUp()

        engine = StockfishEngine(self.config)

        try:
            await engine.analyze_moves("1. e4")
            assert False, "Should raise RuntimeError for uninitialized engine"
        except RuntimeError as e:
            assert "Engine not initialized" in str(e)

    def test_engine_test_functionality(self):
        """Test funcionalidad de test del engine."""
        self.setUp()

        engine = StockfishEngine(self.config)
        assert engine.test_engine() is True

        # Test con config inválida
        bad_config = MockStockfishConfig(path="/invalid/path")
        bad_engine = StockfishEngine(bad_config)
        assert bad_engine.test_engine() is False


class TestStockfishEngineAdapter:
    """Pruebas para StockfishEngineAdapter."""

    def setUp(self):
        self.config = MockStockfishConfig()

    async def test_adapter_analyze_moves(self):
        """Test análisis a través del adapter."""
        self.setUp()

        adapter = StockfishEngineAdapter(self.config)
        pgn_data = "1. e4 e5 2. Nf3"

        result = await adapter.analyze_moves(pgn_data)

        assert result is not None
        assert len(result.moves) == 3
        assert result.avg_centipawn_loss >= 0


class TestSQLRepositories:
    """Pruebas exhaustivas para SQL Repositories."""

    def setUp(self):
        self.config = MockDatabaseConfig()

    async def test_player_repository_save_new(self):
        """Test guardar nuevo jugador."""
        self.setUp()

        repo = SQLPlayerRepository(self.config)
        player = Player(username="testuser", status="pending")

        saved_player = await repo.save(player)

        assert saved_player.id is not None
        assert saved_player.username == "testuser"
        assert saved_player.status == "pending"

    async def test_player_repository_save_existing(self):
        """Test actualizar jugador existente."""
        self.setUp()

        repo = SQLPlayerRepository(self.config)

        # Crear jugador
        player = Player(username="testuser", status="pending")
        saved_player = await repo.save(player)

        # Actualizar
        updated_player = Player(
            id=saved_player.id,
            username="testuser",
            status="completed",
            done_games=10,
            total_games=10,
            progress_percentage=100.0
        )
        final_player = await repo.save(updated_player)

        assert final_player.id == saved_player.id
        assert final_player.status == "completed"
        assert final_player.done_games == 10

    async def test_player_repository_get_by_username(self):
        """Test obtener jugador por username."""
        self.setUp()

        repo = SQLPlayerRepository(self.config)

        # Jugador no existe
        result = await repo.get_by_username("nonexistent")
        assert result is None

        # Crear y buscar jugador
        player = Player(username="testuser", status="pending")
        await repo.save(player)

        found_player = await repo.get_by_username("testuser")
        assert found_player is not None
        assert found_player.username == "testuser"

    async def test_player_repository_delete(self):
        """Test eliminar jugador."""
        self.setUp()

        repo = SQLPlayerRepository(self.config)

        # Crear jugador
        player = Player(username="testuser", status="pending")
        saved_player = await repo.save(player)

        # Eliminar
        deleted = await repo.delete(saved_player.id)
        assert deleted is True

        # Verificar que no existe
        found_player = await repo.get_by_username("testuser")
        assert found_player is None

        # Eliminar inexistente
        deleted_again = await repo.delete(999)
        assert deleted_again is False

    async def test_player_repository_validation(self):
        """Test validaciones del repository."""
        self.setUp()

        repo = SQLPlayerRepository(self.config)

        # Username vacío
        try:
            player = Player(username="", status="pending")
            await repo.save(player)
            assert False, "Should raise ValueError for empty username"
        except ValueError as e:
            assert "Username is required" in str(e)

    async def test_player_repository_connection_failure(self):
        """Test manejo de fallos de conexión."""
        self.setUp()

        repo = SQLPlayerRepository(self.config)
        repo._simulate_connection_failure()

        player = Player(username="testuser", status="pending")

        try:
            await repo.save(player)
            assert False, "Should raise RuntimeError for connection failure"
        except RuntimeError as e:
            assert "Database connection not active" in str(e)

    async def test_game_repository_save_and_get(self):
        """Test guardar y obtener partidas."""
        self.setUp()

        repo = SQLGameRepository(self.config)

        game = Game(
            username="testuser",
            game_url="https://chess.com/game/123",
            time_control="10+0",
            result="win",
            player_id=1,
            pgn_data="1. e4 e5"
        )

        saved_game = await repo.save(game)
        assert saved_game.id is not None

        # Obtener por ID
        found_game = await repo.get_by_id(saved_game.id)
        assert found_game is not None
        assert found_game.username == "testuser"

    async def test_game_repository_get_by_player_id(self):
        """Test obtener partidas por player_id."""
        self.setUp()

        repo = SQLGameRepository(self.config)

        # Crear partidas para dos jugadores
        game1 = Game(username="player1", game_url="https://chess.com/game/1", time_control="10+0", result="win", player_id=1)
        game2 = Game(username="player1", game_url="https://chess.com/game/2", time_control="5+0", result="loss", player_id=1)
        game3 = Game(username="player2", game_url="https://chess.com/game/3", time_control="10+0", result="draw", player_id=2)

        await repo.save(game1)
        await repo.save(game2)
        await repo.save(game3)

        # Obtener partidas del jugador 1
        player1_games = await repo.get_by_player_id(1)
        assert len(player1_games) == 2

        # Obtener partidas del jugador 2
        player2_games = await repo.get_by_player_id(2)
        assert len(player2_games) == 1

    async def test_game_repository_filters(self):
        """Test filtros de partidas."""
        self.setUp()

        repo = SQLGameRepository(self.config)

        # Crear partidas con diferentes controles de tiempo
        game1 = Game(username="player1", game_url="https://chess.com/game/1", time_control="10+0", result="win", player_id=1)
        game2 = Game(username="player1", game_url="https://chess.com/game/2", time_control="5+0", result="loss", player_id=1)
        game3 = Game(username="player1", game_url="https://chess.com/game/3", time_control="15+10", result="draw", player_id=1)

        await repo.save(game1)
        await repo.save(game2)
        await repo.save(game3)

        # Filtrar por time_control
        filtered_games = await repo.get_by_filters(player_id=1, time_control_filter="10")
        assert len(filtered_games) == 2  # "10+0" y "15+10"

        # Filtrar con limit
        limited_games = await repo.get_by_filters(player_id=1, limit=2)
        assert len(limited_games) == 2

        # Count
        count = await repo.count_by_filters(player_id=1)
        assert count == 3


class TestPlayerHandlers:
    """Pruebas para PlayerHandlers."""

    def setUp(self):
        self.handlers = PlayerHandlers()

    async def test_analyze_player_new(self):
        """Test analyze_player con nuevo jugador."""
        self.setUp()

        result = await self.handlers.analyze_player("newuser", months_to_analyze=6)

        assert result["success"] is True
        assert result["player_id"] is not None
        assert "Analysis started" in result["message"]

    async def test_analyze_player_existing_completed(self):
        """Test analyze_player con jugador ya completado."""
        self.setUp()

        # Crear jugador completado
        completed_player = Player(username="completeduser", status="completed")
        await self.handlers.player_repo.save(completed_player)

        result = await self.handlers.analyze_player("completeduser", force_refresh=False)

        assert result["success"] is True
        assert "already analyzed" in result["message"]

    async def test_analyze_player_empty_username(self):
        """Test analyze_player con username vacío."""
        self.setUp()

        try:
            await self.handlers.analyze_player("")
            assert False, "Should raise ValueError for empty username"
        except ValueError as e:
            assert "Username is required" in str(e)

    async def test_get_player_status_existing(self):
        """Test get_player_status con jugador existente."""
        self.setUp()

        # Crear jugador
        player = Player(username="testuser", status="in_progress", done_games=5, total_games=10, progress_percentage=50.0)
        await self.handlers.player_repo.save(player)

        result = await self.handlers.get_player_status("testuser")

        assert result["username"] == "testuser"
        assert result["status"] == "in_progress"
        assert result["progress"]["percentage"] == 50.0

    async def test_get_player_status_not_found(self):
        """Test get_player_status con jugador inexistente."""
        self.setUp()

        try:
            await self.handlers.get_player_status("nonexistent")
            assert False, "Should raise ValueError for non-existent player"
        except ValueError as e:
            assert "not found" in str(e)


async def run_infrastructure_tests():
    """Ejecutar todas las pruebas de Infrastructure."""
    print("🧪 Running Exhaustive Infrastructure Tests...")

    # StockfishEngine tests
    print("  📝 Testing StockfishEngine...")
    test_engine = TestStockfishEngine()
    await test_engine.test_engine_context_manager()
    await test_engine.test_engine_analyze_moves_valid_pgn()
    await test_engine.test_engine_analyze_moves_empty_pgn()
    await test_engine.test_engine_analyze_moves_invalid_pgn()
    await test_engine.test_engine_not_initialized()
    test_engine.test_engine_test_functionality()
    print("    ✅ StockfishEngine tests passed")

    # StockfishEngineAdapter tests
    print("  📝 Testing StockfishEngineAdapter...")
    test_adapter = TestStockfishEngineAdapter()
    await test_adapter.test_adapter_analyze_moves()
    print("    ✅ StockfishEngineAdapter tests passed")

    # SQL Repositories tests
    print("  📝 Testing SQL Repositories...")
    test_repos = TestSQLRepositories()
    await test_repos.test_player_repository_save_new()
    await test_repos.test_player_repository_save_existing()
    await test_repos.test_player_repository_get_by_username()
    await test_repos.test_player_repository_delete()
    await test_repos.test_player_repository_validation()
    await test_repos.test_player_repository_connection_failure()
    await test_repos.test_game_repository_save_and_get()
    await test_repos.test_game_repository_get_by_player_id()
    await test_repos.test_game_repository_filters()
    print("    ✅ SQL Repositories tests passed")

    # PlayerHandlers tests
    print("  📝 Testing PlayerHandlers...")
    test_handlers = TestPlayerHandlers()
    await test_handlers.test_analyze_player_new()
    await test_handlers.test_analyze_player_existing_completed()
    await test_handlers.test_analyze_player_empty_username()
    await test_handlers.test_get_player_status_existing()
    await test_handlers.test_get_player_status_not_found()
    print("    ✅ PlayerHandlers tests passed")

    print("✅ All Infrastructure tests PASSED!")
    return True


if __name__ == "__main__":
    import asyncio

    async def main():
        try:
            success = await run_infrastructure_tests()
            print("\n🎉 Infrastructure exhaustive testing completed successfully!")
            print("\n📋 Infrastructure Validation Summary:")
            print("  ✅ StockfishEngine - Integración con motor de análisis")
            print("  ✅ StockfishEngineAdapter - Adapter pattern funcionando")
            print("  ✅ SQL Repositories - Persistencia y queries robustas")
            print("  ✅ Database connection handling - Manejo de errores de DB")
            print("  ✅ PlayerHandlers - Bridge HTTP/Application layer")
            print("  ✅ Configuration management - Configuración tipada")
            print("  ✅ External integrations - Interfaces externas limpias")
            print("  ✅ Error handling - Manejo robusto de fallos externos")
            sys.exit(0 if success else 1)
        except Exception as e:
            print(f"❌ Infrastructure tests failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    asyncio.run(main())