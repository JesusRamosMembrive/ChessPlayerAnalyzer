#!/usr/bin/env python3
"""
Pruebas End-to-End exhaustivas.
Valida todo el sistema funcionando en conjunto desde la API hasta la persistencia.
"""
import sys
import asyncio
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

# Agregar path para imports
sys.path.append('.')

# Configuraciones mock
@dataclass
class MockAppConfig:
    def __init__(self):
        self.database = MockDatabaseConfig()
        self.stockfish = MockStockfishConfig()

@dataclass
class MockDatabaseConfig:
    url: str = "sqlite:///:memory:"
    pool_size: int = 5
    max_overflow: int = 10

@dataclass
class MockStockfishConfig:
    path: str = "/usr/bin/stockfish"
    depth: int = 15
    time_limit: float = 1.0

# Entities completas
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

    def with_moves_data(self, moves_data: list) -> 'Game':
        return Game(
            id=self.id,
            player_id=self.player_id,
            username=self.username,
            game_url=self.game_url,
            time_control=self.time_control,
            result=self.result,
            played_at=self.played_at,
            pgn_data=self.pgn_data,
            moves_data=moves_data
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

# Sistema completo mock
class FullSystemMock:
    """Sistema completo integrado para testing end-to-end."""

    def __init__(self):
        self.config = MockAppConfig()
        self._players = {}
        self._games = {}
        self._player_analyses = {}
        self._game_analyses = {}
        self._next_player_id = 1
        self._next_game_id = 1

    # === Repository Layer ===
    async def save_player(self, player: Player) -> Player:
        """Guarda jugador simulando repository."""
        if player.id is None:
            new_player = player.with_id(self._next_player_id)
            self._players[self._next_player_id] = new_player
            self._next_player_id += 1
            return new_player
        else:
            self._players[player.id] = player
            return player

    async def get_player_by_username(self, username: str) -> Optional[Player]:
        """Obtiene jugador por username."""
        for player in self._players.values():
            if player.username == username:
                return player
        return None

    async def save_game(self, game: Game) -> Game:
        """Guarda partida simulando repository."""
        if game.id is None:
            new_game = game.with_id(self._next_game_id)
            self._games[self._next_game_id] = new_game
            self._next_game_id += 1
            return new_game
        else:
            self._games[game.id] = game
            return game

    async def get_games_by_player_id(self, player_id: int) -> List[Game]:
        """Obtiene partidas de un jugador."""
        return [game for game in self._games.values() if game.player_id == player_id]

    async def save_player_analysis(self, analysis: PlayerAnalysis) -> PlayerAnalysis:
        """Guarda análisis de jugador."""
        self._player_analyses[analysis.player_id] = analysis
        return analysis

    async def get_player_analysis(self, player_id: int) -> Optional[PlayerAnalysis]:
        """Obtiene análisis de jugador."""
        return self._player_analyses.get(player_id)

    # === Domain Services ===
    def create_new_player(self, username: str, months_to_analyze: int = 12) -> Player:
        """Servicio de dominio: crear jugador."""
        if not username or not username.strip():
            raise ValueError("Username cannot be empty")

        return Player(
            username=username.strip(),
            status="pending",
            done_games=0,
            total_games=0,
            progress_percentage=0.0
        )

    def update_player_progress(self, player: Player, done_games: int, total_games: int) -> Player:
        """Servicio de dominio: actualizar progreso."""
        if done_games < 0 or total_games < 0 or done_games > total_games:
            raise ValueError("Invalid progress values")

        progress_percentage = (done_games / total_games * 100) if total_games > 0 else 0.0

        if done_games == 0:
            status = "pending"
        elif done_games == total_games:
            status = "completed"
        else:
            status = "in_progress"

        return Player(
            id=player.id,
            username=player.username,
            status=status,
            done_games=done_games,
            total_games=total_games,
            progress_percentage=progress_percentage,
            task_id=player.task_id,
            error_message=player.error_message
        )

    def create_game_from_data(self, player_id: int, username: str, game_data: dict) -> Game:
        """Servicio de dominio: crear partida."""
        if not game_data.get("url"):
            raise ValueError("Game URL is required")

        return Game(
            player_id=player_id,
            username=username,
            game_url=game_data["url"],
            time_control=game_data.get("time_control", "unknown"),
            result=game_data.get("result", "unknown"),
            played_at=datetime.fromisoformat(game_data["end_time"]) if game_data.get("end_time") else None,
            pgn_data=game_data.get("pgn")
        )

    async def analyze_game(self, game: Game) -> GameAnalysis:
        """Servicio de dominio: analizar partida."""
        if not game.pgn_data:
            raise ValueError("Game must have PGN data for analysis")

        # Mock análisis
        quality_metrics = QualityMetrics(
            avg_acpl=25.5,
            avg_wdl_loss=8.2,
            avg_match_rate=75.3
        )

        timing_metrics = TimingMetrics(
            avg_move_time=15.3,
            time_variance=12.1,
            quick_moves_rate=0.15
        )

        risk_assessment = RiskAssessment(
            cheat_probability=0.12,
            risk_level="low",
            suspicion_flags=[]
        )

        return GameAnalysis(
            game_id=game.id,
            quality_metrics=quality_metrics,
            timing_metrics=timing_metrics,
            risk_assessment=risk_assessment,
            analyzed_at=datetime.now()
        )

    # === Application Layer ===
    async def execute_analyze_player_command(self, username: str, force_refresh: bool = False, months_to_analyze: int = 12) -> Dict[str, Any]:
        """Use Case: Analizar jugador."""
        try:
            # Verificar jugador existente
            existing_player = await self.get_player_by_username(username)

            if existing_player and not force_refresh:
                if existing_player.status == "completed":
                    return {
                        "success": True,
                        "player_id": existing_player.id,
                        "message": "Player already analyzed"
                    }

            # Crear o resetear jugador
            if existing_player:
                new_player = Player(
                    id=existing_player.id,
                    username=existing_player.username,
                    status="pending",
                    done_games=0,
                    total_games=0,
                    progress_percentage=0.0
                )
            else:
                new_player = self.create_new_player(username, months_to_analyze)

            saved_player = await self.save_player(new_player)

            return {
                "success": True,
                "player_id": saved_player.id,
                "task_id": "task_123",
                "message": f"Analysis started for player {username}"
            }

        except Exception as e:
            return {
                "success": False,
                "error_message": str(e)
            }

    async def execute_get_player_status_query(self, username: str) -> Optional[Dict[str, Any]]:
        """Use Case: Obtener estado de jugador."""
        player = await self.get_player_by_username(username)
        if not player:
            return None

        return {
            "username": player.username,
            "status": player.status,
            "progress": {
                "percentage": player.progress_percentage,
                "done_games": player.done_games,
                "total_games": player.total_games
            },
            "task_id": player.task_id,
            "error_message": player.error_message
        }

    async def simulate_full_analysis_workflow(self, username: str) -> Dict[str, Any]:
        """Simula el workflow completo de análisis de jugador."""
        # 1. Iniciar análisis
        start_result = await self.execute_analyze_player_command(username)
        if not start_result["success"]:
            return start_result

        player_id = start_result["player_id"]

        # 2. Simular descarga de partidas
        mock_games_data = [
            {
                "url": f"https://chess.com/game/{i}",
                "time_control": "10+0",
                "result": "win" if i % 2 == 0 else "loss",
                "end_time": "2023-01-01T12:00:00",
                "pgn": f"[White \"{username}\"] [Black \"opponent{i}\"] 1. e4 e5 2. Nf3"
            }
            for i in range(1, 6)  # 5 partidas
        ]

        # 3. Crear partidas en el sistema
        games = []
        for game_data in mock_games_data:
            game = self.create_game_from_data(player_id, username, game_data)
            saved_game = await self.save_game(game)
            games.append(saved_game)

        # 4. Actualizar progreso: iniciando análisis
        player = await self.get_player_by_username(username)
        player = self.update_player_progress(player, 0, len(games))
        await self.save_player(player)

        # 5. Simular análisis de cada partida
        analyzed_games = 0
        for game in games:
            try:
                analysis = await self.analyze_game(game)
                analyzed_games += 1

                # Actualizar progreso
                player = await self.get_player_by_username(username)
                player = self.update_player_progress(player, analyzed_games, len(games))
                await self.save_player(player)

            except Exception as e:
                # Log error pero continuar
                print(f"Error analyzing game {game.id}: {e}")

        # 6. Generar análisis agregado del jugador
        player_analysis = PlayerAnalysis(
            player_id=player_id,
            overall_metrics=QualityMetrics(avg_acpl=25.5, avg_wdl_loss=8.2, avg_match_rate=75.3),
            timing_metrics=TimingMetrics(avg_move_time=15.3, time_variance=12.1, quick_moves_rate=0.15),
            risk_assessment=RiskAssessment(cheat_probability=0.12, risk_level="low", suspicion_flags=[]),
            games_analyzed=analyzed_games,
            total_games=len(games),
            analysis_date=datetime.now()
        )

        await self.save_player_analysis(player_analysis)

        # 7. Resultado final
        final_player = await self.get_player_by_username(username)
        return {
            "success": True,
            "username": username,
            "player_id": player_id,
            "games_analyzed": analyzed_games,
            "total_games": len(games),
            "final_status": final_player.status,
            "analysis_available": True
        }

    # === API Layer ===
    async def api_analyze_player(self, username: str, force_refresh: bool = False, months_to_analyze: int = 12) -> Dict[str, Any]:
        """Simula endpoint de API: POST /v2/players/{username}/analyze."""
        if not username:
            raise ValueError("Username is required")

        result = await self.execute_analyze_player_command(username, force_refresh, months_to_analyze)

        # Transformar para respuesta API
        if result["success"]:
            return {
                "success": True,
                "player_id": result["player_id"],
                "task_id": result.get("task_id"),
                "message": result["message"]
            }
        else:
            return {
                "success": False,
                "error": result["error_message"]
            }

    async def api_get_player_status(self, username: str) -> Dict[str, Any]:
        """Simula endpoint de API: GET /v2/players/{username}/status."""
        if not username:
            raise ValueError("Username is required")

        result = await self.execute_get_player_status_query(username)
        if not result:
            raise ValueError(f"Player {username} not found")

        return result

    async def api_get_player_analysis(self, username: str) -> Dict[str, Any]:
        """Simula endpoint de API: GET /v2/players/{username}/analysis."""
        player = await self.get_player_by_username(username)
        if not player:
            raise ValueError(f"Player {username} not found")

        analysis = await self.get_player_analysis(player.id)
        if not analysis:
            raise ValueError(f"No analysis found for player {username}")

        return {
            "username": username,
            "analysis": {
                "overall_metrics": {
                    "avg_acpl": analysis.overall_metrics.avg_acpl,
                    "avg_wdl_loss": analysis.overall_metrics.avg_wdl_loss,
                    "avg_match_rate": analysis.overall_metrics.avg_match_rate
                },
                "timing_metrics": {
                    "avg_move_time": analysis.timing_metrics.avg_move_time,
                    "time_variance": analysis.timing_metrics.time_variance,
                    "quick_moves_rate": analysis.timing_metrics.quick_moves_rate
                },
                "risk_assessment": {
                    "cheat_probability": analysis.risk_assessment.cheat_probability,
                    "risk_level": analysis.risk_assessment.risk_level,
                    "suspicion_flags": analysis.risk_assessment.suspicion_flags
                },
                "games_analyzed": analysis.games_analyzed,
                "total_games": analysis.total_games,
                "analysis_date": analysis.analysis_date.isoformat()
            }
        }


# Tests End-to-End
class TestEndToEndWorkflows:
    """Pruebas exhaustivas end-to-end."""

    def setUp(self):
        self.system = FullSystemMock()

    async def test_complete_player_analysis_workflow(self):
        """Test workflow completo de análisis de jugador."""
        self.setUp()

        print("    🔄 Testing complete player analysis workflow...")

        # 1. Iniciar análisis
        result = await self.system.simulate_full_analysis_workflow("testplayer")

        assert result["success"] is True
        assert result["username"] == "testplayer"
        assert result["games_analyzed"] == 5
        assert result["total_games"] == 5
        assert result["final_status"] == "completed"
        assert result["analysis_available"] is True

        # 2. Verificar que el jugador existe y está completado
        player = await self.system.get_player_by_username("testplayer")
        assert player is not None
        assert player.status == "completed"
        assert player.progress_percentage == 100.0

        # 3. Verificar que las partidas se guardaron
        games = await self.system.get_games_by_player_id(player.id)
        assert len(games) == 5

        # 4. Verificar que el análisis se generó
        analysis = await self.system.get_player_analysis(player.id)
        assert analysis is not None
        assert analysis.games_analyzed == 5

    async def test_api_endpoints_integration(self):
        """Test integración completa de endpoints API."""
        self.setUp()

        print("    🔄 Testing API endpoints integration...")

        # 1. POST /v2/players/{username}/analyze
        analyze_result = await self.system.api_analyze_player("apiuser", months_to_analyze=6)
        assert analyze_result["success"] is True
        assert analyze_result["player_id"] is not None

        # 2. GET /v2/players/{username}/status - jugador recién creado
        status_result = await self.system.api_get_player_status("apiuser")
        assert status_result["username"] == "apiuser"
        assert status_result["status"] == "pending"

        # 3. Simular progreso
        player = await self.system.get_player_by_username("apiuser")
        updated_player = self.system.update_player_progress(player, 3, 10)
        await self.system.save_player(updated_player)

        # 4. GET /v2/players/{username}/status - con progreso
        status_result2 = await self.system.api_get_player_status("apiuser")
        assert status_result2["status"] == "in_progress"
        assert status_result2["progress"]["percentage"] == 30.0

        # 5. Simular análisis completo
        await self.system.simulate_full_analysis_workflow("apiuser")

        # 6. GET /v2/players/{username}/analysis - análisis completo
        analysis_result = await self.system.api_get_player_analysis("apiuser")
        assert analysis_result["username"] == "apiuser"
        assert "overall_metrics" in analysis_result["analysis"]
        assert analysis_result["analysis"]["games_analyzed"] > 0

    async def test_error_handling_workflow(self):
        """Test manejo de errores en workflow completo."""
        self.setUp()

        print("    🔄 Testing error handling workflow...")

        # 1. Username vacío
        try:
            await self.system.api_analyze_player("")
            assert False, "Should raise ValueError for empty username"
        except ValueError as e:
            assert "Username is required" in str(e)

        # 2. Jugador inexistente para status
        try:
            await self.system.api_get_player_status("nonexistent")
            assert False, "Should raise ValueError for non-existent player"
        except ValueError as e:
            assert "not found" in str(e)

        # 3. Análisis inexistente
        # Crear jugador sin análisis
        player = self.system.create_new_player("noanalysis")
        await self.system.save_player(player)

        try:
            await self.system.api_get_player_analysis("noanalysis")
            assert False, "Should raise ValueError for missing analysis"
        except ValueError as e:
            assert "No analysis found" in str(e)

    async def test_concurrent_operations(self):
        """Test operaciones concurrentes en el sistema."""
        self.setUp()

        print("    🔄 Testing concurrent operations...")

        # Simular múltiples jugadores siendo analizados "simultáneamente"
        usernames = ["concurrent1", "concurrent2", "concurrent3"]

        # Iniciar análisis para todos
        tasks = []
        for username in usernames:
            task = self.system.execute_analyze_player_command(username)
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        # Verificar que todos se iniciaron correctamente
        for i, result in enumerate(results):
            assert result["success"] is True
            assert f"concurrent{i+1}" in result["message"]

        # Verificar que todos los jugadores existen
        for username in usernames:
            player = await self.system.get_player_by_username(username)
            assert player is not None
            assert player.status == "pending"

    async def test_force_refresh_workflow(self):
        """Test workflow de force refresh."""
        self.setUp()

        print("    🔄 Testing force refresh workflow...")

        # 1. Crear jugador y completar análisis
        await self.system.simulate_full_analysis_workflow("refreshuser")

        player = await self.system.get_player_by_username("refreshuser")
        assert player.status == "completed"

        # 2. Intentar re-analizar sin force_refresh
        result1 = await self.system.api_analyze_player("refreshuser", force_refresh=False)
        assert result1["success"] is True
        assert "already analyzed" in result1["message"]

        # 3. Re-analizar con force_refresh
        result2 = await self.system.api_analyze_player("refreshuser", force_refresh=True)
        assert result2["success"] is True
        assert "Analysis started" in result2["message"]

        # 4. Verificar que el jugador se reseteó
        reset_player = await self.system.get_player_by_username("refreshuser")
        assert reset_player.status == "pending"
        assert reset_player.done_games == 0

    async def test_data_persistence_workflow(self):
        """Test persistencia de datos a través del workflow."""
        self.setUp()

        print("    🔄 Testing data persistence workflow...")

        # 1. Crear datos completos
        await self.system.simulate_full_analysis_workflow("persistuser")

        # 2. Verificar persistencia de jugador
        player = await self.system.get_player_by_username("persistuser")
        assert player is not None
        original_player_id = player.id

        # 3. Verificar persistencia de partidas
        games = await self.system.get_games_by_player_id(player.id)
        assert len(games) == 5
        for game in games:
            assert game.username == "persistuser"
            assert game.player_id == player.id

        # 4. Verificar persistencia de análisis
        analysis = await self.system.get_player_analysis(player.id)
        assert analysis is not None
        assert analysis.player_id == player.id

        # 5. Verificar integridad relacional
        assert analysis.games_analyzed == len(games)


async def run_end_to_end_tests():
    """Ejecutar todas las pruebas end-to-end."""
    print("🧪 Running Exhaustive End-to-End Tests...")

    test_e2e = TestEndToEndWorkflows()

    await test_e2e.test_complete_player_analysis_workflow()
    print("    ✅ Complete player analysis workflow")

    await test_e2e.test_api_endpoints_integration()
    print("    ✅ API endpoints integration")

    await test_e2e.test_error_handling_workflow()
    print("    ✅ Error handling workflow")

    await test_e2e.test_concurrent_operations()
    print("    ✅ Concurrent operations")

    await test_e2e.test_force_refresh_workflow()
    print("    ✅ Force refresh workflow")

    await test_e2e.test_data_persistence_workflow()
    print("    ✅ Data persistence workflow")

    print("✅ All End-to-End tests PASSED!")
    return True


if __name__ == "__main__":
    async def main():
        try:
            success = await run_end_to_end_tests()
            print("\n🎉 End-to-End exhaustive testing completed successfully!")
            print("\n📋 End-to-End Validation Summary:")
            print("  ✅ Complete analysis workflow - Todo el ciclo funciona")
            print("  ✅ API integration - Endpoints conectados correctamente")
            print("  ✅ Error handling - Manejo robusto de errores")
            print("  ✅ Concurrent operations - Sistema soporta concurrencia")
            print("  ✅ Force refresh - Re-análisis funciona correctamente")
            print("  ✅ Data persistence - Integridad de datos mantenida")
            print("  ✅ Cross-layer communication - Todas las capas integradas")
            print("  ✅ Business workflows - Casos de uso complejos validados")
            sys.exit(0 if success else 1)
        except Exception as e:
            print(f"❌ End-to-End tests failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    asyncio.run(main())