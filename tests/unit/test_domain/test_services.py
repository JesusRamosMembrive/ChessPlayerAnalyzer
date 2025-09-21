"""
Tests unitarios para servicios de dominio.
"""
import pytest
from unittest.mock import Mock, AsyncMock
from datetime import datetime

from app.domain.entities.player import Player, PlayerStatus
from app.domain.entities.game import Game, MoveData
from app.domain.entities.analysis import PlayerAnalysis, GameAnalysis
from app.domain.services.player_service import PlayerService
from app.domain.services.analysis_service import AnalysisService
from app.domain.services.game_service import GameService
from app.domain.value_objects.metrics import (
    QualityMetrics, TimingMetrics, OpeningMetrics, EndgameMetrics,
    RiskAssessment, PerformanceMetrics, PhaseQuality, ClutchAccuracy,
    TimeComplexity, Benchmark
)


class TestAnalysisService:
    """Tests para AnalysisService."""

    def setup_method(self):
        """Setup para cada test."""
        self.analysis_service = AnalysisService()

    def test_analyze_game_basic(self):
        """Test análisis básico de partida."""
        # Crear datos de test
        game = Game(
            id=1,
            pgn="1. e4 e5 2. Nf3 Nc6 3. Bb5 1-0",
            white_username="alice",
            black_username="bob",
            move_times=[1000, 2000, 1500]  # milliseconds
        )

        moves_data = [
            MoveData(1, "e4", "e4", 0, 15, 15),
            MoveData(2, "e5", "e5", 0, -15, -15),
            MoveData(3, "Nf3", "Nf3", 5, 15, 10)
        ]

        # Ejecutar análisis
        result = self.analysis_service.analyze_game(game, moves_data)

        # Verificar resultado
        assert isinstance(result, GameAnalysis)
        assert result.game_id == 1
        assert isinstance(result.quality_metrics, QualityMetrics)
        assert isinstance(result.timing_metrics, TimingMetrics)
        assert result.analyzed_at is not None

    def test_analyze_game_empty_moves_raises_error(self):
        """Test que analyze_game falla con moves vacíos."""
        game = Game(id=1, pgn="1. e4 1-0")

        with pytest.raises(ValueError, match="No moves data provided"):
            self.analysis_service.analyze_game(game, [])

    def test_calculate_quality_metrics_perfect_game(self):
        """Test cálculo de métricas con partida perfecta."""
        moves_data = [
            MoveData(1, "e4", "e4", 0),  # Perfect moves
            MoveData(2, "e5", "e5", 0),
            MoveData(3, "Nf3", "Nf3", 0)
        ]

        metrics = self.analysis_service._calculate_quality_metrics(moves_data)

        assert metrics.avg_acpl == 0.0
        assert metrics.avg_match_rate == 1.0  # All moves match best

    def test_calculate_quality_metrics_with_errors(self):
        """Test cálculo de métricas con errores."""
        moves_data = [
            MoveData(1, "e4", "e4", 0),      # Perfect
            MoveData(2, "e6", "e5", 20),     # Small error
            MoveData(3, "f3", "Nf3", 50)     # Bigger error
        ]

        metrics = self.analysis_service._calculate_quality_metrics(moves_data)

        assert metrics.avg_acpl > 0
        assert metrics.avg_match_rate < 1.0
        assert 0 <= metrics.avg_match_rate <= 1

    def test_detect_suspicious_quality(self):
        """Test detección de calidad sospechosa."""
        # Métricas normales
        normal_metrics = QualityMetrics(
            avg_acpl=15.0, avg_wdl_loss=0.05, robust_loss=0.02,
            avg_match_rate=0.75, avg_ipr=2200.0
        )
        assert not self.analysis_service._detect_suspicious_quality(normal_metrics)

        # Métricas sospechosas (demasiado perfectas)
        suspicious_metrics = QualityMetrics(
            avg_acpl=3.0, avg_wdl_loss=0.01, robust_loss=0.0,
            avg_match_rate=0.98, avg_ipr=2800.0
        )
        assert self.analysis_service._detect_suspicious_quality(suspicious_metrics)

    def test_analyze_player_basic(self):
        """Test análisis básico de jugador."""
        # Crear datos de test
        game_analyses = [
            self._create_sample_game_analysis(1),
            self._create_sample_game_analysis(2),
            self._create_sample_game_analysis(3)
        ]

        games = [
            Game(id=1, pgn="1. e4 e5 1-0", created_at=datetime(2023, 1, 1)),
            Game(id=2, pgn="1. d4 d5 1-0", created_at=datetime(2023, 1, 2)),
            Game(id=3, pgn="1. Nf3 Nf6 1-0", created_at=datetime(2023, 1, 3))
        ]

        # Ejecutar análisis
        result = self.analysis_service.analyze_player("testuser", game_analyses, games)

        # Verificar resultado
        assert isinstance(result, PlayerAnalysis)
        assert result.username == "testuser"
        assert result.games_analyzed == 3
        assert isinstance(result.quality_metrics, QualityMetrics)
        assert isinstance(result.risk_assessment, RiskAssessment)
        assert result.analyzed_at is not None

    def test_analyze_player_empty_analyses_raises_error(self):
        """Test que analyze_player falla sin análisis."""
        with pytest.raises(ValueError, match="No game analyses provided"):
            self.analysis_service.analyze_player("testuser", [], [])

    def test_calculate_risk_assessment(self):
        """Test cálculo de evaluación de riesgo."""
        # Métricas de bajo riesgo
        low_risk_quality = QualityMetrics(
            avg_acpl=25.0, avg_wdl_loss=0.1, robust_loss=0.05,
            avg_match_rate=0.60, avg_ipr=1800.0
        )
        low_risk_timing = TimingMetrics(60.0, 100.0, 0.5, 5)
        low_risk_opening = OpeningMetrics(0.7, 8.0, 5, 0.3)
        low_risk_performance = PerformanceMetrics(0.0, 0.0, [25.0, 24.0, 26.0])

        risk = self.analysis_service._calculate_risk_assessment(
            low_risk_quality, low_risk_timing, low_risk_opening, low_risk_performance
        )

        assert risk.risk_score < 50
        assert not risk.risk_factors["low_acpl"]

        # Métricas de alto riesgo
        high_risk_quality = QualityMetrics(
            avg_acpl=5.0, avg_wdl_loss=0.01, robust_loss=0.0,
            avg_match_rate=0.95, avg_ipr=2700.0
        )
        high_risk_timing = TimingMetrics(30.0, 10.0, 0.95, 0)
        high_risk_opening = OpeningMetrics(0.05, 15.0, 1, 0.1)

        high_risk = self.analysis_service._calculate_risk_assessment(
            high_risk_quality, high_risk_timing, high_risk_opening, low_risk_performance
        )

        assert high_risk.risk_score > 70
        assert high_risk.risk_factors["low_acpl"]
        assert high_risk.risk_factors["high_match_rate"]

    def _create_sample_game_analysis(self, game_id: int) -> GameAnalysis:
        """Helper para crear análisis de partida de muestra."""
        return GameAnalysis(
            game_id=game_id,
            quality_metrics=QualityMetrics(15.0, 0.05, 0.02, 0.75, 2200.0),
            timing_metrics=TimingMetrics(45.0, 200.0, 0.7, 3),
            opening_metrics=OpeningMetrics(0.6, 8.0, 4, 0.25),
            endgame_metrics=EndgameMetrics(None, None, None),
            analyzed_at=datetime.utcnow()
        )


class TestPlayerService:
    """Tests para PlayerService."""

    def setup_method(self):
        """Setup para cada test."""
        self.player_repo_mock = Mock()
        self.analysis_repo_mock = Mock()
        self.player_service = PlayerService(self.player_repo_mock, self.analysis_repo_mock)

    @pytest.mark.asyncio
    async def test_request_analysis_new_player(self):
        """Test solicitar análisis de jugador nuevo."""
        # Mock setup
        self.player_repo_mock.get_by_username = AsyncMock(return_value=None)
        self.player_repo_mock.save = AsyncMock()

        # Ejecutar
        result = await self.player_service.request_analysis("newuser")

        # Verificar
        self.player_repo_mock.get_by_username.assert_called_once_with("newuser")
        self.player_repo_mock.save.assert_called_once()

        # Verificar que el jugador returned tiene el estado correcto
        saved_player = self.player_repo_mock.save.call_args[0][0]
        assert saved_player.username == "newuser"
        assert saved_player.status == PlayerStatus.PENDING
        assert saved_player.last_task_id is not None

    @pytest.mark.asyncio
    async def test_request_analysis_player_already_pending(self):
        """Test solicitar análisis de jugador ya en proceso."""
        # Mock setup - jugador existente en estado pending
        existing_player = Player(username="testuser", status=PlayerStatus.PENDING)
        self.player_repo_mock.get_by_username = AsyncMock(return_value=existing_player)

        # Debe fallar
        with pytest.raises(ValueError, match="already being analyzed"):
            await self.player_service.request_analysis("testuser")

    @pytest.mark.asyncio
    async def test_request_analysis_force_refresh(self):
        """Test solicitar análisis con force_refresh."""
        # Mock setup - jugador existente completado
        existing_player = Player(username="testuser", status=PlayerStatus.READY)
        self.player_repo_mock.get_by_username = AsyncMock(return_value=existing_player)
        self.player_repo_mock.save = AsyncMock()

        # Ejecutar con force_refresh
        result = await self.player_service.request_analysis("testuser", force_refresh=True)

        # Debe funcionar
        self.player_repo_mock.save.assert_called_once()
        saved_player = self.player_repo_mock.save.call_args[0][0]
        assert saved_player.status == PlayerStatus.PENDING

    @pytest.mark.asyncio
    async def test_update_analysis_progress(self):
        """Test actualización de progreso."""
        # Mock setup
        player = Player(username="testuser", status=PlayerStatus.PENDING)
        self.player_repo_mock.get_by_username = AsyncMock(return_value=player)
        self.player_repo_mock.save = AsyncMock()

        # Ejecutar
        result = await self.player_service.update_analysis_progress("testuser", 5, 10)

        # Verificar
        self.player_repo_mock.save.assert_called_once()
        saved_player = self.player_repo_mock.save.call_args[0][0]
        assert saved_player.done_games == 5
        assert saved_player.total_games == 10
        assert saved_player.progress == 50

    @pytest.mark.asyncio
    async def test_complete_analysis(self):
        """Test completar análisis."""
        # Mock setup
        player = Player(username="testuser", status=PlayerStatus.PENDING)
        analysis = self._create_sample_player_analysis("testuser")

        self.player_repo_mock.get_by_username = AsyncMock(return_value=player)
        self.player_repo_mock.save = AsyncMock()
        self.analysis_repo_mock.save_player_analysis = AsyncMock()

        # Ejecutar
        result = await self.player_service.complete_analysis("testuser", analysis)

        # Verificar
        self.analysis_repo_mock.save_player_analysis.assert_called_once_with(analysis)
        self.player_repo_mock.save.assert_called_once()

        saved_player = self.player_repo_mock.save.call_args[0][0]
        assert saved_player.status == PlayerStatus.READY
        assert saved_player.progress == 100

    def test_validate_analysis_request_valid(self):
        """Test validación de request válido."""
        result = self.player_service.validate_analysis_request("validuser")

        assert result["valid"] is True
        assert "Valid request" in result["reason"]

    def test_validate_analysis_request_empty_username(self):
        """Test validación con username vacío."""
        result = self.player_service.validate_analysis_request("")

        assert result["valid"] is False
        assert "cannot be empty" in result["reason"]

    def test_validate_analysis_request_invalid_characters(self):
        """Test validación con caracteres inválidos."""
        result = self.player_service.validate_analysis_request("user@invalid")

        assert result["valid"] is False
        assert "invalid characters" in result["reason"]

    def _create_sample_player_analysis(self, username: str) -> PlayerAnalysis:
        """Helper para crear análisis de jugador de muestra."""
        return PlayerAnalysis(
            username=username,
            games_analyzed=5,
            quality_metrics=QualityMetrics(15.0, 0.05, 0.02, 0.75, 2200.0),
            timing_metrics=TimingMetrics(45.0, 200.0, 0.7, 3),
            opening_metrics=OpeningMetrics(0.6, 8.0, 4, 0.25),
            endgame_metrics=EndgameMetrics(None, None, None),
            performance_metrics=PerformanceMetrics(0.0, 0.0, [15.0, 14.0, 16.0]),
            phase_quality=PhaseQuality(12.0, 18.0, 15.0, 0.01, 0.02, 0.015, 0.015),
            clutch_accuracy=ClutchAccuracy(None, 0.5),
            time_complexity=TimeComplexity(None),
            benchmark=Benchmark(50, 60),
            risk_assessment=RiskAssessment(25, {"low_acpl": False}, 75)
        )


class TestGameService:
    """Tests para GameService."""

    def setup_method(self):
        """Setup para cada test."""
        self.game_repo_mock = Mock()
        self.game_service = GameService(self.game_repo_mock)

    def test_parse_pgn_to_game_valid(self):
        """Test parsing de PGN válido."""
        pgn = '''[White "alice"]
[Black "bob"]
[WhiteElo "1500"]
[BlackElo "1600"]
[TimeControl "600+5"]
[Termination "Normal"]
[Date "2023.12.25"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 1-0'''

        result = self.game_service._parse_pgn_to_game(pgn, "alice")

        assert result is not None
        assert result.white_username == "alice"
        assert result.black_username == "bob"
        assert result.white_elo == 1500
        assert result.black_elo == 1600
        assert result.time_control == "600+5"
        assert result.termination == "Normal"

    def test_parse_pgn_to_game_player_not_in_game(self):
        """Test parsing cuando el jugador no participa."""
        pgn = '''[White "alice"]
[Black "bob"]

1. e4 e5 1-0'''

        result = self.game_service._parse_pgn_to_game(pgn, "charlie")

        assert result is None

    def test_is_valid_game_valid(self):
        """Test validación de partida válida."""
        game = Game(
            id=1,
            pgn="[White \"alice\"][Black \"bob\"] 1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 1-0",
            white_username="alice",
            black_username="bob"
        )

        assert self.game_service._is_valid_game(game, "alice") is True

    def test_is_valid_game_too_short(self):
        """Test validación de partida muy corta."""
        game = Game(
            id=1,
            pgn="[White \"alice\"][Black \"bob\"] 1. e4 1-0",  # Solo 1 movimiento
            white_username="alice",
            black_username="bob"
        )

        assert self.game_service._is_valid_game(game, "alice") is False

    def test_is_valid_game_player_not_participant(self):
        """Test validación cuando jugador no participa."""
        game = Game(
            id=1,
            pgn="1. e4 e5 2. Nf3 Nc6 1-0",
            white_username="alice",
            black_username="bob"
        )

        assert self.game_service._is_valid_game(game, "charlie") is False

    def test_validate_pgn_valid(self):
        """Test validación de PGN válido."""
        pgn = '''[White "alice"]
[Black "bob"]
[Result "1-0"]

1. e4 e5 2. Nf3 1-0'''

        result = self.game_service.validate_pgn(pgn)

        assert result["valid"] is True

    def test_validate_pgn_missing_headers(self):
        """Test validación de PGN con headers faltantes."""
        pgn = '''[White "alice"]

1. e4 e5 1-0'''  # Falta Black y Result

        result = self.game_service.validate_pgn(pgn)

        assert result["valid"] is False
        assert "Missing required headers" in result["reason"]

    def test_validate_pgn_no_moves(self):
        """Test validación de PGN sin movimientos."""
        pgn = '''[White "alice"]
[Black "bob"]
[Result "1-0"]'''

        result = self.game_service.validate_pgn(pgn)

        assert result["valid"] is False
        assert "No moves found" in result["reason"]