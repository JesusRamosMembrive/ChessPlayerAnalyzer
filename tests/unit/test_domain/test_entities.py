"""
Tests unitarios para entidades de dominio.
"""
import pytest
from datetime import datetime
from app.domain.entities.player import Player, PlayerStatus
from app.domain.entities.game import Game, MoveData


class TestPlayer:
    """Tests para entidad Player."""

    def test_create_player_default_status(self):
        """Test creación de player con estado por defecto."""
        player = Player(username="testuser")
        assert player.username == "testuser"
        assert player.status == PlayerStatus.NOT_ANALYZED
        assert player.progress == 0

    def test_mark_as_pending(self):
        """Test marcar player como pendiente."""
        player = Player(username="testuser")
        player.mark_as_pending("task-123")

        assert player.status == PlayerStatus.PENDING
        assert player.last_task_id == "task-123"
        assert player.requested_at is not None
        assert player.error is None

    def test_update_progress(self):
        """Test actualización de progreso."""
        player = Player(username="testuser")
        player.update_progress(done_games=5, total_games=10)

        assert player.done_games == 5
        assert player.total_games == 10
        assert player.progress == 50

    def test_update_progress_zero_total(self):
        """Test actualización de progreso con total cero."""
        player = Player(username="testuser")
        player.update_progress(done_games=0, total_games=0)

        assert player.progress == 0  # No división por cero

    def test_mark_as_ready(self):
        """Test marcar player como listo."""
        player = Player(username="testuser")
        player.mark_as_ready()

        assert player.status == PlayerStatus.READY
        assert player.finished_at is not None
        assert player.progress == 100
        assert player.error is None

    def test_mark_as_error(self):
        """Test marcar player con error."""
        player = Player(username="testuser")
        player.mark_as_error("Connection failed")

        assert player.status == PlayerStatus.ERROR
        assert player.error == "Connection failed"
        assert player.finished_at is not None

    def test_is_ready_for_analysis(self):
        """Test verificación si player está listo para análisis."""
        player = Player(username="testuser")

        # NOT_ANALYZED debe estar listo
        assert player.is_ready_for_analysis() is True

        # PENDING no debe estar listo
        player.status = PlayerStatus.PENDING
        assert player.is_ready_for_analysis() is False

        # ERROR debe estar listo (para reintentar)
        player.status = PlayerStatus.ERROR
        assert player.is_ready_for_analysis() is True

    def test_can_be_refreshed(self):
        """Test verificación si se puede refrescar análisis."""
        player = Player(username="testuser")

        # NOT_ANALYZED no se puede refrescar (no hay nada que refrescar)
        assert player.can_be_refreshed() is False

        # READY se puede refrescar
        player.status = PlayerStatus.READY
        assert player.can_be_refreshed() is True

        # ERROR se puede refrescar
        player.status = PlayerStatus.ERROR
        assert player.can_be_refreshed() is True


class TestGame:
    """Tests para entidad Game."""

    def test_create_game(self):
        """Test creación de game básica."""
        game = Game(
            id=1,
            pgn="1. e4 e5 2. Nf3 1-0",
            white_username="player1",
            black_username="player2"
        )
        assert game.id == 1
        assert game.white_username == "player1"

    def test_get_player_color(self):
        """Test obtener color del jugador."""
        game = Game(
            id=1,
            pgn="1. e4 e5 1-0",
            white_username="alice",
            black_username="bob"
        )

        assert game.get_player_color("alice") == "white"
        assert game.get_player_color("bob") == "black"
        assert game.get_player_color("charlie") is None

    def test_get_player_elo(self):
        """Test obtener ELO del jugador."""
        game = Game(
            id=1,
            pgn="1. e4 e5 1-0",
            white_username="alice",
            black_username="bob",
            white_elo=1500,
            black_elo=1600
        )

        assert game.get_player_elo("alice") == 1500
        assert game.get_player_elo("bob") == 1600
        assert game.get_player_elo("charlie") is None

    def test_is_analyzed(self):
        """Test verificación si game está analizada."""
        game = Game(id=1, pgn="1. e4 e5 1-0")

        # Sin moves, no está analizada
        assert game.is_analyzed() is False

        # Con moves vacías, no está analizada
        game.moves = []
        assert game.is_analyzed() is False

        # Con moves, está analizada
        game.moves = [MoveData(1, "e4", "e4", 0)]
        assert game.is_analyzed() is True

    def test_get_total_moves(self):
        """Test obtener total de movimientos."""
        game = Game(id=1, pgn="1. e4 e5 1-0")

        # Sin moves
        assert game.get_total_moves() == 0

        # Con moves
        game.moves = [
            MoveData(1, "e4", "e4", 0),
            MoveData(2, "e5", "e5", 0)
        ]
        assert game.get_total_moves() == 2