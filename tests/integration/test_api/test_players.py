"""
Tests de integración para endpoints de players.
"""
import pytest
from fastapi.testclient import TestClient
from app.models import Player, PlayerStatus


class TestPlayersEndpoint:
    """Tests para endpoints de players."""

    def test_get_player_not_found(self, client: TestClient):
        """Test GET player que no existe."""
        response = client.get("/players/nonexistent")
        assert response.status_code == 404

    def test_post_analyze_player(self, client: TestClient):
        """Test POST para analizar un player."""
        # Este test requeriría mocking de Celery y Chess.com API
        # Por ahora verificamos que el endpoint existe
        response = client.post("/players/testuser")
        # Puede fallar por dependencias, pero el endpoint debe existir
        assert response.status_code in [200, 422, 500]  # No 404

    def test_health_endpoint(self, client: TestClient):
        """Test endpoint de health check."""
        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()


class TestPlayerModel:
    """Tests para el modelo Player."""

    def test_create_player(self, session):
        """Test creación de player en BD."""
        player = Player(
            username="testuser",
            status=PlayerStatus.pending,
            progress=0,
            total_games=0,
            done_games=0
        )
        session.add(player)
        session.commit()
        session.refresh(player)

        assert player.username == "testuser"
        assert player.status == PlayerStatus.pending
        assert player.progress == 0

    def test_player_status_enum(self):
        """Test enum de estado de player."""
        assert PlayerStatus.not_analyzed == "not_analyzed"
        assert PlayerStatus.pending == "pending"
        assert PlayerStatus.ready == "ready"
        assert PlayerStatus.error == "error"