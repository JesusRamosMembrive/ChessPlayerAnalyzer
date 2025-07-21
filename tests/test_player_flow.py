from pathlib import Path


def test_analyze_game(client):
    pgn_path = Path(__file__).parent / "data" / "sample_user.pgn"
    pgn_text = pgn_path.read_text()
    resp = client.post("/analyze", json={"pgn": pgn_text})
    assert resp.status_code == 200
    data = resp.json()
    assert "game_id" in data

    resp2 = client.get(f"/games/{data['game_id']}")
    assert resp2.status_code == 200
    assert resp2.json()["id"] == data["game_id"]
