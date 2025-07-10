from pathlib import Path
import time # Added
from unittest.mock import patch, MagicMock, ANY, call # Added ANY and call

from fastapi import status as http_status # Renamed for clarity
from app.models import Player, PlayerStatus, Game # Added
from app.main import app # Added app for dependency_overrides
from app.database import get_session # Added get_session for dependency_overrides


def test_analyze_game(client): # Existing test
    pgn_path = Path(__file__).parent / "data" / "sample_user.pgn"
    pgn_text = pgn_path.read_text()
    resp = client.post("/analyze", json={"pgn": pgn_text})
    assert resp.status_code == 200 # Changed to use http_status if preferred, but 200 is fine
    data = resp.json()
    assert "game_id" in data

    resp2 = client.get(f"/games/{data['game_id']}")
    assert resp2.status_code == 200
    assert resp2.json()["id"] == data["game_id"]


def test_stop_player_analysis_successful_quick_stop(client):
    """
    Tests POST /players/{username}/stop endpoint.
    Simulates tasks stopping quickly.
    """
    username = "testuser_stoppable"
    main_task_id = "main_celery_task_abc123"
    additional_task_id = "additional_celery_task_def456"

    # 1. Prepare Mock Player and Game data
    mock_player = Player(
        username=username,
        status=PlayerStatus.pending,
        progress=50,
        last_task_id=main_task_id,
        id=1 # Assuming an ID is needed for DB operations if not using username as PK in all contexts
    )
    # In SQLModel, related objects are often handled by relationship,
    # but for deletion, we just need to ensure the query for games returns something.
    mock_game1 = Game(id=101, white_username=username, black_username="opponent1", pgn="pgn1")
    mock_game2 = Game(id=102, white_username="opponent2", black_username=username, pgn="pgn2")
    games_for_player = [mock_game1, mock_game2]

    # 2. Mock Database Session
    mock_db_session = MagicMock()
    mock_db_session.get.return_value = mock_player

    # Mock the query for deleting games: session.exec(select(models.Game)...).all()
    mock_query_result = MagicMock()
    mock_query_result.all.return_value = games_for_player
    mock_db_session.exec.return_value = mock_query_result

    # Override FastAPI dependency for get_session
    app.dependency_overrides[get_session] = lambda: mock_db_session

    # 3. Mocks for Celery, Redis, and time.sleep
    with patch('app.main.celery_app.control.revoke') as mock_celery_revoke, \
         patch('app.main.celery_app.control.inspect') as mock_celery_inspect, \
         patch('app.main.AsyncResult') as MockCeleryAsyncResult, \
         patch('app.main.redis_client') as mock_redis_client, \
         patch('time.sleep', return_value=None) as mock_time_sleep:

        # Configure mock_celery_inspect
        mock_inspector_instance = MagicMock()
        mock_inspector_instance.active.return_value = {
            "worker1": [
                {"id": main_task_id, "args": f"('{username}',)", "kwargs": "{}"}, # Main task might appear here too
                {"id": additional_task_id, "args": f"('{username}', 'some_arg')", "kwargs": "{}"},
                {"id": "unrelated_task_789", "args": "('other_user',)", "kwargs": "{}"}
            ]
        }
        mock_celery_inspect.return_value = mock_inspector_instance

        # Configure MockCeleryAsyncResult
        # Main task stops after first poll, additional task also stops quickly
        def async_result_side_effect(task_id_param, app_param=None):
            instance = MagicMock(id=task_id_param)
            if task_id_param == main_task_id:
                instance.state = "REVOKED" # Or SUCCESS/FAILURE
                instance.ready.return_value = True
            elif task_id_param == additional_task_id:
                instance.state = "REVOKED"
                instance.ready.return_value = True
            else: # Should not happen if logic is correct
                instance.state = "PENDING"
                instance.ready.return_value = False
            return instance
        MockCeleryAsyncResult.side_effect = async_result_side_effect

        # 4. Perform the API call
        response = client.post(f"/players/{username}/stop")

        # 5. Assertions
        assert response.status_code == http_status.HTTP_200_OK
        data = response.json()
        assert data["username"] == username
        assert data["status"] == "stopped"
        assert data["message"] == "Analysis stopped and all data removed successfully"
        assert data["games_deleted"] == len(games_for_player)

        # Verify Redis cancellation key
        mock_redis_client.set.assert_called_once_with(f"cancel:{username}", "true", ex=300)

        # Verify Celery revoke calls
        # Call for main_task_id
        mock_celery_revoke.assert_any_call(main_task_id, terminate=True, signal='SIGKILL')
        # Call for additional_task_id (usually Celery's revoke for a list of tasks)
        # The implementation collects additional tasks into a list.
        # If only one additional task, it might be called as `revoke([additional_task_id], ...)`
        # or `revoke(additional_task_id, ...)` if the list has one item and gets unpacked.
        # Let's check if revoke was called with additional_task_id in its first argument.

        # A more robust way to check calls when order or exact list structure might vary:
        found_revoke_call_for_additional = False
        for call_args in mock_celery_revoke.call_args_list:
            args, kwargs = call_args
            if args and args[0] == additional_task_id: # Direct call
                 found_revoke_call_for_additional = True
                 assert kwargs.get('terminate') is True
                 assert kwargs.get('signal') == 'SIGKILL'
                 break
            if args and isinstance(args[0], list) and additional_task_id in args[0]: # Call with a list
                 found_revoke_call_for_additional = True
                 assert kwargs.get('terminate') is True
                 assert kwargs.get('signal') == 'SIGKILL'
                 break
        assert found_revoke_call_for_additional, f"Revoke for additional task {additional_task_id} not found"


        # Verify AsyncResult was checked for both tasks
        MockCeleryAsyncResult.assert_any_call(main_task_id, app=ANY)
        MockCeleryAsyncResult.assert_any_call(additional_task_id, app=ANY)

        # Verify time.sleep was called at least once by the polling loop (even if tasks stop fast, one poll happens)
        # If tasks stop on the very first check, sleep might not be called.
        # The loop runs: `while tasks_to_monitor and (time.time() - start_wait_time) < MAX_WAIT_TIME:`
        # If tasks_to_monitor becomes empty after the first round of checks, sleep is skipped.
        # So, mock_time_sleep.called could be False if everything stops immediately.
        # For this "quick stop" test, it's plausible sleep isn't called.

        # Verify database deletions
        mock_db_session.get.assert_called_once_with(Player, username)
        # Check that delete was called for the player
        mock_db_session.delete.assert_any_call(mock_player)
        # Check that delete was called for each game
        for game in games_for_player:
            mock_db_session.delete.assert_any_call(game)
        mock_db_session.commit.assert_called_once()

    # Clean up dependency override
    del app.dependency_overrides[get_session]
