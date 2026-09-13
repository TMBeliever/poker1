from unittest.mock import MagicMock, patch
import pytest

from src.competition.protocol import AgentPokerClient, Config, APIError
from src.competition.live_watchman import CompetitionWatchman


def test_watchman_track_hand_progress():
    client = AgentPokerClient(Config(base_url="https://test.com"))
    watchman = CompetitionWatchman(
        client=client,
        competition_id="comp_test",
        idle_seconds=1,
        round_hands=2,
    )

    # First hand start
    obs1 = {
        "agentId": "hero",
        "table": {
            "bigBlind": 100,
            "hand": {"id": "hand_1"},
            "players": [{"agentId": "hero", "stack": 1000}],
        },
    }
    watchman._track_hand_progress(obs1)
    assert watchman.total_hands == 0
    assert watchman.prev_hand_id == "hand_1"

    # Second hand (hand_1 ended with hero stack 1200 -> +200 chips = +2 BB)
    obs2 = {
        "agentId": "hero",
        "table": {
            "bigBlind": 100,
            "hand": {"id": "hand_2"},
            "players": [{"agentId": "hero", "stack": 1200}],
        },
    }
    watchman._track_hand_progress(obs2)
    assert watchman.total_hands == 1
    assert watchman.session_net_bb == 2.0
    assert watchman.prev_hand_id == "hand_2"


def test_watchman_make_action_body():
    client = AgentPokerClient(Config(base_url="https://test.com"))
    watchman = CompetitionWatchman(client=client, competition_id="comp_test")
    obs = {
        "competitionId": "comp_test",
        "table": {"id": "tbl_99"},
        "actionRequest": {
            "id": "act_001",
            "allowedActions": [{"type": "check"}],
        },
    }
    body = watchman._make_action_body(obs)
    assert body["competitionId"] == "comp_test"
    assert body["tableId"] == "tbl_99"
    assert body["actionRequestId"] == "act_001"
    assert "decision" in body


def test_watchman_action_retry_on_409():
    client = AgentPokerClient(Config(base_url="https://test.com"))
    watchman = CompetitionWatchman(client=client, competition_id="comp_test")
    watchman.table_id = "tbl_99"

    # Mock client.action raising 409 stale_action_request
    err = APIError(409, "stale_action_request", "Expired")
    with patch.object(client, "action", side_effect=err):
        with patch.object(watchman, "_observe_current", return_value={"status": "seated"}) as mock_obs:
            body = {"tableId": "tbl_99", "decision": {"type": "fold"}}
            res = watchman._action_with_retry(body, {})
            assert res == {"status": "seated"}
            mock_obs.assert_called_once()
