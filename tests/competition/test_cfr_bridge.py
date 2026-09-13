import numpy as np
import pytest
import torch

from src.competition.cfr_bridge import CFRCompetitionBridge, card_str_to_index
from src.competition.soul import SoulManager


def test_card_str_to_index():
    # '2c' should be 0 * 13 + 0 = 0
    assert card_str_to_index("2c") == 0
    # 'Ac' should be 0 * 13 + 12 = 12
    assert card_str_to_index("Ac") == 12
    # '2d' should be 1 * 13 + 0 = 13
    assert card_str_to_index("2d") == 13
    # 'As' should be 3 * 13 + 12 = 51
    assert card_str_to_index("As") == 51
    # Invalid
    assert card_str_to_index("") == 0
    assert card_str_to_index("xyz") == 0


def test_encode_observation_shape():
    bridge = CFRCompetitionBridge(model_path="non_existent_path.pt")
    mock_obs = {
        "competitionId": "comp_123",
        "agentId": "hero_agent",
        "table": {
            "id": "tbl_456",
            "dealerSeat": 2,
            "bigBlind": 200,
            "initialStake": 20000,
            "hand": {
                "id": "hand_789",
                "pot": 1200,
                "communityCards": ["Ah", "Kd", "2c"],
                "actions": [
                    {"type": "bet", "amount": 200},
                    {"type": "call", "amount": 200},
                ],
            },
            "players": [
                {
                    "seat": 0,
                    "agentId": "hero_agent",
                    "stack": 19600,
                    "handState": {
                        "holeCards": ["As", "Ks"],
                        "currentBet": 200,
                        "status": "active",
                    },
                },
                {
                    "seat": 1,
                    "agentId": "villain_1",
                    "stack": 18000,
                    "handState": {
                        "holeCards": [],
                        "currentBet": 200,
                        "status": "active",
                    },
                },
            ],
        },
        "actionRequest": {
            "id": "act_req_01",
            "allowedActions": [
                {"type": "check"},
                {"type": "bet", "minAmount": 200, "maxAmount": 19600},
            ],
        },
    }

    feat = bridge.encode_observation(mock_obs)
    assert isinstance(feat, np.ndarray)
    assert feat.shape == (156,)
    # Check hole cards encoded: 'As' (51) and 'Ks' (50) should be 1.0
    assert feat[card_str_to_index("As")] == 1.0
    assert feat[card_str_to_index("Ks")] == 1.0


def test_choose_action_clamping():
    bridge = CFRCompetitionBridge(model_path="non_existent_path.pt")
    mock_obs = {
        "competitionId": "comp_123",
        "agentId": "hero_agent",
        "table": {
            "id": "tbl_456",
            "dealerSeat": 0,
            "bigBlind": 200,
            "initialStake": 20000,
            "hand": {
                "id": "hand_789",
                "pot": 600,
                "communityCards": ["Th", "9h", "8h"],
            },
            "players": [
                {
                    "seat": 0,
                    "agentId": "hero_agent",
                    "stack": 19400,
                    "handState": {
                        "holeCards": ["Ah", "Kh"],
                        "currentBet": 0,
                        "status": "active",
                    },
                }
            ],
        },
        "actionRequest": {
            "id": "act_req_02",
            "allowedActions": [
                {"type": "check"},
                {"type": "bet", "minAmount": 200, "maxAmount": 5000},
            ],
        },
    }

    res = bridge.choose_action(mock_obs)
    assert "decision" in res
    dec = res["decision"]
    assert dec["type"] in ("check", "bet")
    if dec["type"] == "bet":
        assert 200 <= dec["amount"] <= 5000


def test_soul_manager(tmp_path):
    soul_file = tmp_path / "SOUL.md"
    sm = SoulManager(soul_path=soul_file)
    assert soul_file.exists()
    assert "ApexPoker" in sm.content

    # Test chat generation length constraint <= 140 chars
    for act in ["raise", "call", "fold", "check", "allIn"]:
        for _ in range(20):
            chat = sm.generate_chat(act, street="turn")
            if chat:
                assert len(chat) <= 140
