"""Unit and integration tests for Phase 7 Opponent Modeling and Phase 8 League."""

import pytest
import pokers as pkrs
import torch

from src.agents.archetypes import (
    AdaptiveRegAgent,
    CallingStationAgent,
    LAGAgent,
    ManiacAgent,
    NitAgent,
    OverblufferAgent,
    OverfolderAgent,
    TAGAgent,
)
from src.league.league import LeagueManager
from src.opponent_modeling.dynamic_opponent_model import DynamicOpponentEncoder, OpponentTracker


def test_archetypes_choose_legal_actions():
    state = pkrs.State.from_seed(n_players=6, button=0, sb=1.0, bb=2.0, stake=200.0, seed=42)
    archetypes = [
        NitAgent(0),
        TAGAgent(1),
        LAGAgent(2),
        CallingStationAgent(3),
        ManiacAgent(4),
        OverfolderAgent(5),
        OverblufferAgent(0),
        AdaptiveRegAgent(1),
    ]

    for agent in archetypes:
        action = agent.choose_action(state)
        assert action.action in state.legal_actions


def test_opponent_tracker_and_dynamic_encoder():
    tracker = OpponentTracker()
    tracker.record_hand_start(player_id=1)

    # Record some actions
    tracker.record_action(player_id=1, action=pkrs.Action(pkrs.ActionEnum.Raise), is_preflop=True)
    tracker.record_action(player_id=1, action=pkrs.Action(pkrs.ActionEnum.Fold), is_facing_raise=True)

    stats = tracker.get_or_create_stats(player_id=1)
    assert stats.hands_tracked == 1
    assert stats.vpip == 1.0
    assert stats.pfr == 1.0
    assert stats.fold_rate == 0.5

    vec = tracker.get_summary_vector(player_id=1)
    assert vec.shape == (7,)

    # Dynamic encoder forward
    encoder = DynamicOpponentEncoder(gru_output_dim=64, stats_dim=7, embedding_dim=32)
    action_seq = torch.randn(2, 5, 4)
    state_ctx = torch.randn(2, 5, 20)
    stats_tensor = torch.randn(2, 7)

    emb = encoder(action_seq, state_ctx, stats_tensor)
    assert emb.shape == (2, 32)


def test_league_manager_population_tournament(tmp_path):
    save_dir = str(tmp_path / "league")
    league = LeagueManager(save_dir=save_dir, seed=42)

    assert len(league.archetypes) == 8

    # Run league tournament
    results = league.run_league_tournament(
        prelim_rounds=2,
        hands_per_round=2,
        seed=101,
    )

    assert "main_agent_metrics" in results
    assert results["total_prelim_players"] == 120
    assert len(results["final_standings"]) == 6
    assert results["champion"] is not None
