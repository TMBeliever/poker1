"""Unit and integration tests for preliminary stage of TournamentEnv."""

import pytest
from src.agents.random_agent import RandomAgent
from src.tournament.state import Stage
from src.tournament.tournament_env import TournamentEnv


def test_preliminary_round_progression_and_accounting():
    # Run 4 rounds, 2 hands each to test R1-R3 random + R4 Swiss transition quickly
    env = TournamentEnv(
        num_players=120,
        table_size=6,
        prelim_rounds=4,
        hands_per_prelim_round=2,
        seed=42,
    )
    agents = {i: RandomAgent(player_id=0) for i in range(120)}

    prelim_standings = env.run_preliminary(agents)

    assert len(prelim_standings) == 120
    assert env.stage == Stage.PRELIMINARY
    assert env.current_round == 4
    assert env.current_hand == 8  # 4 rounds * 2 hands

    # Check zero-sum invariant across all 120 players
    total_net = sum(p.stage_net_bb for p in prelim_standings)
    assert pytest.approx(total_net, abs=1e-4) == 0.0

    # Top 12 should be strictly ranked
    for i in range(len(prelim_standings) - 1):
        p_curr = prelim_standings[i]
        p_next = prelim_standings[i + 1]
        assert p_curr.stage_net_bb >= p_next.stage_net_bb or (
            p_curr.stage_net_bb == p_next.stage_net_bb and p_curr.rebuy_count <= p_next.rebuy_count
        )


def test_preliminary_state_observation():
    env = TournamentEnv(
        num_players=120,
        table_size=6,
        prelim_rounds=2,
        hands_per_prelim_round=1,
        seed=100,
    )
    agents = {i: RandomAgent(player_id=0) for i in range(120)}
    env.run_preliminary(agents)

    # Inspect state for top player
    top_player_id = env.prelim_standings[0].player_id
    state = env.get_tournament_state(top_player_id)

    assert state.stage == Stage.PRELIMINARY
    assert state.rank == 1
    assert state.cutoff_rank == 12
    assert state.rank_margin_bb >= 0.0
    assert state.cumulative_net_bb == env.players[top_player_id].cumulative_net_bb
