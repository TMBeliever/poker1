"""Deterministic full-tournament end-to-end regression tests."""

import pytest
from src.agents.random_agent import RandomAgent
from src.tournament.state import Stage
from src.tournament.tournament_env import TournamentEnv


def test_deterministic_full_tournament_execution():
    # Deterministic tournament run
    env1 = TournamentEnv(
        num_players=120,
        table_size=6,
        prelim_rounds=2,
        hands_per_prelim_round=2,
        semifinal_hands=2,
        final_hands=3,
        seed=12345,
    )
    agents1 = {i: RandomAgent(player_id=0) for i in range(120)}
    res1 = env1.run_full_tournament(agents1)

    assert env1.stage == Stage.FINISHED
    assert res1["total_prelim_players"] == 120
    assert len(res1["preliminary_top12"]) == 12
    assert len(res1["semifinal_table_a"]) == 6
    assert len(res1["semifinal_table_b"]) == 6
    assert len(res1["final_standings"]) == 6
    assert res1["champion"] == res1["final_standings"][0]

    # Run again with identical seed -> must produce bit-for-bit identical results
    env2 = TournamentEnv(
        num_players=120,
        table_size=6,
        prelim_rounds=2,
        hands_per_prelim_round=2,
        semifinal_hands=2,
        final_hands=3,
        seed=12345,
    )
    agents2 = {i: RandomAgent(player_id=0) for i in range(120)}
    res2 = env2.run_full_tournament(agents2)

    assert res1["champion"].player_id == res2["champion"].player_id
    assert pytest.approx(res1["champion"].stage_net_bb, abs=1e-5) == res2["champion"].stage_net_bb
    for p1, p2 in zip(res1["final_standings"], res2["final_standings"]):
        assert p1.player_id == p2.player_id
        assert pytest.approx(p1.stage_net_bb, abs=1e-5) == p2.stage_net_bb


def test_seed_sensitivity():
    # Different seed should generate a different run
    env1 = TournamentEnv(
        num_players=120,
        prelim_rounds=2,
        hands_per_prelim_round=2,
        semifinal_hands=2,
        final_hands=3,
        seed=111,
    )
    env2 = TournamentEnv(
        num_players=120,
        prelim_rounds=2,
        hands_per_prelim_round=2,
        semifinal_hands=2,
        final_hands=3,
        seed=999,
    )
    agents = {i: RandomAgent(player_id=0) for i in range(120)}
    res1 = env1.run_full_tournament(agents)
    res2 = env2.run_full_tournament(agents)

    # Prelim rank 1 should almost certainly differ or have different net BB
    top_1 = res1["preliminary_top12"][0]
    top_2 = res2["preliminary_top12"][0]
    assert (top_1.player_id != top_2.player_id) or (top_1.stage_net_bb != top_2.stage_net_bb)
