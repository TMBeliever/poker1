"""Unit and integration tests for semifinal stage and snake seeding."""

import pytest
from src.agents.random_agent import RandomAgent
from src.tournament.state import Stage, PlayerRecord
from src.tournament.tournament_env import TournamentEnv
from src.tournament.advancement import AdvancementManager


def test_snake_seeding_exact_mapping():
    mgr = AdvancementManager()
    # Create 12 players with ranks 1..12
    players = [
        PlayerRecord(player_id=i, stage_net_bb=float(100 - i))
        for i in range(1, 13)
    ]
    # Ranks: player 1 is rank 1, player 2 is rank 2, ..., player 12 is rank 12
    table_a, table_b = mgr.advance_preliminary_to_semifinal(players)

    # A: 1, 4, 5, 8, 9, 12
    assert [p.player_id for p in table_a] == [1, 4, 5, 8, 9, 12]
    # B: 2, 3, 6, 7, 10, 11
    assert [p.player_id for p in table_b] == [2, 3, 6, 7, 10, 11]

    # Verify 100 BB reset
    for p in table_a + table_b:
        assert p.stack_bb == 100.0
        assert p.stage_net_bb == 0.0
        assert p.rebuy_count == 0


def test_semifinal_round_execution():
    env = TournamentEnv(
        num_players=120,
        prelim_rounds=1,
        hands_per_prelim_round=1,
        semifinal_hands=4,
        seed=42,
    )
    agents = {i: RandomAgent(player_id=0) for i in range(120)}
    env.run_preliminary(agents)
    standings_a, standings_b = env.run_semifinal(agents)

    assert len(standings_a) == 6
    assert len(standings_b) == 6
    assert env.stage == Stage.SEMIFINAL

    # Check zero-sum per table
    net_a = sum(p.stage_net_bb for p in standings_a)
    net_b = sum(p.stage_net_bb for p in standings_b)
    assert pytest.approx(net_a, abs=1e-4) == 0.0
    assert pytest.approx(net_b, abs=1e-4) == 0.0


def test_semifinal_to_final_advancement():
    mgr = AdvancementManager()
    table_a = [PlayerRecord(player_id=i, stage_net_bb=float(i * 10)) for i in range(1, 7)]
    table_b = [PlayerRecord(player_id=i, stage_net_bb=float(i * 10)) for i in range(7, 13)]

    # Top 3 from A: IDs 6, 5, 4 (net 60, 50, 40)
    # Top 3 from B: IDs 12, 11, 10 (net 120, 110, 100)
    final_table = mgr.advance_semifinal_to_final(table_a, table_b)

    assert len(final_table) == 6
    final_ids = {p.player_id for p in final_table}
    assert final_ids == {6, 5, 4, 12, 11, 10}

    # Verify 100 BB reset for final
    for p in final_table:
        assert p.stack_bb == 100.0
        assert p.stage_net_bb == 0.0
