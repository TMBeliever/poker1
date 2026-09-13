"""Comprehensive automated rule validation suite for ApexPoker tournament rules.

Systematically validates all 18 official tournament invariants specified in the problem definition:
1. 初始人数 = 120
2. 桌数 = 20
3. 每桌 = 6
4. 预赛 = 200 hands
5. R1-R3 random
6. R4-R10 Swiss
7. Top12 qualification
8. Snake seeding (A=[1,4,5,8,9,12], B=[2,3,6,7,10,11])
9. 半决赛 2 tables
10. 每桌 20 hands in semifinal
11. Top3 each advance from Table A and Table B
12. 最终 6 players
13. Final = 30 hands
14. Champion = rank 1 in final
15. tie-break hierarchy
16. rebuy triggers and replenishment
17. score accounting & zero-sum invariance
18. stage reset to 100 BB
"""

import random
import pytest
from src.agents.random_agent import RandomAgent
from src.tournament.state import Stage, PlayerRecord
from src.tournament.rebuy import RebuyManager
from src.tournament.ranking import RankingEngine
from src.tournament.swiss import SwissPairing
from src.tournament.advancement import AdvancementManager
from src.tournament.tournament_env import TournamentEnv


def test_rule_1_2_3_player_and_table_counts():
    """Rules 1, 2, 3: Initial players = 120, Tables = 20, Table size = 6."""
    env = TournamentEnv(num_players=120, table_size=6, seed=42)
    assert len(env.players) == 120
    assert len(env.active_players) == 120
    
    # Check pairing yields exactly 20 tables with 6 players each
    tables = env.swiss_pairing.pair_round(env.active_players, round_no=1, rng=env.rng)
    assert len(tables) == 20
    for table in tables:
        assert len(table) == 6


def test_rule_4_preliminary_200_hands():
    """Rule 4: Preliminary stage is exactly 200 hands (10 rounds * 20 hands)."""
    env = TournamentEnv(num_players=120, table_size=6, seed=42)
    assert env.prelim_rounds == 10
    assert env.hands_per_prelim_round == 20
    total_prelim_hands = env.prelim_rounds * env.hands_per_prelim_round
    assert total_prelim_hands == 200


def test_rule_5_and_6_r1_r3_random_and_r4_r10_swiss():
    """Rules 5 & 6: R1-R3 random pairing, R4-R10 Swiss pairing."""
    ranking = RankingEngine()
    swiss = SwissPairing(table_size=6, ranking_engine=ranking)
    # Create 120 players with descending net profit: Player 0 has highest, 119 has lowest
    players = [PlayerRecord(player_id=i, stage_net_bb=float(120 - i)) for i in range(120)]
    rng = random.Random(42)

    # R1, R2, R3 should shuffle randomly (top players not strictly grouped)
    tables_r1 = swiss.pair_round(players, round_no=1, rng=rng)
    table_0_ids_r1 = {p.player_id for p in tables_r1[0]}
    assert table_0_ids_r1 != {0, 1, 2, 3, 4, 5}  # Random, not top 6

    # R4, R5, ..., R10 must group top 6 players at Table 0
    tables_r4 = swiss.pair_round(players, round_no=4, rng=rng)
    table_0_ids_r4 = {p.player_id for p in tables_r4[0]}
    assert table_0_ids_r4 == {0, 1, 2, 3, 4, 5}

    table_1_ids_r4 = {p.player_id for p in tables_r4[1]}
    assert table_1_ids_r4 == {6, 7, 8, 9, 10, 11}


def test_rule_7_and_8_top12_and_snake_seeding():
    """Rules 7 & 8: Top 12 qualify and are snake-seeded into Table A and Table B.
    
    Table A = 1, 4, 5, 8, 9, 12
    Table B = 2, 3, 6, 7, 10, 11
    """
    adv = AdvancementManager()
    # 120 players with rank equal to player_id (Player 1 has +1000, Player 120 has -1000)
    players = [
        PlayerRecord(player_id=i, stage_net_bb=float(1000 - i * 10))
        for i in range(1, 121)
    ]
    table_a, table_b = adv.advance_preliminary_to_semifinal(players)

    # 1-indexed ranks match player_id
    assert [p.player_id for p in table_a] == [1, 4, 5, 8, 9, 12]
    assert [p.player_id for p in table_b] == [2, 3, 6, 7, 10, 11]


def test_rule_9_and_10_semifinal_tables_and_hands():
    """Rules 9 & 10: Semifinal has exactly 2 tables of 6 and 20 hands per table."""
    env = TournamentEnv(num_players=120, seed=42)
    assert env.semifinal_hands == 20
    
    # Fast test of semifinal stage structure
    env.prelim_standings = [
        PlayerRecord(player_id=i, stage_net_bb=float(1000 - i * 10))
        for i in range(1, 121)
    ]
    env.semifinal_hands = 2  # Smoke run with 2 hands
    agents = {i: RandomAgent(player_id=0) for i in range(1, 121)}
    semi_a, semi_b = env.run_semifinal(agents)

    assert len(semi_a) == 6
    assert len(semi_b) == 6
    assert len(env.current_tables) == 2


def test_rule_11_and_12_top3_each_and_final_6_players():
    """Rules 11 & 12: Top 3 from each semifinal table advance to a 6-player final."""
    adv = AdvancementManager()
    table_a = [PlayerRecord(player_id=i, stage_net_bb=float(i * 10)) for i in range(1, 7)]
    table_b = [PlayerRecord(player_id=i, stage_net_bb=float(i * 10)) for i in range(7, 13)]

    final_table = adv.advance_semifinal_to_final(table_a, table_b)
    assert len(final_table) == 6
    # Table A top 3: 6, 5, 4; Table B top 3: 12, 11, 10
    final_ids = {p.player_id for p in final_table}
    assert final_ids == {6, 5, 4, 12, 11, 10}


def test_rule_13_and_14_final_hands_and_champion():
    """Rules 13 & 14: Final is 30 hands, and net BB rank 1 is Champion."""
    env = TournamentEnv(num_players=120, seed=42)
    assert env.final_hands == 30

    adv = AdvancementManager()
    final_table = [
        PlayerRecord(player_id=10, stage_net_bb=50.0),
        PlayerRecord(player_id=20, stage_net_bb=180.0),  # Winner
        PlayerRecord(player_id=30, stage_net_bb=-30.0),
        PlayerRecord(player_id=40, stage_net_bb=-80.0),
        PlayerRecord(player_id=50, stage_net_bb=-20.0),
        PlayerRecord(player_id=60, stage_net_bb=-100.0),
    ]
    champ, standings = adv.determine_champion(final_table)
    assert champ.player_id == 20
    assert standings[0].player_id == 20


def test_rule_15_tie_break_hierarchy():
    """Rule 15: Deterministic tie-break hierarchy (Net BB -> Rebuys -> Hands -> Player ID)."""
    ranking = RankingEngine()
    
    # Case 1: Same net BB, fewer rebuys wins
    p_a = PlayerRecord(player_id=1, stage_net_bb=100.0, rebuy_count=1)
    p_b = PlayerRecord(player_id=2, stage_net_bb=100.0, rebuy_count=0)
    assert ranking.rank_players([p_a, p_b])[0].player_id == 2

    # Case 2: Same net BB & rebuys, more hands played wins
    p_c = PlayerRecord(player_id=3, stage_net_bb=100.0, rebuy_count=0, stage_hands_played=50)
    p_d = PlayerRecord(player_id=4, stage_net_bb=100.0, rebuy_count=0, stage_hands_played=60)
    assert ranking.rank_players([p_c, p_d])[0].player_id == 4

    # Case 3: All equal, lower player_id wins
    p_e = PlayerRecord(player_id=5, stage_net_bb=100.0, rebuy_count=0, stage_hands_played=60)
    p_f = PlayerRecord(player_id=8, stage_net_bb=100.0, rebuy_count=0, stage_hands_played=60)
    assert ranking.rank_players([p_f, p_e])[0].player_id == 5


def test_rule_16_and_17_rebuy_and_score_accounting():
    """Rules 16 & 17: Auto-rebuy on bust and zero-sum conservation of net BB."""
    rebuy_mgr = RebuyManager(starting_stack_bb=100.0)
    players = [PlayerRecord(player_id=i, stack_bb=100.0) for i in range(6)]

    # Hand with large swing: Player 0 wins +200 BB, Player 1 loses -100 BB (bust), Player 2 loses -100 BB (bust)
    deltas = [200.0, -100.0, -100.0, 0.0, 0.0, 0.0]
    for p, d in zip(players, deltas):
        rebuy_mgr.apply_hand_result(p, d)

    # Players 1 and 2 must have rebought
    assert players[1].rebuy_count == 1
    assert players[2].rebuy_count == 1
    assert players[1].stack_bb == 100.0
    assert players[2].stack_bb == 100.0

    # Net profits must reflect loss
    assert players[1].stage_net_bb == -100.0
    assert players[2].stage_net_bb == -100.0
    assert players[0].stage_net_bb == 200.0

    # Zero-sum property holds strictly
    assert pytest.approx(sum(p.stage_net_bb for p in players), abs=1e-5) == 0.0


def test_rule_18_stage_reset_100bb():
    """Rule 18: Stacks are reset to exactly 100 BB at semifinal and final boundaries."""
    adv = AdvancementManager()
    prelim_players = [
        PlayerRecord(player_id=i, stack_bb=float(300 + i * 20), stage_net_bb=float(i * 50))
        for i in range(12)
    ]
    table_a, table_b = adv.advance_preliminary_to_semifinal(prelim_players)

    for p in table_a + table_b:
        assert p.stack_bb == 100.0
        assert p.stage_net_bb == 0.0
        assert p.rebuy_count == 0

    final_table = adv.advance_semifinal_to_final(table_a, table_b)
    for p in final_table:
        assert p.stack_bb == 100.0
        assert p.stage_net_bb == 0.0
        assert p.rebuy_count == 0
