"""Unit tests for SwissPairing in preliminary rounds."""

import random
import pytest
from src.tournament.state import PlayerRecord
from src.tournament.swiss import SwissPairing
from src.tournament.ranking import RankingEngine


def test_pairing_table_count_and_sizes():
    pairing = SwissPairing(table_size=6)
    players = [PlayerRecord(player_id=i) for i in range(120)]
    rng = random.Random(42)

    # Round 1
    tables = pairing.pair_round(players, round_no=1, rng=rng)
    assert len(tables) == 20
    for table in tables:
        assert len(table) == 6

    # Verify all 120 unique players are present
    assigned_ids = [p.player_id for table in tables for p in table]
    assert len(set(assigned_ids)) == 120


def test_r1_to_r3_random_behavior():
    pairing = SwissPairing(table_size=6)
    players = [PlayerRecord(player_id=i, stage_net_bb=float(i)) for i in range(120)]
    rng = random.Random(123)

    tables_r1 = pairing.pair_round(players, round_no=1, rng=rng)
    tables_r2 = pairing.pair_round(players, round_no=2, rng=rng)

    # R1 and R2 table 0 should not be identical under random shuffle
    ids_t0_r1 = {p.player_id for p in tables_r1[0]}
    ids_t0_r2 = {p.player_id for p in tables_r2[0]}
    assert ids_t0_r1 != ids_t0_r2


def test_r4_to_r10_swiss_performance_grouping():
    ranking = RankingEngine()
    pairing = SwissPairing(table_size=6, ranking_engine=ranking)
    # Player 119 has highest net BB, Player 0 has lowest
    players = [PlayerRecord(player_id=i, stage_net_bb=float(i)) for i in range(120)]
    rng = random.Random(999)

    tables = pairing.pair_round(players, round_no=4, rng=rng)
    assert len(tables) == 20

    # Table 0 should contain top 6 players (IDs 114 to 119)
    table_0_ids = {p.player_id for p in tables[0]}
    expected_top6_ids = {114, 115, 116, 117, 118, 119}
    assert table_0_ids == expected_top6_ids

    # Table 1 should contain next 6 players (IDs 108 to 113)
    table_1_ids = {p.player_id for p in tables[1]}
    expected_next6_ids = {108, 109, 110, 111, 112, 113}
    assert table_1_ids == expected_next6_ids

    # Table 19 should contain bottom 6 players (IDs 0 to 5)
    table_19_ids = {p.player_id for p in tables[19]}
    expected_bottom6_ids = {0, 1, 2, 3, 4, 5}
    assert table_19_ids == expected_bottom6_ids


def test_swiss_reproducibility_with_seed():
    pairing = SwissPairing(table_size=6)
    players1 = [PlayerRecord(player_id=i, stage_net_bb=float(i)) for i in range(120)]
    players2 = [PlayerRecord(player_id=i, stage_net_bb=float(i)) for i in range(120)]

    rng1 = random.Random(42)
    rng2 = random.Random(42)

    tables1 = pairing.pair_round(players1, round_no=5, rng=rng1)
    tables2 = pairing.pair_round(players2, round_no=5, rng=rng2)

    for t1, t2 in zip(tables1, tables2):
        assert [p.player_id for p in t1] == [p.player_id for p in t2]
