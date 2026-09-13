"""Unit tests for RankingEngine and deterministic tie-break hierarchy."""

import pytest
from src.tournament.state import PlayerRecord, Stage
from src.tournament.ranking import RankingEngine


def test_ranking_by_net_bb():
    engine = RankingEngine()
    p1 = PlayerRecord(player_id=1, stage_net_bb=50.0)
    p2 = PlayerRecord(player_id=2, stage_net_bb=150.0)
    p3 = PlayerRecord(player_id=3, stage_net_bb=-20.0)

    ranked = engine.rank_players([p1, p2, p3])
    assert [p.player_id for p in ranked] == [2, 1, 3]


def test_tie_break_fewer_rebuys():
    engine = RankingEngine()
    # Same net BB (+50.0), but p1 has 0 rebuys while p2 has 1 rebuy
    p1 = PlayerRecord(player_id=1, stage_net_bb=50.0, rebuy_count=0)
    p2 = PlayerRecord(player_id=2, stage_net_bb=50.0, rebuy_count=1)

    ranked = engine.rank_players([p2, p1])
    assert [p.player_id for p in ranked] == [1, 2]


def test_tie_break_hands_played():
    engine = RankingEngine()
    # Same net BB and rebuys, but p2 played 100 hands vs p1 played 80
    p1 = PlayerRecord(player_id=1, stage_net_bb=50.0, rebuy_count=1, stage_hands_played=80)
    p2 = PlayerRecord(player_id=2, stage_net_bb=50.0, rebuy_count=1, stage_hands_played=100)

    ranked = engine.rank_players([p1, p2])
    assert [p.player_id for p in ranked] == [2, 1]


def test_tie_break_deterministic_player_id():
    engine = RankingEngine()
    # Identical stats, player_id 5 vs 10 -> player 5 wins
    p5 = PlayerRecord(player_id=5, stage_net_bb=0.0, rebuy_count=0, stage_hands_played=20)
    p10 = PlayerRecord(player_id=10, stage_net_bb=0.0, rebuy_count=0, stage_hands_played=20)

    ranked = engine.rank_players([p10, p5])
    assert [p.player_id for p in ranked] == [5, 10]


def test_cutoff_and_margin_computation():
    engine = RankingEngine()
    players = [
        PlayerRecord(player_id=i, stage_net_bb=float(100 - i * 10))
        for i in range(1, 21)
    ]
    # Ranks: player 1 has +90, player 12 has -20, player 13 has -30
    standings = engine.compute_standings_and_margins(players, stage=Stage.PRELIMINARY)

    assert standings[1]["rank"] == 1
    assert standings[1]["cutoff_rank"] == 12
    # Reference cutoff is player 12 net BB (-20)
    # Player 1 margin = 90 - (-20) = 110.0
    assert standings[1]["rank_margin_bb"] == pytest.approx(110.0)

    assert standings[12]["rank"] == 12
    assert standings[12]["rank_margin_bb"] == pytest.approx(0.0)

    assert standings[13]["rank"] == 13
    # Player 13 margin = -30 - (-20) = -10.0
    assert standings[13]["rank_margin_bb"] == pytest.approx(-10.0)


def test_table_rank():
    engine = RankingEngine()
    table_players = [
        PlayerRecord(player_id=10, stage_net_bb=-10.0),
        PlayerRecord(player_id=11, stage_net_bb=40.0),
        PlayerRecord(player_id=12, stage_net_bb=20.0),
    ]
    table_ranks = engine.compute_table_rank(table_players)
    assert table_ranks[11] == 1
    assert table_ranks[12] == 2
    assert table_ranks[10] == 3
