"""Unit tests for Phase 5 TournamentRewardCalculator."""

import pytest
from src.tournament.state import Stage
from src.training.tournament_reward import TournamentRewardCalculator, TournamentRewardBreakdown


def test_reward_component_independence():
    calc = TournamentRewardCalculator()
    res = calc.calculate(
        stage=Stage.PRELIMINARY,
        hand_delta_bb=25.0,
        current_rank=5,
        cutoff_rank=12,
        rank_margin_bb=30.0,
        cumulative_net_bb=80.0,
        hands_remaining=10,
        is_stage_terminal=False,
    )

    assert isinstance(res, TournamentRewardBreakdown)
    assert res.hand_reward == pytest.approx(0.25)
    assert res.net_bb == 80.0
    assert res.rank == 5
    assert res.rank_margin == 30.0
    assert res.final_reward == 0.0

    d = res.to_dict()
    assert "hand_reward" in d
    assert "stage_reward" in d
    assert "final_reward" in d
    assert "net_bb" in d
    assert "rank" in d
    assert "rank_margin" in d
    assert "total_reward" in d


def test_preliminary_qualification_bonus():
    calc = TournamentRewardCalculator(top12_bonus=1.0)
    # Qualified (rank 8 <= 12)
    res_qual = calc.calculate(
        stage=Stage.PRELIMINARY,
        hand_delta_bb=0.0,
        current_rank=8,
        cutoff_rank=12,
        rank_margin_bb=15.0,
        cumulative_net_bb=50.0,
        hands_remaining=0,
        is_stage_terminal=True,
    )
    assert res_qual.stage_reward == 1.0

    # Missed qualification (rank 15 > 12)
    res_elim = calc.calculate(
        stage=Stage.PRELIMINARY,
        hand_delta_bb=0.0,
        current_rank=15,
        cutoff_rank=12,
        rank_margin_bb=-20.0,
        cumulative_net_bb=-40.0,
        hands_remaining=0,
        is_stage_terminal=True,
    )
    assert res_elim.stage_reward == -0.5


def test_semifinal_top3_bonus():
    calc = TournamentRewardCalculator(top3_bonus=2.0)
    # Top 3 qualifier
    res_top3 = calc.calculate(
        stage=Stage.SEMIFINAL,
        hand_delta_bb=10.0,
        current_rank=2,
        cutoff_rank=3,
        rank_margin_bb=20.0,
        cumulative_net_bb=30.0,
        hands_remaining=0,
        is_stage_terminal=True,
    )
    assert res_top3.stage_reward == 2.0


def test_final_champion_bonus():
    calc = TournamentRewardCalculator(champion_bonus=5.0)
    # Champion (rank 1)
    res_champ = calc.calculate(
        stage=Stage.FINAL,
        hand_delta_bb=50.0,
        current_rank=1,
        cutoff_rank=1,
        rank_margin_bb=40.0,
        cumulative_net_bb=150.0,
        hands_remaining=0,
        is_stage_terminal=True,
    )
    assert res_champ.final_reward == 5.0

    # Runner-up (rank 2)
    res_runner_up = calc.calculate(
        stage=Stage.FINAL,
        hand_delta_bb=-20.0,
        current_rank=2,
        cutoff_rank=1,
        rank_margin_bb=-40.0,
        cumulative_net_bb=40.0,
        hands_remaining=0,
        is_stage_terminal=True,
    )
    assert res_runner_up.final_reward < 5.0
