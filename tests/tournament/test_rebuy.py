"""Unit tests for RebuyManager and chip accounting."""

import pytest
from src.tournament.state import PlayerRecord
from src.tournament.rebuy import RebuyManager


def test_initial_state_and_profit():
    mgr = RebuyManager(starting_stack_bb=100.0)
    p = PlayerRecord(player_id=1, stack_bb=100.0)
    assert mgr.compute_net_bb(p) == 0.0
    assert p.rebuy_count == 0


def test_positive_and_negative_hand_deltas():
    mgr = RebuyManager(starting_stack_bb=100.0)
    p = PlayerRecord(player_id=1, stack_bb=100.0)

    # Win 35 BB
    rebought = mgr.apply_hand_result(p, 35.0)
    assert not rebought
    assert p.stack_bb == 135.0
    assert p.rebuy_count == 0
    assert p.stage_net_bb == 35.0

    # Lose 20 BB
    rebought = mgr.apply_hand_result(p, -20.0)
    assert not rebought
    assert p.stack_bb == 115.0
    assert p.rebuy_count == 0
    assert p.stage_net_bb == 15.0


def test_auto_rebuy_trigger_and_accounting():
    mgr = RebuyManager(starting_stack_bb=100.0)
    p = PlayerRecord(player_id=1, stack_bb=100.0)

    # Lose full 100 BB stack -> bust
    rebought = mgr.apply_hand_result(p, -100.0)
    assert rebought is True
    assert p.rebuy_count == 1
    # Stack replenished back to 100 BB
    assert p.stack_bb == 100.0
    # Net BB should be -100.0
    assert p.stage_net_bb == -100.0
    assert p.cumulative_net_bb == -100.0

    # Win 50 BB after rebuy
    rebought = mgr.apply_hand_result(p, 50.0)
    assert not rebought
    assert p.stack_bb == 150.0
    assert p.rebuy_count == 1
    # Net BB: (150 - 100) - (1 * 100) = -50.0
    assert p.stage_net_bb == -50.0


def test_multiple_rebuys():
    mgr = RebuyManager(starting_stack_bb=100.0)
    p = PlayerRecord(player_id=2, stack_bb=100.0)

    for i in range(3):
        # Bust each time
        rebought = mgr.apply_hand_result(p, -100.0)
        assert rebought is True

    assert p.rebuy_count == 3
    assert p.stack_bb == 100.0
    assert p.stage_net_bb == -300.0


def test_zero_sum_accounting_invariance():
    mgr = RebuyManager(starting_stack_bb=100.0)
    players = [PlayerRecord(player_id=i, stack_bb=100.0) for i in range(6)]

    # Hand 1: Player 0 wins 100 BB, Player 1 loses 100 BB (busts), others 0
    deltas = [100.0, -100.0, 0.0, 0.0, 0.0, 0.0]
    for p, d in zip(players, deltas):
        mgr.apply_hand_result(p, d)

    assert players[1].rebuy_count == 1
    total_net = sum(p.stage_net_bb for p in players)
    assert pytest.approx(total_net, abs=1e-5) == 0.0
