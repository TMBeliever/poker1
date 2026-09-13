"""Unit tests for Phase 9 HighLeverageReplayBuffer."""

import numpy as np
import pytest
from src.training.counterfactual_replay import HighLeverageExperience, HighLeverageReplayBuffer


def test_high_leverage_priority_computation():
    buffer = HighLeverageReplayBuffer(capacity=1000)

    # Standard experience
    exp_normal = HighLeverageExperience(
        poker_vec=np.zeros(156),
        tourn_vec=None,
        action_type=1,
        bet_size=1.0,
        regret=1.0,
        pot_size_bb=10.0,
        stack_bb=100.0,
        is_bubble=False,
    )
    mult_normal = buffer.compute_leverage_multiplier(exp_normal)
    assert mult_normal == 1.0

    # High-leverage bubble experience
    exp_bubble = HighLeverageExperience(
        poker_vec=np.zeros(156),
        tourn_vec=None,
        action_type=1,
        bet_size=1.0,
        regret=1.0,
        pot_size_bb=80.0,
        stack_bb=15.0,
        is_bubble=True,
        is_late_stage=True,
    )
    mult_bubble = buffer.compute_leverage_multiplier(exp_bubble)
    assert mult_bubble > 3.0


def test_buffer_add_and_prioritized_sample():
    buffer = HighLeverageReplayBuffer(capacity=500)

    for i in range(50):
        exp = HighLeverageExperience(
            poker_vec=np.zeros(156),
            tourn_vec=None,
            action_type=i % 3,
            bet_size=1.0,
            regret=float(i * 0.1),
            is_bubble=(i > 40),
        )
        buffer.add(exp)

    samples, indices, weights = buffer.sample(batch_size=16)
    assert len(samples) == 16
    assert len(indices) == 16
    assert len(weights) == 16
    assert weights.max() == pytest.approx(1.0)
