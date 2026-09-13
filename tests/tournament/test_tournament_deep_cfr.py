"""Unit tests for Phase 4 TournamentDeepCFRAgent."""

import os
import numpy as np
import pytest
import pokers as pkrs
import torch

from src.tournament.state import Stage, TournamentState
from src.tournament.tournament_deep_cfr import TournamentDeepCFRAgent, AGENT_TYPE_TOURNAMENT


def test_agent_initialization_and_action_choice():
    agent = TournamentDeepCFRAgent(player_id=0, num_players=6, device="cpu")
    state = pkrs.State.from_seed(n_players=6, button=0, sb=1.0, bb=2.0, stake=200.0, seed=42)

    # 1. Action without tournament context
    act1 = agent.choose_action(state, tournament_context=None)
    assert act1.action in state.legal_actions

    # 2. Action with tournament context
    ctx = TournamentState(
        stage=Stage.PRELIMINARY,
        hand_no=10,
        hands_remaining=190,
        player_id=0,
        rank=5,
        cutoff_rank=12,
        rank_margin_bb=35.0,
        cumulative_net_bb=40.0,
        rebuy_count=0,
        table_rank=1,
        table_id=0,
        stack_bb=140.0,
        effective_stack_bb=90.0,
        table_strength=-10.0,
    )
    act2 = agent.choose_action(state, tournament_context=ctx)
    assert act2.action in state.legal_actions


def test_agent_training_steps():
    agent = TournamentDeepCFRAgent(player_id=0, num_players=6, memory_size=500, device="cpu")
    poker_vec = agent.poker_encoder.encode(pkrs.State.from_seed(6, 0, 1.0, 2.0, 200.0, 1))
    tourn_vec = agent.tourn_encoder.encode(
        TournamentState(
            stage=Stage.SEMIFINAL,
            hand_no=1,
            hands_remaining=19,
            player_id=0,
            rank=2,
            cutoff_rank=3,
            rank_margin_bb=10.0,
            cumulative_net_bb=10.0,
            rebuy_count=0,
            table_rank=2,
            table_id=0,
            stack_bb=110.0,
            effective_stack_bb=100.0,
            table_strength=0.0,
        )
    )

    # Populate memory
    for i in range(70):
        agent.record_advantage_experience(
            poker_vec=poker_vec,
            tourn_vec=tourn_vec,
            action_type=i % 3,
            bet_size=1.0,
            regret=float(i % 5),
        )
        agent.record_strategy_experience(
            poker_vec=poker_vec,
            tourn_vec=tourn_vec,
            action_probs=np.array([0.2, 0.5, 0.3]),
            bet_size=1.0,
        )

    adv_loss = agent.train_advantage_network(batch_size=32)
    strat_loss = agent.train_strategy_network(batch_size=32)

    assert isinstance(adv_loss, float)
    assert adv_loss >= 0.0
    assert isinstance(strat_loss, float)
    assert strat_loss >= 0.0


def test_checkpoint_save_and_load(tmp_path):
    agent = TournamentDeepCFRAgent(player_id=0, num_players=6, device="cpu")
    agent.iteration_count = 42

    save_file = str(tmp_path / "tourn_checkpoint.pt")
    agent.save_checkpoint(save_file)
    assert os.path.exists(save_file)

    # Reload into fresh agent
    new_agent = TournamentDeepCFRAgent(player_id=0, num_players=6, device="cpu")
    new_agent.load_checkpoint(save_file)
    assert new_agent.iteration_count == 42


def test_load_base_checkpoint():
    base_ckpt = "models/phase0_test/checkpoint_iter_2.pt"
    if os.path.exists(base_ckpt):
        agent = TournamentDeepCFRAgent(
            player_id=0,
            num_players=6,
            device="cpu",
            base_checkpoint_path=base_ckpt,
        )
        assert agent is not None
