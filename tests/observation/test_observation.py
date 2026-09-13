"""Unit tests for Phase 3 Observation Layer (PokerEncoder, TournamentEncoder, Fusion)."""

import numpy as np
import pytest
import torch
import pokers as pkrs

from src.tournament.state import Stage, TournamentState
from src.observation.poker_encoder import PokerEncoder
from src.observation.tournament_encoder import TournamentEncoder, TournamentEmbeddingNet
from src.observation.fusion import TournamentStateFusion, TournamentAwarePokerNetwork


def test_poker_encoder():
    encoder = PokerEncoder()
    state = pkrs.State.from_seed(n_players=6, button=0, sb=1.0, bb=2.0, stake=200.0, seed=42)

    vec = encoder.encode(state, player_id=0)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (156,)
    assert vec.dtype == np.float64 or vec.dtype == np.float32

    tensor = encoder.encode_tensor(state, player_id=0)
    assert tensor.shape == (1, 156)
    assert tensor.dtype == torch.float32


def test_tournament_encoder():
    ctx = TournamentState(
        stage=Stage.SEMIFINAL,
        hand_no=15,
        hands_remaining=5,
        player_id=3,
        rank=2,
        cutoff_rank=3,
        rank_margin_bb=25.0,
        cumulative_net_bb=40.0,
        rebuy_count=1,
        table_rank=2,
        table_id=1,
        stack_bb=120.0,
        effective_stack_bb=90.0,
        table_strength=10.0,
    )

    vec = TournamentEncoder.encode(ctx)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (13,)
    # Stage is SEMIFINAL (index 1 in [PRELIMINARY, SEMIFINAL, FINAL, FINISHED])
    assert vec[0] == 0.0
    assert vec[1] == 1.0  # SEMIFINAL
    assert vec[2] == 0.0
    assert vec[3] == 0.0

    tensor = TournamentEncoder.encode_tensor(ctx)
    assert tensor.shape == (1, 13)


def test_tournament_embedding_net():
    net = TournamentEmbeddingNet(input_dim=13, embedding_dim=64)
    x = torch.randn(4, 13)
    out = net(x)
    assert out.shape == (4, 64)


def test_fusion_and_tournament_aware_poker_net_forward():
    poker_net = TournamentAwarePokerNetwork(
        poker_input_size=156,
        tourn_input_size=13,
        tourn_emb_dim=64,
        hidden_size=256,
        num_actions=3,
    )

    batch_size = 8
    x_poker = torch.randn(batch_size, 156)
    x_tourn = torch.randn(batch_size, 13)

    logits, sizing = poker_net(x_poker, x_tourn)
    assert logits.shape == (batch_size, 3)
    assert sizing.shape == (batch_size, 1)

    # Bet sizing must be strictly in [0.1, 3.0]
    assert torch.all(sizing >= 0.1)
    assert torch.all(sizing <= 3.0)


def test_backward_gradient_flow_through_tournament_encoder():
    poker_net = TournamentAwarePokerNetwork(
        poker_input_size=156,
        tourn_input_size=13,
        tourn_emb_dim=64,
    )

    x_poker = torch.randn(2, 156, requires_grad=True)
    x_tourn = torch.randn(2, 13, requires_grad=True)

    logits, sizing = poker_net(x_poker, x_tourn)
    loss = logits.sum() + sizing.sum()
    loss.backward()

    # Check gradients in tournament encoder
    for param in poker_net.tourn_net.parameters():
        assert param.grad is not None
        assert not torch.isnan(param.grad).any()

    # Check gradients in fusion layer
    for param in poker_net.fusion.parameters():
        assert param.grad is not None
        assert not torch.isnan(param.grad).any()


def test_forward_without_tournament_context_fallback():
    poker_net = TournamentAwarePokerNetwork()
    x_poker = torch.randn(2, 156)

    # Calling with x_tourn=None should not crash and return valid outputs
    logits, sizing = poker_net(x_poker, x_tourn=None)
    assert logits.shape == (2, 3)
    assert sizing.shape == (2, 1)
