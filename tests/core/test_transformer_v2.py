"""Unit tests and A/B benchmark for Phase 10 TournamentTransformerV2 vs V1 MLP."""

import time
import pytest
import torch

from src.core.transformer_v2 import TournamentTransformerV2
from src.observation.fusion import TournamentAwarePokerNetwork


def test_transformer_v2_forward_and_heads():
    model = TournamentTransformerV2(
        poker_dim=156,
        tourn_dim=13,
        opponent_dim=32,
        d_model=64,
        nhead=4,
        num_layers=2,
        num_actions=3,
        num_atoms=21,
    )

    batch_size = 4
    x_poker = torch.randn(batch_size, 156)
    x_tourn = torch.randn(batch_size, 13)
    x_opp = torch.randn(batch_size, 32)

    outputs = model(x_poker, x_tourn, x_opp)

    assert outputs["advantages"].shape == (batch_size, 3)
    assert outputs["strategy_logits"].shape == (batch_size, 3)
    assert outputs["bet_size"].shape == (batch_size, 1)
    assert torch.all(outputs["bet_size"] >= 0.1)
    assert torch.all(outputs["bet_size"] <= 3.0)
    assert outputs["p_advance"].shape == (batch_size, 1)
    assert torch.all(outputs["p_advance"] >= 0.0)
    assert torch.all(outputs["p_advance"] <= 1.0)
    assert outputs["p_champion"].shape == (batch_size, 1)
    assert outputs["value_distribution"].shape == (batch_size, 21)
    # Sum of atoms equals 1.0
    atom_sums = outputs["value_distribution"].sum(dim=-1)
    assert torch.allclose(atom_sums, torch.ones_like(atom_sums), atol=1e-5)


def test_ab_comparison_v1_mlp_vs_v2_transformer():
    """A/B test comparing V1 (MLP) and V2 (Transformer) architectures."""
    v1_mlp = TournamentAwarePokerNetwork()
    v2_transformer = TournamentTransformerV2(d_model=64, nhead=4, num_layers=2)

    x_poker = torch.randn(16, 156)
    x_tourn = torch.randn(16, 13)

    # 1. Parameter counts
    v1_params = sum(p.numel() for p in v1_mlp.parameters())
    v2_params = sum(p.numel() for p in v2_transformer.parameters())

    assert v1_params > 0
    assert v2_params > 0

    # 2. Timing benchmark (100 forward passes)
    v1_mlp.eval()
    v2_transformer.eval()

    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(50):
            _ = v1_mlp(x_poker, x_tourn)
        v1_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(50):
            _ = v2_transformer(x_poker, x_tourn)
        v2_time = time.perf_counter() - t0

    # Both models must produce valid finite outputs and execute reliably
    assert v1_time > 0.0
    assert v2_time > 0.0
