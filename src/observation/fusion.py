"""Fusion layer combining Poker state features with Tournament context embeddings."""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple

from src.core.model import PokerNetwork
from src.observation.poker_encoder import PokerEncoder
from src.observation.tournament_encoder import TournamentEncoder, TournamentEmbeddingNet


class TournamentStateFusion:
    """Combines PokerState vector with TournamentState vector into unified inputs."""

    def __init__(self, poker_dim: int = PokerEncoder.FEATURE_DIM, tourn_dim: int = TournamentEncoder.RAW_DIM):
        self.poker_dim = poker_dim
        self.tourn_dim = tourn_dim
        self.total_dim = poker_dim + tourn_dim

    def fuse_vectors(self, poker_vec: np.ndarray, tourn_vec: np.ndarray) -> np.ndarray:
        """Concatenate raw poker and tournament feature vectors."""
        return np.concatenate([poker_vec, tourn_vec], axis=-1)

    def fuse_tensors(self, poker_tensor: torch.Tensor, tourn_tensor: torch.Tensor) -> torch.Tensor:
        """Concatenate poker and tournament tensors along the last dimension."""
        return torch.cat([poker_tensor, tourn_tensor], dim=-1)


class TournamentAwarePokerNetwork(nn.Module):
    """Wrapper architecture integrating Tournament Embedding with existing Deep CFR PokerNetwork.
    
    Data Flow:
    x_tourn -> TournamentEmbeddingNet -> tourn_emb (64-dim)
    [x_poker (156-dim), tourn_emb (64-dim)] -> Fusion Layer -> fused_features (input_size)
    fused_features -> Existing Deep CFR PokerNetwork -> (action_logits, bet_size)
    """

    def __init__(
        self,
        poker_input_size: int = PokerEncoder.FEATURE_DIM,
        tourn_input_size: int = TournamentEncoder.RAW_DIM,
        tourn_emb_dim: int = 64,
        hidden_size: int = 256,
        num_actions: int = 3,
        base_network: Optional[PokerNetwork] = None,
    ):
        super().__init__()
        self.poker_input_size = poker_input_size
        self.tourn_emb_dim = tourn_emb_dim
        
        # 1. Tournament Encoder & Embedding
        self.tourn_net = TournamentEmbeddingNet(
            input_dim=tourn_input_size,
            embedding_dim=tourn_emb_dim,
        )

        # 2. Fusion Adapter Layer (Modulates action policy logits and sizing)
        self.num_actions = num_actions
        self.fusion = nn.Sequential(
            nn.Linear(num_actions + tourn_emb_dim, 64),
            nn.Tanh(),
            nn.Linear(64, num_actions),
        )
        self.fusion[-1].weight.data.zero_()
        self.fusion[-1].bias.data.zero_()

        self.sizing_fusion = nn.Sequential(
            nn.Linear(1 + tourn_emb_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
        self.sizing_fusion[-1].weight.data.zero_()
        self.sizing_fusion[-1].bias.data.zero_()

        self.alpha = nn.Parameter(torch.tensor(1.0))

        # 3. Existing Deep CFR Backbone (reused intact)
        self.base_poker_net = base_network or PokerNetwork(
            input_size=poker_input_size,
            hidden_size=hidden_size,
            num_actions=num_actions,
        )

    def freeze_backbone(self) -> None:
        """Freeze base poker network weights to prevent policy distortion."""
        for param in self.base_poker_net.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze base poker network weights."""
        for param in self.base_poker_net.parameters():
            param.requires_grad = True

    def forward(
        self,
        x_poker: torch.Tensor,
        x_tourn: Optional[torch.Tensor] = None,
        opponent_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through Backbone -> Tournament Embedding -> Policy Modulation."""
        base_logits, base_sizing = self.base_poker_net(x_poker, opponent_features=opponent_features)
        if x_tourn is None:
            # Bypass modulation completely when tournament context is omitted
            return base_logits, base_sizing

        tourn_emb = self.tourn_net(x_tourn)
        delta_logits = self.fusion(torch.cat([base_logits, tourn_emb], dim=-1))
        delta_sizing = self.sizing_fusion(torch.cat([base_sizing, tourn_emb], dim=-1))

        fused_logits = base_logits + self.alpha * delta_logits
        fused_sizing = torch.clamp(base_sizing + delta_sizing, 0.1, 3.0)
        return fused_logits, fused_sizing
