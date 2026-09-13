"""Tournament state feature encoder and embedding network."""

import math
import numpy as np
import torch
import torch.nn as nn
from src.tournament.state import Stage, TournamentState


class TournamentEncoder:
    """Encodes TournamentState into a normalized feature vector.
    
    Feature schema (13 dimensions):
    - stage: 4-dim one-hot (PRELIMINARY, SEMIFINAL, FINAL, FINISHED)
    - hands_remaining: 1-dim normalized by 200.0
    - rank: 1-dim normalized by 120.0
    - cutoff_rank: 1-dim normalized by 120.0
    - rank_margin: 1-dim non-linear tanh normalized by 100.0 BB
    - cumulative_net_bb: 1-dim non-linear tanh normalized by 200.0 BB
    - rebuy_count: 1-dim normalized by 10.0
    - table_strength: 1-dim non-linear tanh normalized by 200.0 BB
    - effective_stack: 1-dim normalized by 100.0 BB
    - table_rank: 1-dim normalized by 6.0
    """

    RAW_DIM = 13

    @staticmethod
    def encode(context: TournamentState) -> np.ndarray:
        """Convert a TournamentState instance into a normalized numpy vector."""
        # 1. Stage one-hot (4-dim)
        stage_enc = np.zeros(4, dtype=np.float32)
        stage_map = {
            Stage.PRELIMINARY: 0,
            Stage.SEMIFINAL: 1,
            Stage.FINAL: 2,
            Stage.FINISHED: 3,
        }
        stage_idx = stage_map.get(context.stage, 0)
        stage_enc[stage_idx] = 1.0

        # 2. Continuous indicators
        hands_norm = np.clip(context.hands_remaining / 200.0, 0.0, 1.0)
        rank_norm = np.clip(context.rank / 120.0, 0.0, 1.0)
        cutoff_norm = np.clip(context.cutoff_rank / 120.0, 0.0, 1.0)
        margin_norm = math.tanh(context.rank_margin_bb / 100.0)
        net_norm = math.tanh(context.cumulative_net_bb / 200.0)
        rebuy_norm = np.clip(context.rebuy_count / 10.0, 0.0, 1.0)
        strength_norm = math.tanh(context.table_strength / 200.0)
        stack_norm = np.clip(context.effective_stack_bb / 100.0, 0.0, 5.0)
        table_rank_norm = np.clip(context.table_rank / 6.0, 0.0, 1.0)

        continuous_feats = np.array(
            [
                hands_norm,
                rank_norm,
                cutoff_norm,
                margin_norm,
                net_norm,
                rebuy_norm,
                strength_norm,
                stack_norm,
                table_rank_norm,
            ],
            dtype=np.float32,
        )

        return np.concatenate([stage_enc, continuous_feats])

    @staticmethod
    def encode_tensor(context: TournamentState, device: str = "cpu") -> torch.Tensor:
        """Convert TournamentState to a torch tensor of shape (1, RAW_DIM)."""
        vec = TournamentEncoder.encode(context)
        return torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)


class TournamentEmbeddingNet(nn.Module):
    """Small MLP that projects raw tournament indicators into a dense tournament embedding."""

    def __init__(self, input_dim: int = TournamentEncoder.RAW_DIM, embedding_dim: int = 64):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass generating dense tournament embeddings."""
        return self.net(x)
