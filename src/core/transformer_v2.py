"""Phase 10: Tournament Transformer V2 Architecture.

Integrates:
- Card Encoder (Hole Cards + Public Community Cards)
- Action History Encoder
- Position Encoder
- Tournament Context Encoder
- Opponent Profile Encoder
↓
Tournament Multi-Head Attention Transformer Backbone
↓
Multi-task Output Heads:
1. Advantage Head (Fold, Check/Call, Raise)
2. Strategy Head (Action probabilities + Continuous bet sizing)
3. P(Advance) Head (Probability of qualifying past current stage)
4. P(Champion) Head (Probability of winning overall tournament)
5. Value Distribution Head (Categorical / atom distribution)
"""

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CardEncoder(nn.Module):
    """Embeds 52-card hole and community cards into token embeddings."""
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.card_embedding = nn.Embedding(53, d_model)  # 52 cards + 1 padding/none

    def forward(self, card_indices: torch.Tensor) -> torch.Tensor:
        return self.card_embedding(card_indices)


class TournamentTransformerV2(nn.Module):
    """Transformer V2 multi-task architecture for tournament poker decision making."""

    def __init__(
        self,
        poker_dim: int = 156,
        tourn_dim: int = 13,
        opponent_dim: int = 32,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        num_actions: int = 3,
        num_atoms: int = 21,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_actions = num_actions
        self.num_atoms = num_atoms

        # Input projection tokens
        self.poker_token_proj = nn.Linear(poker_dim, d_model)
        self.tourn_token_proj = nn.Linear(tourn_dim, d_model)
        self.opp_token_proj = nn.Linear(opponent_dim, d_model)

        # Learnable position tokens: Poker, Tournament, Opponent tokens
        self.pos_emb = nn.Parameter(torch.randn(1, 3, d_model) * 0.02)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output Heads
        # 1. Advantage Head (Regret values for each action)
        self.advantage_head = nn.Linear(d_model, num_actions)

        # 2. Strategy Heads (Policy logits + Bet size)
        self.strategy_action_head = nn.Linear(d_model, num_actions)
        self.strategy_sizing_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # 3. Auxiliary Tournament Heads
        self.p_advance_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.p_champion_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # 4. Value Distribution Head (Atom distribution)
        self.value_distribution_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, num_atoms),
            nn.Softmax(dim=-1),
        )

    def forward(
        self,
        x_poker: torch.Tensor,
        x_tourn: Optional[torch.Tensor] = None,
        x_opponent: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through multi-task Transformer architecture."""
        batch_size = x_poker.size(0)

        # Prepare tokens
        poker_token = self.poker_token_proj(x_poker).unsqueeze(1)  # (B, 1, d_model)
        
        if x_tourn is not None:
            tourn_token = self.tourn_token_proj(x_tourn).unsqueeze(1)
        else:
            tourn_token = torch.zeros(batch_size, 1, self.d_model, device=x_poker.device)

        if x_opponent is not None:
            opp_token = self.opp_token_proj(x_opponent).unsqueeze(1)
        else:
            opp_token = torch.zeros(batch_size, 1, self.d_model, device=x_poker.device)

        # Concatenate 3 tokens along sequence length: (B, 3, d_model)
        seq = torch.cat([poker_token, tourn_token, opp_token], dim=1) + self.pos_emb

        # Transformer attention pass
        enc_seq = self.transformer(seq)  # (B, 3, d_model)

        # Global representation from pooled tokens
        pooled = enc_seq.mean(dim=1)  # (B, d_model)

        # Output predictions
        adv_logits = self.advantage_head(pooled)
        strat_logits = self.strategy_action_head(pooled)
        bet_size = 0.1 + 2.9 * self.strategy_sizing_head(pooled)
        p_advance = self.p_advance_head(pooled)
        p_champ = self.p_champion_head(pooled)
        val_dist = self.value_distribution_head(pooled)

        return {
            "advantages": adv_logits,
            "strategy_logits": strat_logits,
            "bet_size": bet_size,
            "p_advance": p_advance,
            "p_champion": p_champ,
            "value_distribution": val_dist,
        }
