"""Tournament-Aware Deep CFR Agent integrating Tournament Context into decision making."""

from collections import deque
import copy
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pokers as pkrs
import torch
import torch.nn.functional as F
import torch.optim as optim

from src.core.deep_cfr import PrioritizedMemory, linear_cfr_weights, weighted_strategy_cross_entropy
from src.core.model import PokerNetwork, encode_state
from src.observation.poker_encoder import PokerEncoder
from src.observation.tournament_encoder import TournamentEncoder
from src.observation.fusion import TournamentAwarePokerNetwork
from src.tournament.state import Stage, TournamentState
from src.utils.actions import (
    action_type_to_pokers_action,
    legal_action_types,
    raise_bounds,
    sanitize_action,
)
from src.utils.checkpoints import attach_checkpoint_metadata, load_checkpoint

AGENT_TYPE_TOURNAMENT = "tournament_aware"


class TournamentDeepCFRAgent:
    """Deep CFR Agent equipped with tournament situational awareness.
    
    Observes:
    - Current Stage (Preliminary / Semifinal / Final)
    - Hands remaining in stage
    - Current rank and qualification cutoff rank
    - Distance to cutoff (rank_margin_bb)
    - Cumulative net BB and rebuy count
    - Table strength and effective stack
    
    All strategic adaptations are learned directly from state and regret minimization,
    not hard-coded heuristics.
    """

    def __init__(
        self,
        player_id: int = 0,
        num_players: int = 6,
        memory_size: int = 200000,
        device: str = "cpu",
        base_checkpoint_path: Optional[str] = None,
    ):
        self.player_id = player_id
        self.num_players = num_players
        self.device = device
        self.num_actions = 3
        self.min_bet_size = 0.1
        self.max_bet_size = 3.0

        self.poker_encoder = PokerEncoder()
        self.tourn_encoder = TournamentEncoder()

        # Build Tournament-aware networks
        self.advantage_net = TournamentAwarePokerNetwork(
            poker_input_size=PokerEncoder.FEATURE_DIM,
            tourn_input_size=TournamentEncoder.RAW_DIM,
            tourn_emb_dim=64,
            hidden_size=256,
            num_actions=self.num_actions,
        ).to(device)
        self.optimizer = optim.Adam(self.advantage_net.parameters(), lr=1e-5, weight_decay=1e-5)

        self.strategy_net = TournamentAwarePokerNetwork(
            poker_input_size=PokerEncoder.FEATURE_DIM,
            tourn_input_size=TournamentEncoder.RAW_DIM,
            tourn_emb_dim=64,
            hidden_size=256,
            num_actions=self.num_actions,
        ).to(device)
        self.strategy_optimizer = optim.Adam(self.strategy_net.parameters(), lr=5e-5, weight_decay=1e-5)

        # Experience replay buffers
        self.advantage_memory = PrioritizedMemory(memory_size)
        self.strategy_memory = deque(maxlen=memory_size)

        self.iteration_count = 0
        self.local_training_iteration = 0

        # Load pretrained base checkpoint weights if provided
        if base_checkpoint_path:
            self.load_base_checkpoint(base_checkpoint_path)

    def load_base_checkpoint(self, path: str, freeze_backbone: bool = True) -> None:
        """Transfer base 6-Max poker knowledge from standard checkpoint into backbone."""
        ckpt = load_checkpoint(path, map_location=self.device)
        if "advantage_net" in ckpt:
            self.advantage_net.base_poker_net.load_state_dict(ckpt["advantage_net"])
        if "strategy_net" in ckpt:
            self.strategy_net.base_poker_net.load_state_dict(ckpt["strategy_net"])
        if freeze_backbone:
            self.advantage_net.freeze_backbone()
            self.strategy_net.freeze_backbone()
            self.optimizer = optim.Adam(
                [p for p in self.advantage_net.parameters() if p.requires_grad],
                lr=1e-4,
                weight_decay=1e-5,
            )
            self.strategy_optimizer = optim.Adam(
                [p for p in self.strategy_net.parameters() if p.requires_grad],
                lr=1e-4,
                weight_decay=1e-5,
            )

    def choose_action(
        self,
        state: pkrs.State,
        tournament_context: Optional[TournamentState] = None,
        strict: bool = False,
        **kwargs,
    ) -> pkrs.Action:
        """Choose an action given current poker state and tournament context."""
        legal_types = legal_action_types(state)
        if not legal_types:
            return pkrs.Action(pkrs.ActionEnum.Fold)

        poker_vec = self.poker_encoder.encode(state, player_id=self.player_id)
        x_poker = torch.tensor(poker_vec, dtype=torch.float32, device=self.device).unsqueeze(0)

        x_tourn = None
        if tournament_context is not None:
            tourn_vec = self.tourn_encoder.encode(tournament_context)
            x_tourn = torch.tensor(tourn_vec, dtype=torch.float32, device=self.device).unsqueeze(0)

        self.strategy_net.eval()
        with torch.no_grad():
            action_logits, bet_size_pred = self.strategy_net(x_poker, x_tourn)

        logits = action_logits.squeeze(0)
        # Mask illegal actions
        masked_logits = torch.full_like(logits, -float("inf"))
        for act_type in legal_types:
            masked_logits[act_type] = logits[act_type]

        probs = F.softmax(masked_logits, dim=-1).cpu().numpy()
        # Handle all-masked edge case
        if np.isnan(probs).any() or probs.sum() == 0:
            probs = np.zeros(self.num_actions)
            for act_type in legal_types:
                probs[act_type] = 1.0 / len(legal_types)

        # Sample action from strategy probabilities
        chosen_action_type = int(np.random.choice(self.num_actions, p=probs))
        bet_size_ratio = float(bet_size_pred.squeeze().cpu().item())
        bet_size_ratio = max(self.min_bet_size, min(self.max_bet_size, bet_size_ratio))

        raw_action = action_type_to_pokers_action(
            chosen_action_type,
            state,
            bet_size_multiplier=bet_size_ratio,
            min_bet_size=self.min_bet_size,
            max_bet_size=self.max_bet_size,
            strict=strict,
        )
        return sanitize_action(state, raw_action, strict=strict)

    def record_advantage_experience(
        self,
        poker_vec: np.ndarray,
        tourn_vec: Optional[np.ndarray],
        action_type: int,
        bet_size: float,
        regret: float,
        priority: Optional[float] = None,
    ) -> None:
        """Store an experience tuple in prioritized advantage replay."""
        experience = (poker_vec, tourn_vec, action_type, bet_size, regret)
        self.advantage_memory.add(experience, priority=priority)

    def record_strategy_experience(
        self,
        poker_vec: np.ndarray,
        tourn_vec: Optional[np.ndarray],
        action_probs: np.ndarray,
        bet_size: float,
        weight: float = 1.0,
    ) -> None:
        """Store a strategy sample in reservoir strategy memory."""
        self.strategy_memory.append((poker_vec, tourn_vec, action_probs, bet_size, weight))

    def train_advantage_network(self, batch_size: int = 64) -> float:
        """Train advantage network from prioritized memory."""
        if len(self.advantage_memory.buffer) < batch_size:
            return 0.0

        samples, indices, weights = self.advantage_memory.sample(batch_size)
        poker_list, tourn_list, act_list, size_list, regret_list = zip(*samples)

        x_poker = torch.tensor(np.array(poker_list), dtype=torch.float32, device=self.device)
        if tourn_list[0] is not None:
            x_tourn = torch.tensor(np.array(tourn_list), dtype=torch.float32, device=self.device)
        else:
            x_tourn = None

        target_actions = torch.tensor(act_list, dtype=torch.long, device=self.device)
        target_regrets = torch.tensor(regret_list, dtype=torch.float32, device=self.device)
        weight_tensors = torch.tensor(weights, dtype=torch.float32, device=self.device)

        self.advantage_net.train()
        self.optimizer.zero_grad()
        action_logits, bet_sizes = self.advantage_net(x_poker, x_tourn)

        # Regret MSE loss on target action
        pred_regrets = action_logits.gather(1, target_actions.unsqueeze(1)).squeeze(1)
        loss = (weight_tensors * F.mse_loss(pred_regrets, target_regrets, reduction="none")).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.advantage_net.parameters(), max_norm=0.5)
        self.optimizer.step()
        return float(loss.item())

    def train_strategy_network(self, batch_size: int = 64) -> float:
        """Train strategy network on historical average policy."""
        if len(self.strategy_memory) < batch_size:
            return 0.0

        samples = random.sample(self.strategy_memory, batch_size)
        poker_list, tourn_list, probs_list, size_list, weight_list = zip(*samples)

        x_poker = torch.tensor(np.array(poker_list), dtype=torch.float32, device=self.device)
        if tourn_list[0] is not None:
            x_tourn = torch.tensor(np.array(tourn_list), dtype=torch.float32, device=self.device)
        else:
            x_tourn = None

        target_probs = torch.tensor(np.array(probs_list), dtype=torch.float32, device=self.device)
        weight_tensors = torch.tensor(weight_list, dtype=torch.float32, device=self.device)

        self.strategy_net.train()
        self.strategy_optimizer.zero_grad()
        action_logits, bet_sizes = self.strategy_net(x_poker, x_tourn)

        # Cross-entropy with target action probabilities
        log_probs = F.log_softmax(action_logits, dim=-1)
        ce_loss = -(target_probs * log_probs).sum(dim=-1)
        loss = (weight_tensors * ce_loss).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.strategy_net.parameters(), max_norm=0.5)
        self.strategy_optimizer.step()
        return float(loss.item())

    def save_checkpoint(self, path: str, **extra_metadata) -> None:
        """Save tournament-aware agent checkpoint."""
        state = {
            "iteration": self.iteration_count,
            "advantage_net": self.advantage_net.state_dict(),
            "strategy_net": self.strategy_net.state_dict(),
            "min_bet_size": self.min_bet_size,
            "max_bet_size": self.max_bet_size,
        }
        state.update(extra_metadata)
        ckpt = attach_checkpoint_metadata(state, self, AGENT_TYPE_TOURNAMENT)
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str) -> None:
        """Load tournament-aware agent checkpoint or transfer from base checkpoint."""
        ckpt = load_checkpoint(path, map_location=self.device)
        adv_state = ckpt.get("advantage_net", {})
        if any(k.startswith("base_poker_net") for k in adv_state.keys()):
            # Native tournament checkpoint
            self.advantage_net.load_state_dict(adv_state, strict=False)
            self.strategy_net.load_state_dict(ckpt.get("strategy_net", {}), strict=False)
        else:
            # Standard base checkpoint -> transfer into base poker backbone
            self.load_base_checkpoint(path)
        self.iteration_count = ckpt.get("iteration", 0)
