"""Curriculum Training Framework for ApexPoker Tournament AI."""

from dataclasses import dataclass, field
from enum import Enum, auto
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pokers as pkrs
import torch

from src.agents.random_agent import RandomAgent
from src.tournament.state import Stage, TournamentState, PlayerRecord
from src.tournament.tournament_env import TournamentEnv
from src.tournament.tournament_deep_cfr import TournamentDeepCFRAgent
from src.training.tournament_reward import TournamentRewardCalculator


class CurriculumPhase(Enum):
    P0_BASE = auto()
    P1_CONTEXT = auto()
    P2_PRELIMINARY = auto()
    P3_SEMIFINAL = auto()
    P4_FINAL = auto()
    P5_FULL = auto()

    def __str__(self) -> str:
        return self.name


@dataclass
class CurriculumConfig:
    """Curriculum hyperparameter and schedule configuration."""
    iterations: int = 5
    batch_size: int = 32
    save_dir: str = "models/curriculum"
    device: str = "cpu"
    seed: int = 42
    base_checkpoint_path: Optional[str] = None


class CurriculumTrainer:
    """Sequentially trains TournamentDeepCFRAgent across the 6 progressive curriculum stages."""

    def __init__(self, agent: TournamentDeepCFRAgent, config: Optional[CurriculumConfig] = None):
        self.agent = agent
        self.config = config or CurriculumConfig()
        # Calibrated tournament reward calculator: hand_scale=0.08 (8x higher chip value)
        self.reward_calculator = TournamentRewardCalculator(hand_scale=0.08, margin_weight=0.15)
        os.makedirs(self.config.save_dir, exist_ok=True)

    def _train_stage_scenarios(
        self,
        phase_name: str,
        stage_choice: Stage,
        iterations: int,
        cutoff_rank: int,
        total_players: int,
        hands_max: int,
    ) -> Dict[str, Any]:
        losses = []
        rng = np.random.RandomState(self.config.seed + hash(phase_name) % 10000)

        for it in range(iterations):
            hands_rem = int(rng.randint(1, hands_max + 1))
            rank = int(rng.randint(1, total_players + 1))
            margin = float((cutoff_rank - rank) * rng.uniform(1.5, 12.0))
            stack_bb = float(rng.uniform(8.0, 140.0))
            effective_stack_bb = min(stack_bb, float(rng.uniform(8.0, 100.0)))

            ctx = TournamentState(
                stage=stage_choice,
                hand_no=int(rng.randint(1, 100)),
                hands_remaining=hands_rem,
                player_id=self.agent.player_id,
                rank=rank,
                cutoff_rank=cutoff_rank,
                rank_margin_bb=margin,
                cumulative_net_bb=margin,
                rebuy_count=int(rng.choice([0, 1, 2], p=[0.8, 0.15, 0.05])),
                table_rank=min(6, max(1, rank % 6 + 1)),
                table_id=0,
                stack_bb=stack_bb,
                effective_stack_bb=effective_stack_bb,
                table_strength=float(rng.uniform(-0.5, 0.5)),
            )

            state = pkrs.State.from_seed(
                n_players=min(6, total_players),
                button=int(it % min(6, total_players)),
                sb=1.0,
                bb=2.0,
                stake=stack_bb * 2.0,
                seed=self.config.seed + it,
            )

            poker_vec = self.agent.poker_encoder.encode(state, player_id=self.agent.player_id)
            tourn_vec = self.agent.tourn_encoder.encode(ctx)

            # Query base poker policy (frozen anchor)
            with torch.no_grad():
                x_p_tensor = torch.tensor(poker_vec, dtype=torch.float32, device=self.agent.device).unsqueeze(0)
                base_logits, _ = self.agent.strategy_net.base_poker_net(x_p_tensor)
                base_probs = torch.softmax(base_logits, dim=-1).squeeze(0).cpu().numpy()

            # Situation-dependent modulation:
            is_bubble_ahead = (rank <= cutoff_rank) and (margin > 8.0) and (hands_rem < 15)
            is_bubble_behind = (rank > cutoff_rank) and (margin < -2.0) and (hands_rem < 15)
            is_short_stack = (stack_bb <= 22.0) or is_bubble_behind
            is_deep_ahead = (stack_bb >= 90.0 and rank <= cutoff_rank and margin > 8.0) or is_bubble_ahead

            target_probs = base_probs.copy()
            if is_short_stack:
                # Short stack or behind cutoff: Push/fold aggression to build stack or survive
                target_probs[2] += 0.28  # Raise +28%
                target_probs[0] -= 0.12  # Fold -12%
                target_probs[1] -= 0.16  # Call -16% (eliminate limp/flat-calling)
            elif is_deep_ahead:
                # Deep stack or ahead of bubble: Avoid coinflips, preserve chip lead
                target_probs[0] += 0.25  # Fold +25%
                target_probs[2] -= 0.15  # Raise -15%
                target_probs[1] -= 0.10  # Call -10%
            else:
                # Normal rounds: retain base model's proven winning TAG aggression!
                pass

            target_probs = np.clip(target_probs, 0.05, 0.92)
            target_probs /= target_probs.sum()

            regrets = np.log(np.maximum(1e-4, target_probs)) - np.log(np.maximum(1e-4, base_probs))

            for a in range(self.agent.num_actions):
                self.agent.record_advantage_experience(
                    poker_vec=poker_vec,
                    tourn_vec=tourn_vec,
                    action_type=a,
                    bet_size=1.0,
                    regret=float(regrets[a]),
                )
            self.agent.record_strategy_experience(
                poker_vec=poker_vec,
                tourn_vec=tourn_vec,
                action_probs=target_probs,
                bet_size=1.0,
            )

            adv_loss = self.agent.train_advantage_network(batch_size=min(32, len(self.agent.advantage_memory.buffer)))
            strat_loss = self.agent.train_strategy_network(batch_size=min(32, len(self.agent.strategy_memory)))
            losses.append((adv_loss, strat_loss))

        ckpt_path = os.path.join(self.config.save_dir, f"{phase_name.lower()}.pt")
        self.agent.save_checkpoint(ckpt_path, curriculum_phase=phase_name)
        return {"phase": phase_name, "iterations": iterations, "losses": losses, "checkpoint": ckpt_path}

    def train_p1_context_sensitivity(self, iterations: int = 100) -> Dict[str, Any]:
        """P1: Focus on learning tournament context representations and ICM reward signals."""
        return self._train_stage_scenarios(
            phase_name="curriculum_p1",
            stage_choice=Stage.PRELIMINARY,
            iterations=iterations,
            cutoff_rank=12,
            total_players=120,
            hands_max=50,
        )

    def train_p2_preliminary(self, iterations: int = 80) -> Dict[str, Any]:
        """P2: Preliminary stage simulation (120 players, 200 hands, Swiss, Top 12)."""
        return self._train_stage_scenarios(
            phase_name="curriculum_p2",
            stage_choice=Stage.PRELIMINARY,
            iterations=iterations,
            cutoff_rank=12,
            total_players=120,
            hands_max=200,
        )

    def train_p3_semifinal(self, iterations: int = 60) -> Dict[str, Any]:
        """P3: Semifinal stage simulation (12 players, 2 tables of 6, Top 3 each)."""
        return self._train_stage_scenarios(
            phase_name="curriculum_p3",
            stage_choice=Stage.SEMIFINAL,
            iterations=iterations,
            cutoff_rank=3,
            total_players=12,
            hands_max=20,
        )

    def train_p4_final(self, iterations: int = 60) -> Dict[str, Any]:
        """P4: Final stage simulation (6 players, 30 hands, Champion)."""
        return self._train_stage_scenarios(
            phase_name="curriculum_p4",
            stage_choice=Stage.FINAL,
            iterations=iterations,
            cutoff_rank=1,
            total_players=6,
            hands_max=30,
        )

    def train_p5_full(self, iterations: int = 80) -> Dict[str, Any]:
        """P5: End-to-end full tournament joint training across all stages."""
        rng = np.random.RandomState(self.config.seed + 555)
        losses = []
        for it in range(iterations):
            stg = rng.choice([Stage.PRELIMINARY, Stage.SEMIFINAL, Stage.FINAL])
            cutoff = 12 if stg == Stage.PRELIMINARY else (3 if stg == Stage.SEMIFINAL else 1)
            tot_p = 120 if stg == Stage.PRELIMINARY else (12 if stg == Stage.SEMIFINAL else 6)
            h_max = 50 if stg == Stage.PRELIMINARY else (20 if stg == Stage.SEMIFINAL else 30)
            res = self._train_stage_scenarios(
                phase_name="curriculum_p5_full",
                stage_choice=stg,
                iterations=1,
                cutoff_rank=cutoff,
                total_players=tot_p,
                hands_max=h_max,
            )
            losses.extend(res.get("losses", []))

        ckpt_path = os.path.join(self.config.save_dir, "curriculum_p5_full.pt")
        self.agent.save_checkpoint(ckpt_path, curriculum_phase="P5_FULL")
        return {
            "phase": "P5_FULL",
            "iterations": iterations,
            "losses": losses,
            "checkpoint": ckpt_path,
        }

    def run_full_curriculum(self) -> Dict[str, Any]:
        """Run all curriculum stages sequentially: P1 -> P2 -> P3 -> P4 -> P5."""
        p1_res = self.train_p1_context_sensitivity(iterations=self.config.iterations)
        p2_res = self.train_p2_preliminary(iterations=1)
        p3_res = self.train_p3_semifinal(iterations=1)
        p4_res = self.train_p4_final(iterations=1)
        p5_res = self.train_p5_full(iterations=1)

        return {
            "P1": p1_res,
            "P2": p2_res,
            "P3": p3_res,
            "P4": p4_res,
            "P5": p5_res,
        }
