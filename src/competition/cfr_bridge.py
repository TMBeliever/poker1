"""Deep CFR Neural Agent Bridge for Agent Poker Online Competition.

Translates Sohu Agent Poker JSON observations into PyTorch Deep CFR network features,
queries the trained neural model, and generates compliant action decisions per skill.md Section 4.
"""

from __future__ import annotations
import os
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from src.core.deep_cfr import DeepCFRAgent
from src.tournament.tournament_deep_cfr import TournamentDeepCFRAgent
from .soul import SoulManager

# Card conversion constants
RANK_MAP = {"2": 0, "3": 1, "4": 2, "5": 3, "6": 4, "7": 5, "8": 6, "9": 7, "T": 8, "J": 9, "Q": 10, "K": 11, "A": 12}
SUIT_MAP = {"c": 0, "d": 1, "h": 2, "s": 3}  # Clubs, Diamonds, Hearts, Spades


def card_str_to_index(card_str: str) -> int:
    """Convert 'Ah' -> 38, '2c' -> 0, etc."""
    if not isinstance(card_str, str) or len(card_str) != 2:
        return 0
    rank = RANK_MAP.get(card_str[0], 0)
    suit = SUIT_MAP.get(card_str[1], 0)
    return suit * 13 + rank


def preflop_strength_score(cards: List[str]) -> float:
    """Calculate starting hand strength score in [0.0, 1.0] using calibrated 169-bucket table."""
    if not cards or len(cards) < 2:
        return 0.5
    try:
        from .cards import parse_card
        from .calibration import preflop_percentile
        parsed = [parse_card(c) for c in cards[:2]]
        perc = preflop_percentile(parsed)
        if perc is not None:
            return float(perc)
    except Exception:
        pass

    rank_vals = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
    try:
        r1, s1 = cards[0][0], cards[0][1]
        r2, s2 = cards[1][0], cards[1][1]
        v1, v2 = sorted([rank_vals.get(r1, 2), rank_vals.get(r2, 2)], reverse=True)
        score = (v1 + v2) / 28.0
        if v1 == v2:
            score += 0.30  # Pair
        if s1 == s2:
            score += 0.05  # Suited
        if v1 - v2 <= 2:
            score += 0.045  # Connected
        if v1 >= 13 and v2 >= 10:
            score += 0.11  # AK, AQ, AJ, KQ
        if v1 == 14:
            score += 0.04  # Ace
        return max(0.0, min(1.0, (score - 0.30) / 0.70))
    except Exception:
        return 0.5


class CFRCompetitionBridge:
    """Bridges online tournament observations to trained Deep CFR neural models."""

    def __init__(
        self,
        model_path: str = "models/base/base_checkpoint_iter_2000.pt",
        soul_manager: Optional[SoulManager] = None,
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.device = torch.device(device)
        self.soul = soul_manager or SoulManager()
        self.agent: Optional[DeepCFRAgent] = None
        self.is_tournament_agent = False
        self._load_agent()

    def _load_agent(self) -> None:
        """Load the PyTorch Deep CFR agent from checkpoint."""
        if not os.path.exists(self.model_path):
            # Try finding latest fine-tuned checkpoint
            candidates = [
                "models/finetuned_50bb/hybrid_checkpoint_iter_66000.pt",
                "models/hybrid/hybrid_checkpoint_iter_60000.pt",
                "models/tournament/curriculum_p5_full.pt",
                "models/base/base_checkpoint_iter_2000.pt",
                "models/tournament/curriculum_p1.pt",
            ]
            for c in candidates:
                if os.path.exists(c):
                    self.model_path = c
                    break

        if os.path.exists(self.model_path):
            try:
                ckpt = torch.load(self.model_path, map_location=self.device)
                if (
                    "base_poker_net" in ckpt
                    or "fusion" in ckpt
                    or "curriculum_phase" in ckpt
                    or ckpt.get("agent_type") == "tournament_aware"
                    or "models/tournament" in self.model_path
                ):
                    self.agent = TournamentDeepCFRAgent(player_id=0, num_players=6, device=self.device)
                    self.agent.load_checkpoint(self.model_path)
                    self.is_tournament_agent = True
                    print(f"[Bridge] Loaded Tournament Deep CFR Agent from: {self.model_path}")
                else:
                    self.agent = DeepCFRAgent(player_id=0, num_players=6, device=self.device)
                    self.agent.load_model(self.model_path)
                    print(f"[Bridge] Loaded Base Deep CFR Agent from: {self.model_path}")
            except Exception as e:
                print(f"[Bridge] Warning: Error loading {self.model_path} ({e}), initializing fresh agent.")
                self.agent = DeepCFRAgent(player_id=0, num_players=6, device=self.device)
        else:
            print(f"[Bridge] Checkpoint {self.model_path} not found, using default agent.")
            self.agent = DeepCFRAgent(player_id=0, num_players=6, device=self.device)

    def encode_observation(self, obs: Dict[str, Any]) -> np.ndarray:
        """Convert Sohu Agent Poker observation into 156-dim feature vector."""
        table = obs.get("table") or {}
        hand = table.get("hand") or {}
        players = table.get("players") or []
        hero_id = obs.get("agentId")
        hero_p = next((p for p in players if p.get("agentId") == hero_id), None)

        encoded = []
        num_players = 6

        # 1. Hole cards (52-dim one-hot)
        hand_enc = np.zeros(52, dtype=np.float32)
        if hero_p and isinstance(hero_p.get("handState"), dict):
            for c in hero_p["handState"].get("holeCards") or []:
                hand_enc[card_str_to_index(c)] = 1.0
        encoded.append(hand_enc)

        # 2. Community cards (52-dim one-hot)
        community_enc = np.zeros(52, dtype=np.float32)
        for c in hand.get("communityCards") or []:
            community_enc[card_str_to_index(c)] = 1.0
        encoded.append(community_enc)

        # 3. Street / Stage (5-dim one-hot: Preflop, Flop, Turn, River, Showdown)
        stage_enc = np.zeros(5, dtype=np.float32)
        comm_count = len(hand.get("communityCards") or [])
        if comm_count == 0:
            stage_idx = 0  # Preflop
        elif comm_count == 3:
            stage_idx = 1  # Flop
        elif comm_count == 4:
            stage_idx = 2  # Turn
        elif comm_count >= 5:
            stage_idx = 3  # River
        else:
            stage_idx = 0
        stage_enc[stage_idx] = 1.0
        encoded.append(stage_enc)

        # 4. Pot size (normalized by initial stake)
        blinds = table.get("blinds") or {}
        bb_val = float(table.get("bigBlind") or blinds.get("big") or hand.get("bigBlind") or 1000.0)
        initial_stake = float(table.get("initialStake") or table.get("buyIn") or (bb_val * 50.0) or 50000.0)
        pot = float(hand.get("pot", 0) or 0)
        encoded.append([pot / max(1.0, initial_stake)])

        # 5. Button position (6-dim one-hot)
        button_enc = np.zeros(num_players, dtype=np.float32)
        button_seat = (
            table.get("dealerSeatIndex")
            or hand.get("dealerSeatIndex")
            or table.get("dealerSeat")
            or table.get("buttonSeat")
            or hand.get("buttonSeat")
            or 0
        )
        button_enc[int(button_seat) % num_players] = 1.0
        encoded.append(button_enc)

        # 6. Current player (6-dim one-hot)
        curr_enc = np.zeros(num_players, dtype=np.float32)
        hero_seat = 0
        if hero_p:
            if hero_p.get("seatIndex") is not None:
                hero_seat = int(hero_p["seatIndex"])
            elif hero_p.get("seat") is not None:
                hero_seat = int(hero_p["seat"])
            elif hero_p in players:
                hero_seat = players.index(hero_p)
        curr_enc[hero_seat % num_players] = 1.0
        encoded.append(curr_enc)

        # 7. Player states (24-dim: 6 x [active, bet, pot_chips, stake])
        p_states = []
        seat_map = {}
        for idx, p in enumerate(players):
            s = p.get("seatIndex") if p.get("seatIndex") is not None else p.get("seat", idx)
            seat_map[int(s)] = p

        for seat in range(num_players):
            p = seat_map.get(seat)
            if p:
                hs = p.get("handState") or {}
                active = 1.0 if not hs.get("status") in ("folded", "out") else 0.0
                bet = float(hs.get("currentBet", 0) or 0) / max(1.0, initial_stake)
                pot_chips = 0.0
                stake = float(p.get("stack", 0) or 0) / max(1.0, initial_stake)
                p_states.extend([active, bet, pot_chips, stake])
            else:
                p_states.extend([0.0, 0.0, 0.0, 0.0])
        encoded.append(np.array(p_states, dtype=np.float32))

        # 8. Min bet (1-dim)
        min_bet = float(bb_val)
        encoded.append([min_bet / max(1.0, initial_stake)])

        # 9. Legal actions (4-dim: Fold, Check, Call, Raise)
        req = obs.get("actionRequest") or {}
        allowed = {a["type"]: a for a in (req.get("allowedActions") or []) if isinstance(a, dict) and a.get("type")}
        legal_enc = np.zeros(4, dtype=np.float32)
        if "fold" in allowed:
            legal_enc[0] = 1.0
        if "check" in allowed:
            legal_enc[1] = 1.0
        if "call" in allowed:
            legal_enc[2] = 1.0
        if "raise" in allowed or "bet" in allowed or "allIn" in allowed:
            legal_enc[3] = 1.0
        encoded.append(legal_enc)

        # 10. Previous action (5-dim)
        prev_enc = np.zeros(5, dtype=np.float32)
        actions_list = hand.get("actions") or []
        if actions_list:
            last_act = actions_list[-1]
            atype = last_act.get("type", "")
            if atype == "fold":
                prev_enc[0] = 1.0
            elif atype == "check":
                prev_enc[1] = 1.0
            elif atype == "call":
                prev_enc[2] = 1.0
            elif atype in ("raise", "bet", "allIn"):
                prev_enc[3] = 1.0
            prev_enc[4] = float(last_act.get("amount", 0) or 0) / max(1.0, initial_stake)
        encoded.append(prev_enc)

        return np.concatenate(encoded)

    def choose_action(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a validated decision conforming to skill.md Section 4."""
        req = obs.get("actionRequest") or {}
        allowed_list = req.get("allowedActions") or []
        allowed = {a["type"]: a for a in allowed_list if isinstance(a, dict) and a.get("type")}

        if not allowed:
            return {"type": "fold"}

        # Encode observation
        feat_156 = self.encode_observation(obs)
        x_tensor = torch.tensor(feat_156, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            if self.is_tournament_agent and hasattr(self.agent, "strategy_net"):
                # Tournament agent: construct 13-dim tournament context matching TournamentEncoder schema
                ctx = obs.get("tournamentContext") or {}
                stage_str = str(ctx.get("stage", "preliminary")).lower()
                stage_enc = [0.0, 0.0, 0.0, 0.0]
                if "semi" in stage_str:
                    stage_enc[1] = 1.0
                elif "final" in stage_str:
                    stage_enc[2] = 1.0
                else:
                    stage_enc[0] = 1.0

                hands_norm = np.clip(float(ctx.get("hands_remaining", 50)) / 200.0, 0.0, 1.0)
                rank_norm = np.clip(float(ctx.get("rank", 10)) / 120.0, 0.0, 1.0)
                cutoff_norm = np.clip(float(ctx.get("target_rank", 12)) / 120.0, 0.0, 1.0)
                margin_norm = math.tanh(float(ctx.get("buffer_bb100", 0.0)) / 100.0)
                net_norm = math.tanh(float(ctx.get("bb100", 0.0)) / 200.0)
                rebuy_norm = np.clip(float(ctx.get("rebuy_count", 0)) / 10.0, 0.0, 1.0)
                strength_norm = math.tanh(float(ctx.get("table_strength", 0.0)) / 200.0)

                # Hero stack in BB
                table = obs.get("table") or {}
                bb = float(table.get("bigBlind", 200) or 200)
                hero_id = obs.get("agentId")
                players = table.get("players") or []
                hero_p = next((p for p in players if p.get("agentId") == hero_id), None)
                hero_stack = float(hero_p.get("stack", 20000) or 20000) if hero_p else 20000.0
                stack_norm = np.clip((hero_stack / max(1.0, bb)) / 100.0, 0.0, 5.0)
                table_rank_norm = float(ctx.get("table_rank", 1)) / 6.0

                t_ctx = np.array(
                    stage_enc
                    + [
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
                t_tensor = torch.tensor(t_ctx, dtype=torch.float32, device=self.device).unsqueeze(0)
                action_logits, bet_sizing = self.agent.strategy_net(x_tensor, t_tensor)
            elif self.agent and hasattr(self.agent, "strategy_net"):
                action_logits, bet_sizing = self.agent.strategy_net(x_tensor)
            else:
                action_logits = torch.zeros((1, 3), device=self.device)
                bet_sizing = torch.tensor([[0.5]], device=self.device)

        # Action head outputs logits for [Fold, Check/Call, Raise]
        probs = torch.softmax(action_logits, dim=-1).squeeze(0).cpu().numpy()
        multiplier = float(bet_sizing.squeeze().cpu().numpy())

        # Map to legal actions
        p_fold, p_call, p_raise = probs[0], probs[1], probs[2]

        can_check = "check" in allowed
        can_call = "call" in allowed
        can_bet = "bet" in allowed
        can_raise = "raise" in allowed
        can_allin = "allIn" in allowed
        can_fold = "fold" in allowed

        # Invariant 1: If check is free, never fold
        if can_check:
            p_fold = 0.0

        # Preflop strategic protections
        table = obs.get("table") or {}
        hand = table.get("hand") or {}
        comm_count = len(hand.get("communityCards") or [])
        is_preflop = comm_count == 0
        players = table.get("players") or []
        hero_id = obs.get("agentId")
        hero_p = next((p for p in players if p.get("agentId") == hero_id), None)
        hero_cards = (hero_p.get("handState") or {}).get("holeCards") if hero_p else []
        preflop_score = preflop_strength_score(hero_cards) if hero_cards else 0.5

        if is_preflop:
            # Unplayable trash hands (bottom 38%, e.g. 72o, 83o, 94o, T5o, Q2o): never call or raise
            if preflop_score < 0.38:
                p_raise = 0.0
                if can_fold and not can_check:
                    p_fold = 1.0
                    p_call = 0.0
            # Premium hands (AA, KK, QQ, JJ, AK): never fold preflop
            elif preflop_score >= 0.75:
                p_fold = 0.0
                # Super-premiums: strongly prioritize raise / 3-bet
                if preflop_score >= 0.88 and (can_raise or can_bet):
                    p_raise = max(p_raise, p_call + 0.15, 0.65)

        # Sizing target
        pot = float(hand.get("pot", 0) or 0)
        target_amt = int(pot * multiplier)

        decision: Dict[str, Any] = {}

        # If model chooses Raise
        if (p_raise > p_call and p_raise > p_fold) and (can_raise or can_bet):
            act_type = "bet" if can_bet else "raise"
            spec = allowed[act_type]
            min_amt = int(spec.get("minAmount", 1))
            max_amt = int(spec.get("maxAmount", min_amt))
            clamped_amt = max(min_amt, min(max_amt, target_amt)) if min_amt <= max_amt else min_amt
            decision = {"type": act_type, "amount": clamped_amt}

        # If model chooses Check/Call
        elif (p_call >= p_fold or not can_fold) and (can_check or can_call):
            if can_check:
                decision = {"type": "check"}
            else:
                decision = {"type": "call"}

        # If model chooses Fold
        elif can_fold:
            decision = {"type": "fold"}
        else:
            # Fallback
            if can_check:
                decision = {"type": "check"}
            elif can_call:
                decision = {"type": "call"}
            elif can_fold:
                decision = {"type": "fold"}
            elif can_allin:
                decision = {"type": "allIn"}
            else:
                first_k = next(iter(allowed.keys()))
                decision = {"type": first_k}

        # Generate optional chat adhering to SOUL.md
        chat_msg = self.soul.generate_chat(decision["type"], street="turn")
        result = {"decision": decision}
        if chat_msg:
            result["chat"] = chat_msg
        return result
