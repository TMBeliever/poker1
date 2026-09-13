"""Canonical construction of the tournament context the strategy conditions on.

`StrategyAgent._tournament_pressure` reads rank / bb100 / stage / boundaries /
hands remaining, and those values steer safety, attack, bubble and late-stage
behaviour across all three tournament stages:
- Preliminary: 120 players, 10 rounds x 20 hands = 200 hands, Target: Top 12 (10%)
- Semifinal: 12 players, 2 tables of 6, 20 hands, Target: Top 3 per table (50%)
- Final: 6 players, 1 table of 6, 30 hands, Target: 1st Place Champion (16.7%)
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import math
from typing import Any
import warnings

from .context_data import RANK_LADDER, GOLDEN_LINE_BB100, FIELD_SIZE, QUALIFY_RANK


@dataclass
class TournamentContext:
    """Explicit, structured representation of tournament state for policy decisions."""
    stage: str
    rank: int | None
    bb100: float | None
    hands_remaining: int
    total_stage_hands: int
    stage_progress: float
    round_no: int
    target_rank: int
    cutoff_bb100: float | None
    buffer_bb100: float | None
    table_strength: float = 0.0
    context_source: str = "real"
    exploit_confidence: float = 1.0
    rank12_bb100: float | None = None
    rank13_bb100: float | None = None
    rank3_bb100: float | None = None
    rank4_bb100: float | None = None
    leader_bb100: float | None = None
    second_bb100: float | None = None
    rank1_bb100: float | None = None
    rank2_bb100: float | None = None
    cycle_no: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "stage": self.stage,
            "rank": self.rank,
            "bb100": self.bb100,
            "hands_remaining": self.hands_remaining,
            "total_stage_hands": self.total_stage_hands,
            "stage_progress": self.stage_progress,
            "round_no": self.round_no,
            "target_rank": self.target_rank,
            "cutoff_bb100": self.cutoff_bb100,
            "buffer_bb100": self.buffer_bb100,
            "table_strength": self.table_strength,
            "context_source": self.context_source,
            "exploit_confidence": self.exploit_confidence,
            "rank12_bb100": self.rank12_bb100,
            "rank13_bb100": self.rank13_bb100,
            "rank3_bb100": self.rank3_bb100,
            "rank4_bb100": self.rank4_bb100,
            "leader_bb100": self.leader_bb100,
            "second_bb100": self.second_bb100,
            "rank1_bb100": self.rank1_bb100,
            "rank2_bb100": self.rank2_bb100,
        }
        if self.cycle_no is not None:
            d["cycle_no"] = self.cycle_no
        if self.extra:
            d.update(self.extra)
        return d


def rank_from_bb100(bb100: float) -> int:
    """Implied field rank for a given BB/100, from the calibrated ladder."""
    if not RANK_LADDER:
        return FIELD_SIZE
    pts = RANK_LADDER
    if bb100 <= pts[0][0]:
        return int(pts[0][1])
    if bb100 >= pts[-1][0]:
        return int(pts[-1][1])
    for i in range(1, len(pts)):
        x0, y0 = pts[i - 1]
        x1, y1 = pts[i]
        if bb100 <= x1:
            if x1 == x0:
                return int(round(y1))
            t = (bb100 - x0) / (x1 - x0)
            return int(round(y0 + t * (y1 - y0)))
    return int(pts[-1][1])


def boundary_bb100(rank: int) -> float:
    """BB/100 required to achieve rank or better (inverse of rank_from_bb100)."""
    if not RANK_LADDER:
        return GOLDEN_LINE_BB100
    pts = RANK_LADDER
    if rank <= pts[-1][1]:
        return float(pts[-1][0])
    if rank >= pts[0][1]:
        return float(pts[0][0])
    for i, (bb, r) in enumerate(pts):
        if r <= rank:
            if r == rank:
                return float(bb)
            prev_bb, prev_r = pts[i - 1]
            t = (rank - prev_r) / (r - prev_r)
            return float(prev_bb + t * (bb - prev_bb))
    return float(pts[0][0])


def build_context(
    rank: Any,
    bb100: Any,
    rank12_bb100: Any = None,
    rank13_bb100: Any = None,
    hands_remaining: Any = 50,
    round_no: Any = 1,
    cycle_no: Any = None,
    table_strength: float = 0.0,
    stage: str | None = None,
    target_rank: int | None = None,
    total_stage_hands: int | None = None,
    rank3_bb100: Any = None,
    rank4_bb100: Any = None,
    leader_bb100: Any = None,
    second_bb100: Any = None,
    rank1_bb100: Any = None,
    rank2_bb100: Any = None,
    context_source: str = "real",
    exploit_confidence: float = 1.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Unified canonical construction of tournament context dict with stage awareness."""
    r_no = int(round_no) if round_no is not None else 1

    # Infer stage if not explicitly passed
    if stage is None:
        if r_no == 11:
            stage = "semifinal"
        elif r_no == 12:
            stage = "final"
        else:
            stage = "preliminary"

    # Stage-specific defaults
    if stage == "semifinal":
        t_rank = target_rank if target_rank is not None else 3
        tot_hands = total_stage_hands if total_stage_hands is not None else 20
        cutoff = rank4_bb100 if rank is not None and int(rank) <= 3 else rank3_bb100
    elif stage == "final":
        t_rank = target_rank if target_rank is not None else 1
        tot_hands = total_stage_hands if total_stage_hands is not None else 30
        c1 = leader_bb100 if leader_bb100 is not None else rank1_bb100
        c2 = second_bb100 if second_bb100 is not None else rank2_bb100
        cutoff = c2 if rank is not None and int(rank) == 1 else c1
    else:
        stage = "preliminary"
        t_rank = target_rank if target_rank is not None else QUALIFY_RANK
        tot_hands = total_stage_hands if total_stage_hands is not None else 200
        cutoff = rank13_bb100 if rank is not None and int(rank) <= QUALIFY_RANK else rank12_bb100

    rem = max(0, int(hands_remaining)) if hands_remaining is not None else 0
    progress = max(0.0, min(1.0, 1.0 - (rem / max(1, tot_hands))))

    buf: float | None = None
    if bb100 is not None and cutoff is not None:
        try:
            buf = float(bb100) - float(cutoff)
        except (TypeError, ValueError):
            buf = None

    t_ctx = TournamentContext(
        stage=stage,
        rank=int(rank) if rank is not None else None,
        bb100=float(bb100) if bb100 is not None else None,
        hands_remaining=rem,
        total_stage_hands=tot_hands,
        stage_progress=progress,
        round_no=r_no,
        target_rank=t_rank,
        cutoff_bb100=float(cutoff) if cutoff is not None else None,
        buffer_bb100=buf,
        table_strength=float(table_strength or 0.0),
        context_source=context_source,
        exploit_confidence=exploit_confidence,
        rank12_bb100=rank12_bb100,
        rank13_bb100=rank13_bb100,
        rank3_bb100=rank3_bb100,
        rank4_bb100=rank4_bb100,
        leader_bb100=leader_bb100,
        second_bb100=second_bb100,
        rank1_bb100=rank1_bb100 if rank1_bb100 is not None else leader_bb100,
        rank2_bb100=rank2_bb100 if rank2_bb100 is not None else second_bb100,
        cycle_no=cycle_no,
        extra=kwargs,
    )
    return t_ctx.to_dict()


def build_preliminary_context(
    rank: int | None,
    bb100: float | None,
    rank12_bb100: float | None,
    rank13_bb100: float | None,
    hands_remaining: int,
    round_no: int,
    table_strength: float = 0.0,
    cycle_no: int | None = None,
) -> dict[str, Any]:
    return build_context(
        rank=rank,
        bb100=bb100,
        rank12_bb100=rank12_bb100,
        rank13_bb100=rank13_bb100,
        hands_remaining=hands_remaining,
        round_no=round_no,
        cycle_no=cycle_no,
        table_strength=table_strength,
        stage="preliminary",
        target_rank=12,
        total_stage_hands=200,
    )


def build_semifinal_context(
    rank: int | None,
    bb100: float | None,
    rank3_bb100: float | None,
    rank4_bb100: float | None,
    hands_remaining: int,
    table_strength: float = 0.0,
) -> dict[str, Any]:
    return build_context(
        rank=rank,
        bb100=bb100,
        hands_remaining=hands_remaining,
        round_no=11,
        table_strength=table_strength,
        stage="semifinal",
        target_rank=3,
        total_stage_hands=20,
        rank3_bb100=rank3_bb100,
        rank4_bb100=rank4_bb100,
    )


def build_final_context(
    rank: int | None,
    bb100: float | None,
    leader_bb100: float | None,
    second_bb100: float | None,
    hands_remaining: int,
    table_strength: float = 0.0,
) -> dict[str, Any]:
    return build_context(
        rank=rank,
        bb100=bb100,
        hands_remaining=hands_remaining,
        round_no=12,
        table_strength=table_strength,
        stage="final",
        target_rank=1,
        total_stage_hands=30,
        leader_bb100=leader_bb100,
        second_bb100=second_bb100,
        rank1_bb100=leader_bb100,
        rank2_bb100=second_bb100,
    )


def synthetic_context(bb100: float, hands_remaining: int, round_no: int,
                      cycle_no: int | None = None, table_strength: float = 0.0) -> dict[str, Any]:
    """Context derived from the agent's own win rate, for when standings are unavailable."""
    warnings.warn(
        "Using synthetic context because standings are unavailable; ladder-based rank estimation in effect.",
        UserWarning,
        stacklevel=2,
    )
    rank = rank_from_bb100(bb100)
    return build_context(
        rank=rank,
        bb100=bb100,
        rank12_bb100=boundary_bb100(QUALIFY_RANK),
        rank13_bb100=boundary_bb100(QUALIFY_RANK + 1),
        hands_remaining=max(0, int(hands_remaining)),
        round_no=round_no,
        cycle_no=cycle_no,
        table_strength=table_strength,
        context_source="synthetic",
        exploit_confidence=0.5,
    )


def context_from_standings(rows: list[dict[str, Any]], hero_id: str | None,
                           hands_remaining: int, round_no: int,
                           cycle_no: int | None = None) -> dict[str, Any] | None:
    """Build the context from real competition standings, if they carry what we need."""
    parsed: list[tuple[str, float]] = []
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        aid = r.get("agentId") or r.get("id") or (r.get("agent") or {}).get("agentId")
        val = r.get("bb100")
        if val is None:
            bb = r.get("bb")
            hands = r.get("hands")
            if bb is not None and hands:
                val = float(bb) / float(hands) * 100.0
        if aid is None or val is None:
            continue
        try:
            parsed.append((str(aid), float(val)))
        except (TypeError, ValueError):
            continue
    if len(parsed) < QUALIFY_RANK:
        return None
    parsed.sort(key=lambda x: -x[1])
    rank = next((i for i, (aid, _) in enumerate(parsed, 1) if aid == hero_id), None)
    if rank is None:
        return None
    r12 = parsed[QUALIFY_RANK - 1][1]
    r13 = parsed[QUALIFY_RANK][1] if len(parsed) > QUALIFY_RANK else r12
    return build_context(
        rank,
        parsed[rank - 1][1],
        r12,
        r13,
        hands_remaining,
        round_no,
        cycle_no,
        context_source="real",
        exploit_confidence=1.0,
    )


@dataclass
class OpponentStrengthSignal:
    """Explicit decomposition of opponent strength into behavioral, performance, and confidence components."""
    behavioral_skill_signal: float
    performance_signal: float
    confidence: float
    posterior_strength: float


class TableStrengthModel:
    """Bayesian opponent skill posterior table strength model.
    
    Principles:
    1. Strict Hero Exclusion: Hero's own statistics, win rate, and archetype are
       strictly excluded from the table strength computation. Hero cannot influence
       their own perceived environment.
    2. Comprehensive Opponent Skill Posterior: Evaluates opponents based on:
       - behavioral_skill_signal (VPIP, PFR, 3-Bet, AF, archetype)
       - performance_signal (chip count, net BB, BB/100)
       - confidence (sample size Bayesian shrinkage weight)
       - posterior_strength (weighted combination shrunk towards neutral)
       Performance alone is strictly prohibited from being treated as true strength.
    3. Standardized Scale [-1.0, +1.0]:
       - -1.0: Extremely soft Fish table (multiple Stations / Passives)
       -  0.0: Neutral / balanced table (or unobserved / insufficient sample)
       - +1.0: Extremely tough Shark table (disciplined TAGs, LAGs, Nits)
    """

    @classmethod
    def evaluate_opponent_signals(cls, data: dict[str, Any] | Any) -> OpponentStrengthSignal:
        """Evaluate and split an individual opponent's strength into explicit signals."""
        if not data:
            return OpponentStrengthSignal(0.0, 0.0, 0.0, 0.0)

        if isinstance(data, dict):
            p = data
        else:
            p = getattr(data, "__dict__", {})

        hands = int(p.get("hands", 0) or 0)
        if hands <= 0:
            return OpponentStrengthSignal(0.0, 0.0, 0.0, 0.0)

        # Sample size shrinkage: confidence in [0, 1]
        prior_weight = 10.0
        confidence = hands / (hands + prior_weight)

        # 1. Behavioral Skill Signal (archetype, VPIP, PFR gap, AF, 3-Bet)
        arch = str(p.get("archetype", "")).lower()
        is_station = bool(p.get("is_station")) or "station" in arch
        is_passive = bool(p.get("is_passive")) or "passive" in arch
        is_nit = bool(p.get("is_nit")) or "nit" in arch
        is_maniac = bool(p.get("is_maniac")) or "maniac" in arch
        is_tag = "tag" in arch
        is_lag = "lag" in arch

        if is_station:
            arch_score = -0.95
        elif is_passive:
            arch_score = -0.85
        elif is_tag:
            arch_score = 0.85
        elif is_lag:
            arch_score = 0.75
        elif is_nit:
            arch_score = 0.55
        elif is_maniac:
            arch_score = 0.45
        elif "balanced" in arch:
            arch_score = 0.10
        else:
            arch_score = 0.0

        vpip = float(p.get("vpip", 0.25) or 0.25)
        pfr = float(p.get("pfr", 0.18) or 0.18)
        gap = vpip - pfr
        if gap > 0.12 or vpip > 0.38:
            gap_score = -min(1.0, max(0.0, (gap - 0.08) / 0.15 + (vpip - 0.30) / 0.20))
        elif vpip <= 0.28 and pfr >= 0.15:
            gap_score = 0.70
        else:
            gap_score = 0.0

        af = float(p.get("af", 1.5) or 1.5)
        if af < 0.9:
            af_score = -0.80
        elif af > 2.2:
            af_score = 0.60
        else:
            af_score = 0.10

        # 2. Performance Signal (BB/100, chip count, net BB)
        bb100_val = p.get("bb_100")
        if bb100_val is None:
            bb100_val = p.get("bb100")
        if bb100_val is None:
            net_bb = p.get("net_bb")
            if net_bb is not None and hands > 0:
                bb100_val = (float(net_bb) / hands) * 100.0
            else:
                bb100_val = 0.0
        bb100_val = float(bb100_val or 0.0)
        performance_signal = max(-1.0, min(1.0, math.tanh(bb100_val / 30.0)))

        # 3. Behavioral Skill Signal & Posterior Strength
        if arch_score != 0.0:
            raw_beh = (0.45 * arch_score + 0.15 * gap_score + 0.10 * af_score) / 0.70
            raw_combined = 0.70 * raw_beh + 0.30 * performance_signal
        else:
            raw_beh = (0.30 * gap_score + 0.25 * af_score) / 0.55
            raw_combined = 0.55 * raw_beh + 0.45 * performance_signal

        behavioral_skill_signal = max(-1.0, min(1.0, raw_beh))
        posterior_strength = max(-1.0, min(1.0, round(confidence * raw_combined, 4)))

        return OpponentStrengthSignal(
            behavioral_skill_signal=round(behavioral_skill_signal, 4),
            performance_signal=round(performance_signal, 4),
            confidence=round(confidence, 4),
            posterior_strength=posterior_strength,
        )

    @classmethod
    def evaluate_opponent_skill(cls, data: dict[str, Any] | Any) -> float:
        """Evaluate an individual opponent's skill rating in [-1.0, +1.0]."""
        return cls.evaluate_opponent_signals(data).posterior_strength

    @classmethod
    def compute(
        cls,
        table_players: list[dict[str, Any] | str],
        hero_id: str | None,
        profiles: dict[str, Any] | None = None,
    ) -> float:
        """Compute normalized table strength in [-1.0, +1.0] strictly excluding hero."""
        profiles = profiles or {}
        opponents_scores = []

        for p in table_players:
            if isinstance(p, dict):
                aid = p.get("agentId") or p.get("id") or p.get("agent_id")
                p_data = p
            else:
                aid = str(p)
                p_data = profiles.get(aid, {})

            # Strict Hero Exclusion
            if hero_id is not None and aid == hero_id:
                continue

            # Merge profile data if available
            merged = dict(p_data) if isinstance(p_data, dict) else {}
            if aid and aid in profiles and isinstance(profiles[aid], dict):
                for k, v in profiles[aid].items():
                    if k not in merged or merged[k] is None:
                        merged[k] = v

            score = cls.evaluate_opponent_skill(merged)
            opponents_scores.append(score)

        if not opponents_scores:
            return 0.0

        avg_score = sum(opponents_scores) / len(opponents_scores)
        return round(max(-1.0, min(1.0, avg_score)), 3)

    @classmethod
    def compute_table_strength_map(
        cls,
        group: list[str],
        profiles_or_st: dict[str, Any],
    ) -> dict[str, float]:
        """Compute individualized table strength for every player in group, strictly excluding themselves."""
        ts_map = {}
        for aid in group:
            ts_map[aid] = cls.compute(group, hero_id=aid, profiles=profiles_or_st)
        return ts_map

