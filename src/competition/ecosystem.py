"""Golden Pyramid Ecosystem and Opponent Field Generation for AgentPoker.

Provides realistic 120-player competitive ecosystems:
- Sharks (30%): Exploitative, highly aggressive, high pressure (LAG, Maniac, Top Sharks)
- Regulars (40%): Balanced, tight-aggressive, solid discipline (TAG, Nit, Balanced GTO)
- Fish (30%): Calling stations, passive recreationals, high VPIP, low fold (Station, Passive)
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from random import Random
from typing import Any
import json

from .strategy import StrategyParams
from .training import ARCHETYPES, profile_to_params


SHARK_ARCHETYPES = ["lag", "maniac"]
REGULAR_ARCHETYPES = ["balanced", "tight", "nit"]
FISH_ARCHETYPES = ["station"]


def classify_profile_dict(p: dict[str, Any]) -> str:
    """Classify a player profile dictionary into 'shark', 'regular', or 'fish'."""
    st = p.get("stats") or p
    hands = int(st.get("hands", 0) or 0)
    vpip_c = float(st.get("vpip_count", 0) or 0)
    pfr_c = float(st.get("pfr_count", 0) or 0)
    vpip = (vpip_c / max(1, hands)) if hands > 0 else float(st.get("vpip", 0.25) or 0.25)
    pfr = (pfr_c / max(1, hands)) if hands > 0 else float(st.get("pfr", 0.15) or 0.15)
    af = float(st.get("aggression_factor", 1.5) or 1.5)

    # Calling station / passive fish
    if vpip >= 0.35 and (pfr <= 0.12 or af < 1.0):
        return "fish"
    if vpip >= 0.40 and af < 1.5:
        return "fish"

    # Sharks: high aggression, wide range or loose-aggressive
    if (vpip >= 0.28 and pfr >= 0.18) or af >= 2.5:
        return "shark"

    # Regulars: tight-aggressive or balanced
    return "regular"


@dataclass(frozen=True)
class PyramidRatios:
    """Target ecosystem ratios."""
    shark_ratio: float = 0.30
    regular_ratio: float = 0.40
    fish_ratio: float = 0.30

    def compute_counts(self, total: int) -> tuple[int, int, int]:
        """Compute integer counts (sharks, regulars, fish) that strictly sum to `total`."""
        if total <= 0:
            return 0, 0, 0
        n_sharks = int(round(total * self.shark_ratio))
        n_regulars = int(round(total * self.regular_ratio))
        n_fish = total - n_sharks - n_regulars
        # Ensure non-negative
        if n_fish < 0:
            n_fish = 0
            n_regulars = total - n_sharks
        return n_sharks, n_regulars, n_fish


def load_profile_params_by_tier(
    profiles_path: str | Path | None,
    min_hands: int = 15,
) -> tuple[list[StrategyParams], list[StrategyParams], list[StrategyParams]]:
    """Load profiles and partition into (sharks, regulars, fish)."""
    if not profiles_path:
        return [], [], []
    path = Path(profiles_path)
    if not path.exists():
        return [], [], []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return [], [], []

    raw_list = data if isinstance(data, list) else list(data.values())
    filtered = [p for p in raw_list if isinstance(p, dict) and int(p.get("hands", 0) or (p.get("stats") or {}).get("hands", 0) or 0) >= min_hands]

    sharks: list[StrategyParams] = []
    regulars: list[StrategyParams] = []
    fish: list[StrategyParams] = []

    for p in filtered:
        tier = classify_profile_dict(p)
        params = profile_to_params(p)
        if tier == "shark":
            sharks.append(params)
        elif tier == "fish":
            fish.append(params)
        else:
            regulars.append(params)

    return sharks, regulars, fish


def build_ecosystem_pool(
    count: int,
    mode: str = "pyramid",
    profiles_path: str | Path | None = "models/opponent_profiles.json",
    min_hands: int = 15,
    ratios: PyramidRatios = PyramidRatios(),
    seed: int | None = None,
) -> list[StrategyParams]:
    """Build an opponent pool of exact length `count` matching the specified mode."""
    if count <= 0:
        return []

    rng = Random(seed) if seed is not None else Random()

    shark_archetypes = [ARCHETYPES[k] for k in SHARK_ARCHETYPES]
    regular_archetypes = [ARCHETYPES[k] for k in REGULAR_ARCHETYPES]
    fish_archetypes = [ARCHETYPES[k] for k in FISH_ARCHETYPES]
    all_archetypes = list(ARCHETYPES.values())

    p_sharks, p_regs, p_fish = load_profile_params_by_tier(profiles_path, min_hands)

    def _fill_tier(needed: int, real_pool: list[StrategyParams], arch_pool: list[StrategyParams]) -> list[StrategyParams]:
        res: list[StrategyParams] = []
        if real_pool:
            shuffled = list(real_pool)
            rng.shuffle(shuffled)
            for i in range(min(needed, len(shuffled))):
                res.append(replace(shuffled[i]))
        # Supplement remainder from archetype pool
        rem = needed - len(res)
        for i in range(rem):
            res.append(replace(arch_pool[i % len(arch_pool)]))
        return res

    if mode == "pyramid":
        n_sharks, n_regs, n_fish = ratios.compute_counts(count)
        pool: list[StrategyParams] = []
        pool.extend(_fill_tier(n_sharks, p_sharks, shark_archetypes))
        pool.extend(_fill_tier(n_regs, p_regs, regular_archetypes))
        pool.extend(_fill_tier(n_fish, p_fish, fish_archetypes))
        rng.shuffle(pool)
        return pool[:count]

    elif mode == "sharks":
        return _fill_tier(count, p_sharks, shark_archetypes)

    elif mode == "fish":
        return _fill_tier(count, p_fish, fish_archetypes)

    elif mode == "profiles" and (p_sharks or p_regs or p_fish):
        all_profs = p_sharks + p_regs + p_fish
        rng.shuffle(all_profs)
        out: list[StrategyParams] = []
        for i in range(count):
            out.append(replace(all_profs[i % len(all_profs)]))
        return out

    elif mode == "mix" and (p_sharks or p_regs or p_fish):
        all_profs = p_sharks + p_regs + p_fish
        n_prof = count // 2
        n_arch = count - n_prof
        out = []
        for i in range(n_prof):
            out.append(replace(all_profs[i % len(all_profs)]))
        for i in range(n_arch):
            out.append(replace(all_archetypes[i % len(all_archetypes)]))
        rng.shuffle(out)
        return out

    else:  # "archetypes" fallback
        out = []
        for i in range(count):
            out.append(replace(all_archetypes[i % len(all_archetypes)]))
        return out


def build_120_pyramid_field(
    profiles_path: str | Path | None = "models/opponent_profiles.json",
    min_hands: int = 15,
    seed: int | None = 42,
) -> list[StrategyParams]:
    """Convenience helper to generate standard 120-player field under Golden Pyramid ratios."""
    return build_ecosystem_pool(
        count=120,
        mode="pyramid",
        profiles_path=profiles_path,
        min_hands=min_hands,
        ratios=PyramidRatios(0.30, 0.40, 0.30),
        seed=seed,
    )
