"""Native Opponent Profile Manager for Deep CFR.

Responsible for:
1. Connecting to online arena to pull real match hands without external dependencies.
2. Generating and maintaining Bayesian-smoothed opponent profiles.
3. Converting player profiles into normalized tensor features for EnhancedPokerNetwork.
4. Supplying realistic digital-twin tournament fields for 120-player MTT simulations.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.agents.profiled_agent import ProfiledAgent
from src.competition.profiler import OpponentProfiler
from src.competition.protocol import AgentPokerClient, Config


DEFAULT_PROFILES_PATH = "data/profiles/opponent_profiles.json"
DEFAULT_HANDS_PATH = "data/processed/hands.jsonl"


class OpponentProfileManager:
    """Central authority for opponent profiles in the Deep CFR architecture."""

    def __init__(self, profiles_path: str | Path = DEFAULT_PROFILES_PATH):
        self.profiles_path = Path(profiles_path)
        self.profiles: Dict[str, Dict[str, Any]] = {}
        self.name_to_id: Dict[str, str] = {}
        self.load_profiles()

    def load_profiles(self, path: Optional[str | Path] = None) -> int:
        """Load profile database from disk."""
        target = Path(path) if path else self.profiles_path
        if not target.exists():
            return 0
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self.profiles = data
                self.name_to_id = {
                    info.get("name", aid): aid for aid, info in data.items()
                }
                return len(self.profiles)
        except Exception as e:
            print(f"[ProfileManager] Warning: failed to load {target}: {e}")
        return 0

    def pull_online_hands(
        self,
        max_hands: int = 1000,
        save_hands_path: str | Path = DEFAULT_HANDS_PATH,
        out_profiles_path: Optional[str | Path] = None,
        min_hands: int = 30,
    ) -> int:
        """Fetch completed hands directly from online competition API and update profiles."""
        out_path = Path(out_profiles_path) if out_profiles_path else self.profiles_path
        client = AgentPokerClient(Config())
        cid = client.cfg.competition_id
        if not cid:
            raise RuntimeError("AGENTPOKER_COMPETITION_ID is not configured in .env")

        print(f"[ProfileManager] Connecting to {client.cfg.base_url} (competition: {cid})...")
        profiler = OpponentProfiler()
        fetched = profiler.pull_competition_hands(
            client=client,
            cid=cid,
            max_hands=max_hands,
            save_hands_path=save_hands_path,
        )
        print(f"[ProfileManager] Successfully ingested {fetched} hands from online server.")

        profiles = profiler.export(
            out_path=out_path,
            min_hands=min_hands,
            filter_afk=True,
        )
        self.profiles = profiles
        self.name_to_id = {
            info.get("name", aid): aid for aid, info in profiles.items()
        }
        print(f"[ProfileManager] Saved {len(profiles)} cleaned profiles -> {out_path}")
        return len(profiles)

    def get_profile(self, identifier: str) -> Dict[str, Any]:
        """Look up profile by agentId or agent name. Returns baseline defaults if unknown."""
        if identifier in self.profiles:
            return self.profiles[identifier]
        if identifier in self.name_to_id:
            return self.profiles[self.name_to_id[identifier]]
        
        # Fallback population prior
        return {
            "name": identifier,
            "vpip": 0.25,
            "pfr": 0.18,
            "threebet": 0.08,
            "af": 2.0,
            "cbet_flop": 0.55,
            "fold_to_cbet": 0.45,
            "open_size_bb": 2.35,
            "cbet_size": 0.50,
            "value_bet_size": 0.65,
            "raise_size": 0.70,
            "archetype": "Unknown (Population Prior)",
        }

    def to_feature_vector(
        self,
        identifier: str,
        dim: int = 7,
        stack_pressure: float = 1.0,
        stage_val: float = 0.0,
    ) -> np.ndarray:
        """Generate normalized continuous vector for neural network feature conditioning."""
        p = self.get_profile(identifier)
        vpip = float(p.get("vpip", 0.25))
        pfr = float(p.get("pfr", 0.18))
        af = float(p.get("af", 2.0))
        # AFq proxy: AF / (AF + 1.0)
        afq = af / (af + 1.0)
        threebet = float(p.get("threebet", 0.08))
        fold_rate = float(p.get("fold", 0.65))

        if dim == 7:
            return np.array(
                [
                    vpip,
                    pfr,
                    afq,
                    threebet,
                    fold_rate,
                    math.tanh(stack_pressure),
                    stage_val,
                ],
                dtype=np.float32,
            )

        # 20-dim detailed representation for EnhancedPokerNetwork
        v_20 = np.zeros(20, dtype=np.float32)
        v_20[0] = vpip
        v_20[1] = pfr
        v_20[2] = afq
        v_20[3] = threebet
        v_20[4] = fold_rate
        v_20[5] = float(p.get("cbet_flop", 0.55))
        v_20[6] = float(p.get("fold_to_cbet", 0.45))
        v_20[7] = float(p.get("turn_barrel", 0.45))
        v_20[8] = float(p.get("wtsd", 0.30))
        v_20[9] = float(p.get("wsd", 0.50))
        v_20[10] = min(1.0, float(p.get("open_size_bb", 2.5)) / 5.0)
        v_20[11] = float(p.get("cbet_size", 0.5))
        v_20[12] = float(p.get("value_bet_size", 0.65))
        v_20[13] = float(p.get("raise_size", 0.70))
        v_20[14] = math.tanh(stack_pressure)
        v_20[15] = stage_val
        # 16..19 reserved for one-hot archetype indicator
        arch = p.get("archetype", "")
        if "TAG" in arch:
            v_20[16] = 1.0
        elif "LAG" in arch:
            v_20[17] = 1.0
        elif "Nit" in arch:
            v_20[18] = 1.0
        elif "Maniac" in arch or "Station" in arch:
            v_20[19] = 1.0

        return v_20

    def build_field_agents(
        self,
        count: int,
        sb: float = 1.0,
        bb: float = 2.0,
        stake: float = 200.0,
        seed: Optional[int] = None,
    ) -> List[ProfiledAgent]:
        """Instantiate `count` playable ProfiledAgents matching the online field distribution."""
        if count <= 0:
            return []

        profile_list = list(self.profiles.values())
        if not profile_list:
            # Fallback if no profiles exist
            profile_list = [self.get_profile(f"Placeholder_{i}") for i in range(count)]

        agents: List[ProfiledAgent] = []
        for i in range(count):
            p_data = profile_list[i % len(profile_list)]
            agent_seed = None if seed is None else (seed + i * 37)
            agent = ProfiledAgent(
                player_id=i,
                profile=p_data,
                sb=sb,
                bb=bb,
                stake=stake,
                seed=agent_seed,
            )
            agents.append(agent)

        return agents

    def summary(self) -> Dict[str, Any]:
        """Return aggregated summary of known profiles."""
        total = len(self.profiles)
        archetypes: Dict[str, int] = {}
        vpips: List[float] = []
        pfrs: List[float] = []

        for p in self.profiles.values():
            arch = p.get("archetype", "Unknown").split("(")[0].strip()
            archetypes[arch] = archetypes.get(arch, 0) + 1
            vpips.append(float(p.get("vpip", 0.0)))
            pfrs.append(float(p.get("pfr", 0.0)))

        return {
            "total_profiles": total,
            "archetype_distribution": archetypes,
            "mean_vpip": float(np.mean(vpips)) if vpips else 0.0,
            "mean_pfr": float(np.mean(pfrs)) if pfrs else 0.0,
        }
