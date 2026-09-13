"""League Training and Population Manager for ApexPoker."""

from enum import Enum, auto
import os
import random
from typing import Any, Dict, List, Optional

from src.agents.archetypes import (
    AdaptiveRegAgent,
    CallingStationAgent,
    LAGAgent,
    ManiacAgent,
    NitAgent,
    OverblufferAgent,
    OverfolderAgent,
    TAGAgent,
)
from src.agents.profiled_agent import ProfiledAgent
from src.agents.random_agent import RandomAgent
from src.opponent_modeling.profile_manager import OpponentProfileManager
from src.tournament.state import PlayerRecord
from src.tournament.tournament_env import TournamentEnv


class LeagueMemberRole(Enum):
    MAIN = auto()
    PREVIOUS_MAIN = auto()
    FROZEN_CHECKPOINT = auto()
    MAIN_EXPLOITER = auto()
    HUMAN_STYLE = auto()


class LeagueManager:
    """Manages league population, historical frozen checkpoints, and evaluation match-ups.
    
    Roles supported:
    - Main Agent: currently active training model
    - Previous Main: snapshot of preceding iteration
    - Frozen Pool: historical checkpoints preventing policy collapse
    - Main Exploiter: counter-strategy agent targeting Main
    - Human-Style Archetypes: Nit, TAG, LAG, Station, Maniac, Overfolder, Overbluffer, Adaptive Reg
    """

    def __init__(
        self,
        main_agent: Optional[Any] = None,
        save_dir: str = "models/league",
        profiles_path: Optional[str] = None,
        seed: int = 42,
    ):
        self.save_dir = save_dir
        self.seed = seed
        self.rng = random.Random(seed)
        os.makedirs(self.save_dir, exist_ok=True)

        self.main_agent = main_agent
        self.previous_main: Optional[Any] = None
        self.frozen_checkpoints: List[str] = []
        self.main_exploiters: List[Any] = []
        self.profile_manager: Optional[OpponentProfileManager] = None

        if profiles_path and os.path.exists(profiles_path):
            self.load_profiles(profiles_path)
        
        # Instantiate 8 canonical style archetypes
        self.archetypes: Dict[str, Any] = {
            "Nit": NitAgent(player_id=0),
            "TAG": TAGAgent(player_id=0),
            "LAG": LAGAgent(player_id=0),
            "CallingStation": CallingStationAgent(player_id=0),
            "Maniac": ManiacAgent(player_id=0),
            "Overfolder": OverfolderAgent(player_id=0),
            "Overbluffer": OverblufferAgent(player_id=0),
            "AdaptiveReg": AdaptiveRegAgent(player_id=0),
        }

    def load_profiles(self, profiles_path: str) -> int:
        """Load empirical online Bayesian user profiles into the league."""
        self.profile_manager = OpponentProfileManager(profiles_path)
        return len(self.profile_manager.profiles)

    def add_frozen_checkpoint(self, checkpoint_path: str) -> None:
        """Register a frozen historical checkpoint into the league pool."""
        if os.path.exists(checkpoint_path) and checkpoint_path not in self.frozen_checkpoints:
            self.frozen_checkpoints.append(checkpoint_path)

    def update_main_checkpoint(self, checkpoint_path: str, new_main_agent: Any) -> None:
        """Promote new checkpoint to Main, archiving current Main to Previous Main & Frozen Pool."""
        if self.main_agent is not None:
            self.previous_main = self.main_agent

        self.add_frozen_checkpoint(checkpoint_path)
        self.main_agent = new_main_agent

    def sample_opponent_population(self, total_opponents: int) -> List[Any]:
        """Sample a balanced pool of opponents from Real Profiles, Archetypes, and Random."""
        pool = list(self.archetypes.values())
        opponents = []
        has_profiles = bool(self.profile_manager and self.profile_manager.profiles)
        profile_list = list(self.profile_manager.profiles.values()) if has_profiles else []

        for i in range(total_opponents):
            roll = self.rng.random()
            if has_profiles and roll < 0.5:
                # 50% real online opponent profiles
                p_data = self.rng.choice(profile_list)
                opponents.append(ProfiledAgent(player_id=i, profile=p_data, seed=self.rng.randint(0, 100000)))
            elif roll < 0.85:
                # 35% canonical archetypes
                archetype = self.rng.choice(pool)
                opponents.append(archetype)
            else:
                # 15% random exploration agents
                opponents.append(RandomAgent(player_id=i))

        return opponents

    def build_120_tournament_agent_map(
        self,
        main_seat: int = 0,
        main_agent: Optional[Any] = None,
    ) -> Dict[int, Any]:
        """Create an agent map for a 120-player tournament with Main agent and diverse league members."""
        active_main = main_agent or self.main_agent or RandomAgent(player_id=main_seat)
        agent_map = {main_seat: active_main}

        has_profiles = bool(self.profile_manager and self.profile_manager.profiles)
        profile_list = list(self.profile_manager.profiles.values()) if has_profiles else []
        archetype_list = list(self.archetypes.values())

        # Distribute remaining 119 seats among real competitor profiles and archetypes
        for seat in range(120):
            if seat == main_seat:
                continue

            if has_profiles and (seat - 1) < len(profile_list):
                # Seat real competitor digital twin
                p_data = profile_list[(seat - 1) % len(profile_list)]
                agent_map[seat] = ProfiledAgent(
                    player_id=seat,
                    profile=p_data,
                    seed=self.seed + seat * 50,
                )
            else:
                # Rotate across archetypes and baseline agents
                arch_template = archetype_list[seat % len(archetype_list)]
                agent_map[seat] = arch_template.__class__(player_id=seat)

        return agent_map

    def run_league_tournament(
        self,
        main_agent: Optional[Any] = None,
        prelim_rounds: int = 2,
        hands_per_round: int = 2,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run a deterministic league tournament evaluating the Main agent against the population."""
        tourn_seed = seed or self.seed
        env = TournamentEnv(
            num_players=120,
            table_size=6,
            prelim_rounds=prelim_rounds,
            hands_per_prelim_round=hands_per_round,
            semifinal_hands=2,
            final_hands=3,
            seed=tourn_seed,
        )

        agent_map = self.build_120_tournament_agent_map(main_seat=0, main_agent=main_agent)
        results = env.run_full_tournament(agent_map)

        # Extract Main Agent metrics
        main_record = env.players[0]
        results["main_agent_metrics"] = {
            "player_id": 0,
            "cumulative_net_bb": main_record.cumulative_net_bb,
            "rebuy_count": main_record.rebuy_count,
            "is_champion": results["champion"].player_id == 0,
            "made_top12": any(p.player_id == 0 for p in results["preliminary_top12"]),
            "made_final": any(p.player_id == 0 for p in results["final_standings"]),
        }
        return results
