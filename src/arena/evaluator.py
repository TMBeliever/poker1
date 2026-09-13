"""Arena Evaluator executing cross-project tournaments and 6-max head-to-head battles."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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
from src.arena.adapter import build_competitor
from src.tournament.poker_table import PokerTableEnv
from src.tournament.tournament_env import TournamentEnv


DEFAULT_ARCHETYPES = [
    TAGAgent,
    LAGAgent,
    NitAgent,
    CallingStationAgent,
    ManiacAgent,
    AdaptiveRegAgent,
    OverfolderAgent,
    OverblufferAgent,
]


class ArenaEvaluator:
    """Orchestrates head-to-head tournaments and 6-max matches between arbitrary models."""

    def __init__(
        self,
        competitor_specs: List[Tuple[str, str]],
        sb: float = 1.0,
        bb: float = 2.0,
        stake: float = 200.0,
        profiles_path: Optional[str] = "data/profiles/opponent_profiles.json",
    ):
        """Args:
        competitor_specs: List of (display_name, path_or_spec), e.g.:
          [("Our_Stage2", "models/tournament/curriculum_p5_full.pt"),
           ("Final2_Champion", "/Users/liang/files/seft/agentpoker_final 2/models/champion.json"),
           ("Our_Stage1", "models/base/base_checkpoint_iter_2000.pt")]
        profiles_path: Optional path to opponent profiles json file.
        """
        self.competitor_specs = competitor_specs
        self.sb = sb
        self.bb = bb
        self.stake = stake
        self.profiles_path = profiles_path

    def run_mtt_arena(
        self,
        num_tournaments: int = 3,
        seeds: Optional[List[int]] = None,
        total_players: int = 120,
    ) -> Dict[str, Any]:
        """Execute full 120-player Swiss Multi-Stage Tournaments containing all competitors."""
        if seeds is None:
            seeds = [42, 142, 242, 342, 442][:num_tournaments]

        k = len(self.competitor_specs)
        stats = {
            i: {
                "name": self.competitor_specs[i][0],
                "spec": self.competitor_specs[i][1],
                "net_bbs": [],
                "rebuys": [],
                "top12_flags": [],
                "top6_flags": [],
                "champ_flags": [],
                "final_ranks": [],
            }
            for i in range(k)
        }

        print("=" * 75)
        print(" 🏆 STARTING 120-PLAYER CROSS-PROJECT MULTI-STAGE TOURNAMENT ARENA")
        print("=" * 75)
        for i, (name, spec) in enumerate(self.competitor_specs):
            print(f" • Competitor #{i+1} [Seat {i}]: {name} ({spec})")
        print(f" • Remaining {total_players - k} seats filled with canonical archetypes (TAG/LAG/Nit/Station...)")
        print(f" • Tournaments to run: {num_tournaments} (Seeds: {seeds})")
        print("=" * 75)

        for t_idx, s in enumerate(seeds):
            print(f"\n--- Tournament #{t_idx+1} (Seed {s}) ---")
            env = TournamentEnv(
                num_players=total_players,
                table_size=6,
                prelim_rounds=10,
                hands_per_prelim_round=20,
                semifinal_hands=20,
                final_hands=30,
                seed=s,
            )

            # Build agents
            agents = {}
            for i in range(k):
                name, spec = self.competitor_specs[i]
                agent, _ = build_competitor(name, spec, player_id=i, sb=self.sb, bb=self.bb, stake=self.stake)
                agents[i] = agent

            # Fill remaining seats using real online profiles if available
            needed = total_players - k
            if self.profiles_path and os.path.exists(self.profiles_path):
                from src.opponent_modeling.profile_manager import OpponentProfileManager
                pm = OpponentProfileManager(self.profiles_path)
                field_agents = pm.build_field_agents(needed, sb=self.sb, bb=self.bb, stake=self.stake, seed=s)
                for idx, agent in enumerate(field_agents):
                    seat = k + idx
                    agent.player_id = seat
                    agents[seat] = agent
            else:
                for seat in range(k, total_players):
                    arch_cls = DEFAULT_ARCHETYPES[seat % len(DEFAULT_ARCHETYPES)]
                    agents[seat] = arch_cls(seat)

            res = env.run_full_tournament(agents)

            # Gather results
            for i in range(k):
                p_rec = env.players[i]
                stats[i]["net_bbs"].append(p_rec.cumulative_net_bb)
                stats[i]["rebuys"].append(p_rec.rebuy_count)

                made_top12 = any(p.player_id == i for p in res["preliminary_top12"])
                made_top6 = any(p.player_id == i for p in res["final_standings"])
                is_champ = res["champion"].player_id == i

                stats[i]["top12_flags"].append(made_top12)
                stats[i]["top6_flags"].append(made_top6)
                stats[i]["champ_flags"].append(is_champ)

                # Final rank approximation
                if is_champ:
                    rank = 1
                elif made_top6:
                    rank = next(idx + 1 for idx, p in enumerate(res["final_standings"]) if p.player_id == i)
                elif made_top12:
                    rank = 7
                else:
                    rank = 20  # Field rank
                stats[i]["final_ranks"].append(rank)

                print(
                    f"  [{stats[i]['name']:<20}] Net: {p_rec.cumulative_net_bb:+6.1f} BB | "
                    f"Rebuys: {p_rec.rebuy_count} | Top12: {str(made_top12):<5} | FT: {str(made_top6):<5} | Champ: {str(is_champ):<5}"
                )

        # Aggregate summaries
        summary = []
        for i in range(k):
            net_list = stats[i]["net_bbs"]
            mean_net = float(np.mean(net_list))
            summary.append(
                {
                    "rank_order": 0,
                    "name": stats[i]["name"],
                    "spec": stats[i]["spec"],
                    "mean_net_bb": mean_net,
                    "bb_per_100": mean_net / 2.0,  # 200 hands prelim approximation
                    "mean_rebuys": float(np.mean(stats[i]["rebuys"])),
                    "p_top12": float(np.mean(stats[i]["top12_flags"])) * 100.0,
                    "p_top6": float(np.mean(stats[i]["top6_flags"])) * 100.0,
                    "p_champ": float(np.mean(stats[i]["champ_flags"])) * 100.0,
                    "avg_rank": float(np.mean(stats[i]["final_ranks"])),
                    "net_bbs": net_list,
                }
            )

        summary.sort(key=lambda x: x["mean_net_bb"], reverse=True)
        for idx, item in enumerate(summary):
            item["rank_order"] = idx + 1

        return {
            "mode": "120_player_tournament",
            "num_tournaments": num_tournaments,
            "seeds": seeds,
            "leaderboard": summary,
        }

    def run_6max_arena(
        self,
        total_hands: int = 100,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Execute direct 6-max head-to-head ring table battle between the competitors."""
        k = len(self.competitor_specs)
        if k > 6:
            raise ValueError("Direct 6-max ring table supports at most 6 competitors simultaneously.")

        # Build table agents
        agents = []
        names = []
        for i in range(6):
            if i < k:
                name, spec = self.competitor_specs[i]
            else:
                name = f"Benchmark_{DEFAULT_ARCHETYPES[i].__name__}"
                spec = DEFAULT_ARCHETYPES[i].__name__
            agent, disp_name = build_competitor(name, spec, player_id=i, sb=self.sb, bb=self.bb, stake=self.stake)
            agents.append(agent)
            names.append(disp_name)

        print("=" * 75)
        print(f" 🥊 STARTING 6-MAX DIRECT RING TABLE BATTLE ({total_hands} HANDS)")
        print("=" * 75)
        for seat, name in enumerate(names):
            print(f" • Seat {seat}: {name}")
        print("=" * 75)

        table_env = PokerTableEnv(num_players=6, sb=self.sb, bb=self.bb, stake=self.stake)
        cumulative_deltas = [0.0] * 6

        rng = np.random.RandomState(seed)
        for h in range(total_hands):
            button = h % 6
            hand_seed = int(rng.randint(0, 10000000))
            deltas_bb, _ = table_env.play_hand(agents=agents, button=button, seed=hand_seed)
            for i in range(6):
                cumulative_deltas[i] += deltas_bb[i]

        leaderboard = []
        for i in range(6):
            net = cumulative_deltas[i]
            bb100 = (net / max(1, total_hands)) * 100.0
            leaderboard.append(
                {
                    "seat": i,
                    "name": names[i],
                    "total_net_bb": net,
                    "bb_per_100": bb100,
                    "hands": total_hands,
                }
            )

        leaderboard.sort(key=lambda x: x["total_net_bb"], reverse=True)
        for idx, item in enumerate(leaderboard):
            item["rank_order"] = idx + 1

        return {
            "mode": "6max_ring_battle",
            "total_hands": total_hands,
            "seed": seed,
            "leaderboard": leaderboard,
        }

    @staticmethod
    def print_leaderboard(results: Dict[str, Any]) -> None:
        """Render a formatted ASCII leaderboard table with medal indicators."""
        mode = results.get("mode")
        lb = results.get("leaderboard", [])

        medals = ["🥇", "🥈", "🥉", " 4", " 5", " 6"]
        print("\n" + "=" * 85)
        if mode == "120_player_tournament":
            print(f" 🏆 ARENA TOURNAMENT LEADERBOARD ({results.get('num_tournaments')} Tournaments)")
            print("=" * 85)
            print(f"{'Rank':<5} | {'Competitor Name':<28} | {'Net BB':<10} | {'BB/100':<9} | {'Top 12%':<8} | {'FT%':<6} | {'Champ%':<7} | {'Rebuys'}")
            print("-" * 85)
            for item in lb:
                rk = medals[item["rank_order"] - 1] if item["rank_order"] <= 3 else f" #{item['rank_order']}"
                print(
                    f"{rk:<5} | {item['name']:<28} | {item['mean_net_bb']:+8.1f} BB | {item['bb_per_100']:+7.1f} | "
                    f"{item['p_top12']:6.1f}% | {item['p_top6']:5.1f}% | {item['p_champ']:5.1f}% | {item['mean_rebuys']:.2f}"
                )
        else:
            print(f" 🥊 6-MAX HEAD-TO-HEAD BATTLE LEADERBOARD ({results.get('total_hands')} Hands)")
            print("=" * 85)
            print(f"{'Rank':<5} | {'Competitor Name':<28} | {'Total Net BB':<14} | {'BB/100':<12} | {'Hands'}")
            print("-" * 85)
            for item in lb:
                rk = medals[item["rank_order"] - 1] if item["rank_order"] <= 3 else f" #{item['rank_order']}"
                print(
                    f"{rk:<5} | {item['name']:<28} | {item['total_net_bb']:+12.1f} BB | {item['bb_per_100']:+10.1f} | {item['hands']}"
                )
        print("=" * 85 + "\n")
