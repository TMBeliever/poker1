"""Run full 120-player ApexPoker Multi-Stage Tournament from CLI."""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml

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
from src.agents.random_agent import RandomAgent
from src.tournament.state import Stage
from src.tournament.tournament_deep_cfr import TournamentDeepCFRAgent
from src.tournament.tournament_env import TournamentEnv


def build_tournament_population(num_players: int, checkpoint_path: Optional[str] = None):
    archetypes = [
        NitAgent,
        TAGAgent,
        LAGAgent,
        CallingStationAgent,
        ManiacAgent,
        OverfolderAgent,
        OverblufferAgent,
        AdaptiveRegAgent,
    ]
    agents = {}

    # Player 0: Candidate model or strong benchmark
    if checkpoint_path and os.path.exists(checkpoint_path):
        agent0 = TournamentDeepCFRAgent(player_id=0, num_players=6)
        agent0.load_checkpoint(checkpoint_path)
        agents[0] = agent0
    else:
        agents[0] = TAGAgent(0)

    for i in range(1, num_players):
        arch_cls = archetypes[i % len(archetypes)]
        agents[i] = arch_cls(i)

    return agents


def main():
    parser = argparse.ArgumentParser(description="ApexPoker 120-Player Tournament Simulator")
    parser.add_argument("--config", type=str, default="configs/tournament.yaml", help="Path to tournament config")
    parser.add_argument("--players", type=int, default=120, help="Number of players (default 120)")
    parser.add_argument("--seed", type=int, default=42, help="Tournament random seed")
    parser.add_argument("--prelim-rounds", type=int, default=10, help="Number of prelim rounds (default 10)")
    parser.add_argument("--hands-per-round", type=int, default=20, help="Hands per prelim round (default 20)")
    parser.add_argument("--semifinal-hands", type=int, default=20, help="Hands per semifinal table (default 20)")
    parser.add_argument("--final-hands", type=int, default=30, help="Hands in final table (default 30)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional model checkpoint for Player 0")
    args = parser.parse_args()

    # Load YAML config if present
    cfg = {}
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    total_players = cfg.get("num_players", args.players)
    sb = float(cfg.get("small_blind", 500.0))
    bb = float(cfg.get("big_blind", 1000.0))
    starting_stack_bb = float(cfg.get("starting_stack_bb", 50.0))
    seed = cfg.get("seed", args.seed)
    prelim_cfg = cfg.get("preliminary", {})
    prelim_rounds = prelim_cfg.get("num_rounds", args.prelim_rounds)
    hands_per_round = prelim_cfg.get("hands_per_round", args.hands_per_round)

    semi_cfg = cfg.get("semifinal", {})
    semi_hands = semi_cfg.get("hands_per_table", args.semifinal_hands)

    final_cfg = cfg.get("final", {})
    final_hands = final_cfg.get("hands", args.final_hands)

    print("=" * 70)
    print(" APEXPOKER 120-PLAYER MULTI-STAGE TOURNAMENT")
    print("=" * 70)
    print(f" Total Players:        {total_players} (Blinds: SB {sb:.0f} / BB {bb:.0f}, Stack: {starting_stack_bb * bb:.0f} chips = {starting_stack_bb:.0f} BB)")
    print(f" Preliminary Format:   {prelim_rounds} rounds x {hands_per_round} hands = {prelim_rounds * hands_per_round} hands (R1-R3 Random, R4-R10 Swiss)")
    print(f" Semifinal Format:     Top 12 Snake-Seeded into Tables A & B ({semi_hands} hands)")
    print(f" Final Format:         Top 3 from each table -> 6 Players ({final_hands} hands)")
    print(f" Random Seed:          {seed}")
    print("=" * 70)

    env = TournamentEnv(
        num_players=total_players,
        table_size=6,
        starting_stack_bb=starting_stack_bb,
        sb=sb,
        bb=bb,
        prelim_rounds=prelim_rounds,
        hands_per_prelim_round=hands_per_round,
        semifinal_hands=semi_hands,
        final_hands=final_hands,
        seed=seed,
    )

    agents = build_tournament_population(total_players, checkpoint_path=args.checkpoint)

    # 1. Preliminary
    print("\n[1/3] Running Preliminary Stage (120 players across 20 tables)...")
    prelim_standings = env.run_preliminary(agents)
    top12 = prelim_standings[:12]
    print(f"  Preliminary completed! Top Qualifier: Player {top12[0].player_id} ({top12[0].stage_net_bb:+.1f} BB, {top12[0].rebuy_count} rebuys)")
    print("  Top 12 Qualifiers for Semifinals:")
    for rank, p in enumerate(top12, start=1):
        print(f"    Rank {rank:2d}: Player {p.player_id:3d} | Net: {p.stage_net_bb:+8.1f} BB | Rebuys: {p.rebuy_count:2d}")

    # 2. Semifinal
    print("\n[2/3] Running Semifinal Stage (12 players snake-seeded into Tables A & B)...")
    semi_a, semi_b = env.run_semifinal(agents)
    print(f"  Table A Qualifiers: {[p.player_id for p in semi_a[:3]]}")
    print(f"  Table B Qualifiers: {[p.player_id for p in semi_b[:3]]}")

    # 3. Final
    print(f"\n[3/3] Running Final Stage (6 players, {final_hands} hands)...")
    champion, final_standings = env.run_final(agents)

    print("\n" + "=" * 70)
    print(f" TOURNAMENT CHAMPION: Player {champion.player_id} (Final Net: {champion.stage_net_bb:+.1f} BB)")
    print("=" * 70)
    print(" Final Table Standings:")
    for rank, p in enumerate(final_standings, start=1):
        indicator = "★ CHAMPION ★" if rank == 1 else ""
        print(f"  {rank:2d}. Player {p.player_id:3d} | Net: {p.stage_net_bb:+8.1f} BB | Rebuys: {p.rebuy_count:2d} {indicator}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
