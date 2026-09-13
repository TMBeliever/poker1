"""Run League evaluation and population matches from CLI."""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml
from src.league.league import LeagueManager
from src.tournament.tournament_deep_cfr import TournamentDeepCFRAgent


def main():
    parser = argparse.ArgumentParser(description="Run ApexPoker League Evaluation")
    parser.add_argument("--config", type=str, default="configs/league.yaml", help="Path to league config")
    parser.add_argument("--checkpoint", type=str, default="models/tournament/curriculum_p1.pt", help="Main candidate model")
    parser.add_argument("--prelim-rounds", type=int, default=3, help="Preliminary rounds")
    parser.add_argument("--hands-per-round", type=int, default=2, help="Hands per round")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-dir", type=str, default="models/league", help="League directory")
    parser.add_argument("--profiles", type=str, default="data/profiles/opponent_profiles.json", help="Path to online profiles")
    args = parser.parse_args()

    print("=" * 70)
    print(" APEXPOKER LEAGUE POPULATION TOURNAMENT")
    print("=" * 70)

    main_agent = None
    if os.path.exists(args.checkpoint):
        main_agent = TournamentDeepCFRAgent(player_id=0, num_players=6)
        main_agent.load_checkpoint(args.checkpoint)
        print(f"Loaded Main Agent from checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint found; running League with default agent.")

    league = LeagueManager(
        main_agent=main_agent,
        save_dir=args.save_dir,
        profiles_path=args.profiles if os.path.exists(args.profiles) else None,
        seed=args.seed,
    )

    if league.profile_manager:
        print(f"Loaded Online Profiles in League Pool: {len(league.profile_manager.profiles)} competitors")
    print(f"Active Archetypes in League Pool: {list(league.archetypes.keys())}")
    print(f"Executing 120-player League Tournament (seed={args.seed})...")

    results = league.run_league_tournament(
        main_agent=main_agent,
        prelim_rounds=args.prelim_rounds,
        hands_per_round=args.hands_per_round,
        seed=args.seed,
    )

    metrics = results.get("main_agent_metrics", {})
    champion = results["champion"]

    print("\n" + "=" * 70)
    print(" LEAGUE TOURNAMENT RESULTS")
    print("=" * 70)
    print(f" Champion:           Player {champion.player_id} (Final Net: {champion.stage_net_bb:+.1f} BB)")
    print(f" Main Agent (P0):    Net: {metrics.get('cumulative_net_bb', 0.0):+8.1f} BB | Rebuys: {metrics.get('rebuy_count', 0)}")
    print(f" Made Top 12:        {metrics.get('made_top12', False)}")
    print(f" Made Final:         {metrics.get('made_final', False)}")
    print(f" Is Champion:        {metrics.get('is_champion', False)}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
