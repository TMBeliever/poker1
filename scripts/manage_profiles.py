"""CLI Tool to manage, pull, inspect, and evaluate online opponent profiles."""

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.opponent_modeling.profile_manager import OpponentProfileManager, DEFAULT_PROFILES_PATH


def print_profiles_table(profiles: dict):
    if not profiles:
        print("No profiles found in database.")
        return

    print("=" * 95)
    print(f" 🗂️  ONLINE OPPONENT PROFILES DATABASE ({len(profiles)} COMPETITORS)")
    print("=" * 95)
    print(f"{'Name':<20} | {'Hands':<6} | {'VPIP':<7} | {'PFR':<7} | {'3-Bet':<7} | {'AF':<6} | {'Archetype':<24}")
    print("-" * 95)

    for aid, p in profiles.items():
        name = p.get("name", aid)[:18]
        hands = p.get("hands", 0)
        vpip = f"{float(p.get('vpip', 0.0)) * 100:.1f}%"
        pfr = f"{float(p.get('pfr', 0.0)) * 100:.1f}%"
        threebet = f"{float(p.get('threebet', 0.0)) * 100:.1f}%"
        af = f"{float(p.get('af', 0.0)):.2f}"
        arch = p.get("archetype", "Unknown")[:24]
        print(f"{name:<20} | {hands:<6} | {vpip:<7} | {pfr:<7} | {threebet:<7} | {af:<6} | {arch:<24}")

    print("=" * 95)


def main():
    parser = argparse.ArgumentParser(description="Manage Online Opponent Profiles for Deep CFR")
    parser.add_argument("--profiles", type=str, default=DEFAULT_PROFILES_PATH, help="Path to profile database JSON")
    parser.add_argument("--pull", action="store_true", help="Pull latest hands from online arena API and update profiles")
    parser.add_argument("--all", action="store_true", help="Pull all available hands in the competition without limit")
    parser.add_argument("--max-hands", type=int, default=1000, help="Maximum number of hands to pull from online arena (ignored if --all is set)")
    parser.add_argument("--min-hands", type=int, default=15, help="Minimum hand threshold for an agent to be profiled")
    parser.add_argument("--list", action="store_true", help="List all tracked opponent profiles in a formatted table")
    parser.add_argument("--stats", action="store_true", help="Show statistical summary of the competitive field")
    parser.add_argument("--vector", type=str, default=None, help="Display the normalized neural feature vector for an agent name")
    args = parser.parse_args()

    pm = OpponentProfileManager(args.profiles)

    if args.pull:
        limit = None if args.all else args.max_hands
        print(f"[Profiles] Pulling {'ALL available' if args.all else f'up to {args.max_hands}'} hands from online arena...")
        count = pm.pull_online_hands(max_hands=limit, min_hands=args.min_hands)
        print(f"[Profiles] Update complete: {count} active profiles saved to {args.profiles}")

    if args.list:
        print_profiles_table(pm.profiles)

    if args.stats:
        s = pm.summary()
        print("\n" + "=" * 50)
        print(" 📊 COMPETITIVE FIELD DISTRIBUTION")
        print("=" * 50)
        print(f"Total Profiled Competitors : {s['total_profiles']}")
        print(f"Field Mean VPIP            : {s['mean_vpip'] * 100:.1f}%")
        print(f"Field Mean PFR             : {s['mean_pfr'] * 100:.1f}%")
        print("\nArchetype Breakdown:")
        for arch, count in s["archetype_distribution"].items():
            print(f" • {arch:<24}: {count} ({count / max(1, s['total_profiles']) * 100:.1f}%)")
        print("=" * 50 + "\n")

    if args.vector:
        vec7 = pm.to_feature_vector(args.vector, dim=7)
        vec20 = pm.to_feature_vector(args.vector, dim=20)
        p = pm.get_profile(args.vector)
        print("\n" + "=" * 65)
        print(f" 🧬 NEURAL FEATURE VECTOR FOR: {p.get('name', args.vector)}")
        print("=" * 65)
        print(f"Archetype: {p.get('archetype')}")
        print(f"7-Dim Vector (VPIP, PFR, AFq, 3Bet, Fold, Pressure, Stage):")
        print("  ", np.array2string(vec7, precision=3, separator=", "))
        print(f"\n20-Dim Conditioning Vector for EnhancedPokerNetwork:")
        print("  ", np.array2string(vec20, precision=3, separator=", "))
        print("=" * 65 + "\n")

    if not any([args.pull, args.list, args.stats, args.vector]):
        # Default action
        print_profiles_table(pm.profiles)


if __name__ == "__main__":
    main()
