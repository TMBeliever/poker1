"""Run single table 6-Max NLHE simulation from CLI."""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pokers as pkrs
import yaml

from src.agents.archetypes import TAGAgent
from src.agents.random_agent import RandomAgent
from src.tournament.poker_table import PokerTableEnv


def main():
    parser = argparse.ArgumentParser(description="Run 6-Max Poker Simulation")
    parser.add_argument("--config", type=str, default="configs/poker.yaml", help="Path to poker config")
    parser.add_argument("--players", type=int, default=6, help="Number of players")
    parser.add_argument("--hands", type=int, default=10, help="Number of hands to play")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--stake", type=float, default=200.0, help="Starting stack")
    args = parser.parse_args()

    cfg = {}
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    num_players = cfg.get("num_players", args.players)
    sb = float(cfg.get("small_blind", 500.0))
    bb = float(cfg.get("big_blind", 1000.0))
    stake = float(cfg.get("starting_stack_bb", 100.0)) * bb

    print(f"Starting 6-Max Poker Simulation: {args.hands} hands, {num_players} players, SB={sb}, BB={bb}, seed={seed}")
    table_env = PokerTableEnv(num_players=num_players, sb=sb, bb=bb, stake=stake)

    agents = [RandomAgent(i) if i > 0 else TAGAgent(0) for i in range(num_players)]

    cumulative_rewards = [0.0] * num_players
    for h in range(args.hands):
        deltas, state = table_env.play_hand(
            agents=agents,
            button=h % num_players,
            seed=seed + h,
        )
        for i, d in enumerate(deltas):
            cumulative_rewards[i] += d
        print(f"Hand {h + 1:3d} | Pot: {state.pot:6.1f} | Winner net: {max(deltas):+6.1f} BB | Seat 0: {deltas[0]:+6.1f} BB")

    print("\n" + "=" * 50)
    print("Final Cumulative Net BB:")
    for i, net in enumerate(cumulative_rewards):
        print(f"  Seat {i} ({agents[i].name}): {net:+8.2f} BB")
    print("=" * 50)


if __name__ == "__main__":
    main()
