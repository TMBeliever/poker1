"""Comprehensive Multi-Stage Checkpoint Evaluation against Benchmark Pool."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
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
from src.tournament.tournament_deep_cfr import TournamentDeepCFRAgent
from src.tournament.tournament_env import TournamentEnv


def evaluate_checkpoint(
    checkpoint_path: str,
    num_tournaments: int = 3,
    seed: int = 42,
    output_json: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute complete 5-stage benchmark evaluation on a target checkpoint."""
    print("=" * 70)
    print(f" EVALUATING CHECKPOINT: {checkpoint_path}")
    print("=" * 70)

    agent = TournamentDeepCFRAgent(player_id=0, num_players=6)
    if os.path.exists(checkpoint_path):
        agent.load_checkpoint(checkpoint_path)
        print(" Successfully loaded checkpoint weights.")
    else:
        print(f" Warning: Checkpoint path '{checkpoint_path}' not found, evaluating fresh agent.")

    archetype_classes = [
        NitAgent,
        TAGAgent,
        LAGAgent,
        CallingStationAgent,
        ManiacAgent,
        OverfolderAgent,
        OverblufferAgent,
        AdaptiveRegAgent,
    ]

    all_net_bbs = []
    top12_count = 0
    final_count = 0
    champ_count = 0

    for tourn_idx in range(num_tournaments):
        tourn_seed = seed + tourn_idx * 100
        env = TournamentEnv(
            num_players=120,
            table_size=6,
            prelim_rounds=2,
            hands_per_prelim_round=2,
            semifinal_hands=2,
            final_hands=3,
            seed=tourn_seed,
        )

        # Build diverse tournament population
        agents = {0: agent}
        for seat in range(1, 120):
            arch_cls = archetype_classes[seat % len(archetype_classes)]
            agents[seat] = arch_cls(seat)

        res = env.run_full_tournament(agents)
        my_record = env.players[0]
        net_bb = my_record.cumulative_net_bb
        all_net_bbs.append(net_bb)

        made_top12 = any(p.player_id == 0 for p in res["preliminary_top12"])
        made_final = any(p.player_id == 0 for p in res["final_standings"])
        is_champ = res["champion"].player_id == 0

        if made_top12:
            top12_count += 1
        if made_final:
            final_count += 1
        if is_champ:
            champ_count += 1

        print(f"  Tournament {tourn_idx + 1}/{num_tournaments}: Net={net_bb:+7.1f} BB | Top12={made_top12} | Final={made_final} | Champ={is_champ}")

    mean_net = float(np.mean(all_net_bbs))
    variance = float(np.var(all_net_bbs))
    worst_5_pct = float(np.percentile(all_net_bbs, 5))
    p_top12 = top12_count / num_tournaments
    p_top6 = final_count / num_tournaments
    p_champ = champ_count / num_tournaments
    bb_per_100 = mean_net / max(1, 14) * 100.0  # Approx per 100 hands

    metrics = {
        "checkpoint": checkpoint_path,
        "tournaments_evaluated": num_tournaments,
        "mean_net_bb": round(mean_net, 2),
        "bb_per_100": round(bb_per_100, 2),
        "variance": round(variance, 2),
        "worst_5_percent": round(worst_5_pct, 2),
        "p_top12": round(p_top12, 3),
        "p_top6": round(p_top6, 3),
        "p_champion": round(p_champ, 3),
    }

    print("\n" + "=" * 70)
    print(" CHECKPOINT BENCHMARK METRICS SUMMARY")
    print("=" * 70)
    print(f" Mean Net BB:          {metrics['mean_net_bb']:+8.2f} BB")
    print(f" Estimated BB/100:     {metrics['bb_per_100']:+8.2f}")
    print(f" P(Top 12):            {metrics['p_top12'] * 100:6.1f}%")
    print(f" P(Top 6):             {metrics['p_top6'] * 100:6.1f}%")
    print(f" P(Champion):          {metrics['p_champion'] * 100:6.1f}%")
    print(f" Variance:             {metrics['variance']:8.2f}")
    print(f" Worst 5% Net BB:      {metrics['worst_5_percent']:+8.2f} BB")
    print("=" * 70 + "\n")

    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics written to: {output_json}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate ApexPoker Checkpoints across Tournament Benchmarks")
    parser.add_argument("--checkpoint", type=str, default="models/phase0_test/checkpoint_iter_2.pt", help="Checkpoint file to evaluate")
    parser.add_argument("--tournaments", type=int, default=2, help="Number of tournaments")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    parser.add_argument("--json-out", type=str, default=None, help="Optional output JSON path")
    args = parser.parse_args()

    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        num_tournaments=args.tournaments,
        seed=args.seed,
        output_json=args.json_out,
    )


if __name__ == "__main__":
    main()
