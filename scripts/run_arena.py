"""CLI script for cross-project model battles: ApexPoker vs agentpoker_final 2 vs Bots."""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.arena.evaluator import ArenaEvaluator


def parse_competitor_arg(arg: str) -> tuple[str, str]:
    """Parse 'Name=Spec' or just 'Spec'."""
    if "=" in arg:
        name, spec = arg.split("=", 1)
        return name.strip(), spec.strip()
    p = Path(arg)
    return p.stem, arg.strip()


def plot_arena_results(results: dict, save_path: str):
    """Generate a clean visual comparison chart for the arena."""
    lb = results.get("leaderboard", [])
    if not lb:
        return

    sns.set_theme(style="whitegrid")
    plt.rcParams['font.sans-serif'] = ['STHeiti', 'Songti SC', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    names = [item["name"] for item in lb]
    mode = results.get("mode")

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    if mode == "120_player_tournament":
        vals = [item["mean_net_bb"] for item in lb]
        colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in vals]
        bars = ax.barh(names[::-1], vals[::-1], color=colors[::-1], height=0.55, edgecolor="black")
        ax.set_xlabel("120人锦标赛均值净收益 (Net BB)", fontsize=11, fontweight="bold")
        ax.set_title("ApexPoker 跨项目模型竞技场战绩对决 (120人瑞士轮锦标赛)", fontsize=13, fontweight="bold", pad=12)

        for bar, val in zip(bars, vals[::-1]):
            x_pos = val + (5 if val >= 0 else -15)
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2, f"{val:+.1f} BB", va="center", fontsize=10, fontweight="bold")
    else:
        vals = [item["total_net_bb"] for item in lb]
        colors = ["#3498db" if v > 0 else "#e74c3c" for v in vals]
        bars = ax.barh(names[::-1], vals[::-1], color=colors[::-1], height=0.55, edgecolor="black")
        ax.set_xlabel("6人桌总净收益 (Net BB)", fontsize=11, fontweight="bold")
        ax.set_title(f"ApexPoker 6人桌直接交锋擂台战绩 ({results.get('total_hands')} 手牌)", fontsize=13, fontweight="bold", pad=12)

        for bar, val in zip(bars, vals[::-1]):
            x_pos = val + (2 if val >= 0 else -10)
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2, f"{val:+.1f} BB", va="center", fontsize=10, fontweight="bold")

    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison chart saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="ApexPoker Cross-Project Model Arena (ApexPoker vs agentpoker_final 2 vs Bots)")
    parser.add_argument(
        "--mode",
        type=str,
        default="tournament",
        choices=["tournament", "6max"],
        help="Battle mode: 'tournament' (120-player Swiss MTT) or '6max' (direct 6-player ring table)",
    )
    parser.add_argument(
        "--competitors",
        nargs="*",
        default=None,
        help="List of competitors in 'Name=PathOrSpec' format. Supports .pt models, .json models from agentpoker_final 2, and bots (TAG, LAG, Nit).",
    )
    parser.add_argument("--tournaments", type=int, default=3, help="Number of 120-player tournaments to run")
    parser.add_argument("--hands", type=int, default=100, help="Number of hands for 6-max direct battle")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--profiles", type=str, default="data/profiles/opponent_profiles.json", help="Path to online opponent profiles database to populate realistic field")
    parser.add_argument("--json-out", type=str, default="results/arena_report.json", help="Path to save output JSON")
    parser.add_argument("--plot-out", type=str, default="results/arena_comparison.png", help="Path to save chart PNG")
    args = parser.parse_args()

    # Default competitor lineup if none provided
    if not args.competitors:
        default_competitors = [
            ("Our_Stage2_Tournament", "models/tournament/curriculum_p5_full.pt"),
            ("AgentPoker_Champion", "/Users/liang/files/seft/agentpoker_final 2/models/champion.json"),
            ("Our_Stage1_BaseCFR", "models/base/base_checkpoint_iter_2000.pt"),
            ("TAG_Archetype", "TAG"),
        ]
    else:
        default_competitors = [parse_competitor_arg(c) for c in args.competitors]

    evaluator = ArenaEvaluator(default_competitors, profiles_path=args.profiles)

    if args.mode == "tournament":
        mtt_seeds = [args.seed + i * 100 for i in range(args.tournaments)]
        results = evaluator.run_mtt_arena(num_tournaments=args.tournaments, seeds=mtt_seeds)
    else:
        results = evaluator.run_6max_arena(total_hands=args.hands, seed=args.seed)

    evaluator.print_leaderboard(results)

    # Save reports
    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Detailed arena report saved to: {args.json_out}")

    if args.plot_out:
        os.makedirs(os.path.dirname(args.plot_out) or ".", exist_ok=True)
        plot_arena_results(results, args.plot_out)


if __name__ == "__main__":
    main()
