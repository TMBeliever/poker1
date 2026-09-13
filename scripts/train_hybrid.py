"""CLI entrypoint to execute Dual-Track Hybrid Deep CFR Training."""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml
from src.training.train_hybrid import train_hybrid_cfr
from src.opponent_modeling.profile_manager import DEFAULT_PROFILES_PATH


def main():
    parser = argparse.ArgumentParser(description="Dual-Track Hybrid Deep CFR Training (Self-Play + Real Profiles)")
    parser.add_argument("--config", type=str, default="configs/training.yaml", help="Path to config file")
    parser.add_argument("--base-checkpoint", type=str, default="models/long_train_5h/mixed_checkpoint_iter_18000.pt", help="Starting checkpoint")
    parser.add_argument("--checkpoint-dir", type=str, default="models/long_train_5h", help="Historical checkpoint directory for self-evolution")
    parser.add_argument("--profiles", type=str, default=DEFAULT_PROFILES_PATH, help="Path to opponent profiles JSON")
    parser.add_argument("--profile-ratio", type=float, default=0.5, help="Fraction of table filled with real profiles (0.0 to 1.0)")
    parser.add_argument("--iterations", type=int, default=500, help="Number of training iterations")
    parser.add_argument("--traversals", type=int, default=20, help="CFR traversals per iteration")
    parser.add_argument("--refresh-interval", type=int, default=50, help="Opponent pool re-sampling frequency")
    parser.add_argument("--checkpoint-interval", type=int, default=100, help="Save interval")
    parser.add_argument("--eval-interval", type=int, default=50, help="Evaluation interval")
    parser.add_argument("--eval-games", type=int, default=100, help="Hands per evaluation")
    parser.add_argument("--save-dir", type=str, default="models/hybrid", help="Directory to save hybrid models")
    parser.add_argument("--log-dir", type=str, default="logs/hybrid", help="Directory for TensorBoard telemetry")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device ('cpu' or 'cuda')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose diagnostics output")
    args = parser.parse_args()

    # Load YAML config if present
    cfg = {}
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            full_cfg = yaml.safe_load(f) or {}
            cfg = full_cfg.get("hybrid_cfr", {})

    base_checkpoint = args.base_checkpoint or cfg.get("base_checkpoint")
    checkpoint_dir = args.checkpoint_dir or cfg.get("checkpoint_dir", "models/long_train_5h")
    profiles_path = args.profiles or cfg.get("profiles_path", DEFAULT_PROFILES_PATH)
    profile_ratio = args.profile_ratio if args.profile_ratio is not None else cfg.get("profile_ratio", 0.5)
    iterations = args.iterations or cfg.get("iterations", 500)
    traversals = args.traversals or cfg.get("traversals_per_iteration", 20)

    print("=" * 80)
    print(" 🚀 DUAL-TRACK HYBRID DEEP CFR TRAINING (双轨自演化与实战画像特训)")
    print("=" * 80)
    print(f" • Base Model Checkpoint    : {base_checkpoint}")
    print(f" • Self-Play Checkpoint Pool : {checkpoint_dir}")
    print(f" • Online Opponent Profiles  : {profiles_path}")
    print(f" • Dual-Track Mixture Ratio  : {profile_ratio * 100:.1f}% Real Profiles / {(1 - profile_ratio) * 100:.1f}% Self-Play")
    print(f" • Training Iterations       : {iterations} (traversals/iter: {traversals})")
    print(f" • Checkpoint Save Cadence   : Every {args.checkpoint_interval} iterations")
    print(f" • Evaluation Cadence        : Every {args.eval_interval} iterations ({args.eval_games} hands/eval)")
    print(f" • Destination Save Directory: {args.save_dir}")
    print("=" * 80 + "\n")

    agent, summary = train_hybrid_cfr(
        base_checkpoint=base_checkpoint,
        checkpoint_dir=checkpoint_dir,
        profiles_path=profiles_path,
        profile_ratio=profile_ratio,
        additional_iterations=iterations,
        traversals_per_iteration=traversals,
        refresh_interval=args.refresh_interval,
        checkpoint_interval=args.checkpoint_interval,
        evaluation_interval=args.eval_interval,
        eval_games=args.eval_games,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        device=args.device,
        seed=args.seed,
        verbose=args.verbose,
    )

    print("\n" + "=" * 80)
    print(" 🏆 DUAL-TRACK HYBRID TRAINING COMPLETED")
    print("=" * 80)
    print(f" Final Checkpoint           : {summary['latest_checkpoint']}")
    print(f" Final Strategy Loss        : {summary['final_strat_loss']:.4f}" if summary['final_strat_loss'] is not None else " N/A")
    print(f" Profit vs Checkpoint League: {summary['final_profit_vs_checkpoints']:+.2f} BB/hand")
    print(f" Profit vs Real S10 Profiles: {summary['final_profit_vs_profiles']:+.2f} BB/hand")
    print(f" Profit vs Random Baseline  : {summary['final_profit_vs_random']:+.2f} BB/hand")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
