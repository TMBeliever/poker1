"""Train Base 6-Max Deep CFR agent from CLI."""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml
from src.training.train import continue_training, train_deep_cfr


def main():
    parser = argparse.ArgumentParser(description="Train Base Deep CFR Agent")
    parser.add_argument("--config", type=str, default="configs/training.yaml", help="Path to training config")
    parser.add_argument("--checkpoint", type=str, default=None, help="Continue training from checkpoint")
    parser.add_argument("--iterations", type=int, default=None, help="CFR training iterations")
    parser.add_argument("--traversals", type=int, default=None, help="Traversals per iteration")
    parser.add_argument("--save-dir", type=str, default="models/base", help="Model checkpoint directory")
    parser.add_argument("--log-dir", type=str, default="logs/base", help="TensorBoard log directory")
    parser.add_argument("--checkpoint-interval", type=int, default=100, help="Save interval")
    parser.add_argument("--eval-interval", type=int, default=50, help="Eval interval")
    parser.add_argument("--random-eval-games", type=int, default=100, help="Eval games vs random")
    args = parser.parse_args()

    cfg = {}
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            full_cfg = yaml.safe_load(f) or {}
            cfg = full_cfg.get("base_cfr", {})

    iterations = args.iterations or cfg.get("iterations", 100)
    traversals = args.traversals or cfg.get("traversals_per_iteration", 20)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    if args.checkpoint:
        print(f"Continuing Base Deep CFR Training from: {args.checkpoint}")
        print(f"Adding {iterations} iterations, {traversals} traversals/iter")
        agent, losses, profits = continue_training(
            checkpoint_path=args.checkpoint,
            additional_iterations=iterations,
            traversals_per_iteration=traversals,
            save_dir=args.save_dir,
            log_dir=args.log_dir,
            checkpoint_interval=args.checkpoint_interval,
            evaluation_interval=args.eval_interval,
            random_eval_games=args.random_eval_games,
        )
    else:
        print(f"Starting Base Deep CFR Training: {iterations} iterations, {traversals} traversals/iter")
        agent, losses, profits = train_deep_cfr(
            num_iterations=iterations,
            traversals_per_iteration=traversals,
            num_players=6,
            player_id=0,
            save_dir=args.save_dir,
            log_dir=args.log_dir,
            checkpoint_interval=args.checkpoint_interval,
            evaluation_interval=args.eval_interval,
            random_eval_games=args.random_eval_games,
        )

    final_model_path = os.path.join(args.save_dir, f"base_checkpoint_iter_{agent.iteration_count}.pt")
    agent.save_model(final_model_path)
    print(f"\nBase Deep CFR training complete. Model saved to: {final_model_path}")


if __name__ == "__main__":
    main()
