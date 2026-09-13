"""Train Opponent Modeling Deep CFR agent from CLI."""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml
from src.training.train_opponent_modeling import train_deep_cfr_with_opponent_modeling


def main():
    parser = argparse.ArgumentParser(description="Train Opponent Modeling Deep CFR Agent")
    parser.add_argument("--config", type=str, default="configs/training.yaml", help="Path to training config")
    parser.add_argument("--iterations", type=int, default=10, help="Training iterations")
    parser.add_argument("--traversals", type=int, default=10, help="Traversals per iteration")
    parser.add_argument("--save-dir", type=str, default="models/opponent", help="Save directory")
    parser.add_argument("--log-dir", type=str, default="logs/opponent", help="Log directory")
    parser.add_argument("--base-checkpoint", type=str, default=None, help="Pretrained opponent model checkpoint")
    parser.add_argument("--progress-interval", type=int, default=5, help="Progress interval")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    base_ckpt = args.base_checkpoint if args.base_checkpoint and os.path.exists(args.base_checkpoint) else None

    print(f"Starting Opponent Modeling Deep CFR Training: {args.iterations} iterations, {args.traversals} traversals/iter")
    agent, adv_losses, strat_losses, om_losses, profits = train_deep_cfr_with_opponent_modeling(
        num_iterations=args.iterations,
        traversals_per_iteration=args.traversals,
        num_players=6,
        player_id=0,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        checkpoint_path=base_ckpt,
        progress_interval=args.progress_interval,
    )

    final_path = os.path.join(args.save_dir, f"opponent_model_iter_{args.iterations}.pt")
    agent.save_model(final_path)
    print(f"\nOpponent Modeling training complete. Model saved to: {final_path}")


if __name__ == "__main__":
    main()
