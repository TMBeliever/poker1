"""Train Tournament-Aware Deep CFR agent across curriculum phases."""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml
from src.tournament.tournament_deep_cfr import TournamentDeepCFRAgent
from src.training.curriculum import CurriculumConfig, CurriculumTrainer


def main():
    parser = argparse.ArgumentParser(description="Train Tournament Deep CFR Agent via Curriculum")
    parser.add_argument("--config", type=str, default="configs/training.yaml", help="Path to training config")
    parser.add_argument("--phase", type=str, default="all", choices=["P1", "P2", "P3", "P4", "P5", "all"], help="Curriculum phase")
    parser.add_argument("--base-checkpoint", type=str, default="models/phase0_test/checkpoint_iter_2.pt", help="Pretrained base model")
    parser.add_argument("--save-dir", type=str, default="models/tournament", help="Save directory")
    parser.add_argument("--iterations", type=int, default=5, help="Training iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    # Load YAML configuration if present
    cfg = {}
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    os.makedirs(args.save_dir, exist_ok=True)
    curriculum_cfg = CurriculumConfig(
        iterations=args.iterations,
        save_dir=args.save_dir,
        device=args.device,
        seed=args.seed,
        base_checkpoint_path=args.base_checkpoint if os.path.exists(args.base_checkpoint) else None,
    )

    print(f"Initializing Tournament Deep CFR Agent (seed={args.seed}, device={args.device})...")
    agent = TournamentDeepCFRAgent(
        player_id=0,
        num_players=6,
        device=args.device,
        base_checkpoint_path=curriculum_cfg.base_checkpoint_path,
    )
    trainer = CurriculumTrainer(agent, config=curriculum_cfg)

    print(f"Executing Curriculum Training: Phase '{args.phase}'...")
    if args.phase == "all":
        results = trainer.run_full_curriculum()
        print("\nFull Curriculum Training Successfully Completed!")
        for phase_name, res in results.items():
            print(f"  [{phase_name}] Checkpoint: {res.get('checkpoint')}")
    elif args.phase == "P1":
        res = trainer.train_p1_context_sensitivity(iterations=args.iterations)
        print(f"Phase P1 complete: {res['checkpoint']}")
    elif args.phase == "P2":
        res = trainer.train_p2_preliminary(iterations=1)
        print(f"Phase P2 complete: {res['checkpoint']}")
    elif args.phase == "P3":
        res = trainer.train_p3_semifinal(iterations=1)
        print(f"Phase P3 complete: {res['checkpoint']}")
    elif args.phase == "P4":
        res = trainer.train_p4_final(iterations=1)
        print(f"Phase P4 complete: {res['checkpoint']}")
    elif args.phase == "P5":
        res = trainer.train_p5_full(iterations=1)
        print(f"Phase P5 complete: {res['checkpoint']}")


if __name__ == "__main__":
    main()
