"""Unit tests for Phase 6 CurriculumTrainer progression."""

import os
import pytest
from src.tournament.tournament_deep_cfr import TournamentDeepCFRAgent
from src.training.curriculum import CurriculumTrainer, CurriculumConfig


def test_curriculum_sequential_progression(tmp_path):
    save_dir = str(tmp_path / "curriculum_models")
    config = CurriculumConfig(
        iterations=3,
        save_dir=save_dir,
        seed=42,
    )
    agent = TournamentDeepCFRAgent(player_id=0, num_players=6, memory_size=500, device="cpu")
    trainer = CurriculumTrainer(agent, config=config)

    # 1. Test P1: Context sensitivity
    p1_res = trainer.train_p1_context_sensitivity(iterations=2)
    assert p1_res["phase"] in ("P1_CONTEXT", "curriculum_p1")
    assert os.path.exists(p1_res["checkpoint"])

    # 2. Test P2: Preliminary
    p2_res = trainer.train_p2_preliminary(iterations=1)
    assert p2_res["phase"] in ("P2_PRELIMINARY", "curriculum_p2")
    assert os.path.exists(p2_res["checkpoint"])

    # 3. Test P3: Semifinal
    p3_res = trainer.train_p3_semifinal(iterations=1)
    assert p3_res["phase"] in ("P3_SEMIFINAL", "curriculum_p3")
    assert os.path.exists(p3_res["checkpoint"])

    # 4. Test P4: Final
    p4_res = trainer.train_p4_final(iterations=1)
    assert p4_res["phase"] in ("P4_FINAL", "curriculum_p4")
    assert os.path.exists(p4_res["checkpoint"])

    # 5. Test P5: Full tournament
    p5_res = trainer.train_p5_full(iterations=1)
    assert p5_res["phase"] in ("P5_FULL", "curriculum_p5_full")
    assert os.path.exists(p5_res["checkpoint"])

