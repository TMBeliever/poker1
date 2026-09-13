"""Unit tests for Dual-Track Hybrid Deep CFR Training."""

import os
import pytest
from src.core.deep_cfr import DeepCFRAgent
from src.opponent_modeling.profile_manager import OpponentProfileManager
from src.training.train_hybrid import (
    sample_hybrid_opponents,
    evaluate_against_profiles,
    train_hybrid_cfr,
)


@pytest.fixture
def dummy_profiles_path(tmp_path):
    import json
    path = str(tmp_path / "test_profiles.json")
    profiles = {
        "p1": {"name": "Bot_Maniac", "vpip": 0.65, "pfr": 0.50, "af": 4.0, "archetype": "Maniac"},
        "p2": {"name": "Bot_Nit", "vpip": 0.12, "pfr": 0.10, "af": 1.2, "archetype": "Nit"},
        "p3": {"name": "Bot_TAG", "vpip": 0.22, "pfr": 0.18, "af": 2.5, "archetype": "TAG"},
        "p4": {"name": "Bot_LAG", "vpip": 0.32, "pfr": 0.26, "af": 3.0, "archetype": "LAG"},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profiles, f)
    return path


def test_sample_hybrid_opponents_ratios(dummy_profiles_path, tmp_path):
    pm = OpponentProfileManager(dummy_profiles_path)
    ckpt_dir = str(tmp_path / "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)

    # 1. 50% Ratio: 5 opponent seats -> 2 or 3 profile seats
    wrappers, meta = sample_hybrid_opponents(
        checkpoint_dir=ckpt_dir,
        opponent_model_pattern="*.pt",
        profile_manager=pm,
        profile_ratio=0.6,
        device="cpu",
        player_id=0,
        num_players=6,
    )
    assert len(wrappers) == 6
    assert wrappers[0] is None
    for pos in range(1, 6):
        assert wrappers[pos] is not None
    assert len(meta["profile_agents"]) == 3

    # 2. 100% Ratio: all 5 seats profiles
    wrappers_all_prof, meta_all = sample_hybrid_opponents(
        checkpoint_dir=ckpt_dir,
        opponent_model_pattern="*.pt",
        profile_manager=pm,
        profile_ratio=1.0,
        device="cpu",
        player_id=0,
        num_players=6,
    )
    assert len(meta_all["profile_agents"]) == 5


def test_train_hybrid_cfr_smoke(dummy_profiles_path, tmp_path):
    save_dir = str(tmp_path / "models_hybrid")
    log_dir = str(tmp_path / "logs_hybrid")

    agent, summary = train_hybrid_cfr(
        base_checkpoint=None,
        checkpoint_dir=save_dir,
        profiles_path=dummy_profiles_path,
        profile_ratio=0.5,
        additional_iterations=2,
        traversals_per_iteration=2,
        refresh_interval=1,
        checkpoint_interval=1,
        evaluation_interval=1,
        eval_games=2,
        save_dir=save_dir,
        log_dir=log_dir,
        device="cpu",
        seed=123,
    )

    assert summary["final_iteration"] == 2
    assert summary["total_iterations_trained"] == 2
    assert os.path.exists(summary["latest_checkpoint"])
    assert isinstance(agent, DeepCFRAgent)
