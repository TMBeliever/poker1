"""Dual-Track Hybrid CFR Training System (双轨混合训练系统).

Combines self-evolution (GTO Checkpoint League) with targeted real-world
opponent profile exploitation (Empirical Bayesian Profiles from S10 Online Competition).

- Track 1 (Self-Evolution): Keeps the agent theoretically sound and unexploitable
  by playing against frozen historical checkpoints.
- Track 2 (Real Profile Exploitation): Extracts maximum expected value (EV)
  by exposing the agent to the empirical distributions of real online competitors
  (Maniacs, Nits, LAGs, TAGs, Calling Stations).
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pokers as pkrs
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.agents.profiled_agent import ProfiledAgent
from src.agents.random_agent import RandomAgent
from src.core.deep_cfr import DeepCFRAgent
from src.core.model import set_verbose
from src.opponent_modeling.profile_manager import (
    DEFAULT_PROFILES_PATH,
    OpponentProfileManager,
)
from src.training.train import (
    _PerspectiveAgentWrapper,
    _cfr_traverse_with_opponents,
    _record_traversal_failure_diagnostics,
    _set_training_iterations,
    _should_evaluate,
    _should_save_checkpoint,
    evaluate_against_checkpoint_agents,
    evaluate_against_random,
)
from src.utils.agents import CheckpointAgent
from src.utils.checkpoints import find_checkpoints, load_checkpoint
from src.utils.evaluation import evaluate_agent_matchup, write_evaluation_diagnostics
from src.utils.traversal_diagnostics import TraversalFailure


def sample_hybrid_opponents(
    checkpoint_dir: str,
    opponent_model_pattern: str,
    profile_manager: OpponentProfileManager,
    profile_ratio: float = 0.5,
    device: str = "cpu",
    player_id: int = 0,
    num_players: int = 6,
    rng: Optional[random.Random] = None,
) -> Tuple[List[Optional[_PerspectiveAgentWrapper]], Dict[str, Any]]:
    """Sample a blended 6-max table containing both historical checkpoints and real profiles.

    Args:
        checkpoint_dir: Directory containing historical checkpoints.
        opponent_model_pattern: Glob pattern for checkpoint files.
        profile_manager: Initialized OpponentProfileManager with active profiles.
        profile_ratio: Fraction of opponent seats allocated to real profiles [0.0, 1.0].
        device: PyTorch device ('cpu' or 'cuda').
        player_id: Seat index of the learning agent (0..5).
        num_players: Number of players at table (default 6).
        rng: Optional Random instance.

    Returns:
        opponent_wrappers: List of 6 elements (None at player_id).
        metadata: Table composition information.
    """
    if rng is None:
        rng = random.Random()

    opponent_seats = [pos for pos in range(num_players) if pos != player_id]
    num_opponent_seats = len(opponent_seats)

    # Determine seat counts for profiles vs checkpoints
    num_profiles = int(round(num_opponent_seats * profile_ratio))
    num_profiles = max(0, min(num_opponent_seats, num_profiles))
    num_checkpoints = num_opponent_seats - num_profiles

    # Randomly partition seats
    shuffled_seats = list(opponent_seats)
    rng.shuffle(shuffled_seats)
    profile_seats = shuffled_seats[:num_profiles]
    checkpoint_seats = shuffled_seats[num_profiles:]

    opponent_agents: List[Optional[Any]] = [None] * num_players
    meta: Dict[str, Any] = {
        "profile_agents": [],
        "checkpoint_agents": [],
        "checkpoint_fallback_profiles": [],
        "random_fallback_agents": [],
    }

    # 1. Populate Profile Seats (Exploitation Track)
    available_profiles = list(profile_manager.profiles.values())
    if available_profiles and profile_seats:
        sampled_profiles = rng.sample(
            available_profiles,
            min(len(profile_seats), len(available_profiles)),
        )
        # In case fewer profiles than seats, cycle through
        for idx, seat in enumerate(profile_seats):
            p_data = sampled_profiles[idx % len(sampled_profiles)]
            agent_seed = rng.randint(0, 1_000_000)
            agent = ProfiledAgent(
                player_id=seat,
                profile=p_data,
                seed=agent_seed,
            )
            opponent_agents[seat] = agent
            meta["profile_agents"].append(
                {"seat": seat, "name": p_data.get("name"), "archetype": p_data.get("archetype")}
            )
    else:
        # Fallback if no profiles available
        for seat in profile_seats:
            opponent_agents[seat] = RandomAgent(seat)
            meta["random_fallback_agents"].append(seat)

    # 2. Populate Checkpoint Seats (Self-Evolution Track)
    checkpoint_files = [str(p) for p in find_checkpoints(checkpoint_dir, opponent_model_pattern)]
    if checkpoint_files and checkpoint_seats:
        sampled_ckpts = rng.sample(
            checkpoint_files,
            min(len(checkpoint_seats), len(checkpoint_files)),
        )
        for idx, seat in enumerate(checkpoint_seats):
            ckpt_path = sampled_ckpts[idx % len(sampled_ckpts)]
            agent = CheckpointAgent(
                player_id=seat,
                model_path=ckpt_path,
                device=device,
                sanitize_actions=True,
            )
            opponent_agents[seat] = agent
            meta["checkpoint_agents"].append(
                {"seat": seat, "checkpoint": os.path.basename(ckpt_path)}
            )
    else:
        # Fallback if no checkpoints found: use profiles or random
        for seat in checkpoint_seats:
            if available_profiles:
                p_data = rng.choice(available_profiles)
                opponent_agents[seat] = ProfiledAgent(
                    player_id=seat,
                    profile=p_data,
                    seed=rng.randint(0, 1_000_000),
                )
                meta["checkpoint_fallback_profiles"].append(
                    {"seat": seat, "name": p_data.get("name"), "archetype": p_data.get("archetype")}
                )
            else:
                opponent_agents[seat] = RandomAgent(seat)
                meta["random_fallback_agents"].append(seat)

    # Wrap opponents with perspective wrappers
    opponent_wrappers: List[Optional[_PerspectiveAgentWrapper]] = [None] * num_players
    for pos in opponent_seats:
        if opponent_agents[pos] is not None:
            opponent_wrappers[pos] = _PerspectiveAgentWrapper(opponent_agents[pos])

    return opponent_wrappers, meta


def evaluate_against_profiles(
    agent: DeepCFRAgent,
    profile_manager: OpponentProfileManager,
    num_games: int = 100,
    writer: Optional[SummaryWriter] = None,
    iteration: Optional[int] = None,
    metric_prefix: str = "Evaluation/Profiles",
    seed: int = 42,
) -> float:
    """Evaluate learning agent against 5 real online competitor profiles."""
    rng = random.Random(seed)
    available_profiles = list(profile_manager.profiles.values())
    if not available_profiles:
        return 0.0

    opponents = [None] * 6
    sampled = rng.sample(available_profiles, min(5, len(available_profiles)))
    for pos in range(1, 6):
        p_data = sampled[(pos - 1) % len(sampled)]
        opponents[pos] = ProfiledAgent(
            player_id=pos,
            profile=p_data,
            seed=seed + pos * 100,
        )

    metrics = evaluate_agent_matchup(
        agent,
        opponents,
        num_games=num_games,
        seed_start=seed,
        num_players=6,
        strict=True,
        label="evaluation vs online profiles",
        print_warnings=False,
    )
    if writer is not None and iteration is not None:
        write_evaluation_diagnostics(writer, metrics, iteration, metric_prefix)

    return float(metrics.get("avg_profit", 0.0))


def train_hybrid_cfr(
    base_checkpoint: Optional[str] = "models/long_train_5h/mixed_checkpoint_iter_18000.pt",
    checkpoint_dir: str = "models/long_train_5h",
    profiles_path: str = DEFAULT_PROFILES_PATH,
    profile_ratio: float = 0.5,
    additional_iterations: int = 1000,
    traversals_per_iteration: int = 20,
    refresh_interval: int = 50,
    checkpoint_interval: int = 100,
    evaluation_interval: int = 50,
    eval_games: int = 100,
    save_dir: str = "models/hybrid",
    log_dir: str = "logs/hybrid",
    device: str = "cpu",
    seed: int = 42,
    verbose: bool = False,
    opponent_model_pattern: str = "*checkpoint_iter_*.pt",
) -> Tuple[DeepCFRAgent, Dict[str, Any]]:
    """Execute Dual-Track Hybrid Deep CFR training run.

    Args:
        base_checkpoint: Path to initial starting model.
        checkpoint_dir: Directory containing historical checkpoints for self-evolution.
        profiles_path: Path to JSON opponent profile database.
        profile_ratio: Balance between Real Profiles and Checkpoints (0.0=pure self-play, 1.0=pure exploit).
        additional_iterations: Iterations to execute.
        traversals_per_iteration: CFR tree traversals per iteration.
        refresh_interval: Opponent table re-sampling period.
        checkpoint_interval: Cadence for checkpoint saving.
        evaluation_interval: Cadence for running dual evaluations.
        eval_games: Number of hands per evaluation matchup.
        save_dir: Destination directory for trained checkpoints.
        log_dir: Destination directory for TensorBoard telemetry.
        device: 'cpu' or 'cuda'.
        seed: Random seed.
        verbose: Diagnostic print level.
        opponent_model_pattern: Glob pattern to identify historical checkpoints.

    Returns:
        agent: Trained DeepCFRAgent.
        metrics_summary: Final training metrics dictionary.
    """
    set_verbose(verbose)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    rng = random.Random(seed)

    writer = SummaryWriter(log_dir)

    # 1. Initialize learning agent
    learning_agent = DeepCFRAgent(player_id=0, num_players=6, device=device)
    starting_iteration = 1
    losses: List[float] = []
    profits_vs_checkpoints: List[float] = []
    profits_vs_profiles: List[float] = []
    profits_vs_random: List[float] = []

    if base_checkpoint and os.path.exists(base_checkpoint):
        print(f"[Hybrid CFR] Loading base model from: {base_checkpoint}")
        learning_agent.load_model(base_checkpoint)
        try:
            ckpt_state = load_checkpoint(base_checkpoint, map_location=device)
            starting_iteration = ckpt_state.get("iteration", learning_agent.iteration_count) + 1
        except Exception as exc:
            print(f"[Hybrid CFR] Note: Could not read extra metadata ({exc}), continuing from {learning_agent.iteration_count + 1}")
            starting_iteration = learning_agent.iteration_count + 1

    # 2. Load Profiles
    pm = OpponentProfileManager(profiles_path)
    print(f"[Hybrid CFR] Loaded {len(pm.profiles)} online competitor profiles from {profiles_path}")
    print(f"[Hybrid CFR] Dual-Track configuration: profile_ratio={profile_ratio * 100:.1f}% (Exploit), checkpoint_ratio={(1 - profile_ratio) * 100:.1f}% (Self-Play)")

    # 3. Initial Baseline Evaluation
    print("[Hybrid CFR] Running initial dual-track baseline evaluations...")
    init_prof_vs_ckpt = 0.0
    eval_ckpt_opponents = None
    # Evaluate vs checkpoints if available
    available_ckpts = find_checkpoints(checkpoint_dir, opponent_model_pattern)
    if available_ckpts:
        eval_ckpt_opponents = [None] * 6
        for pos in range(1, 6):
            eval_ckpt_opponents[pos] = CheckpointAgent(
                pos,
                model_path=str(available_ckpts[pos % len(available_ckpts)]),
                device=device,
                sanitize_actions=True,
            )
        init_prof_vs_ckpt = evaluate_against_checkpoint_agents(
            learning_agent,
            eval_ckpt_opponents,
            num_games=eval_games,
            writer=writer,
            iteration=starting_iteration - 1,
            metric_prefix="Evaluation/Checkpoint",
        )

    init_prof_vs_prof = evaluate_against_profiles(
        learning_agent,
        pm,
        num_games=eval_games,
        writer=writer,
        iteration=starting_iteration - 1,
        metric_prefix="Evaluation/Profiles",
        seed=seed,
    )

    init_prof_vs_rand = evaluate_against_random(
        learning_agent,
        num_games=eval_games,
        num_players=6,
        writer=writer,
        iteration=starting_iteration - 1,
        metric_prefix="Evaluation/Random",
    )

    print(f"  Initial vs Checkpoint League : {init_prof_vs_ckpt:+.2f} BB/hand")
    print(f"  Initial vs Real S10 Profiles : {init_prof_vs_prof:+.2f} BB/hand")
    print(f"  Initial vs Random Baseline   : {init_prof_vs_rand:+.2f} BB/hand")

    profits_vs_checkpoints.append(init_prof_vs_ckpt)
    profits_vs_profiles.append(init_prof_vs_prof)
    profits_vs_random.append(init_prof_vs_rand)

    # 4. Main Training Loop
    final_iteration = starting_iteration + additional_iterations - 1
    progress = tqdm(
        range(starting_iteration, final_iteration + 1),
        desc="Dual-Track CFR",
        unit="iter",
        dynamic_ncols=True,
        disable=not sys.stderr.isatty(),
    )

    opponent_wrappers, meta = sample_hybrid_opponents(
        checkpoint_dir=checkpoint_dir,
        opponent_model_pattern=opponent_model_pattern,
        profile_manager=pm,
        profile_ratio=profile_ratio,
        device=device,
        player_id=learning_agent.player_id,
        num_players=6,
        rng=rng,
    )

    last_strat_loss = None
    last_adv_loss = None

    for iteration in progress:
        local_iteration = iteration - starting_iteration + 1
        _set_training_iterations(learning_agent, iteration, local_iteration)
        iter_start = time.time()

        # Dynamic table refresh
        if iteration % refresh_interval == 1 and iteration > starting_iteration:
            opponent_wrappers, meta = sample_hybrid_opponents(
                checkpoint_dir=checkpoint_dir,
                opponent_model_pattern=opponent_model_pattern,
                profile_manager=pm,
                profile_ratio=profile_ratio,
                device=device,
                player_id=learning_agent.player_id,
                num_players=6,
                rng=rng,
            )

        # Execute CFR tree traversals across the hybrid table
        for t in range(traversals_per_iteration):
            button_pos = t % 6
            state = pkrs.State.from_seed(
                n_players=6,
                button=button_pos,
                sb=1.0,
                bb=2.0,
                stake=200.0,
                seed=rng.randint(0, 1_000_000),
            )

            try:
                _cfr_traverse_with_opponents(
                    learning_agent,
                    state,
                    local_iteration,
                    opponent_wrappers,
                    verbose=verbose,
                )
            except TraversalFailure:
                _record_traversal_failure_diagnostics(writer, learning_agent, iteration)
                raise

        traversal_time = time.time() - iter_start
        writer.add_scalar("Time/Traversal", traversal_time, iteration)

        # Train advantage network
        adv_loss = learning_agent.train_advantage_network()
        last_adv_loss = adv_loss
        losses.append(adv_loss)
        writer.add_scalar("Loss/Advantage", adv_loss, iteration)
        writer.add_scalar("Memory/Advantage", len(learning_agent.advantage_memory), iteration)

        # In-loop Dual Evaluation & Strategy update
        if _should_evaluate(iteration, final_iteration, evaluation_interval):
            strat_loss = learning_agent.train_strategy_network()
            last_strat_loss = strat_loss
            writer.add_scalar("Loss/Strategy", strat_loss, iteration)
            writer.add_scalar("Memory/Strategy", len(learning_agent.strategy_memory), iteration)

            # 1. Eval vs Checkpoints
            p_ckpt = 0.0
            if available_ckpts and eval_ckpt_opponents:
                p_ckpt = evaluate_against_checkpoint_agents(
                    learning_agent,
                    eval_ckpt_opponents,
                    num_games=eval_games,
                    writer=writer,
                    iteration=iteration,
                    metric_prefix="Evaluation/Checkpoint",
                )
                writer.add_scalar("Performance/ProfitVsCheckpoint", p_ckpt, iteration)
                profits_vs_checkpoints.append(p_ckpt)

            # 2. Eval vs Profiles
            p_prof = evaluate_against_profiles(
                learning_agent,
                pm,
                num_games=eval_games,
                writer=writer,
                iteration=iteration,
                metric_prefix="Evaluation/Profiles",
                seed=seed + iteration,
            )
            writer.add_scalar("Performance/ProfitVsProfiles", p_prof, iteration)
            profits_vs_profiles.append(p_prof)

            # 3. Eval vs Random
            p_rand = evaluate_against_random(
                learning_agent,
                num_games=eval_games,
                num_players=6,
                writer=writer,
                iteration=iteration,
                metric_prefix="Evaluation/Random",
            )
            writer.add_scalar("Performance/ProfitVsRandom", p_rand, iteration)
            profits_vs_random.append(p_rand)

            progress.set_postfix({
                "adv_loss": f"{adv_loss:.4f}",
                "strat_loss": f"{strat_loss:.4f}",
                "vs_prof": f"{p_prof:+.1f}BB",
                "vs_ckpt": f"{p_ckpt:+.1f}BB",
            })
            print(
                f"[Hybrid CFR] [Iter {iteration}/{final_iteration}] "
                f"adv_loss={adv_loss:.4f}, strat_loss={strat_loss:.4f} | "
                f"vs_prof={p_prof:+.2f} BB/hand, vs_ckpt={p_ckpt:+.2f} BB/hand, vs_rand={p_rand:+.2f} BB/hand",
                flush=True,
            )

        # Save checkpoint
        if _should_save_checkpoint(iteration, final_iteration, checkpoint_interval):
            ckpt_name = f"hybrid_checkpoint_iter_{iteration}.pt"
            ckpt_file = os.path.join(save_dir, ckpt_name)
            torch.save(
                {
                    "iteration": iteration,
                    "profile_ratio": profile_ratio,
                    "adv_loss": last_adv_loss,
                    "strat_loss": last_strat_loss,
                    "advantage_net": learning_agent.advantage_net.state_dict(),
                    "strategy_net": learning_agent.strategy_net.state_dict(),
                    "advantage_model_state_dict": learning_agent.advantage_net.state_dict(),
                    "strategy_model_state_dict": learning_agent.strategy_net.state_dict(),
                    "losses": losses,
                    "profits_vs_checkpoints": profits_vs_checkpoints,
                    "profits_vs_profiles": profits_vs_profiles,
                    "profits_vs_random": profits_vs_random,
                    "schema_version": 1,
                    "agent_type": "standard",
                    "num_players": 6,
                    "player_id": learning_agent.player_id,
                },
                ckpt_file,
            )
            print(f"[Hybrid CFR] 💾 Saved checkpoint to {ckpt_file}", flush=True)

    writer.close()

    summary = {
        "final_iteration": final_iteration,
        "total_iterations_trained": additional_iterations,
        "profile_ratio": profile_ratio,
        "final_adv_loss": last_adv_loss,
        "final_strat_loss": last_strat_loss,
        "final_profit_vs_checkpoints": profits_vs_checkpoints[-1] if profits_vs_checkpoints else 0.0,
        "final_profit_vs_profiles": profits_vs_profiles[-1] if profits_vs_profiles else 0.0,
        "final_profit_vs_random": profits_vs_random[-1] if profits_vs_random else 0.0,
        "latest_checkpoint": os.path.join(save_dir, f"hybrid_checkpoint_iter_{final_iteration}.pt"),
    }
    return learning_agent, summary
