"""Unit and integration tests for final stage and champion crowning."""

import pytest
from src.agents.random_agent import RandomAgent
from src.tournament.state import Stage, PlayerRecord
from src.tournament.tournament_env import TournamentEnv
from src.tournament.advancement import AdvancementManager


def test_champion_determination_by_net_bb():
    mgr = AdvancementManager()
    final_players = [
        PlayerRecord(player_id=1, stage_net_bb=-40.0),
        PlayerRecord(player_id=2, stage_net_bb=85.0),
        PlayerRecord(player_id=3, stage_net_bb=120.0),
        PlayerRecord(player_id=4, stage_net_bb=-10.0),
        PlayerRecord(player_id=5, stage_net_bb=-55.0),
        PlayerRecord(player_id=6, stage_net_bb=-100.0),
    ]

    champ, standings = mgr.determine_champion(final_players)
    assert champ.player_id == 3
    assert champ.stage_net_bb == 120.0
    assert standings[0].player_id == 3
    assert standings[1].player_id == 2


def test_final_stage_execution():
    env = TournamentEnv(
        num_players=120,
        prelim_rounds=1,
        hands_per_prelim_round=1,
        semifinal_hands=1,
        final_hands=5,
        seed=77,
    )
    agents = {i: RandomAgent(player_id=0) for i in range(120)}
    env.run_preliminary(agents)
    env.run_semifinal(agents)
    champ, final_standings = env.run_final(agents)

    assert env.stage == Stage.FINISHED
    assert len(final_standings) == 6
    assert champ == final_standings[0]
    assert env.champion == champ

    # Zero-sum check on final table
    net_final = sum(p.stage_net_bb for p in final_standings)
    assert pytest.approx(net_final, abs=1e-4) == 0.0
