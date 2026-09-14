from __future__ import annotations
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Any
import yaml


class ExecutionMode(str, Enum):
    DEV = "DEV"
    TEST = "TEST"
    OFFICIAL = "OFFICIAL"


@dataclass(frozen=True)
class TournamentConfig:
    """Standardized tournament configuration for AgentPoker.

    Defaults strictly adhere to the official 120-player tournament format:
    - Field: 120 players, 20 tables (6-max)
    - Blinds: SB 500, BB 1000, Starting Stack 100 BB (100,000 chips)
    - Preliminary: 10 rounds x 20 hands = 200 hands, continuous stacks, auto-rebuy
    - Semifinal: Top 12 qualify, 2 tables of 6, 20 hands, stack reset 100 BB
    - Final: Top 6 qualify, 1 table of 6, 30 hands, stack reset 100 BB
    """
    field_size: int = 120
    small_blind: int = 500
    big_blind: int = 1000
    starting_stack_bb: int = 100
    seats_per_table: int = 6
    preliminary_rounds: int = 10
    hands_per_round: int = 20
    min_completion_rate: float = 0.80
    semifinal_qualifiers: int = 12
    semifinal_tables: int = 2
    semifinal_hands: int = 20
    final_qualifiers: int = 6
    final_hands: int = 30
    execution_mode: ExecutionMode | None = None

    def __post_init__(self) -> None:
        if self.field_size < self.semifinal_qualifiers:
            raise ValueError(f"field_size ({self.field_size}) must be >= semifinal_qualifiers ({self.semifinal_qualifiers})")
        if self.field_size % self.seats_per_table != 0:
            raise ValueError(f"field_size ({self.field_size}) must be divisible by seats_per_table ({self.seats_per_table})")
        if self.semifinal_qualifiers != self.semifinal_tables * self.seats_per_table:
            raise ValueError(f"semifinal_qualifiers ({self.semifinal_qualifiers}) must equal semifinal_tables * seats_per_table")
        if self.final_qualifiers != self.seats_per_table:
            raise ValueError(f"final_qualifiers ({self.final_qualifiers}) must equal seats_per_table ({self.seats_per_table})")
        if not (0.0 < self.min_completion_rate <= 1.0):
            raise ValueError(f"min_completion_rate ({self.min_completion_rate}) must be in (0.0, 1.0]")
        if self.small_blind <= 0 or self.big_blind <= self.small_blind:
            raise ValueError(f"Invalid blinds: SB={self.small_blind}, BB={self.big_blind}")

        # Resolve execution mode
        if self.execution_mode is None:
            if (
                self.field_size == 120
                and self.seats_per_table == 6
                and self.preliminary_rounds == 10
                and self.hands_per_round == 20
                and self.semifinal_hands == 20
                and self.final_hands == 30
            ):
                mode = ExecutionMode.OFFICIAL
            else:
                mode = ExecutionMode.TEST
            object.__setattr__(self, "execution_mode", mode)

        if self.execution_mode == ExecutionMode.OFFICIAL:
            if self.field_size != 120:
                raise ValueError(f"OFFICIAL execution mode strictly requires field_size == 120, got {self.field_size}")
            if self.seats_per_table != 6:
                raise ValueError(f"OFFICIAL execution mode strictly requires seats_per_table == 6, got {self.seats_per_table}")
            if self.total_preliminary_hands != 200:
                raise ValueError(f"OFFICIAL execution mode strictly requires 200 preliminary hands, got {self.total_preliminary_hands}")
            if self.semifinal_hands != 20:
                raise ValueError(f"OFFICIAL execution mode strictly requires 20 semifinal hands, got {self.semifinal_hands}")
            if self.final_hands != 30:
                raise ValueError(f"OFFICIAL execution mode strictly requires 30 final hands, got {self.final_hands}")

    @property
    def starting_chips(self) -> int:
        return self.starting_stack_bb * self.big_blind

    @property
    def total_preliminary_hands(self) -> int:
        return self.preliminary_rounds * self.hands_per_round

    @property
    def min_hands_required(self) -> int:
        return int(self.total_preliminary_hands * self.min_completion_rate)

    @property
    def num_tables(self) -> int:
        return self.field_size // self.seats_per_table

    @property
    def top12_rate_baseline(self) -> float:
        return self.semifinal_qualifiers / self.field_size

    @property
    def final_rate_baseline(self) -> float:
        return self.final_qualifiers / self.field_size

    @property
    def champion_rate_baseline(self) -> float:
        return 1.0 / self.field_size

    @property
    def baseline_fitness(self) -> float:
        """Theoretical zero-sum equilibrium fitness under current weighting."""
        # fit = 0.20 * top_rate + 0.20 * final_rate + 0.25 * champ_rate + 0.05 * rank_norm + 0.30 * bb_factor
        # Under equilibrium: rank_norm = 0.5, bb_factor = 0.5
        return (
            0.20 * self.top12_rate_baseline
            + 0.20 * self.final_rate_baseline
            + 0.25 * self.champion_rate_baseline
            + 0.05 * 0.50
            + 0.30 * 0.50
        )

    @classmethod
    def official_120(cls) -> TournamentConfig:
        """Official 120-player tournament configuration."""
        return cls(field_size=120, execution_mode=ExecutionMode.OFFICIAL)

    @classmethod
    def fast_36(cls, mode: ExecutionMode = ExecutionMode.DEV) -> TournamentConfig:
        """Fast 36-player research profile (same structure, scaled down field). Allowed only in DEV/TEST."""
        if mode == ExecutionMode.OFFICIAL:
            raise ValueError("fast_36 cannot be used in OFFICIAL execution mode.")
        return cls(field_size=36, execution_mode=mode)

    @classmethod
    def smoke_24(cls, mode: ExecutionMode = ExecutionMode.TEST) -> TournamentConfig:
        """Minimal 24-player smoke test profile. Allowed only in DEV/TEST."""
        if mode == ExecutionMode.OFFICIAL:
            raise ValueError("smoke_24 cannot be used in OFFICIAL execution mode.")
        return cls(field_size=24, execution_mode=mode)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TournamentConfig:
        """Load configuration from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        # Map common legacy aliases
        mapping = {
            "small_blind": "small_blind",
            "big_blind": "big_blind",
            "starting_stack_bb": "starting_stack_bb",
            "seats": "seats_per_table",
            "seats_per_table": "seats_per_table",
            "preliminary_rounds": "preliminary_rounds",
            "rounds": "preliminary_rounds",
            "hands_per_round": "hands_per_round",
            "hpr": "hands_per_round",
            "minimum_completion": "min_completion_rate",
            "min_completion": "min_completion_rate",
            "min_completion_rate": "min_completion_rate",
            "field_size": "field_size",
            "semifinal_qualifiers": "semifinal_qualifiers",
            "semifinal_tables": "semifinal_tables",
            "semifinal_hands": "semifinal_hands",
            "final_qualifiers": "final_qualifiers",
            "final_hands": "final_hands",
            "execution_mode": "execution_mode",
        }
        kwargs: dict[str, Any] = {}
        for k, v in data.items():
            if k in mapping:
                if mapping[k] == "execution_mode" and v is not None:
                    kwargs["execution_mode"] = ExecutionMode(v)
                else:
                    kwargs[mapping[k]] = v
        return cls(**kwargs)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        data = asdict(self)
        if self.execution_mode is not None:
            data["execution_mode"] = self.execution_mode.value
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
