"""Occupancy-gated restarts from real, apple-boundary training states.

Only start states are reused. PPO never replays actions, rewards or old gradients.
Banks stay inside workers; the parent receives only scalar episode metadata.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from snake_rl.config import TrajectoryRestartConfig
from snake_rl.env import DIRECTIONS, Observation, Position, SnakeEnv


@dataclass(frozen=True)
class TrajectoryStart:
    """Geometry captured just after an apple; not an exact process checkpoint."""

    snake: tuple[Position, ...]
    direction: Position
    food: Position
    width: int
    height: int

    @classmethod
    def capture(cls, env: SnakeEnv) -> TrajectoryStart:
        if env._episode_over or env.food is None or env.steps_since_food != 0:
            raise ValueError("La capture exige un état actif juste après une pomme.")
        state = cls(tuple(env.snake), env.direction, env.food, env.config.width, env.config.height)
        state.validate(env)
        return state

    def validate(self, env: SnakeEnv) -> None:
        if (self.width, self.height) != (env.config.width, env.config.height):
            raise ValueError("La trajectoire appartient à une autre taille de carte.")
        if not 3 <= len(self.snake) < self.width * self.height:
            raise ValueError("Longueur de corps invalide dans la trajectoire.")
        if len(set(self.snake)) != len(self.snake) or self.food in self.snake:
            raise ValueError("Chevauchement du corps ou de la pomme dans la trajectoire.")
        if any(
            not (0 <= x < self.width and 0 <= y < self.height) for x, y in (*self.snake, self.food)
        ):
            raise ValueError("Les coordonnées de la trajectoire sortent de la carte.")
        if any(
            abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1
            for a, b in zip(self.snake, self.snake[1:], strict=False)
        ):
            raise ValueError("Le corps de la trajectoire doit être contigu.")
        heading = tuple(a - b for a, b in zip(self.snake[0], self.snake[1], strict=True))
        if self.direction not in DIRECTIONS.values() or self.direction != heading:
            raise ValueError(
                "La direction ne correspond pas à la tête et au cou de la trajectoire."
            )

    def restart(self, env: SnakeEnv) -> tuple[Observation, dict[str, Any]]:
        """After a normal reset, replace geometry and reset episode-local counters.

        Preserve the worker's fresh RNG: future apples need not repeat the source
        trajectory. At an apple boundary the loop history contains only this state.
        """
        self.validate(env)
        env.snake = list(self.snake)
        env.direction, env.food = self.direction, self.food
        env.initial_length = len(self.snake)
        env.score = env.steps = env.steps_since_food = env.longest_steps_without_food = 0
        env._episode_over = False
        env._start_type = "trajectory"
        env._state_visits = {env._state_signature(): 1}
        return env._observation(), env._info()


class TrajectoryRestartWrapper(gym.Wrapper):
    """Bounded, worker-local reservoir; never changes a running episode."""

    def __init__(self, env: gym.Env, config: TrajectoryRestartConfig) -> None:
        super().__init__(env)
        self.settings = config
        self.snake_env: SnakeEnv = env.unwrapped
        if self.snake_env.start_states.enabled:
            raise ValueError(
                "Ne mélangez pas les départs structurés et les reprises de trajectoires."
            )
        self.lengths = config.lengths(self.snake_env.config.width * self.snake_env.config.height)
        self.banks: list[list[TrajectoryStart]] = [[] for _ in self.lengths]
        self.seen = [0] * len(self.lengths)
        self.stage = 0
        self.episode_stage = 0
        self.eligible = True
        self.rng = np.random.default_rng(0)
        self.starts = self.restarts = self.fallbacks = 0

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        if not self.settings.enabled:
            return observation, info
        if seed is not None:
            self.rng = np.random.default_rng(np.random.SeedSequence([seed, 7319]))
        self.starts += 1
        self.episode_stage = self.stage
        self.eligible = self.stage == 0
        if self.stage and self.rng.random() >= self.settings.standard_probability:
            bank = self.banks[self.stage - 1]
            if bank:
                observation, info = bank[int(self.rng.integers(len(bank)))].restart(self.snake_env)
                self.restarts += 1
                self.eligible = True
            else:
                self.fallbacks += 1
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if not self.settings.enabled:
            return observation, reward, terminated, truncated, info
        length = info["length"]
        if (
            not (terminated or truncated)
            and info["steps_since_food"] == 0
            and length in self.lengths
        ):
            index = self.lengths.index(length)
            state = TrajectoryStart.capture(self.snake_env)
            self.seen[index] += 1
            bank = self.banks[index]
            if len(bank) < self.settings.reservoir_size:
                bank.append(state)
            else:
                slot = int(self.rng.integers(self.seen[index]))
                if slot < len(bank):
                    bank[slot] = state
        if terminated or truncated:
            target = (
                self.lengths[self.episode_stage]
                if self.episode_stage < len(self.lengths)
                else self.snake_env.config.width * self.snake_env.config.height
            )
            info.update(
                trajectory_stage=self.episode_stage,
                trajectory_eligible=self.eligible,
                trajectory_success=length >= target,
            )
        return observation, reward, terminated, truncated, info

    def set_trajectory_stage(self, stage: int) -> None:
        if not 0 <= stage <= len(self.lengths):
            raise ValueError("Palier de trajectoire invalide.")
        self.stage = stage  # Applied only at the next reset.

    def trajectory_status(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "bank_sizes": [len(bank) for bank in self.banks],
            "starts": self.starts,
            "restarts": self.restarts,
            "fallbacks": self.fallbacks,
        }


class PromotionGate:
    """A rolling window of completed, stage-matched training episodes, not a test score."""

    def __init__(self, config: TrajectoryRestartConfig) -> None:
        self.config = config
        self.stage = 0
        self.outcomes: deque[bool] = deque(maxlen=config.promotion_window)

    def observe(self, info: dict[str, Any]) -> None:
        if info.get("trajectory_stage") == self.stage and info.get("trajectory_eligible", False):
            self.outcomes.append(bool(info["trajectory_success"]))

    @property
    def success_rate(self) -> float:
        return sum(self.outcomes) / len(self.outcomes) if self.outcomes else 0.0

    def promote(self, *, available: bool) -> bool:
        if (
            self.stage < len(self.config.fractions)
            and available
            and len(self.outcomes) == self.config.promotion_window
            and self.success_rate >= self.config.promotion_success_rate
        ):
            self.stage += 1
            self.outcomes.clear()
            return True
        return False


class TrajectoryRestartCallback(BaseCallback):
    """Promote at rollout boundaries; leave standard-start model selection untouched."""

    def __init__(
        self,
        config: TrajectoryRestartConfig,
        output: Path,
        indices: list[int] | None = None,
        prefix: str = "trajectory",
    ) -> None:
        super().__init__()
        self.gate = PromotionGate(config)
        self.output = output
        self.indices = indices
        self.prefix = prefix
        self.promotions: list[dict[str, Any]] = []  # At most one entry per configured fraction.

    def _on_step(self) -> bool:
        for index, (done, info) in enumerate(
            zip(self.locals.get("dones", []), self.locals.get("infos", []), strict=True)
        ):
            if done and (self.indices is None or index in self.indices):
                self.gate.observe(info)
        return True

    def _on_rollout_end(self) -> None:
        workers = self.training_env.env_method("trajectory_status", indices=self.indices)
        stage = self.gate.stage
        available = stage < len(self.gate.config.fractions) and any(
            worker["bank_sizes"][stage] for worker in workers
        )
        rate, count = self.gate.success_rate, len(self.gate.outcomes)
        if self.gate.promote(available=available):
            self.training_env.env_method(
                "set_trajectory_stage", self.gate.stage, indices=self.indices
            )
            self.promotions.append(
                {
                    "transitions": self.num_timesteps,
                    "stage": self.gate.stage,
                    "fraction": self.gate.config.fractions[stage],
                    "success_rate": rate,
                    "episodes": count,
                }
            )
        self.logger.record(f"{self.prefix}/stage", self.gate.stage)
        self.logger.record(f"{self.prefix}/gate_success_rate", self.gate.success_rate)
        self.logger.record(f"{self.prefix}/gate_episodes", len(self.gate.outcomes))
        for key in ("starts", "restarts", "fallbacks"):
            self.logger.record(f"{self.prefix}/{key}", sum(worker[key] for worker in workers))
        self.logger.record(f"{self.prefix}/bank_states", sum(sum(w["bank_sizes"]) for w in workers))
        self._write_summary(workers)

    def _on_training_end(self) -> None:
        self._write_summary(self.training_env.env_method("trajectory_status", indices=self.indices))

    def _write_summary(self, workers: list[dict[str, Any]]) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "transitions": self.num_timesteps,
                    "stage": self.gate.stage,
                    "promotions": self.promotions,
                    "gate_episodes": len(self.gate.outcomes),
                    "gate_success_rate": self.gate.success_rate,
                    "workers": workers,
                    "resume_behavior": "fresh banks and stage zero in each new run",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
