"""Concise human-readable training progress for PPO."""

from __future__ import annotations

import math
import sys
import time
from collections import deque
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from tqdm.auto import tqdm


class TrainingProgressCallback(BaseCallback):
    """Display progress, gameplay results and the latest PPO diagnostics."""

    def __init__(self, total_timesteps: int, rollout_size: int, n_epochs: int) -> None:
        super().__init__(verbose=0)
        self.total_timesteps = total_timesteps
        self.rollout_size = rollout_size
        self.n_epochs = n_epochs
        self.episodes = 0
        self.scores: deque[float] = deque(maxlen=100)
        self.rewards: deque[float] = deque(maxlen=100)
        self.lengths: deque[float] = deque(maxlen=100)
        self.initial_lengths: deque[float] = deque(maxlen=100)
        self.loops: deque[float] = deque(maxlen=100)
        self.collisions: deque[float] = deque(maxlen=100)
        self._last_steps = 0
        self._started_at = 0.0
        self._progress: Any = None

    def _on_training_start(self) -> None:
        self._last_steps = self.model.num_timesteps
        self._started_at = time.monotonic()
        self._progress = tqdm(
            total=self.total_timesteps,
            initial=min(self.model.num_timesteps, self.total_timesteps),
            desc="PPO collecte",
            unit="step",
            dynamic_ncols=True,
            smoothing=0.1,
            disable=not sys.stderr.isatty(),
        )

    def _on_rollout_start(self) -> None:
        self._progress.set_description("PPO collecte", refresh=True)

    def _on_step(self) -> bool:
        current_steps = min(self.model.num_timesteps, self.total_timesteps)
        increment = max(current_steps - self._last_steps, 0)
        if increment:
            self._progress.update(increment)
            self._last_steps = current_steps

        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        for done, info in zip(dones, infos, strict=False):
            if not done:
                continue
            self.episodes += 1
            if "score" in info:
                self.scores.append(float(info["score"]))
            if "initial_length" in info:
                self.initial_lengths.append(float(info["initial_length"]))
            self.loops.append(float(bool(info.get("looped", False))))
            self.collisions.append(float(info.get("collision") is not None))
            episode = info.get("episode")
            if episode:
                self.rewards.append(float(episode["r"]))
                self.lengths.append(float(episode["l"]))

        if increment and (self.model.num_timesteps % max(self.rollout_size // 8, 1) == 0):
            self._refresh_postfix()
        return True

    def _on_rollout_end(self) -> None:
        self._record_game_metrics()
        self._refresh_postfix()
        self._progress.set_description("PPO optimisation", refresh=True)

    def _record_game_metrics(self) -> None:
        if self.scores:
            self.logger.record("game/score_mean_100", float(np.mean(self.scores)))
        if self.rewards:
            self.logger.record("game/reward_mean_100", float(np.mean(self.rewards)))
        if self.lengths:
            self.logger.record("game/length_mean_100", float(np.mean(self.lengths)))
        if self.initial_lengths:
            self.logger.record("game/initial_length_mean_100", float(np.mean(self.initial_lengths)))
        if self.loops:
            self.logger.record("game/loop_rate_100", float(np.mean(self.loops)))
        if self.collisions:
            self.logger.record("game/collision_rate_100", float(np.mean(self.collisions)))
        self.logger.record("game/episodes", self.episodes)

    def _refresh_postfix(self) -> None:
        elapsed = max(time.monotonic() - self._started_at, 1e-6)
        values = self.model.logger.name_to_value
        rollout_count = math.ceil(self.total_timesteps / self.rollout_size)
        epochs_left = max(rollout_count * self.n_epochs - self.model._n_updates, 0)
        postfix: dict[str, str | int] = {
            "fps": f"{self.model.num_timesteps / elapsed:.0f}",
            "games": self.episodes,
            "score": self._mean(self.scores),
            "loop": self._mean(self.loops),
            "loss": self._metric(values, "train/loss"),
            "kl": self._metric(values, "train/approx_kl"),
            "epochs_left": epochs_left,
        }
        self._progress.set_postfix(postfix, refresh=True)

    def _on_training_end(self) -> None:
        if self._progress is not None:
            self._record_game_metrics()
            self._refresh_postfix()
            self._progress.close()
            self.logger.dump(self.model.num_timesteps)

    @staticmethod
    def _mean(values: deque[float]) -> str:
        return "-" if not values else f"{np.mean(values):.2f}"

    @staticmethod
    def _metric(values: dict[str, Any], name: str) -> str:
        value = values.get(name)
        return "-" if value is None else f"{float(value):.4g}"
