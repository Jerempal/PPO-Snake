"""Évaluation fondée sur le score et arrêt anticipé de l'entraînement Snake."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv


class ScoreEvalCallback(BaseCallback):
    """Select checkpoints by apples eaten and stop after a score plateau."""

    def __init__(
        self,
        eval_env: VecEnv,
        eval_freq: int,
        n_eval_episodes: int,
        best_model_dir: Path,
        deterministic: bool,
        patience: int,
        min_score_improvement: float,
        seed: int,
        map_sizes: list[int] | None = None,
    ) -> None:
        super().__init__(verbose=0)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_model_dir = best_model_dir
        self.deterministic = deterministic
        self.patience = patience
        self.min_score_improvement = min_score_improvement
        self.seed = seed
        self.map_sizes = map_sizes
        self.selection_metric = "normalized_mean_score" if map_sizes else "mean_score"
        self.best_mean_score = float("-inf")
        self.evaluations_without_improvement = 0
        self.history: list[dict[str, float | int]] = []

    def _on_training_start(self) -> None:
        """Protect the starting policy before applying any gradient update."""
        self._evaluate_and_record(count_plateau=False)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq:
            return True
        return self._evaluate_and_record(count_plateau=True)

    def _on_training_end(self) -> None:
        """Evaluate the final policy when the cadence does not land on the last step."""
        last_evaluated_step = int(self.history[-1]["timesteps"]) if self.history else -1
        if last_evaluated_step != self.num_timesteps:
            self._evaluate_and_record(count_plateau=False)

    def _evaluate_and_record(self, *, count_plateau: bool) -> bool:
        metrics = self._evaluate()
        selection_score = metrics[self.selection_metric]
        improved = selection_score > self.best_mean_score + self.min_score_improvement
        if improved:
            self.best_mean_score = selection_score
            self.evaluations_without_improvement = 0
            self.best_model_dir.mkdir(parents=True, exist_ok=True)
            self.model.save(self.best_model_dir / "best_model")
        elif count_plateau:
            self.evaluations_without_improvement += 1

        record: dict[str, float | int] = {
            "timesteps": self.num_timesteps,
            **metrics,
            "selection_score": selection_score,
            "best_mean_score": self.best_mean_score,
            "evaluations_without_improvement": self.evaluations_without_improvement,
        }
        self.history.append(record)
        self._write_summary()
        for name, value in metrics.items():
            self.logger.record(f"eval/{name}", value)
        self.logger.record("eval/best_mean_score", self.best_mean_score)
        self.logger.record("eval/selection_score", selection_score)
        self.logger.record(
            "eval/evaluations_without_improvement",
            self.evaluations_without_improvement,
        )
        return not count_plateau or self.evaluations_without_improvement < self.patience

    def _evaluate(self) -> dict[str, float]:
        scores: list[float] = []
        rewards: list[float] = []
        lengths: list[float] = []
        by_map = {size: [] for size in self.map_sizes or []}
        wins = {size: [] for size in self.map_sizes or []}
        # Give every worker a fixed quota. Stopping after the first N completed
        # episodes over-represents short failures when vectorized games have
        # very different durations.
        episode_quotas = np.full(
            self.eval_env.num_envs,
            self.n_eval_episodes // self.eval_env.num_envs,
            dtype=np.int64,
        )
        episode_quotas[: self.n_eval_episodes % self.eval_env.num_envs] += 1
        completed_episodes = np.zeros(self.eval_env.num_envs, dtype=np.int64)
        self.eval_env.seed(self.seed)
        observation = self.eval_env.reset()
        episode_rewards = np.zeros(self.eval_env.num_envs, dtype=np.float64)
        episode_lengths = np.zeros(self.eval_env.num_envs, dtype=np.int64)
        while np.any(completed_episodes < episode_quotas):
            action, _ = self.model.predict(observation, deterministic=self.deterministic)
            observation, reward, done, infos = self.eval_env.step(action)
            episode_rewards += np.asarray(reward, dtype=np.float64)
            episode_lengths += 1
            for index, finished in enumerate(done):
                if bool(finished) and completed_episodes[index] < episode_quotas[index]:
                    score = float(infos[index]["score"])
                    scores.append(score)
                    if self.map_sizes:
                        size = self.map_sizes[index]
                        by_map[size].append(score)
                        wins[size].append(float(infos[index].get("won", False)))
                    rewards.append(float(episode_rewards[index]))
                    lengths.append(float(episode_lengths[index]))
                    completed_episodes[index] += 1
                if bool(finished):
                    episode_rewards[index] = 0.0
                    episode_lengths[index] = 0
        metrics = {
            "mean_score": float(np.mean(scores)),
            "max_score": float(np.max(scores)),
            "score_q25": float(np.quantile(scores, 0.25)),
            "mean_reward": float(np.mean(rewards)),
            "mean_length": float(np.mean(lengths)),
        }
        if self.map_sizes:
            normalized = []
            for size, values in by_map.items():
                mean = float(np.mean(values))
                normalized.append(mean / (size * size - 3))
                metrics.update(
                    {
                        f"map_{size}_mean_score": mean,
                        f"map_{size}_normalized_score": normalized[-1],
                        f"map_{size}_win_rate": float(np.mean(wins[size])),
                        f"map_{size}_episodes": len(values),
                    }
                )
            metrics["normalized_mean_score"] = float(np.mean(normalized))
            metrics["worst_map_normalized_score"] = min(normalized)
        return metrics

    def _write_summary(self) -> None:
        payload: dict[str, Any] = {
            "selection_metric": self.selection_metric,
            "best_selection_score": self.best_mean_score,
            "best_mean_score": self.best_mean_score,
            "evaluations_without_improvement": self.evaluations_without_improvement,
            "stopped_early": self.evaluations_without_improvement >= self.patience,
            "history": self.history,
        }
        self.best_model_dir.mkdir(parents=True, exist_ok=True)
        (self.best_model_dir / "evaluation_summary.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
