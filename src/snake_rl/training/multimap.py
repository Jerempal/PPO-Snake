"""Static equal worker allocation: equal transitions, never equal episode counts."""

import json
from dataclasses import replace
from functools import partial
from pathlib import Path

from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from snake_rl.env import SnakeEnv
from snake_rl.training.trajectory_restart import TrajectoryRestartWrapper


def _worker(config, size, rank, monitor_dir, evaluation):
    environment = replace(config.environment, width=size, height=size)
    if evaluation:
        environment = replace(environment, loop_max_state_visits=None)
    env = SnakeEnv(environment, config.rewards)
    env = Monitor(env, filename=str(monitor_dir / str(rank)) if monitor_dir else None)
    if evaluation:
        return TimeLimit(env, max_episode_steps=config.evaluation.horizon)
    if config.trajectory_restart.enabled:
        env = TrajectoryRestartWrapper(env, config.trajectory_restart)
    return env


def make_multimap_env(config, run_dir: Path, *, evaluation=False):
    sizes = config.map_sizes
    count = config.training.n_envs
    if evaluation:
        count = min(config.evaluation.n_envs // len(sizes), config.evaluation.episodes) * len(sizes)
    assignment = [sizes[i % len(sizes)] for i in range(count)]
    monitor_dir = None if evaluation else run_dir / "monitor"
    if monitor_dir:
        monitor_dir.mkdir(parents=True, exist_ok=True)
    factories = [
        partial(_worker, config, size, i, monitor_dir, evaluation)
        for i, size in enumerate(assignment)
    ]
    cls = SubprocVecEnv if config.training.env_backend == "subprocess" else DummyVecEnv
    env = cls(factories)
    env.seed(config.training.seed + (10_000 if evaluation else 0))
    return env, assignment


class MapExposureCallback(BaseCallback):
    def __init__(self, assignment, output: Path):
        super().__init__()
        self.assignment = assignment
        self.output = output

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        exposure = {
            str(size): self.num_timesteps * self.assignment.count(size) // len(self.assignment)
            for size in sorted(set(self.assignment))
        }
        for size, count in exposure.items():
            self.logger.record(f"maps/{size}/transitions", count)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(
            json.dumps(
                {"total_transitions": self.num_timesteps, "transitions_per_map": exposure}, indent=2
            )
            + "\n"
        )

    def _on_training_end(self):
        self._on_rollout_end()
