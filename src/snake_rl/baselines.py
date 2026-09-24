"""Solveurs déterministes et point d'entrée de leur évaluation."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from snake_rl.config import ExperimentConfig, evaluation_start_states, load_config
from snake_rl.env import Position, RelativeAction, SnakeEnv
from snake_rl.evaluation_protocol import FIXED_HORIZON, PROTOCOLS, prepare_evaluation
from snake_rl.expert import SafePlanner
from snake_rl.metrics import summarize_episodes


class BaselinePolicy(Protocol):
    def action(self, env: SnakeEnv) -> int: ...


class HamiltonianPolicy:
    """Follow a Hamiltonian cycle and take only order-preserving shortcuts."""

    def __init__(self) -> None:
        self._orientations: dict[int, int] = {}
        self._cycles: dict[tuple[int, int], tuple[list[Position], dict[Position, int]]] = {}

    def action(self, env: SnakeEnv) -> int:
        cycle, indices = self._cycle(env)
        identity = id(env)
        if env.steps == 0 or identity not in self._orientations:
            self._orientations[identity] = self._initial_orientation(env, indices, len(cycle))
        orientation = self._orientations[identity]
        head = env.snake[0]
        food = env.food
        food_distance = (
            self._forward_distance(head, food, indices, orientation, len(cycle))
            if food is not None
            else len(cycle)
        )

        candidates: list[tuple[int, int]] = []
        for action in range(env.action_space.n):
            dx, dy = env.direction_for_action(action)
            candidate = (head[0] + dx, head[1] + dy)
            if env.collision_type(candidate) is not None:
                continue
            distance = self._forward_distance(head, candidate, indices, orientation, len(cycle))
            eats = candidate == food
            fills_board = eats and len(env.snake) + 1 == len(cycle)
            body_after_move = env.snake if eats else env.snake[:-1]
            retained_distance = min(
                self._forward_distance(head, segment, indices, orientation, len(cycle))
                for segment in body_after_move[1:]
            )
            if not 0 < distance < retained_distance and not fills_board:
                continue
            next_clearance = min(
                self._forward_distance(candidate, segment, indices, orientation, len(cycle))
                for segment in body_after_move
            )
            if next_clearance > 1 or fills_board:
                candidates.append((action, distance))

        if not candidates:
            for fallback in (
                RelativeAction.STRAIGHT,
                RelativeAction.LEFT,
                RelativeAction.RIGHT,
            ):
                dx, dy = env.direction_for_action(int(fallback))
                if env.collision_type((head[0] + dx, head[1] + dy)) is None:
                    return int(fallback)
            return int(RelativeAction.STRAIGHT)

        before_food = [candidate for candidate in candidates if candidate[1] <= food_distance]
        if before_food:
            return max(before_food, key=lambda candidate: candidate[1])[0]
        return min(candidates, key=lambda candidate: candidate[1])[0]

    def _cycle(self, env: SnakeEnv) -> tuple[list[Position], dict[Position, int]]:
        dimensions = (env.config.width, env.config.height)
        if dimensions not in self._cycles:
            cells, is_cycle = env.start_path()
            if not is_cycle:
                raise ValueError("Le solveur hamiltonien exige au moins une dimension paire.")
            self._cycles[dimensions] = (cells, {position: i for i, position in enumerate(cells)})
        return self._cycles[dimensions]

    @staticmethod
    def _initial_orientation(env: SnakeEnv, indices: dict[Position, int], size: int) -> int:
        head_index = indices[env.snake[0]]
        neck_index = indices[env.snake[1]]
        difference = (head_index - neck_index) % size
        if difference == 1:
            return 1
        if difference == size - 1:
            return -1
        raise ValueError("Le serpent initial n'est pas ordonné sur le cycle hamiltonien.")

    @staticmethod
    def _forward_distance(
        start: Position,
        target: Position,
        indices: dict[Position, int],
        orientation: int,
        size: int,
    ) -> int:
        return (orientation * (indices[target] - indices[start])) % size


class HamiltonianCyclePolicy(HamiltonianPolicy):
    """Follow the Hamiltonian cycle exactly, without shortcuts or fallbacks."""

    def action(self, env: SnakeEnv) -> int:
        cycle, indices = self._cycle(env)
        identity = id(env)
        if env.steps == 0 or identity not in self._orientations:
            self._orientations[identity] = self._initial_orientation(env, indices, len(cycle))
        orientation = self._orientations[identity]
        head = env.snake[0]
        target = cycle[(indices[head] + orientation) % len(cycle)]
        for action in range(env.action_space.n):
            dx, dy = env.direction_for_action(action)
            if (head[0] + dx, head[1] + dy) == target:
                return action
        raise RuntimeError("La prochaine case du cycle hamiltonien exige un demi-tour interdit.")


def evaluate_baseline(
    policy: BaselinePolicy,
    config: ExperimentConfig,
    *,
    episodes: int,
    seed: int,
    start_state_mode: str,
    max_episode_steps: int | None = None,
    protocol: str = "environment",
) -> dict[str, object]:
    """Evaluate a deterministic policy with the neural evaluation protocol."""
    if episodes < 1:
        raise ValueError("Le nombre de parties doit être positif.")
    if start_state_mode not in {"standard", "configured", "long"}:
        raise ValueError("start_state_mode doit être 'standard', 'configured' ou 'long'.")
    if max_episode_steps is not None and max_episode_steps < 1:
        raise ValueError("max_episode_steps doit être positif.")
    start_states = evaluation_start_states(config.start_states, start_state_mode)
    returns: list[float] = []
    scores: list[int] = []
    lengths: list[int] = []
    initial_lengths: list[int] = []
    reasons: list[str] = []
    longest_food_gaps: list[int] = []
    milestones = (
        tuple(step for step in (1_000, 5_000, 10_000) if step <= max_episode_steps)
        if max_episode_steps is not None
        else ()
    )
    scores_at_steps: dict[int, list[int]] = {step: [] for step in milestones}
    env = SnakeEnv(config.environment, config.rewards, start_states)
    try:
        for episode in range(episodes):
            _, info = env.reset(seed=seed + episode)
            episode_return = 0.0
            terminated = truncated = False
            reached_horizon = False
            episode_scores_at_steps: dict[int, int] = {}
            while not (terminated or truncated or reached_horizon):
                _, reward, terminated, truncated, info = env.step(policy.action(env))
                episode_return += reward
                for step in milestones:
                    if int(info["steps"]) == step:
                        episode_scores_at_steps[step] = int(info["score"])
                reached_horizon = bool(
                    max_episode_steps is not None
                    and int(info["steps"]) >= max_episode_steps
                    and not (terminated or truncated)
                )
            returns.append(episode_return)
            scores.append(int(info["score"]))
            lengths.append(int(info["steps"]))
            initial_lengths.append(int(info["initial_length"]))
            longest_food_gaps.append(int(info["longest_steps_without_food"]))
            for step in milestones:
                scores_at_steps[step].append(episode_scores_at_steps.get(step, int(info["score"])))
            reasons.append(
                "win"
                if info["won"]
                else "loop"
                if info["looped"]
                else "horizon"
                if reached_horizon
                else str(info["collision"] or "terminated")
            )
    finally:
        env.close()

    return {
        "evaluated_at": datetime.now(UTC).isoformat(),
        "policy": type(policy).__name__,
        "episodes": episodes,
        "seed": seed,
        "deterministic": True,
        "start_state_mode": start_state_mode,
        "protocol": protocol,
        "max_episode_steps": max_episode_steps,
        **summarize_episodes(
            returns=returns,
            scores=scores,
            lengths=lengths,
            initial_lengths=initial_lengths,
            termination_reasons=reasons,
            bootstrap_seed=seed + 1_000_000,
            longest_food_gaps=longest_food_gaps,
            scores_at_steps=scores_at_steps or None,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        choices=("planner", "hamiltonian", "hamiltonian-cycle"),
        required=True,
    )
    parser.add_argument("--config", type=Path, default=Path("configs/ppo.toml"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=50_000)
    parser.add_argument(
        "--start-states",
        choices=("standard", "configured", "long"),
        default="standard",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--protocol", choices=PROTOCOLS, default=FIXED_HORIZON)
    parser.add_argument("--horizon", type=int, default=10_000)
    args = parser.parse_args()
    policy: BaselinePolicy
    if args.policy == "planner":
        policy = SafePlanner()
    elif args.policy == "hamiltonian-cycle":
        policy = HamiltonianCyclePolicy()
    else:
        policy = HamiltonianPolicy()
    resolved_config = load_config(args.config)
    evaluation_config, max_episode_steps = prepare_evaluation(
        resolved_config,
        protocol=args.protocol,
        horizon=args.horizon,
    )
    metrics = evaluate_baseline(
        policy,
        evaluation_config,
        episodes=args.episodes,
        seed=args.seed,
        start_state_mode=args.start_states,
        max_episode_steps=max_episode_steps,
        protocol=args.protocol,
    )
    metrics["schema_version"] = 2
    metrics["config"] = evaluation_config.to_dict()
    rendered = json.dumps(metrics, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
