"""Play Snake manually or watch a trained reinforcement-learning agent."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from snake_rl.cli import add_map_arguments, apply_map_arguments
from snake_rl.config import ExperimentConfig, load_config, load_resolved_config
from snake_rl.env import DIRECTIONS, Heading, RelativeAction, SnakeEnv
from snake_rl.evaluation_protocol import FIXED_HORIZON, prepare_evaluation


def play(config: ExperimentConfig, model_path: Path | None, seed: int) -> None:
    try:
        import pygame
    except ImportError as exc:
        raise ImportError("Le jeu nécessite : pip install 'snake-rl[play]'") from exc

    model: Any = None
    if model_path is not None:
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise ImportError(
                "Model playback requires: pip install 'snake-rl[train,play]'"
            ) from exc
        model = PPO.load(model_path)

    env = SnakeEnv(config.environment, config.rewards, render_mode="human")
    observation, _ = env.reset(seed=seed)
    desired_direction = DIRECTIONS[Heading.RIGHT]
    key_actions = {
        pygame.K_UP: Heading.UP,
        pygame.K_DOWN: Heading.DOWN,
        pygame.K_LEFT: Heading.LEFT,
        pygame.K_RIGHT: Heading.RIGHT,
    }

    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    running = False
                elif event.type == pygame.KEYDOWN and event.key in key_actions:
                    desired_direction = DIRECTIONS[key_actions[event.key]]
                    action = int(env.direction, desired_direction)

            if not running:
                break
            if model is not None:
                predicted, _ = model.predict(observation, deterministic=True)
                action = int(predicted)
            # else:
            #     action = int(_relative_action(
            #         env.direction, desired_direction))

            observation, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(f"Partie terminée : score={info['score']}, actions={info['steps']}")
                pygame.time.wait(350)
                seed += 1
                observation, _ = env.reset(seed=seed)
                desired_direction = DIRECTIONS[Heading.RIGHT]
    finally:
        env.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ppo.toml"))
    parser.add_argument("--model", type=Path, help="Optional trained neural model.")
    parser.add_argument("--seed", type=int, default=42)
    add_map_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = args.config
    if args.model is not None and config_path == Path("configs/ppo.toml"):
        resolved = _find_run_config(args.model)
        if resolved is not None:
            config_path = resolved
    config = (
        load_resolved_config(config_path)
        if config_path.suffix == ".json"
        else load_config(config_path)
    )
    config = apply_map_arguments(config, args)
    config, _ = prepare_evaluation(config, protocol=FIXED_HORIZON, horizon=10_000)
    play(config, args.model, args.seed)


def _relative_action(current: tuple[int, int], desired: tuple[int, int]) -> RelativeAction:
    if desired == (current[1], -current[0]):
        return RelativeAction.LEFT
    if desired == (-current[1], current[0]):
        return RelativeAction.RIGHT
    return RelativeAction.STRAIGHT


def _find_run_config(model_path: Path) -> Path | None:
    for directory in (model_path.resolve().parent, *model_path.resolve().parents):
        candidate = directory / "resolved_config.json"
        if candidate.exists():
            return candidate
    return None


if __name__ == "__main__":
    main()
