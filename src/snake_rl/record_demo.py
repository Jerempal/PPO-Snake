"""Enregistre un GIF comparant PPO au cycle hamiltonien strict."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from snake_rl.baselines import HamiltonianCyclePolicy
from snake_rl.config import load_config, load_resolved_config
from snake_rl.env import SnakeEnv
from snake_rl.evaluate import _find_run_config
from snake_rl.evaluation_protocol import FIXED_HORIZON, prepare_evaluation


def record_demo(
    *,
    model_path: Path,
    config_path: Path | None,
    output: Path,
    seed: int,
    steps: int,
    frame_interval: int,
) -> None:
    """Lance les deux agents avec la même graine et enregistre un GIF comparatif."""
    try:
        from PIL import Image, ImageDraw
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise ImportError(
            "L'enregistrement nécessite : pip install 'snake-rl[train,play]'"
        ) from exc
    if steps < 1 or frame_interval < 1:
        raise ValueError("steps et frame_interval doivent être positifs.")

    config_path = config_path or _find_run_config(model_path)
    if config_path is None:
        raise ValueError("Configuration du modèle introuvable ; indiquez --config.")
    config = (
        load_resolved_config(config_path)
        if config_path.suffix == ".json"
        else load_config(config_path)
    )
    config, _ = prepare_evaluation(
        config,
        protocol=FIXED_HORIZON,
        horizon=10_000,
    )
    model = PPO.load(model_path)
    neural_env = SnakeEnv(config.environment, config.rewards, render_mode="rgb_array")
    cycle_env = SnakeEnv(config.environment, config.rewards, render_mode="rgb_array")
    neural_observation, neural_info = neural_env.reset(seed=seed)
    _, cycle_info = cycle_env.reset(seed=seed)
    hamiltonian = HamiltonianCyclePolicy()
    neural_finished = cycle_finished = False
    frames = []

    def panel(env: SnakeEnv, label: str, info: dict[str, object], finished: bool):
        frame = env.render()
        assert frame is not None
        image = Image.fromarray(frame).resize((320, 320), Image.Resampling.NEAREST)
        canvas = Image.new("RGB", (320, 354), (20, 24, 33))
        canvas.paste(image, (0, 34))
        suffix = " - victoire" if info.get("won") else " - termine" if finished else ""
        ImageDraw.Draw(canvas).text(
            (8, 10),
            f"{label} | pommes {info['score']} | pas {info['steps']}{suffix}",
            fill=(235, 238, 245),
        )
        return canvas

    try:
        for step in range(steps):
            if not neural_finished:
                action, _ = model.predict(neural_observation, deterministic=True)
                neural_observation, _, terminated, truncated, neural_info = neural_env.step(
                    int(np.asarray(action).item())
                )
                neural_finished = terminated or truncated
            if not cycle_finished:
                _, _, terminated, truncated, cycle_info = cycle_env.step(
                    hamiltonian.action(cycle_env)
                )
                cycle_finished = terminated or truncated
            if step % frame_interval == 0 or (neural_finished and cycle_finished):
                left = panel(neural_env, "PPO", neural_info, neural_finished)
                right = panel(cycle_env, "Cycle strict", cycle_info, cycle_finished)
                combined = Image.new("RGB", (644, 354), (10, 12, 18))
                combined.paste(left, (0, 0))
                combined.paste(right, (324, 0))
                frames.append(combined)
            if neural_finished and cycle_finished:
                break
    finally:
        neural_env.close()
        cycle_env.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=90,
        loop=0,
        optimize=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Par défaut : configuration du modèle.")
    parser.add_argument("--output", type=Path, default=Path("assets/gameplay.gif"))
    parser.add_argument("--seed", type=int, default=60_000)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--frame-interval", type=int, default=20)
    args = parser.parse_args()
    record_demo(
        model_path=args.model,
        config_path=args.config,
        output=args.output,
        seed=args.seed,
        steps=args.steps,
        frame_interval=args.frame_interval,
    )


if __name__ == "__main__":
    main()
