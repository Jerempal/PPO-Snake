"""Évalue un modèle d'apprentissage par renforcement avec des graines fixes."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from snake_rl.cli import add_map_arguments, apply_map_arguments
from snake_rl.config import (
    ExperimentConfig,
    evaluation_start_states,
    load_config,
    load_resolved_config,
)
from snake_rl.env import Observation, SnakeEnv
from snake_rl.evaluation_protocol import FIXED_HORIZON, PROTOCOLS, prepare_evaluation
from snake_rl.metrics import summarize_episodes


def _batch_observations(observations: list[Observation]) -> Observation:
    """Empile les observations hybrides pour une prédiction par lot."""
    return {key: np.stack([obs[key] for obs in observations]) for key in observations[0]}


def evaluate(
    model_path: Path,
    config: ExperimentConfig,
    episodes: int,
    seed: int,
    deterministic: bool,
    start_state_mode: str = "standard",
    n_envs: int = 1,
    max_episode_steps: int | None = None,
    protocol: str = "environment",
) -> dict[str, object]:
    try:
        from stable_baselines3 import PPO

    except ImportError as exc:
        raise ImportError("L'evaluation necessite : pip install 'snake-rl[train]'") from exc

    if episodes < 1:
        raise ValueError("Le nombre de parties doit être positif.")
    if n_envs < 1:
        raise ValueError("Le nombre d'environnements doit être positif.")
    if max_episode_steps is not None and max_episode_steps < 1:
        raise ValueError("Le nombre maximal d'actions doit être positif.")
    if start_state_mode not in {"standard", "configured", "long"}:
        raise ValueError("start_state_mode doit être 'standard', 'configured' ou 'long'.")
    model = PPO.load(model_path)
    start_states = evaluation_start_states(config.start_states, start_state_mode)
    worker_count = min(n_envs, episodes)
    all_envs = [
        SnakeEnv(config.environment, config.rewards, start_states) for _ in range(worker_count)
    ]
    active_envs = list(all_envs)
    episode_indices = list(range(worker_count))
    observations = [
        env.reset(seed=seed + episode)[0]
        for env, episode in zip(active_envs, episode_indices, strict=True)
    ]
    episode_returns = [0.0] * worker_count
    returns = [0.0] * episodes
    scores = [0] * episodes
    lengths = [0] * episodes
    initial_lengths = [0] * episodes
    termination_reasons = [""] * episodes
    longest_food_gaps = [0] * episodes
    milestones = (
        tuple(step for step in (1_000, 5_000, 10_000) if step <= max_episode_steps)
        if max_episode_steps is not None
        else ()
    )
    scores_at_steps = {step: [0] * episodes for step in milestones}
    next_episode = worker_count
    try:
        while active_envs:
            actions, _ = model.predict(
                _batch_observations(observations), deterministic=deterministic
            )

            next_envs: list[SnakeEnv] = []
            next_indices: list[int] = []
            next_observations: list[Observation] = []
            next_returns: list[float] = []
            transitions = zip(
                active_envs,
                episode_indices,
                episode_returns,
                np.asarray(actions).reshape(-1),
                strict=True,
            )
            for env, episode, observation_return, action in transitions:
                observation, reward, terminated, truncated, info = env.step(int(action))
                observation_return += reward
                for step in milestones:
                    if int(info["steps"]) == step:
                        scores_at_steps[step][episode] = int(info["score"])
                reached_horizon = bool(
                    max_episode_steps is not None
                    and int(info["steps"]) >= max_episode_steps
                    and not (terminated or truncated)
                )
                episode_finished = terminated or truncated or reached_horizon
                if episode_finished:
                    returns[episode] = observation_return
                    scores[episode] = int(info["score"])
                    lengths[episode] = int(info["steps"])
                    initial_lengths[episode] = int(info["initial_length"])
                    longest_food_gaps[episode] = int(info["longest_steps_without_food"])
                    for step in milestones:
                        if int(info["steps"]) < step:
                            scores_at_steps[step][episode] = int(info["score"])
                    termination_reasons[episode] = (
                        "win"
                        if info["won"]
                        else "loop"
                        if info["looped"]
                        else "horizon"
                        if reached_horizon
                        else str(info["collision"] or "terminated")
                    )
                    if next_episode >= episodes:
                        continue
                    episode = next_episode
                    next_episode += 1
                    observation, _ = env.reset(seed=seed + episode)
                    observation_return = 0.0

                next_envs.append(env)
                next_indices.append(episode)
                next_observations.append(observation)
                next_returns.append(observation_return)

            active_envs = next_envs
            episode_indices = next_indices
            observations = next_observations
            episode_returns = next_returns
    finally:
        for env in all_envs:
            env.close()

    result: dict[str, object] = {
        "schema_version": 2,
        "evaluated_at": datetime.now(UTC).isoformat(),
        "model": str(model_path),
        "algorithm": "PPO",
        "episodes": episodes,
        "n_envs": worker_count,
        "seed": seed,
        "deterministic": deterministic,
        "start_state_mode": start_state_mode,
        "protocol": protocol,
        "max_episode_steps": max_episode_steps,
        **summarize_episodes(
            returns=returns,
            scores=scores,
            lengths=lengths,
            initial_lengths=initial_lengths,
            termination_reasons=termination_reasons,
            bootstrap_seed=seed + 1_000_000,
            longest_food_gaps=longest_food_gaps,
            scores_at_steps=scores_at_steps or None,
        ),
    }
    result["config"] = config.to_dict()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model", type=Path, help="Chemin vers un modèle Stable-Baselines3 entraîné."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Configuration facultative ; par défaut, resolved_config.json près du modèle.",
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument(
        "--n-envs",
        type=int,
        help="Évaluer ce nombre de parties en parallèle (défaut : evaluation.n_envs).",
    )
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument(
        "--stochastic", action="store_true", help="Échantillonner les actions pendant l'évaluation."
    )
    parser.add_argument(
        "--start-states",
        choices=("standard", "configured", "long"),
        default="standard",
        help=(
            "Utiliser les départs courts classiques ou la distribution définie dans "
            "la configuration."
        ),
    )
    parser.add_argument(
        "--output", type=Path, help="Chemin facultatif du fichier JSON de résultats."
    )
    parser.add_argument(
        "--protocol",
        choices=PROTOCOLS,
        default=FIXED_HORIZON,
        help="Règles d'évaluation ; l'horizon fixe borne chaque partie (défaut).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=10_000,
        help="Budget d'actions utilisé par le protocole à horizon fixe.",
    )
    add_map_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = args.config or _find_run_config(args.model)
    if config_path is None:
        config = load_config(Path("configs/ppo.toml"))
        print("Configuration du run introuvable ; utilisation de configs/ppo.toml.")
    elif config_path.suffix == ".json":
        config = load_resolved_config(config_path)
        print(f"Configuration du modèle : {config_path}")
    else:
        config = load_config(config_path)
    resolved_config = apply_map_arguments(config, args)
    evaluation_config, max_episode_steps = prepare_evaluation(
        resolved_config,
        protocol=args.protocol,
        horizon=args.horizon,
    )
    metrics = evaluate(
        args.model,
        evaluation_config,
        args.episodes,
        args.seed,
        deterministic=not args.stochastic,
        start_state_mode=args.start_states,
        n_envs=args.n_envs or resolved_config.evaluation.n_envs,
        max_episode_steps=max_episode_steps,
        protocol=args.protocol,
    )
    rendered = json.dumps(metrics, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


def _find_run_config(model_path: Path) -> Path | None:
    """Find the closest resolved configuration above a saved model."""
    resolved_model = model_path.resolve()
    for directory in (resolved_model.parent, *resolved_model.parents):
        candidate = directory / "resolved_config.json"
        if candidate.exists():
            return candidate
    return None


if __name__ == "__main__":
    main()
