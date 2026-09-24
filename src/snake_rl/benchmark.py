"""Compare PPO et solveurs Snake avec les mêmes graines de parties."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from snake_rl.baselines import HamiltonianCyclePolicy, HamiltonianPolicy, evaluate_baseline
from snake_rl.cli import add_map_arguments, apply_map_arguments
from snake_rl.config import ExperimentConfig, load_config, load_resolved_config
from snake_rl.evaluate import _find_run_config, evaluate
from snake_rl.evaluation_protocol import FIXED_HORIZON, prepare_evaluation
from snake_rl.expert import SafePlanner

POLICIES = {
    "hamiltonian-cycle": HamiltonianCyclePolicy,
    "hamiltonian": HamiltonianPolicy,
    "planner": SafePlanner,
}


def run_benchmark(
    config: ExperimentConfig,
    *,
    ppo_model: Path | None,
    episodes: int,
    seed: int,
    n_envs: int = 32,
    horizon: int = 10_000,
    policies: tuple[str, ...] = ("hamiltonian-cycle",),
) -> dict:
    """Évaluation fraîche, sans cache ; les checkpoints sont lus seulement."""
    if episodes < 1 or n_envs < 1:
        raise ValueError("episodes et n_envs doivent être positifs.")
    if not policies or any(name not in POLICIES for name in policies):
        raise ValueError("Choisissez au moins un solveur pris en charge.")
    config, limit = prepare_evaluation(config, protocol=FIXED_HORIZON, horizon=horizon)
    agents = {}
    if ppo_model is not None:
        started = perf_counter()
        agents["ppo"] = evaluate(
            ppo_model,
            config,
            episodes,
            seed,
            True,
            n_envs=n_envs,
            max_episode_steps=limit,
            protocol=FIXED_HORIZON,
        )
        agents["ppo"]["elapsed_seconds"] = perf_counter() - started
    for name in policies:
        started = perf_counter()
        agents[name] = evaluate_baseline(
            POLICIES[name](),
            config,
            episodes=episodes,
            seed=seed,
            start_state_mode="standard",
            max_episode_steps=limit,
            protocol=FIXED_HORIZON,
        )
        agents[name]["elapsed_seconds"] = perf_counter() - started
    return {
        "config": config.to_dict(),
        "episodes": episodes,
        "seed": seed,
        "horizon": horizon,
        "agents": agents,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--ppo-model", type=Path)
    parser.add_argument("--policy", choices=tuple(POLICIES), default="hamiltonian-cycle")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--n-envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=60_000)
    parser.add_argument("--horizon", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=Path("reports/benchmark.json"))
    add_map_arguments(parser)
    args = parser.parse_args()
    path = args.config or (_find_run_config(args.ppo_model) if args.ppo_model else None)
    if args.ppo_model and path is None:
        parser.error("Configuration du run introuvable ; indiquez explicitement --config.")
    path = path or Path("configs/ppo.toml")
    config = load_resolved_config(path) if path.suffix == ".json" else load_config(path)
    result = run_benchmark(
        apply_map_arguments(config, args),
        ppo_model=args.ppo_model,
        episodes=args.episodes,
        seed=args.seed,
        n_envs=args.n_envs,
        horizon=args.horizon,
        policies=(args.policy,),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for name, metrics in result["agents"].items():
        wins = metrics["termination_counts"].get("win", 0)
        print(f"{name} : {metrics['score_mean']:.2f} pommes, {wins}/{args.episodes} victoires")
    print(f"Résultats : {args.output}")


if __name__ == "__main__":
    main()
