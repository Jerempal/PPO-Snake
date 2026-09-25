"""Compare le cycle strict aux évaluations PPO existantes (aucun entraînement)."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from snake_rl.baselines import HamiltonianCyclePolicy, evaluate_baseline
from snake_rl.config import load_config
from snake_rl.evaluation_protocol import FIXED_HORIZON, prepare_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppo-reports", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/hamiltonian_comparison.json"))
    args = parser.parse_args()
    config, horizon = prepare_evaluation(
        load_config("configs/multimap.toml"), protocol=FIXED_HORIZON, horizon=10_000
    )
    result = {
        "protocol": {
            "episode_seeds": [93000, 93099],
            "horizon": horizon,
            "initial_length": 3,
            "ppo_training_seeds": [42, 43, 44],
            "ppo_checkpoint": "best_model",
            "note": "PPO : 3 modèles x 100 parties ; cycle strict : 100 parties par taille. "
            "Les mêmes graines ne garantissent pas les mêmes pommes après divergence des corps.",
        },
        "maps": {},
    }
    for size in (8, 10, 12):
        entry = {}
        for method in ("direct", "trajectory"):
            scores, lengths, wins = [], [], 0
            for seed in (42, 43, 44):
                path = args.ppo_reports / f"multimap-{method}-{seed}-map{size}.json"
                report = json.loads(path.read_text(encoding="utf-8"))
                assert report["episodes"] == 100 and report["seed"] == 93000
                assert report["max_episode_steps"] == horizon
                assert report["config"]["environment"]["width"] == size
                assert report["config"]["environment"]["height"] == size
                assert report["config"]["environment"]["loop_max_state_visits"] is None
                assert report["initial_lengths"] == [3] * 100
                assert report["deterministic"]
                scores.extend(report["scores"])
                lengths.extend(report["episode_lengths"])
                wins += report["termination_reasons"].count("win")
            entry[method] = {
                "episodes": len(scores),
                "mean_apples": float(np.mean(scores)),
                "wins": wins,
                "win_rate": wins / len(scores),
                "apples_per_1000_actions": 1000 * sum(scores) / sum(lengths),
            }
        print(f"Cycle strict : {size}x{size}, 100 parties...", flush=True)
        evaluation = evaluate_baseline(
            HamiltonianCyclePolicy(),
            replace(config, environment=replace(config.environment, width=size, height=size)),
            episodes=100,
            seed=93000,
            start_state_mode="standard",
            max_episode_steps=horizon,
            protocol=FIXED_HORIZON,
        )
        entry["hamiltonian_cycle"] = {
            "episodes": 100,
            "mean_apples": evaluation["score_mean"],
            "wins": evaluation["termination_counts"].get("win", 0),
            "win_rate": evaluation["termination_counts"].get("win", 0) / 100,
            "apples_per_1000_actions": evaluation["apples_per_1000_steps"],
            "scores": evaluation["scores"],
            "episode_lengths": evaluation["episode_lengths"],
            "termination_reasons": evaluation["termination_reasons"],
        }
        result["maps"][str(size)] = entry
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(entry["hamiltonian_cycle"]["win_rate"], flush=True)


if __name__ == "__main__":
    main()
