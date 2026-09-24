from pathlib import Path

import pytest

from snake_rl.benchmark import main, run_benchmark
from snake_rl.config import load_config


def test_same_protocol_for_ppo_and_solver(monkeypatch):
    calls = []

    def ppo(model, config, episodes, seed, deterministic, **kwargs):
        calls.append((config, episodes, seed, kwargs))
        assert deterministic
        return {"score_mean": 0, "termination_counts": {"wall": episodes}}

    def solver(policy, config, *, episodes, seed, **kwargs):
        calls.append((config, episodes, seed, kwargs))
        return {"score_mean": 1, "termination_counts": {"win": episodes}}

    monkeypatch.setattr("snake_rl.benchmark.evaluate", ppo)
    monkeypatch.setattr("snake_rl.benchmark.evaluate_baseline", solver)
    result = run_benchmark(
        load_config("configs/smoke.toml"),
        ppo_model=Path("model.zip"),
        episodes=2,
        seed=42,
        horizon=100,
    )
    assert set(result["agents"]) == {"ppo", "hamiltonian-cycle"}
    assert calls[0][:3] == calls[1][:3]
    assert all(c[0].environment.loop_max_state_visits is None for c in calls)
    assert all(c[3]["max_episode_steps"] == 100 for c in calls)
    assert "provenance" not in result


def test_solver_only_and_invalid_parameters():
    config = load_config("configs/smoke.toml")
    result = run_benchmark(config, ppo_model=None, episodes=2, seed=1, horizon=4)
    assert result["agents"]["hamiltonian-cycle"]["episode_lengths"] == [4, 4]
    for options in ({"episodes": 0}, {"n_envs": 0}, {"policies": ("unknown",)}):
        with pytest.raises(ValueError):
            run_benchmark(config, **({"ppo_model": None, "episodes": 1, "seed": 1} | options))


def test_cli_output(tmp_path, monkeypatch):
    output = tmp_path / "benchmark.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "snake-benchmark",
            "--config",
            "configs/smoke.toml",
            "--episodes",
            "1",
            "--horizon",
            "4",
            "--output",
            str(output),
        ],
    )
    main()
    assert output.is_file()
