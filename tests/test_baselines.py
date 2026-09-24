import json
from dataclasses import replace

import pytest

from snake_rl.baselines import (
    HamiltonianCyclePolicy,
    HamiltonianPolicy,
    evaluate_baseline,
    main,
)
from snake_rl.config import (
    EnvConfig,
    RewardConfig,
    StartStateConfig,
    evaluation_start_states,
    load_config,
)
from snake_rl.env import SnakeEnv


def test_baseline_cli_records_resolved_and_effective_configs(monkeypatch, capsys) -> None:
    def fake_evaluate(*args, **kwargs):
        assert args[1].environment.loop_max_state_visits is None
        return {"score_mean": 1.0}

    monkeypatch.setattr("snake_rl.baselines.evaluate_baseline", fake_evaluate)
    monkeypatch.setattr(
        "sys.argv",
        [
            "snake-baseline-evaluate",
            "--policy",
            "hamiltonian",
            "--config",
            "configs/ppo.toml",
            "--episodes",
            "1",
            "--horizon",
            "1",
        ],
    )

    main()

    result = json.loads(capsys.readouterr().out)
    assert result["config"]["environment"]["loop_max_state_visits"] is None
    assert "provenance" not in result


def test_hamiltonian_policy_requires_an_even_board_dimension() -> None:
    env = SnakeEnv(
        EnvConfig(width=9, height=9, grid_size=9),
        RewardConfig(),
    )
    env.reset(seed=1)

    with pytest.raises(ValueError, match="dimension paire"):
        HamiltonianPolicy().action(env)


def test_hamiltonian_policy_preserves_order_without_collisions() -> None:
    config = load_config("configs/ppo.toml")
    starts = StartStateConfig(
        enabled=True,
        standard_probability=0.5,
        min_length_fraction=0.25,
        max_length_fraction=0.65,
        length_distribution="normal",
        mean_length_fraction=0.45,
        std_length_fraction=0.08,
    )
    env = SnakeEnv(config.environment, config.rewards, starts)
    policy = HamiltonianPolicy()

    for seed in range(4):
        env.reset(seed=seed)
        terminated = truncated = False
        info = env._info()
        for _ in range(500):
            if terminated or truncated:
                break
            action = policy.action(env)
            dx, dy = env.direction_for_action(action)
            head_x, head_y = env.snake[0]
            assert env.collision_type((head_x + dx, head_y + dy)) is None
            _, _, terminated, truncated, info = env.step(action)
        assert info["collision"] is None


@pytest.mark.parametrize(
    ("seed", "configured"),
    [(60_009, False), (60_013, True)],
)
def test_hamiltonian_policy_regression_traps_stay_collision_free(
    seed: int, configured: bool
) -> None:
    config = load_config("configs/ppo.toml")
    config = replace(config, environment=replace(config.environment, loop_max_state_visits=None))
    mode = "configured" if configured else "standard"
    starts = evaluation_start_states(config.start_states, mode)
    env = SnakeEnv(config.environment, config.rewards, starts)
    policy = HamiltonianPolicy()
    env.reset(seed=seed)
    terminated = truncated = False
    info = env._info()

    for _ in range(500):
        _, _, terminated, truncated, info = env.step(policy.action(env))
        if terminated or truncated:
            break

    assert info["collision"] is None


def test_baseline_evaluation_uses_common_metric_schema() -> None:
    config = load_config("configs/ppo.toml")
    config = replace(
        config,
        environment=EnvConfig(
            width=6,
            height=6,
            grid_size=6,
        ),
    )
    metrics = evaluate_baseline(
        HamiltonianPolicy(),
        config,
        episodes=3,
        seed=100,
        start_state_mode="standard",
    )

    assert metrics["episodes"] == 3
    assert metrics["groups"]["standard"]["episodes"] == 3
    assert metrics["groups"]["long"] is None
    assert len(metrics["score_mean_ci95"]) == 2


def test_hamiltonian_positive_control_solves_exact_75_percent_start() -> None:
    config = load_config("configs/ppo_12.toml")
    config = replace(
        config,
        environment=replace(config.environment, loop_max_state_visits=None),
        start_states=StartStateConfig(
            standard_probability=0.0, min_length_fraction=0.75, max_length_fraction=0.75
        ),
    )

    metrics = evaluate_baseline(
        HamiltonianCyclePolicy(),
        config,
        episodes=30,
        seed=63000,
        start_state_mode="configured",
        max_episode_steps=10000,
        protocol="fixed-horizon",
    )

    assert metrics["initial_lengths"] == [108] * 30
    assert metrics["scores"] == [36] * 30
    assert metrics["termination_counts"] == {"win": 30}
