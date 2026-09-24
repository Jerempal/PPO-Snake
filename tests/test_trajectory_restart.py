from dataclasses import replace
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from snake_rl.config import EnvConfig, TrajectoryRestartConfig, load_config
from snake_rl.env import RelativeAction, SnakeEnv
from snake_rl.training.trajectory_restart import (
    PromotionGate,
    TrajectoryRestartCallback,
    TrajectoryRestartWrapper,
    TrajectoryStart,
)


def make_restart_env(*, enabled=True, reservoir_size=2):
    return TrajectoryRestartWrapper(
        SnakeEnv(EnvConfig(width=8, height=8, grid_size=12)),
        TrajectoryRestartConfig(
            enabled=enabled,
            fractions=(4 / 64, 5 / 64),
            standard_probability=0.25,
            promotion_window=4,
            reservoir_size=reservoir_size,
        ),
    )


def capture_first_threshold(wrapper):
    wrapper.reset(seed=42)
    raw = wrapper.snake_env
    # A valid controlled apple placement; the state itself is reached by a real step.
    raw.food = (5, 4)
    observation, _, terminated, truncated, _ = wrapper.step(RelativeAction.STRAIGHT)
    assert not terminated and not truncated
    assert len(wrapper.banks[0]) >= 1
    return observation


def test_fraction_thresholds_are_occupancy_not_apples():
    settings = TrajectoryRestartConfig()
    assert settings.lengths(144) == (15, 36, 72, 108)
    assert settings.lengths(64) == (7, 16, 32, 48)
    with pytest.raises(ValueError, match="longueurs distinctes"):
        settings.lengths(25)
    config = load_config("configs/ppo.toml")
    assert not config.trajectory_restart.enabled


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fractions": []},
        {"fractions": [0.5, 0.4]},
        {"fractions": [float("nan")]},
        {"standard_probability": 0},
        {"promotion_window": 0},
        {"reservoir_size": 1.5},
        {"promotion_success_rate": 0},
    ],
)
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        TrajectoryRestartConfig(**kwargs)


def test_disabled_wrapper_preserves_observations_rewards_and_rng():
    plain = SnakeEnv(EnvConfig(width=8, height=8, grid_size=12))
    wrapped = make_restart_env(enabled=False)
    for seed in (1, 2, 3):
        a, ai = plain.reset(seed=seed)
        b, bi = wrapped.reset(seed=seed)
        assert ai == bi
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        for action in [1, 0, 1, 2, 1, 1, 1, 1]:
            a, ar, at, ax, ai = plain.step(action)
            b, br, bt, bx, bi = wrapped.step(action)
            assert (ar, at, ax, ai) == (br, bt, bx, bi)
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])
            if at or ax:
                break
        assert (
            plain.np_random.bit_generator.state == wrapped.snake_env.np_random.bit_generator.state
        )
    assert not any(wrapped.banks)


def test_restart_preserves_observation_but_starts_new_episode():
    wrapper = make_restart_env()
    expected = capture_first_threshold(wrapper)
    state = wrapper.banks[0][0]
    wrapper.snake_env.reset(seed=99)
    rng_before = wrapper.snake_env.np_random.bit_generator.state
    actual, info = state.restart(wrapper.snake_env)
    for key in actual:
        np.testing.assert_array_equal(actual[key], expected[key])
    assert info["score"] == 0
    assert info["length"] == info["initial_length"] == 4
    assert info["start_type"] == "trajectory"
    assert info["steps"] == info["steps_since_food"] == 0
    assert wrapper.snake_env.np_random.bit_generator.state == rng_before
    assert wrapper.snake_env._state_visits == {wrapper.snake_env._state_signature(): 1}
    assert wrapper.observation_space.contains(actual)


def test_capture_and_restore_reject_invalid_states():
    wrapper = make_restart_env()
    capture_first_threshold(wrapper)
    state = wrapper.banks[0][0]
    for bad in (
        replace(state, food=state.snake[0]),
        replace(state, snake=(*state.snake[:-1], state.snake[0])),
        replace(state, direction=(0, -1)),
        replace(state, food=(8, 0)),
        replace(state, snake=((7, 7), *state.snake[1:])),
        replace(state, width=10),
    ):
        with pytest.raises(ValueError):
            bad.restart(wrapper.snake_env)
    wrapper.snake_env._episode_over = True
    with pytest.raises(ValueError):
        TrajectoryStart.capture(wrapper.snake_env)


def test_bounded_banks_mixed_starts_and_fallback():
    wrapper = make_restart_env()
    for _ in range(8):
        capture_first_threshold(wrapper)
    assert len(wrapper.banks[0]) == 2
    wrapper.set_trajectory_stage(1)
    types = set()
    for _ in range(60):
        _, info = wrapper.reset()
        types.add(info["start_type"])
    assert types == {"standard", "trajectory"}
    wrapper.set_trajectory_stage(2)
    for _ in range(10):
        _, info = wrapper.reset()
        assert info["start_type"] == "standard"
        assert not wrapper.eligible
    assert wrapper.fallbacks > 0
    with pytest.raises(ValueError):
        wrapper.set_trajectory_stage(3)


def test_promotion_needs_window_success_and_available_bank():
    gate = PromotionGate(TrajectoryRestartConfig(promotion_window=4, promotion_success_rate=0.5))
    for success in (True, False, True):
        gate.observe(
            {"trajectory_stage": 0, "trajectory_eligible": True, "trajectory_success": success}
        )
    assert not gate.promote(available=True)
    gate.observe({"trajectory_stage": 0, "trajectory_eligible": True, "trajectory_success": False})
    assert not gate.promote(available=False)
    assert gate.promote(available=True)
    assert gate.stage == 1 and not gate.outcomes
    gate.observe({"trajectory_stage": 0, "trajectory_eligible": True, "trajectory_success": True})
    gate.observe({"trajectory_stage": 1, "trajectory_eligible": False, "trajectory_success": True})
    assert not gate.outcomes
    for _ in range(30):
        gate.observe(
            {"trajectory_stage": 1, "trajectory_eligible": True, "trajectory_success": False}
        )
    assert len(gate.outcomes) == 4
    assert not gate.promote(available=True)


def test_stage_change_does_not_change_an_active_episode():
    wrapper = make_restart_env()
    capture_first_threshold(wrapper)
    wrapper.set_trajectory_stage(1)
    for _ in range(10):
        _, _, terminated, truncated, info = wrapper.step(RelativeAction.STRAIGHT)
        if terminated or truncated:
            break
    assert info["trajectory_stage"] == 0
    assert info["trajectory_eligible"] and info["trajectory_success"]


@pytest.mark.parametrize("backend", [DummyVecEnv, SubprocVecEnv])
def test_vectorized_worker_integration_without_training(backend):
    kwargs = {"start_method": "spawn"} if backend is SubprocVecEnv else {}
    vec = backend([partial(make_restart_env), partial(make_restart_env)], **kwargs)
    try:
        vec.seed(42)
        observations = vec.reset()
        assert observations["grid"].shape == (2, 7, 12, 12)
        vec.env_method("set_trajectory_stage", 1)
        assert all(s["stage"] == 1 for s in vec.env_method("trajectory_status"))
        for _ in range(5):
            _, _, dones, infos = vec.step(np.array([1, 1]))
            if any(dones):
                assert "trajectory_stage" in infos[0]
                break
    finally:
        vec.close()


def test_callback_records_scalar_metrics_without_training(tmp_path):
    wrapper = make_restart_env()
    capture_first_threshold(wrapper)
    vec = DummyVecEnv([lambda: wrapper])
    try:
        callback = TrajectoryRestartCallback(wrapper.settings, tmp_path / "restart.json")
        callback.model = SimpleNamespace(get_env=lambda: vec, logger=configure(str(tmp_path), []))
        callback.num_timesteps = 100
        callback.locals = {
            "dones": [True],
            "infos": [
                {
                    "trajectory_stage": 0,
                    "trajectory_eligible": True,
                    "trajectory_success": True,
                }
            ],
        }
        for _ in range(4):
            callback._on_step()
        callback._on_rollout_end()
        assert callback.gate.stage == wrapper.stage == 1
        assert callback.promotions[0]["transitions"] == 100
        assert (tmp_path / "restart.json").is_file()
    finally:
        vec.close()


def test_monitor_resets_reward_and_length_for_restart():
    settings = TrajectoryRestartConfig(enabled=True, fractions=(4 / 64,))
    vec = make_vec_env(
        SnakeEnv,
        n_envs=1,
        seed=42,
        env_kwargs={"env_config": EnvConfig(width=8, height=8, grid_size=12)},
        wrapper_class=TrajectoryRestartWrapper,
        wrapper_kwargs={"config": settings},
    )
    try:
        vec.reset()
        wrapper = vec.envs[0]
        capture_first_threshold(wrapper)
        vec.env_method("set_trajectory_stage", 1)
        for _ in range(50):
            vec.reset()
            if wrapper.snake_env._start_type == "trajectory":
                break
        assert wrapper.snake_env.initial_length == 4
        for steps in range(1, 10):
            _, _, dones, infos = vec.step(np.array([1]))
            if dones[0]:
                assert infos[0]["episode"]["l"] == steps
                assert infos[0]["score"] == infos[0]["length"] - 4
                break
    finally:
        vec.close()
