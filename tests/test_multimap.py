import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from stable_baselines3.common.logger import configure

from snake_rl.config import load_config
from snake_rl.score_evaluation import ScoreEvalCallback
from snake_rl.training.multimap import MapExposureCallback, make_multimap_env
from snake_rl.training.trajectory_restart import TrajectoryRestartCallback


def small_config(backend="dummy", restart=True):
    c = load_config("configs/multimap.toml")
    return replace(
        c,
        training=replace(c.training, n_envs=3, n_steps=4, batch_size=6, env_backend=backend),
        evaluation=replace(c.evaluation, n_envs=3, episodes=2, horizon=4),
        trajectory_restart=replace(c.trajectory_restart, enabled=restart),
    )


@pytest.mark.parametrize("backend", ["dummy", "subprocess"])
def test_workers_keep_size_and_exposure_even_when_episodes_end(backend, tmp_path):
    config = small_config(backend)
    vec, assignment = make_multimap_env(config, tmp_path)
    evaluation, eval_assignment = make_multimap_env(config, tmp_path, evaluation=True)
    try:
        assert assignment == eval_assignment == [8, 10, 12]
        observations = vec.reset()
        assert observations["grid"].shape == (3, 7, 12, 12)
        np.testing.assert_allclose(observations["vector"][:, -1], [3 / 64, 3 / 100, 3 / 144])
        for _ in range(20):
            observations, _, _, _ = vec.step(np.ones(3, dtype=int))
        assert [c.width for c in vec.get_attr("config")] == [8, 10, 12]
        vec.env_method("set_trajectory_stage", 1, indices=[0])
        assert [s["stage"] for s in vec.env_method("trajectory_status")] == [1, 0, 0]
        assert all(not s.enabled for s in evaluation.get_attr("start_states"))
        assert all(c.loop_max_state_visits is None for c in evaluation.get_attr("config"))
        evaluation.reset()
        for _ in range(4):
            _, _, dones, _ = evaluation.step(np.ones(3, dtype=int))
        assert all(dones)
    finally:
        vec.close()
        evaluation.close()


def test_promotion_callback_ignores_other_sizes_and_exposure_is_exact(tmp_path):
    config = small_config()
    vec, assignment = make_multimap_env(config, tmp_path)
    try:
        model = SimpleNamespace(get_env=lambda: vec, logger=configure(str(tmp_path), []))
        callback = TrajectoryRestartCallback(
            config.trajectory_restart, tmp_path / "gate.json", indices=[1], prefix="trajectory/10"
        )
        callback.model = model
        callback.locals = {
            "dones": [True, True, True],
            "infos": [
                {"trajectory_stage": 0, "trajectory_eligible": True, "trajectory_success": success}
                for success in [True, False, True]
            ],
        }
        callback._on_step()
        assert list(callback.gate.outcomes) == [False]
        callback._on_rollout_end()
        assert "trajectory/10/stage" in model.logger.name_to_value
        exposure = MapExposureCallback(assignment, tmp_path / "exposure.json")
        exposure.model = model
        exposure.num_timesteps = 90
        exposure._on_rollout_end()
        data = json.loads((tmp_path / "exposure.json").read_text())
        assert data["transitions_per_map"] == {"8": 30, "10": 30, "12": 30}
    finally:
        vec.close()


class EvalWorkers:
    num_envs = 3

    def seed(self, seed):
        pass

    def reset(self):
        self.steps = 0
        return np.zeros((3, 1))

    def step(self, action):
        self.steps += 1
        return (
            np.zeros((3, 1)),
            np.zeros(3),
            np.array([True, self.steps % 2 == 0, self.steps % 3 == 0]),
            [
                {"score": 61, "won": True},
                {"score": 0, "won": False},
                {"score": 0, "won": False},
            ],
        )


def test_normalized_selection_and_per_map_quotas(tmp_path):
    saved = []
    logger = configure(str(tmp_path), [])
    callback = ScoreEvalCallback(
        EvalWorkers(), 1, 6, tmp_path, True, 100, 0, 123, map_sizes=[8, 10, 12]
    )
    callback.model = SimpleNamespace(
        predict=lambda obs, **k: (np.zeros(3, dtype=int), None),
        save=lambda p: saved.append(p),
        logger=logger,
    )
    result = callback._evaluate()
    assert [result[f"map_{s}_episodes"] for s in [8, 10, 12]] == [2, 2, 2]
    assert result["normalized_mean_score"] == pytest.approx(1 / 3)
    assert result["map_8_win_rate"] == 1
    assert result["worst_map_normalized_score"] == 0
    callback._evaluate_and_record(count_plateau=False)
    # More raw apples, but worse normalized result: do NOT select this model.
    callback._evaluate = lambda: {"mean_score": 100 / 3, "normalized_mean_score": 100 / 141 / 3}
    callback._evaluate_and_record(count_plateau=True)
    assert len(saved) == 1
    payload = json.loads((tmp_path / "evaluation_summary.json").read_text())
    assert payload["selection_metric"] == "normalized_mean_score"


def test_configuration_requires_balanced_workers_and_valid_sizes():
    config = load_config("configs/multimap.toml")
    assert config.training.total_timesteps == 9_000_000
    with pytest.raises(ValueError, match="divisible"):
        replace(config, training=replace(config.training, n_envs=64, batch_size=2048))
    assert replace(config, map_sizes=(8, 9, 12)).map_sizes == (8, 9, 12)


@pytest.mark.parametrize("restart", [False, True])
def test_training_wiring_no_optimization(monkeypatch, tmp_path, restart):
    from snake_rl.train import train

    def no_learning(model, *, callback, **kwargs):
        callbacks = callback.callbacks
        gates = [c for c in callbacks if isinstance(c, TrajectoryRestartCallback)]
        assert len(gates) == (3 if restart else 0)
        assert [g.indices for g in gates] == ([[0], [1], [2]] if restart else [])
        evaluator = next(c for c in callbacks if isinstance(c, ScoreEvalCallback))
        assert evaluator.map_sizes == [8, 10, 12]
        assert evaluator.n_eval_episodes == 6
        assert evaluator.selection_metric == "normalized_mean_score"
        return model

    monkeypatch.setattr("stable_baselines3.PPO.learn", no_learning)
    monkeypatch.setattr("snake_rl.report.generate_reports", lambda *a, **k: None)
    train(small_config(restart=restart), tmp_path, run_name="wiring")
