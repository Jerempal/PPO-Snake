from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from snake_rl.config import load_config
from snake_rl.evaluate import evaluate
from snake_rl.evaluation_protocol import prepare_evaluation


class _StraightModel:
    def predict(self, observations, **kwargs):
        del kwargs
        batch_size = len(observations["grid"])
        return np.ones(batch_size, dtype=np.int64), None


def test_evaluation_records_raw_episodes_and_effective_config(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"checkpoint")
    monkeypatch.setattr(PPO, "load", lambda path: _StraightModel())
    resolved_config = load_config("configs/ppo.toml")
    config, horizon = prepare_evaluation(
        resolved_config,
        protocol="fixed-horizon",
        horizon=5,
    )

    result = evaluate(
        model_path,
        config,
        episodes=2,
        seed=123,
        deterministic=True,
        n_envs=2,
        max_episode_steps=horizon,
        protocol="fixed-horizon",
    )

    assert result["schema_version"] == 2
    assert result["algorithm"] == "PPO"
    assert result["scores"] == [0, 0]
    assert result["episode_lengths"] == [5, 5]
    assert result["termination_reasons"] == ["horizon", "horizon"]
    assert result["config"]["environment"]["loop_max_state_visits"] is None
    assert "command" not in result
    assert "provenance" not in result
    assert "model_artifact" not in result
