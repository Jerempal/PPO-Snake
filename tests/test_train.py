import json
from dataclasses import replace
from pathlib import Path

import pytest

from snake_rl.config import TrainingConfig, load_config, load_resolved_config
from snake_rl.policy import CompactHybridFeaturesExtractor
from snake_rl.train import (
    _finalize_run_manifest,
    _learning_rate,
    _write_run_manifest,
    main,
    policy_spec,
    train,
)


def test_linear_learning_rate_schedule_uses_progress_remaining() -> None:
    config = load_config("configs/ppo.toml")
    schedule = _learning_rate(config)

    assert callable(schedule)
    assert schedule(1.0) == pytest.approx(1e-3)
    assert schedule(0.5) == pytest.approx(5.5e-4)
    assert schedule(0.0) == pytest.approx(1e-4)


def test_constant_learning_rate_remains_a_float() -> None:
    config = load_config("configs/ppo.toml")
    config = replace(
        config,
        training=TrainingConfig(
            learning_rate=3e-4, learning_rate_schedule="constant", final_learning_rate=None
        ),
    )

    assert _learning_rate(config) == pytest.approx(3e-4)


def test_feed_forward_policy_uses_compact_hybrid_extractor() -> None:
    config = load_config("configs/ppo.toml")

    policy_name, kwargs = policy_spec(config)

    assert policy_name == "MultiInputPolicy"
    assert kwargs["features_extractor_class"] is CompactHybridFeaturesExtractor


def test_training_cli_applies_normalized_distance_beta(monkeypatch) -> None:
    captured = {}

    def fake_train(config, *args, **kwargs):
        del args, kwargs
        captured["config"] = config

    monkeypatch.setattr("snake_rl.train.train", fake_train)
    monkeypatch.setattr(
        "sys.argv",
        [
            "snake-train",
            "--config",
            "configs/ppo.toml",
            "--food-reward",
            "1",
            "--distance-beta",
            "0.3",
        ],
    )

    main()

    assert captured["config"].rewards.food == 1.0
    assert captured["config"].rewards.distance_beta == 0.3


def test_training_cli_applies_epoch_override(monkeypatch) -> None:
    captured = {}

    def fake_train(config, *args, **kwargs):
        del args, kwargs
        captured["config"] = config

    monkeypatch.setattr("snake_rl.train.train", fake_train)
    monkeypatch.setattr(
        "sys.argv",
        ["snake-train", "--config", "configs/ppo.toml", "--n-epochs", "6"],
    )

    main()

    assert captured["config"].training.n_epochs == 6


def test_run_config_and_minimal_status_do_not_touch_checkpoints(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    parent = tmp_path / "parent.zip"
    parent.write_bytes(b"parent")
    initializer = tmp_path / "initializer.zip"
    initializer.write_bytes(b"initializer")
    config = load_config("configs/ppo.toml")

    _write_run_manifest(run_dir, config)

    assert load_resolved_config(run_dir / "resolved_config.json") == config
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert set(metadata) == {"status", "created_at"}
    assert metadata["status"] == "running"
    assert "provenance" not in metadata

    best_model = run_dir / "best_model" / "best_model.zip"
    best_model.parent.mkdir()
    best_model.write_bytes(b"best")
    final_model = run_dir / "final_model.zip"
    final_model.write_bytes(b"final")
    initialized_model = run_dir / "initialized_model.zip"
    initialized_model.write_bytes(b"initialized")
    _finalize_run_manifest(run_dir, status="complete")

    assert not (run_dir / "artifacts.json").exists()
    assert best_model.read_bytes() == b"best"
    assert final_model.read_bytes() == b"final"
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "complete"


def test_training_setup_failure_finalizes_manifest(monkeypatch, tmp_path: Path) -> None:
    def fail_make_vec_env(*args, **kwargs):
        raise RuntimeError("setup failed")

    monkeypatch.setattr("stable_baselines3.common.env_util.make_vec_env", fail_make_vec_env)

    with pytest.raises(RuntimeError, match="setup failed"):
        train(load_config("configs/smoke.toml"), tmp_path, run_name="failed-setup")

    run_dir = tmp_path / "failed-setup"
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "failed"
    assert not (run_dir / "artifacts.json").exists()


def test_training_rejects_missing_parent_before_creating_run(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Parent model"):
        train(
            load_config("configs/smoke.toml"),
            tmp_path,
            run_name="missing-parent",
            resume_from=tmp_path / "missing.zip",
        )

    assert not (tmp_path / "missing-parent").exists()


def test_training_rejects_incompatible_initialization_modes(tmp_path: Path) -> None:
    model = tmp_path / "model.zip"
    model.write_bytes(b"model")
    config = load_config("configs/smoke.toml")

    with pytest.raises(ValueError, match="exclusifs"):
        train(
            config,
            tmp_path,
            run_name="conflicting-initialization",
            resume_from=model,
            initialize_policy_from=model,
        )
    with pytest.raises(ValueError, match="nécessite"):
        train(config, tmp_path, run_name="missing-initializer", initialize_only=True)

    assert not (tmp_path / "conflicting-initialization").exists()
    assert not (tmp_path / "missing-initializer").exists()


def test_training_cli_forwards_cross_size_initialization(monkeypatch) -> None:
    captured = {}

    def fake_train(config, *args, **kwargs):
        del config, args
        captured.update(kwargs)

    monkeypatch.setattr("snake_rl.train.train", fake_train)
    monkeypatch.setattr(
        "sys.argv",
        [
            "snake-train",
            "--config",
            "configs/ppo.toml",
            "--initialize-policy-from",
            "source.zip",
            "--initialize-only",
        ],
    )

    main()

    assert captured["initialize_policy_from"] == Path("source.zip")
    assert captured["initialize_only"] is True


def test_training_cli_can_request_post_training_evaluations(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("snake_rl.train.train", lambda *a, **k: tmp_path)
    monkeypatch.setattr("snake_rl.report.generate_reports", lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr(
        "sys.argv",
        ["snake-train", "--config", "configs/smoke.toml", "--report-evaluate"],
    )
    main()
    assert calls == [(([tmp_path],), {"evaluate_models": True})]


@pytest.mark.parametrize(
    "flag, enabled",
    [
        ("--trajectory-restart", True),
        ("--no-trajectory-restart", False),
    ],
)
def test_training_cli_can_toggle_trajectory_restart(monkeypatch, flag, enabled):
    captured = []
    monkeypatch.setattr("snake_rl.train.train", lambda config, *a, **k: captured.append(config))
    monkeypatch.setattr("sys.argv", ["snake-train", "--config", "configs/ppo.toml", flag])
    main()
    assert captured[0].trajectory_restart.enabled is enabled


@pytest.mark.parametrize("enabled", [False, True])
def test_training_wires_curriculum_only_for_training_without_learning(
    monkeypatch, tmp_path, enabled
):
    from snake_rl.training.trajectory_restart import (
        TrajectoryRestartCallback,
        TrajectoryRestartWrapper,
    )

    config = load_config("configs/ppo.toml")
    config = replace(
        config,
        training=replace(config.training, n_envs=1, env_backend="dummy"),
        evaluation=replace(config.evaluation, n_envs=1, episodes=1),
        trajectory_restart=replace(config.trajectory_restart, enabled=enabled),
    )
    checked = []

    def no_learning(model, *, callback, **kwargs):
        callbacks = callback.callbacks
        assert isinstance(model.get_env().envs[0], TrajectoryRestartWrapper) is enabled
        assert any(isinstance(c, TrajectoryRestartCallback) for c in callbacks) is enabled
        evaluation = next(c for c in callbacks if type(c).__name__ == "ScoreEvalCallback")
        assert not isinstance(evaluation.eval_env.envs[0], TrajectoryRestartWrapper)
        assert not evaluation.eval_env.envs[0].unwrapped.start_states.enabled
        checked.append(True)
        return model

    monkeypatch.setattr("stable_baselines3.PPO.learn", no_learning)
    monkeypatch.setattr("snake_rl.report.generate_reports", lambda *a, **k: None)
    train(config, tmp_path, run_name="wiring-only")
    assert checked == [True]
