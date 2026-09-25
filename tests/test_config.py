import json
from dataclasses import replace
from pathlib import Path

import pytest

from snake_rl.config import (
    EnvConfig,
    RewardConfig,
    StartStateConfig,
    TrainingConfig,
    evaluation_start_states,
    load_config,
    load_resolved_config,
)

ROOT = Path(__file__).parents[1]


def test_all_maintained_configs_load():
    for path in (ROOT / "configs").glob("*.toml"):
        config = load_config(path)
        assert not config.start_states.enabled
        assert config.training.n_envs * config.training.n_steps % config.training.batch_size == 0


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"width": 4}, "au moins 5"),
        ({"width": 10, "height": 13}, "couvrir"),
        ({"grid_size": 10}, "couvrir"),
        ({"loop_max_state_visits": 1}, "au moins égal à 2"),
    ],
)
def test_invalid_environment(kwargs, message):
    with pytest.raises(ValueError, match=message):
        EnvConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 7},
        {"hidden_size": 0},
        {"hidden_size": 1},
        {"env_backend": "other"},
        {"target_kl": 0},
        {"learning_rate": float("nan")},
        {"gamma": 0},
        {"clip_range": 2},
        {"n_epochs": 0},
        {"final_learning_rate": None},
        {"learning_rate_schedule": "constant"},
        {"ent_coef": -1},
    ],
)
def test_invalid_training_config(kwargs):
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)


def test_normalized_distance():
    reward = RewardConfig(distance_beta=0.3)
    assert reward.distance_coefficient(20, 20) == pytest.approx(0.3 / 38)
    assert reward.distance_coefficient(8, 8) == pytest.approx(0.3 / 14)
    for value in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            RewardConfig(distance_beta=value)


def test_structured_starts_are_evaluation_only():
    config = load_config(ROOT / "configs/ppo.toml")
    with pytest.raises(ValueError, match="réservés à l'évaluation"):
        replace(config, start_states=replace(config.start_states, enabled=True))
    assert not evaluation_start_states(config.start_states, "standard").enabled
    assert evaluation_start_states(config.start_states, "configured").enabled
    long = evaluation_start_states(config.start_states, "long")
    assert long.enabled and long.standard_probability == 0
    with pytest.raises(ValueError):
        evaluation_start_states(config.start_states, "invalid")
    with pytest.raises(ValueError):
        StartStateConfig(length_distribution="other")
    with pytest.raises(ValueError):
        StartStateConfig(min_length_fraction=0.8, max_length_fraction=0.2)


def test_resolved_config_round_trip(tmp_path):
    config = load_config(ROOT / "configs/ppo.toml")
    assert json.loads(json.dumps(config.to_dict())) == config.to_dict()
    path = tmp_path / "resolved_config.json"
    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    assert load_resolved_config(path) == config


def test_loader_rejects_obsolete_keys_and_missing_sections(tmp_path):
    from snake_rl.config import experiment_config_from_dict

    raw = load_config(ROOT / "configs/ppo.toml").to_dict()
    for section, key, value in [
        ("training", "recurrent_type", "none"),
        ("environment", "observation_frame", "egocentric"),
        ("rewards", "safety", 0),
    ]:
        changed = {**raw, section: {**raw[section], key: value}}
        with pytest.raises(ValueError, match="Clés inconnues"):
            experiment_config_from_dict(changed)
    with pytest.raises(ValueError, match="Sections de configuration inconnues"):
        experiment_config_from_dict({**raw, "restart_curriculum": {}})
    del raw["training"]
    with pytest.raises(ValueError, match="Sections de configuration manquantes"):
        experiment_config_from_dict(raw)
