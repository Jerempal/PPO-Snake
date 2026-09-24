import pytest

from snake_rl.config import load_config
from snake_rl.evaluation_protocol import prepare_evaluation


def test_fixed_horizon_disables_training_termination_rules() -> None:
    config = load_config("configs/ppo.toml")

    prepared, horizon = prepare_evaluation(config, protocol="fixed-horizon", horizon=10_000)

    assert horizon == 10_000
    assert prepared.environment.loop_max_state_visits is None


def test_evaluation_protocol_validates_inputs() -> None:
    config = load_config("configs/ppo.toml")

    with pytest.raises(ValueError, match="inconnu"):
        prepare_evaluation(config, protocol="unknown", horizon=10_000)
    with pytest.raises(ValueError, match="positif"):
        prepare_evaluation(config, protocol="fixed-horizon", horizon=0)
