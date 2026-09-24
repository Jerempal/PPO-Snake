import argparse
from pathlib import Path

import pytest

from snake_rl.cli import add_map_arguments, apply_map_arguments
from snake_rl.config import load_config, with_environment_overrides

ROOT = Path(__file__).parents[1]


def reference_config():
    return load_config(ROOT / "configs" / "ppo.toml")


def test_square_map_override_preserves_loop_guard() -> None:
    config = with_environment_overrides(reference_config(), map_size=20)

    assert config.environment.width == 20
    assert config.environment.height == 20
    assert config.environment.loop_max_state_visits == 3


def test_invalid_cli_dimension_is_not_silently_ignored() -> None:
    with pytest.raises(ValueError, match="au moins 5"):
        with_environment_overrides(reference_config(), map_size=0)


def test_cli_arguments_apply_to_config() -> None:
    parser = argparse.ArgumentParser()
    add_map_arguments(parser)

    args = parser.parse_args(["--map-size", "18"])
    config = apply_map_arguments(reference_config(), args)

    assert config.environment.width == 18
    assert config.environment.height == 18
    assert config.environment.grid_size == 20
    assert config.environment.loop_max_state_visits == 3


def test_grid_canvas_override_supports_larger_maps() -> None:
    config = with_environment_overrides(reference_config(), map_size=24, grid_size=24)

    assert config.environment.grid_size == 24


def test_smaller_map_preserves_trained_observation_shape() -> None:
    training = reference_config()
    smaller = with_environment_overrides(training, map_size=12)

    assert smaller.environment.grid_size == 20
