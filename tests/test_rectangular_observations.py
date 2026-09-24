import argparse

import numpy as np
import pytest

from snake_rl.cli import add_map_arguments, apply_map_arguments
from snake_rl.config import EnvConfig, load_config
from snake_rl.env import SnakeEnv


@pytest.mark.parametrize("width,height", [(9, 9), (11, 11), (8, 12), (12, 8), (9, 11)])
@pytest.mark.parametrize("direction", [(0, -1), (1, 0), (0, 1), (-1, 0)])
def test_observations_fit_and_rotate(width, height, direction):
    env = SnakeEnv(EnvConfig(width=width, height=height, grid_size=12))
    env.reset(seed=42)
    env.direction = direction
    grid = env._grid_observation()
    vector = env._vector_observation()
    assert grid.shape == (7, 12, 12)
    assert np.count_nonzero(grid[4] == 0) == width * height
    assert grid[0].sum() == 1
    assert grid[3].sum() == 1
    for channel in (0, 1, 2, 3):
        assert not np.any(grid[channel] * grid[4])
    assert np.all((vector[:4] >= 0) & (vector[:4] <= 1))
    turns = env._egocentric_quarter_turns()
    expected = np.zeros((height, width))
    x, y = env.snake[0]
    expected[y, x] = 1
    expected = np.rot90(expected, turns)
    ey, ex = np.argwhere(expected == 1)[0]
    np.testing.assert_allclose(
        vector[:2], [ex / (expected.shape[1] - 1), ey / (expected.shape[0] - 1)]
    )
    env.close()


def test_rectangle_cli_and_conflicts():
    parser = argparse.ArgumentParser()
    add_map_arguments(parser)
    base = load_config("configs/multimap.toml")
    config = apply_map_arguments(base, parser.parse_args(["--width", "9", "--height", "11"]))
    assert (config.environment.width, config.environment.height) == (9, 11)
    with pytest.raises(ValueError, match="Utilisez"):
        apply_map_arguments(base, parser.parse_args(["--map-size", "9", "--width", "11"]))


@pytest.mark.parametrize("width,height", [(8, 12), (12, 8), (9, 11)])
def test_rectangular_episodes(width, height):
    env = SnakeEnv(EnvConfig(width=width, height=height, grid_size=12))
    observation, _ = env.reset(seed=42)
    for step in range(100):
        assert env.observation_space.contains(observation)
        observation, _, terminated, truncated, _ = env.step(step % 3)
        if terminated or truncated:
            observation, _ = env.reset(seed=42 + step)
    env.close()
