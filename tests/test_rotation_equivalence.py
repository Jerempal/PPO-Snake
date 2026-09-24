"""Regression checks for physical rotations, including known odd-padding behavior."""

import numpy as np
import pytest

from snake_rl.config import EnvConfig, StartStateConfig
from snake_rl.env import SnakeEnv


@pytest.mark.parametrize("width,height", [(8, 10), (8, 12), (10, 12), (9, 11), (9, 9)])
@pytest.mark.parametrize("turns", [1, 2, 3])
def test_physical_rotation_equivalence(width, height, turns):
    source = SnakeEnv(
        EnvConfig(width=width, height=height, grid_size=12),
        start_state_config=StartStateConfig(enabled=True, standard_probability=0),
    )
    source.reset(seed=95123)
    snake, food, direction = source.snake, source.food, source.direction
    w, h = width, height
    for _ in range(turns):
        snake = [(y, w - 1 - x) for x, y in snake]
        food = (food[1], w - 1 - food[0])
        direction = (direction[1], -direction[0])
        w, h = h, w
    target = SnakeEnv(EnvConfig(width=w, height=h, grid_size=12))
    target.reset(seed=0)
    target.snake, target.food, target.direction = snake, food, direction
    a, b = source._observation(), target._observation()
    np.testing.assert_array_equal(a["vector"], b["vector"])
    if width % 2 == height % 2 == 0:
        np.testing.assert_array_equal(a["grid"], b["grid"])
    # Odd padding is currently orientation-dependent, but only a translation.
    aligned = []
    for observation in (a, b):
        grid = observation["grid"]
        y, x = np.argwhere(grid[4] == 0).min(axis=0)
        aligned.append(np.roll(grid, (-int(y), -int(x)), axis=(1, 2)))
    np.testing.assert_array_equal(*aligned)
    source.close()
    target.close()
