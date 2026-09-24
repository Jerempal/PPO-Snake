import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from snake_rl import (
    EnvConfig,
    RelativeAction,
    RewardConfig,
    SnakeEnv,
    StartStateConfig,
)


def test_gymnasium_contract() -> None:
    env = SnakeEnv()
    check_env(env, skip_render_check=True)
    env.close()


def test_reset_is_deterministic_and_valid() -> None:
    first = SnakeEnv()
    second = SnakeEnv()

    first_observation, first_info = first.reset(seed=123)
    second_observation, second_info = second.reset(seed=123)

    if isinstance(first_observation, dict):
        assert isinstance(second_observation, dict)
        for key in first_observation:
            np.testing.assert_array_equal(first_observation[key], second_observation[key])
    else:
        np.testing.assert_array_equal(first_observation, second_observation)
    assert first.food == second.food
    assert first.observation_space.contains(first_observation)
    assert first.food not in first.snake
    assert first_info == second_info


def test_raycast_observation_reports_wall_and_body_clearance() -> None:
    env = SnakeEnv(EnvConfig(width=5, height=5, grid_size=5))
    env.reset(seed=0)
    env.snake = [(2, 2), (2, 3)]
    env.direction = (0, -1)
    env.food = (4, 4)

    observation = env._observation()["vector"]

    assert observation.shape == (33,)
    normalization = np.sqrt(5**2 + 5**2)
    np.testing.assert_allclose(observation[16:24], [2 / normalization] * 8)
    np.testing.assert_allclose(observation[24:32], [1, 1, 1, 1, 1 / normalization, 1, 1, 1])


def test_grid_observation_contains_complete_ordered_snake() -> None:
    env = SnakeEnv(EnvConfig(width=5, height=5, grid_size=5, observation_border=1))
    env.reset(seed=0)
    env.snake = [(2, 2), (2, 3), (1, 3), (1, 2)]
    env.direction = (0, -1)
    env.food = (4, 4)

    observation = env._observation()["grid"]

    assert observation.shape == (7, 7, 7)
    assert observation[0, 3, 3] == 1  # head
    assert observation[1, 4, 3] == 1  # body
    assert observation[2, 3, 2] == 1  # tail
    assert observation[3, 5, 5] == 1  # food
    assert observation[4, 0, 0] == 1  # padded wall
    assert observation[4, 1, 1] == 0  # playable cell
    np.testing.assert_allclose(observation[6], len(env.snake) / 25)
    np.testing.assert_allclose(
        [observation[5, 3, 3], observation[5, 4, 3], observation[5, 4, 2]],
        [1.0, 2 / 3, 1 / 3],
    )


def test_fixed_grid_canvas_centers_a_smaller_playable_map() -> None:
    config = EnvConfig(width=12, height=12, grid_size=20)
    env = SnakeEnv(config)
    env.reset(seed=0)
    env.direction = (0, -1)

    observation = env._observation()["grid"]
    offset = (config.grid_size - config.width) // 2

    assert observation.shape == (7, 20, 20)
    assert observation[4, offset, offset] == 0
    assert observation[4, offset + 11, offset + 11] == 0
    assert observation[4, offset - 1, offset] == 1
    assert observation[4, offset + 12, offset] == 1
    head_x, head_y = env.snake[0]
    assert observation[0, head_y + offset, head_x + offset] == 1


def test_fixed_canvas_keeps_observation_space_compatible_across_map_sizes() -> None:
    small = SnakeEnv(EnvConfig(width=12, height=12, grid_size=20))
    large = SnakeEnv(EnvConfig(width=20, height=20, grid_size=20))

    assert small.observation_space == large.observation_space


def test_hybrid_observation_combines_grid_and_raycast() -> None:
    env = SnakeEnv(EnvConfig(width=5, height=5, grid_size=5))
    observation, _ = env.reset(seed=0)

    assert set(observation) == {"grid", "vector"}
    assert observation["grid"].shape == (7, 5, 5)
    assert observation["vector"].shape == (33,)
    assert env.observation_space.contains(observation)


def test_hybrid_observation_adds_body_length() -> None:
    env = SnakeEnv(
        EnvConfig(
            width=5,
            height=5,
            grid_size=5,
        )
    )
    observation, _ = env.reset(seed=0)
    observation = env._observation()

    assert observation["vector"].shape == (33,)
    assert observation["vector"][-1] == pytest.approx(len(env.snake) / 25)
    assert env.observation_space.contains(observation)


def test_egocentric_observation_is_invariant_to_world_rotation() -> None:
    config = EnvConfig(
        width=5,
        height=5,
        grid_size=5,
    )
    up = SnakeEnv(config)
    up.reset(seed=0)

    up.snake = [(2, 2), (2, 3), (1, 3), (1, 4)]
    up.direction = (0, -1)
    up.food = (4, 1)
    up.steps_since_food = 5

    def clockwise(position: tuple[int, int]) -> tuple[int, int]:
        x, y = position
        return 4 - y, x

    up_observation = up._observation()
    np.testing.assert_array_equal(up_observation["vector"][4:8], [1, 0, 0, 0])
    assert up.observation_space.contains(up_observation)

    snake = up.snake
    food = up.food
    direction = up.direction
    for _ in range(3):
        rotated = SnakeEnv(config)
        rotated.reset(seed=0)
        snake = [clockwise(position) for position in snake]
        assert food is not None
        food = clockwise(food)
        dx, dy = direction
        direction = -dy, dx
        rotated.snake = snake
        rotated.direction = direction
        rotated.food = food
        rotated.steps_since_food = up.steps_since_food

        rotated_observation = rotated._observation()

        np.testing.assert_array_equal(up_observation["grid"], rotated_observation["grid"])
        np.testing.assert_allclose(up_observation["vector"], rotated_observation["vector"])
        assert rotated.observation_space.contains(rotated_observation)


def test_centered_fixed_canvas_remains_egocentric_under_rotation() -> None:
    config = EnvConfig(
        width=12,
        height=12,
        grid_size=20,
    )
    original = SnakeEnv(config)
    original.reset(seed=0)
    original.snake = [(5, 5), (5, 6), (4, 6), (4, 7)]
    original.direction = (0, -1)
    original.food = (8, 3)
    expected = original._observation()

    def clockwise(position: tuple[int, int]) -> tuple[int, int]:
        x, y = position
        return config.width - 1 - y, x

    rotated = SnakeEnv(config)
    rotated.reset(seed=0)
    rotated.snake = [clockwise(position) for position in original.snake]
    rotated.direction = (1, 0)
    rotated.food = clockwise(original.food)
    actual = rotated._observation()

    np.testing.assert_array_equal(actual["grid"], expected["grid"])
    np.testing.assert_allclose(actual["vector"], expected["vector"])


def test_action_space_exposes_only_relative_controls() -> None:
    env = SnakeEnv()
    env.reset(seed=0)
    old_head = env.snake[0]

    assert env.action_space.n == 3
    env.step(RelativeAction.STRAIGHT)

    assert env.snake[0] == (old_head[0] + 1, old_head[1])


def test_relative_actions_turn_from_current_heading() -> None:
    env = SnakeEnv()
    env.reset(seed=0)
    old_head = env.snake[0]

    env.step(RelativeAction.LEFT)

    assert env.snake[0] == (old_head[0], old_head[1] - 1)


def test_long_body_start_is_valid_and_reproducible() -> None:
    config = EnvConfig(width=6, height=6, grid_size=6)
    starts = StartStateConfig(
        enabled=True,
        standard_probability=0.0,
        min_length_fraction=0.5,
        max_length_fraction=0.5,
    )
    first = SnakeEnv(config, start_state_config=starts)
    second = SnakeEnv(config, start_state_config=starts)

    _, info = first.reset(seed=123)
    second.reset(seed=123)

    assert len(first.snake) == 18
    assert info["initial_length"] == 18
    assert first.snake == second.snake
    assert len(set(first.snake)) == len(first.snake)
    assert first.food not in first.snake
    assert all(
        abs(ax - bx) + abs(ay - by) == 1
        for (ax, ay), (bx, by) in zip(first.snake, first.snake[1:], strict=False)
    )


def test_normal_long_body_distribution_is_centered_and_bounded() -> None:
    config = EnvConfig(width=20, height=20, grid_size=20)
    starts = StartStateConfig(
        enabled=True,
        standard_probability=0.0,
        min_length_fraction=0.1,
        max_length_fraction=0.8,
        length_distribution="normal",
        mean_length_fraction=0.45,
        std_length_fraction=0.15,
    )
    env = SnakeEnv(config, start_state_config=starts)
    env.reset(seed=123)

    lengths = np.asarray([env._sample_long_body_length(400) for _ in range(2_000)])

    assert lengths.min() >= 40
    assert lengths.max() <= 320
    assert lengths.mean() == pytest.approx(180, abs=5)
    assert 0.4 < np.mean(lengths < 180) < 0.6


def test_food_grows_snake_and_updates_score() -> None:
    env = SnakeEnv()
    env.reset(seed=0)
    head_x, head_y = env.snake[0]
    env.food = (head_x + 1, head_y)

    _, reward, terminated, truncated, info = env.step(RelativeAction.STRAIGHT)

    assert reward > 0
    assert not terminated
    assert not truncated
    assert info["score"] == 1
    assert info["length"] == 4
    assert env.food not in env.snake


def test_wall_collision_terminates_and_requires_reset() -> None:
    env = SnakeEnv()
    env.reset(seed=0)
    env.snake = [(0, 2), (1, 2), (2, 2)]
    env.direction = (-1, 0)

    observation, reward, terminated, truncated, info = env.step(RelativeAction.STRAIGHT)

    assert reward < 0
    assert terminated
    assert not truncated
    assert info["collision"] == "wall"
    assert env.observation_space.contains(observation)
    with pytest.raises(RuntimeError, match=r"appelez reset\(\)"):
        env.step(RelativeAction.STRAIGHT)


def test_food_gap_counter_is_descriptive_only() -> None:
    env = SnakeEnv(EnvConfig())
    env.reset(seed=0)
    env.snake = [(2, 2), (1, 2), (0, 2)]
    env.direction = (1, 0)
    env.food = (5, 5)

    for action in (RelativeAction.RIGHT,) * 4:
        observation, _, terminated, truncated, _ = env.step(action)

    assert not terminated
    assert not truncated
    assert env.steps_since_food == 4
    assert env.longest_steps_without_food == 4
    assert observation["vector"][-1] == pytest.approx(len(env.snake) / 144)


def test_exact_state_repetition_truncates_training_episode() -> None:
    env = SnakeEnv(
        EnvConfig(width=6, height=6, grid_size=6, loop_max_state_visits=2),
        RewardConfig(loop=-0.5, distance_beta=0.0),
    )
    env.reset(seed=0)
    env.snake = [(2, 2), (1, 2), (1, 1)]
    env.direction = (1, 0)
    env.food = (5, 5)
    env._state_visits = {env._state_signature(): 1}

    terminated = truncated = False
    info = env._info()
    total_reward = 0.0
    actions = (RelativeAction.RIGHT,) * 4
    for index in range(30):
        _, reward, terminated, truncated, info = env.step(actions[index % len(actions)])
        total_reward += reward
        if terminated or truncated:
            break

    assert not terminated
    assert truncated
    assert info["looped"]
    assert total_reward < -0.5


def test_distance_shaping_has_no_positive_loop_reward() -> None:
    env = SnakeEnv(EnvConfig(width=6, height=6, grid_size=6))
    env.reset(seed=0)
    env.snake = [(2, 2), (1, 2), (0, 2)]
    env.direction = (1, 0)
    env.food = (5, 5)

    rewards = [env.step(action)[1] for action in (RelativeAction.RIGHT,) * 4]

    assert sum(rewards) == pytest.approx(4 * env.rewards.step)


def test_distance_beta_bounds_shaping_relative_to_map_size() -> None:
    env = SnakeEnv(
        EnvConfig(width=5, height=5, grid_size=5),
        RewardConfig(step=0.0, distance_beta=0.8),
    )
    env.reset(seed=0)
    env.snake = [(2, 2), (1, 2), (0, 2)]
    env.direction = (1, 0)
    env.food = (4, 2)

    _, reward, terminated, truncated, _ = env.step(RelativeAction.STRAIGHT)

    assert not terminated
    assert not truncated
    assert reward == pytest.approx(0.1)


def test_body_collision_terminates() -> None:
    env = SnakeEnv()
    env.reset(seed=0)
    env.snake = [(2, 2), (2, 1), (3, 1), (3, 2), (3, 3)]
    env.direction = (0, 1)

    _, _, terminated, _, info = env.step(RelativeAction.LEFT)

    assert terminated
    assert info["collision"] == "body"


def test_invalid_action_is_rejected() -> None:
    env = SnakeEnv()
    env.reset(seed=0)

    with pytest.raises(ValueError, match="not a valid RelativeAction"):
        env.step(9)


def test_rgb_render_has_expected_shape() -> None:
    config = EnvConfig(width=8, height=8, grid_size=8)
    env = SnakeEnv(config, render_mode="rgb_array")
    env.reset(seed=0)

    frame = env.render()

    assert frame is not None
    assert frame.shape == (8 * 24, 8 * 24, 3)
    assert frame.dtype == np.uint8


def test_full_grid_has_no_food() -> None:
    env = SnakeEnv(EnvConfig(width=5, height=5, grid_size=5))
    env.reset(seed=0)
    env.snake = [(x, y) for y in range(5) for x in range(5)]

    assert env._spawn_food() is None
