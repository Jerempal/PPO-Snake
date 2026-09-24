from snake_rl.config import EnvConfig, RewardConfig, StartStateConfig
from snake_rl.env import RelativeAction, SnakeEnv
from snake_rl.expert import SafePlanner, _shortest_path, _simulate_path


def test_shortest_path_avoids_blocked_cells() -> None:
    path = _shortest_path((0, 0), (2, 0), {(1, 0)}, width=3, height=2)

    assert path == [(0, 1), (1, 1), (2, 1), (2, 0)]


def test_simulated_path_releases_tail_but_grows_on_food() -> None:
    snake = [(2, 1), (1, 1), (0, 1)]

    moved = _simulate_path(snake, [(2, 0)], food=(0, 0))
    ate = _simulate_path(snake, [(2, 0)], food=(2, 0))

    assert moved == ([(2, 0), (2, 1), (1, 1)], False)
    assert ate == ([(2, 0), (2, 1), (1, 1), (0, 1)], True)


def test_planner_takes_safe_visible_food() -> None:
    env = SnakeEnv(
        EnvConfig(width=8, height=8, grid_size=8),
        RewardConfig(),
    )
    env.reset(seed=1)
    env.food = (5, 4)

    action = SafePlanner().action(env)
    assert action == int(RelativeAction.STRAIGHT)
    _, _, terminated, truncated, info = env.step(action)
    assert info["score"] == 1
    assert not terminated and not truncated


def test_planner_returns_legal_action_on_long_body_starts() -> None:
    starts = StartStateConfig(
        enabled=True,
        standard_probability=0.0,
        min_length_fraction=0.4,
        max_length_fraction=0.6,
        length_distribution="normal",
        mean_length_fraction=0.5,
        std_length_fraction=0.05,
    )
    env = SnakeEnv(
        EnvConfig(
            width=20,
            height=20,
            grid_size=20,
        ),
        RewardConfig(),
        starts,
    )
    planner = SafePlanner()

    for seed in range(20):
        env.reset(seed=seed)
        action = planner.action(env)
        dx, dy = env.direction_for_action(action)
        x, y = env.snake[0]
        assert env.collision_type((x + dx, y + dy)) is None
