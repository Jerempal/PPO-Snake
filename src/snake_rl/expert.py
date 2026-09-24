"""Classical safe planner for long-body Snake states."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from snake_rl.env import CARDINAL_DIRECTIONS, Position, RelativeAction, SnakeEnv


@dataclass(frozen=True)
class Candidate:
    """One legal action and the safety properties of its successor state."""

    action: int
    tail_reachable: bool
    reachable_space: int
    tail_distance: int
    food_distance: int


def _neighbors(position: Position, width: int, height: int):
    x, y = position
    for dx, dy in CARDINAL_DIRECTIONS:
        candidate = (x + dx, y + dy)
        if 0 <= candidate[0] < width and 0 <= candidate[1] < height:
            yield candidate


def _shortest_path(
    start: Position,
    goal: Position | None,
    blocked: set[Position],
    width: int,
    height: int,
) -> list[Position] | None:
    """Return positions after the start along one shortest grid path."""
    if goal is None:
        return None
    frontier = deque([start])
    parents: dict[Position, Position | None] = {start: None}
    while frontier:
        current = frontier.popleft()
        if current == goal:
            path = []
            while parents[current] is not None:
                path.append(current)
                parent = parents[current]
                assert parent is not None
                current = parent
            path.reverse()
            return path
        for candidate in _neighbors(current, width, height):
            if candidate not in blocked and candidate not in parents:
                parents[candidate] = current
                frontier.append(candidate)
    return None


def _reachable_space(start: Position, blocked: set[Position], width: int, height: int) -> int:
    return len(_reachable_positions(start, blocked, width, height))


def _reachable_positions(
    start: Position, blocked: set[Position], width: int, height: int
) -> set[Position]:
    frontier = [start]
    reached = {start}
    while frontier:
        current = frontier.pop()
        for candidate in _neighbors(current, width, height):
            if candidate not in blocked and candidate not in reached:
                reached.add(candidate)
                frontier.append(candidate)
    return reached


def _simulate_path(
    snake: list[Position],
    path: list[Position],
    food: Position | None,
) -> tuple[list[Position], bool] | None:
    simulated = list(snake)
    ate_food = False
    for new_head in path:
        eats_now = new_head == food
        collision_body = simulated if eats_now else simulated[:-1]
        if new_head in collision_body:
            return None
        simulated.insert(0, new_head)
        if eats_now:
            ate_food = True
        else:
            simulated.pop()
    return simulated, ate_food


def _tail_path(snake: list[Position], width: int, height: int) -> list[Position] | None:
    return _shortest_path(snake[0], snake[-1], set(snake[1:-1]), width, height)


class SafePlanner:
    """Choose food paths that preserve tail access, otherwise maximize safe space."""

    def action(self, env: SnakeEnv) -> int:
        width, height = env.config.width, env.config.height
        blocked = set(env.snake[1:-1])
        candidates = self.candidates(env)
        safe_candidates = [candidate for candidate in candidates if candidate.tail_reachable]
        action_pool = safe_candidates or candidates
        food_path = _shortest_path(
            env.snake[0],
            env.food,
            blocked,
            width,
            height,
        )
        if food_path:
            simulated = _simulate_path(env.snake, food_path, env.food)
            if simulated is not None:
                food_snake, ate_food = simulated
                if ate_food and _tail_path(food_snake, width, height) is not None:
                    action = self._action_towards(env, food_path[0])
                    if action is not None:
                        return action

        if not candidates:
            return int(RelativeAction.STRAIGHT)
        best = max(
            action_pool,
            key=lambda candidate: (
                candidate.reachable_space,
                -candidate.tail_distance,
                -candidate.food_distance,
                -abs(candidate.action - int(RelativeAction.STRAIGHT)),
            ),
        )
        return best.action

    def candidates(self, env: SnakeEnv) -> list[Candidate]:
        width, height = env.config.width, env.config.height
        candidates = []
        for action in range(env.action_space.n):
            direction = env.direction_for_action(action)
            head_x, head_y = env.snake[0]
            new_head = (head_x + direction[0], head_y + direction[1])
            if env.collision_type(new_head) is not None:
                continue
            simulated = _simulate_path(env.snake, [new_head], env.food)
            assert simulated is not None
            snake, _ = simulated
            blocked = set(snake[1:-1])
            tail_path = _tail_path(snake, width, height)
            food_path = _shortest_path(snake[0], env.food, blocked, width, height)
            candidates.append(
                Candidate(
                    action=action,
                    tail_reachable=tail_path is not None,
                    reachable_space=_reachable_space(snake[0], blocked, width, height),
                    tail_distance=len(tail_path) if tail_path is not None else width * height + 1,
                    food_distance=len(food_path) if food_path is not None else width * height + 1,
                )
            )
        return candidates

    @staticmethod
    def _action_towards(env: SnakeEnv, target: Position) -> int | None:
        head_x, head_y = env.snake[0]
        direction = (target[0] - head_x, target[1] - head_y)
        for action in range(env.action_space.n):
            if env.direction_for_action(action) == direction:
                return action
        return None
