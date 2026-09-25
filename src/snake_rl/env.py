"""Environnement Snake compatible avec Gymnasium."""

from __future__ import annotations

from enum import IntEnum
from math import sqrt
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from snake_rl.config import (
    EnvConfig,
    RewardConfig,
    StartStateConfig,
)

Position = tuple[int, int]


class Heading(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class RelativeAction(IntEnum):
    LEFT = 0
    STRAIGHT = 1
    RIGHT = 2


DIRECTIONS: dict[Heading, Position] = {
    Heading.UP: (0, -1),
    Heading.DOWN: (0, 1),
    Heading.LEFT: (-1, 0),
    Heading.RIGHT: (1, 0),
}

RAY_DIRECTIONS: tuple[Position, ...] = (
    (0, -1),
    (1, -1),
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
)
CARDINAL_DIRECTIONS = tuple(DIRECTIONS.values())
CARDINAL_INDEX = {direction: index for index, direction in enumerate(CARDINAL_DIRECTIONS)}


Observation = dict[str, np.ndarray]


class SnakeEnv(gym.Env[Observation, int]):
    """Environnement Snake déterministe, sans affichage par défaut.

    Chaque observation combine sept plans spatiaux et 33 valeurs de contrôle et d'état.
    La grille et le vecteur sont toujours orientés dans le repère du serpent.
    """

    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 420,
    }

    def __init__(
        self,
        env_config: EnvConfig | None = None,
        reward_config: RewardConfig | None = None,
        start_state_config: StartStateConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if render_mode not in {None, *self.metadata["render_modes"]}:
            raise ValueError(f"Mode de rendu non pris en charge : {render_mode}")

        self.config = env_config or EnvConfig()
        self.rewards = reward_config or RewardConfig()
        self.start_states = start_state_config or StartStateConfig()
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(len(RelativeAction))
        canvas = self.config.grid_size + 2 * self.config.observation_border
        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(0.0, 1.0, shape=(7, canvas, canvas), dtype=np.float32),
                "vector": spaces.Box(0.0, 1.0, shape=(33,), dtype=np.float32),
            }
        )

        self.snake: list[Position] = []
        self.direction = DIRECTIONS[Heading.RIGHT]
        self.food: Position | None = None
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.longest_steps_without_food = 0
        self.initial_length = 3
        self._state_visits: dict[tuple[tuple[Position, ...], Position, Position | None], int] = {}
        self._episode_over = True
        self._start_type = "standard"
        self._pygame: Any = None
        self._window: Any = None
        self._clock: Any = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        del options
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
        if (
            self.start_states.enabled
            and self.np_random.random() >= self.start_states.standard_probability
        ):
            self._set_long_body_start()
            self._start_type = "structured"
        else:
            self._set_normal_start()
            self._start_type = "standard"
        self.initial_length = len(self.snake)
        self.food = self._spawn_food()
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.longest_steps_without_food = 0
        self._state_visits = {self._state_signature(): 1}
        self._episode_over = False

        observation = self._observation()
        info = self._info()
        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(self, action: int) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        if self._episode_over:
            raise RuntimeError("La partie est terminée ; appelez reset() avant step().")

        self.direction = self.direction_for_action(int(action))

        old_distance = self._food_distance()
        head_x, head_y = self.snake[0]
        direction_x, direction_y = self.direction
        new_head = (head_x + direction_x, head_y + direction_y)
        self.steps += 1
        self.steps_since_food += 1
        self.longest_steps_without_food = max(
            self.longest_steps_without_food, self.steps_since_food
        )
        reward = self.rewards.step
        collision = self.collision_type(new_head)
        won = False

        ate_food = False
        if collision is not None:
            reward += self.rewards.death
            terminated = True
        else:
            ate_food = new_head == self.food
            self.snake.insert(0, new_head)
            if ate_food:
                self.score += 1
                self.steps_since_food = 0
                reward += self.rewards.food
                self.food = self._spawn_food()
                won = self.food is None
                if won:
                    reward += self.rewards.win
            else:
                self.snake.pop()

            terminated = won
            if not terminated and not ate_food:
                new_distance = self._food_distance()
                # A potential difference telescopes to zero over any movement loop.
                distance_coefficient = self.rewards.distance_coefficient(
                    self.config.width,
                    self.config.height,
                )
                reward += distance_coefficient * (old_distance - new_distance)

        looped = False
        if not terminated and self.config.loop_max_state_visits is not None:
            if ate_food:
                self._state_visits.clear()
            signature = self._state_signature()
            visits = self._state_visits.get(signature, 0) + 1
            self._state_visits[signature] = visits
            looped = visits >= self.config.loop_max_state_visits

        truncated = looped
        if looped:
            reward += self.rewards.loop
        self._episode_over = terminated or truncated
        info = self._info(collision=collision, won=won, looped=looped)
        observation = self._observation()

        if self.render_mode == "human":
            self.render()
        return observation, float(reward), terminated, truncated, info

    def direction_for_action(self, action: int) -> Position:
        dx, dy = self.direction
        relative_action = RelativeAction(action)
        if relative_action == RelativeAction.LEFT:
            return dy, -dx
        if relative_action == RelativeAction.RIGHT:
            return -dy, dx
        return self.direction

    def _set_normal_start(self) -> None:
        center_x = self.config.width // 2
        center_y = self.config.height // 2
        self.snake = [(center_x, center_y), (center_x - 1, center_y), (center_x - 2, center_y)]
        self.direction = DIRECTIONS[Heading.RIGHT]

    def _set_long_body_start(self) -> None:
        cells, is_cycle = self.start_path()
        area = self.config.width * self.config.height
        target_length = self._sample_long_body_length(area)
        self._set_structured_body(cells, is_cycle, target_length)

    def _set_structured_body(
        self,
        cells: list[Position],
        is_cycle: bool,
        target_length: int,
    ) -> None:
        if is_cycle:
            start = int(self.np_random.integers(len(cells)))
            step = 1 if bool(self.np_random.integers(2)) else -1
            self.snake = [
                cells[(start + step * index) % len(cells)] for index in range(target_length)
            ]
        else:
            cells = list(cells)
            if bool(self.np_random.integers(2)):
                cells.reverse()
            start = int(self.np_random.integers(len(cells) - target_length + 1))
            self.snake = cells[start : start + target_length]
        head_x, head_y = self.snake[0]
        neck_x, neck_y = self.snake[1]
        self.direction = (head_x - neck_x, head_y - neck_y)

    def _sample_long_body_length(self, area: int) -> int:
        """Sample a bounded long-body length without piling probability at the bounds."""
        minimum = max(3, round(area * self.start_states.min_length_fraction))
        maximum = min(area - 1, round(area * self.start_states.max_length_fraction))
        minimum = min(minimum, maximum)
        if self.start_states.length_distribution == "uniform":
            return int(self.np_random.integers(minimum, maximum + 1))

        mean = area * self.start_states.mean_length_fraction
        standard_deviation = area * self.start_states.std_length_fraction
        for _ in range(128):
            candidate = round(float(self.np_random.normal(mean, standard_deviation)))
            if minimum <= candidate <= maximum:
                return candidate
        return min(max(round(mean), minimum), maximum)

    def start_path(self) -> tuple[list[Position], bool]:
        """Build a Hamiltonian cycle when possible, otherwise a serpentine path."""
        width, height = self.config.width, self.config.height
        if height % 2 == 0:
            return self._cycle_with_even_height(width, height), True
        if width % 2 == 0:
            transposed = self._cycle_with_even_height(height, width)
            return [(y, x) for x, y in transposed], True
        return (
            [
                (x, y)
                for y in range(height)
                for x in (range(width) if y % 2 == 0 else range(width - 1, -1, -1))
            ],
            False,
        )

    @staticmethod
    def _cycle_with_even_height(width: int, height: int) -> list[Position]:
        cells = [(0, 0), *((x, 0) for x in range(1, width))]
        for y in range(1, height):
            xs = range(width - 1, 0, -1) if y % 2 else range(1, width)
            cells.extend((x, y) for x in xs)
        cells.extend((0, y) for y in range(height - 1, 0, -1))
        return cells

    def _spawn_food(self) -> Position | None:
        occupied = set(self.snake)
        free_cells = [
            (x, y)
            for y in range(self.config.height)
            for x in range(self.config.width)
            if (x, y) not in occupied
        ]
        if not free_cells:
            return None
        index = int(self.np_random.integers(len(free_cells)))
        return free_cells[index]

    def collision_type(self, position: Position) -> str | None:
        x, y = position
        if x < 0 or x >= self.config.width or y < 0 or y >= self.config.height:
            return "wall"
        if position in self.snake[:-1]:
            return "body"
        return None

    def _food_distance(self) -> int:
        if self.food is None:
            return 0
        head_x, head_y = self.snake[0]
        return abs(head_x - self.food[0]) + abs(head_y - self.food[1])

    def _danger_flags(self) -> list[float]:
        head_x, head_y = self.snake[0]
        return [
            float(self.collision_type((head_x + dx, head_y + dy)) is not None)
            for dx, dy in DIRECTIONS.values()
        ]

    def _observation(self) -> Observation:
        return {"grid": self._grid_observation(), "vector": self._vector_observation()}

    def _vector_observation(self) -> np.ndarray:
        """Retourne le vecteur egocentrique de controle a 33 valeurs."""

        head_x, head_y = self.snake[0]
        if self.food is None:
            food_x = food_y = 0
            relative_food = [0.0, 0.0, 0.0, 0.0]
        else:
            food_x, food_y = self.food
            relative_food = [
                float(food_y < head_y),
                float(food_y > head_y),
                float(food_x < head_x),
                float(food_x > head_x),
            ]

        direction_one_hot = [
            float(self.direction == direction) for direction in DIRECTIONS.values()
        ]
        danger_flags = self._danger_flags()
        quarter_turns = self._egocentric_quarter_turns()
        if quarter_turns:
            head_x, head_y = self._rotate_position(
                (head_x, head_y), quarter_turns, self.config.width, self.config.height
            )
            if self.food is not None:
                food_x, food_y = self._rotate_position(
                    (food_x, food_y), quarter_turns, self.config.width, self.config.height
                )
            direction_one_hot = self._rotate_cardinal_values(direction_one_hot, quarter_turns)
            danger_flags = self._rotate_cardinal_values(danger_flags, quarter_turns)
            relative_food = self._rotate_cardinal_values(relative_food, quarter_turns)
        wall_clearance, body_clearance = self._ray_clearances()
        if quarter_turns:
            wall_clearance = list(np.roll(wall_clearance, -2 * quarter_turns))
            body_clearance = list(np.roll(body_clearance, -2 * quarter_turns))
        rotated_width, rotated_height = (
            (self.config.height, self.config.width)
            if quarter_turns % 2
            else (self.config.width, self.config.height)
        )
        return np.asarray(
            [
                head_x / (rotated_width - 1),
                head_y / (rotated_height - 1),
                food_x / (rotated_width - 1),
                food_y / (rotated_height - 1),
                *direction_one_hot,
                *danger_flags,
                *relative_food,
                *wall_clearance,
                *body_clearance,
                len(self.snake) / (self.config.width * self.config.height),
            ],
            dtype=np.float32,
        )

    def _grid_observation(self) -> np.ndarray:
        """Return head, body, tail, food, wall and ordered-body channels."""
        canvas = self.config.grid_size + 2 * self.config.observation_border
        map_offset = (
            self.config.observation_border + (self.config.grid_size - self.config.width) // 2
        )
        offset_y = (
            self.config.observation_border + (self.config.grid_size - self.config.height) // 2
        )
        observation = np.zeros((7, canvas, canvas), dtype=np.float32)
        observation[4, :, :] = 1.0
        observation[
            4,
            offset_y : offset_y + self.config.height,
            map_offset : map_offset + self.config.width,
        ] = 0.0

        def mark(channel: int, position: Position, value: float = 1.0) -> None:
            x, y = position
            observation[channel, y + offset_y, x + map_offset] = value

        mark(0, self.snake[0])
        if len(self.snake) > 2:
            for segment in self.snake[1:-1]:
                mark(1, segment)
        if len(self.snake) > 1:
            mark(2, self.snake[-1])
        if self.food is not None:
            mark(3, self.food)
        denominator = max(len(self.snake) - 1, 1)
        for index, segment in enumerate(self.snake):
            mark(5, segment, 1.0 - index / denominator)
        observation[6, :, :] = len(self.snake) / (self.config.width * self.config.height)
        quarter_turns = self._egocentric_quarter_turns()
        if quarter_turns:
            observation = np.rot90(observation, quarter_turns, axes=(1, 2)).copy()
        return observation

    def _egocentric_quarter_turns(self) -> int:
        """Return CCW quarter turns that align the current heading with up."""
        return {
            DIRECTIONS[Heading.UP]: 0,
            DIRECTIONS[Heading.RIGHT]: 1,
            DIRECTIONS[Heading.DOWN]: 2,
            DIRECTIONS[Heading.LEFT]: 3,
        }[self.direction]

    @staticmethod
    def _rotate_position(
        position: Position,
        quarter_turns: int,
        size: int,
        height: int | None = None,
    ) -> Position:
        """Rotate a board position, swapping rectangular dimensions each turn."""
        x, y = position
        width, height = size, size if height is None else height
        for _ in range(quarter_turns % 4):
            x, y = y, width - 1 - x
            width, height = height, width
        return x, y

    @staticmethod
    def _rotate_direction(direction: Position, quarter_turns: int) -> Position:
        """Rotate a screen-coordinate direction counter-clockwise."""
        dx, dy = direction
        for _ in range(quarter_turns % 4):
            dx, dy = dy, -dx
        return dx, dy

    @classmethod
    def _rotate_cardinal_values(cls, values: list[float], quarter_turns: int) -> list[float]:
        """Move world-cardinal values into their rotated cardinal slots."""
        rotated = [0.0] * len(CARDINAL_DIRECTIONS)
        for value, direction in zip(values, CARDINAL_DIRECTIONS, strict=True):
            rotated_direction = cls._rotate_direction(direction, quarter_turns)
            rotated[CARDINAL_INDEX[rotated_direction]] = value
        return rotated

    def _ray_clearances(self) -> tuple[list[float], list[float]]:
        """Return normalized wall and body clearance along eight grid rays."""
        head_x, head_y = self.snake[0]
        # normalization = float(max(self.config.width, self.config.height) - 1)
        normalization = sqrt(
            self.config.width * self.config.width + self.config.height * self.config.height
        )
        body = set(self.snake[1:])
        wall_clearance: list[float] = []
        body_clearance: list[float] = []

        for direction_x, direction_y in RAY_DIRECTIONS:
            distance = 0
            first_body_distance: int | None = None
            while True:
                candidate_distance = distance + 1
                x = head_x + direction_x * candidate_distance
                y = head_y + direction_y * candidate_distance
                if x < 0 or x >= self.config.width or y < 0 or y >= self.config.height:
                    break
                distance = candidate_distance
                if first_body_distance is None and (x, y) in body:
                    first_body_distance = distance

            wall_clearance.append(distance / normalization)
            body_clearance.append(
                1.0 if first_body_distance is None else first_body_distance / normalization
            )

        return wall_clearance, body_clearance

    def _info(
        self,
        collision: str | None = None,
        won: bool = False,
        looped: bool = False,
    ) -> dict[str, Any]:
        return {
            "score": self.score,
            "length": len(self.snake),
            "steps": self.steps,
            "steps_since_food": self.steps_since_food,
            "longest_steps_without_food": self.longest_steps_without_food,
            "initial_length": self.initial_length,
            "start_type": self._start_type,
            "collision": collision,
            "won": won,
            "looped": looped,
        }

    def _state_signature(self) -> tuple[tuple[Position, ...], Position, Position | None]:
        """Return the exact Markov state used by the optional training loop guard."""
        return tuple(self.snake), self.direction, self.food

    @staticmethod
    def _opposite(direction: Position) -> Position:
        return -direction[0], -direction[1]

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        frame = self._rgb_frame()
        if self.render_mode == "rgb_array":
            return frame

        if self._pygame is None:
            try:
                import pygame
            except ImportError as exc:
                raise ImportError(
                    "Le rendu humain nécessite : pip install 'snake-rl[play]'"
                ) from exc
            self._pygame = pygame
            pygame.init()
            self._window = pygame.display.set_mode((frame.shape[1], frame.shape[0]))
            pygame.display.set_caption("Snake RL")
            self._clock = pygame.time.Clock()

        self._pygame.event.pump()
        surface = self._pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        self._window.blit(surface, (0, 0))
        self._pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])
        return None

    def _rgb_frame(self) -> np.ndarray:
        cell_size = 24
        supersample = 4
        scaled_cell_size = cell_size * supersample
        frame = np.full(
            (
                self.config.height * scaled_cell_size,
                self.config.width * scaled_cell_size,
                3,
            ),
            (20, 24, 33),
            dtype=np.uint8,
        )

        def paint_capsule(
            start: Position,
            end: Position,
            radius: float,
            color: tuple[int, int, int],
        ) -> None:
            start_x = (start[0] + 0.5) * scaled_cell_size
            start_y = (start[1] + 0.5) * scaled_cell_size
            end_x = (end[0] + 0.5) * scaled_cell_size
            end_y = (end[1] + 0.5) * scaled_cell_size
            minimum_x = max(0, int(min(start_x, end_x) - radius - 1))
            maximum_x = min(frame.shape[1], int(max(start_x, end_x) + radius + 2))
            minimum_y = max(0, int(min(start_y, end_y) - radius - 1))
            maximum_y = min(frame.shape[0], int(max(start_y, end_y) + radius + 2))
            grid_y, grid_x = np.ogrid[minimum_y:maximum_y, minimum_x:maximum_x]
            direction_x = end_x - start_x
            direction_y = end_y - start_y
            length_squared = direction_x * direction_x + direction_y * direction_y
            if length_squared == 0:
                distance_squared = (grid_x - start_x) ** 2 + (grid_y - start_y) ** 2
            else:
                projection = (
                    (grid_x - start_x) * direction_x + (grid_y - start_y) * direction_y
                ) / length_squared
                projection = np.clip(projection, 0.0, 1.0)
                closest_x = start_x + projection * direction_x
                closest_y = start_y + projection * direction_y
                distance_squared = (grid_x - closest_x) ** 2 + (grid_y - closest_y) ** 2
            mask = distance_squared <= radius * radius
            frame[minimum_y:maximum_y, minimum_x:maximum_x][mask] = color

        body_radius = scaled_cell_size * 0.39
        for index in range(len(self.snake) - 1, 0, -1):
            paint_capsule(self.snake[index], self.snake[index - 1], body_radius, (42, 173, 91))
        paint_capsule(self.snake[0], self.snake[0], body_radius, (91, 225, 132))
        if self.food is not None:
            paint_capsule(self.food, self.food, scaled_cell_size * 0.3, (235, 87, 87))
        return (
            frame.reshape(
                self.config.height * cell_size,
                supersample,
                self.config.width * cell_size,
                supersample,
                3,
            )
            .mean(axis=(1, 3))
            .astype(np.uint8)
        )

    def close(self) -> None:
        if self._pygame is not None:
            self._pygame.display.quit()
            self._pygame.quit()
        self._pygame = self._window = self._clock = None
