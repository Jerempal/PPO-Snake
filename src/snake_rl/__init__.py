"""Snake reinforcement-learning package."""

from snake_rl.config import EnvConfig, RewardConfig, StartStateConfig
from snake_rl.env import RelativeAction, SnakeEnv

__all__ = [
    "EnvConfig",
    "RelativeAction",
    "RewardConfig",
    "SnakeEnv",
    "StartStateConfig",
]
__version__ = "0.2.0"
