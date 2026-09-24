"""Compact neural feature extractor used by the Snake policies."""

from __future__ import annotations

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from snake_rl.config import ArchitectureConfig


class ResidualBlock(nn.Module):
    """A small residual block that preserves spatial resolution."""

    def __init__(self, channels: int, normalization: str = "group", groups: int = 4) -> None:
        super().__init__()

        def norm():
            if normalization == "group":
                return nn.GroupNorm(groups, channels)
            if normalization == "batch":
                return nn.BatchNorm2d(channels)
            return nn.Identity()

        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding="same"),
            norm(),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding="same"),
            norm(),
        )
        self.activation = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.layers(inputs))


class CompactGridFeaturesExtractor(BaseFeaturesExtractor):
    """Encode the board with a compact residual CNN and bounded projection."""

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
        architecture: dict | None = None,
    ) -> None:
        super().__init__(observation_space, features_dim)
        channels = observation_space.shape[0]
        self.architecture = ArchitectureConfig(**(architecture or {}))
        spec = self.architecture
        trunk = []
        for output_channels, stride in zip(spec.cnn_channels, spec.strides, strict=True):
            trunk.extend(
                [
                    nn.Conv2d(
                        channels,
                        output_channels,
                        kernel_size=3,
                        stride=stride,
                        padding="same" if stride == 1 else 1,
                    ),
                    nn.ReLU(),
                    ResidualBlock(output_channels, spec.normalization, spec.groups)
                    if spec.residual_blocks
                    else nn.Identity(),
                ]
            )
            channels = output_channels
        self.network = nn.Sequential(
            *trunk,
            nn.AdaptiveAvgPool2d((spec.pool_size, spec.pool_size)),
            nn.Flatten(),
            nn.Linear(channels * spec.pool_size * spec.pool_size, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)


class CompactHybridFeaturesExtractor(BaseFeaturesExtractor):
    """Fuse the CNN board embedding with explicit raycast/state features."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        grid_features_dim: int = 128,
        vector_features_dim: int = 64,
        architecture: dict | None = None,
    ) -> None:
        super().__init__(observation_space, grid_features_dim + vector_features_dim)
        self.grid_extractor = CompactGridFeaturesExtractor(
            observation_space.spaces["grid"],
            grid_features_dim,
            architecture,
        )
        vector_size = observation_space.spaces["vector"].shape[0]
        self.vector_extractor = nn.Sequential(
            nn.Linear(vector_size, vector_features_dim),
            nn.ReLU(),
            nn.Linear(vector_features_dim, vector_features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        grid_features = self.grid_extractor(observations["grid"])
        vector_features = self.vector_extractor(observations["vector"])
        return torch.cat((grid_features, vector_features), dim=1)
