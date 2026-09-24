import torch
from gymnasium import spaces

from snake_rl.policy import (
    CompactGridFeaturesExtractor,
    CompactHybridFeaturesExtractor,
    ResidualBlock,
)


def test_residual_block_preserves_tensor_shape() -> None:
    block = ResidualBlock(8)
    inputs = torch.zeros((2, 8, 10, 10))

    assert block(inputs).shape == inputs.shape


def test_compact_grid_extractor_accepts_different_canvas_sizes() -> None:
    extractor = CompactGridFeaturesExtractor(
        spaces.Box(0.0, 1.0, shape=(6, 14, 14)), features_dim=64
    )

    assert extractor(torch.zeros((2, 6, 14, 14))).shape == (2, 64)


def test_compact_hybrid_extractor_fuses_grid_and_vector_features() -> None:
    space = spaces.Dict(
        {
            "grid": spaces.Box(0.0, 1.0, shape=(6, 20, 20)),
            "vector": spaces.Box(0.0, 1.0, shape=(33,)),
        }
    )
    extractor = CompactHybridFeaturesExtractor(
        space,
        grid_features_dim=64,
        vector_features_dim=32,
    )

    features = extractor(
        {
            "grid": torch.zeros((2, 6, 20, 20)),
            "vector": torch.zeros((2, 33)),
        }
    )

    assert features.shape == (2, 96)
