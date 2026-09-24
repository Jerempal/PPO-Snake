import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from snake_rl.config import ArchitectureConfig, EnvConfig, load_config, load_resolved_config
from snake_rl.env import SnakeEnv
from snake_rl.policy import CompactGridFeaturesExtractor
from snake_rl.train import _check_resume_architecture, policy_spec


@pytest.mark.parametrize("side", [8, 12, 20])
def test_no_border_preserves_all_playable_cells_including_corners(side):
    env = SnakeEnv(EnvConfig(width=side, height=side, grid_size=side))
    env.reset(seed=42)
    env.snake = [(side - 1, side - 1), (side - 2, side - 1), (side - 3, side - 1)]
    env.direction = (1, 0)
    env.food = (0, 0)
    obs = env._observation()
    assert obs["grid"].shape == (7, side, side)
    assert np.count_nonzero(obs["grid"][4]) == 0
    assert obs["grid"][0].sum() == obs["grid"][3].sum() == 1
    assert env.observation_space.contains(obs)


def test_legacy_json_retains_border_without_changing_file(tmp_path):
    raw = load_config("configs/ppo.toml").to_dict()
    del raw["environment"]["observation_border"]
    del raw["architecture"]
    path = tmp_path / "resolved_config.json"
    text = json.dumps(raw)
    path.write_text(text)
    legacy = load_resolved_config(path)
    canvas = raw["environment"]["grid_size"] + 2
    assert SnakeEnv(legacy.environment).observation_space["grid"].shape == (7, canvas, canvas)
    assert path.read_text() == text


@pytest.mark.parametrize("normalization", ["group", "batch", "none"])
def test_recorded_architecture_drives_the_actual_network(normalization, tmp_path):
    config = load_config("configs/ppo.toml")
    spec = ArchitectureConfig(
        cnn_channels=(8, 16, 24), strides=(1, 2, 2), pool_size=3, normalization=normalization
    )
    config = replace(config, architecture=spec)
    name, kwargs = policy_spec(config)
    env = SnakeEnv(config.environment)
    # Construction/save/load only: no learning or optimizer steps.
    model = PPO(name, env, policy_kwargs=kwargs, n_steps=2, batch_size=2, device="cpu")
    assert model.policy.features_extractor.grid_extractor.architecture == spec
    assert model.policy.features_extractor.grid_extractor.network[11].in_features == 24 * 9
    path = tmp_path / "policy.zip"
    obs, _ = env.reset(seed=0)
    before = model.predict(obs, deterministic=True)[0]
    model.save(path)
    loaded = PPO.load(path, device="cpu")
    assert loaded.policy.features_extractor.grid_extractor.architecture == spec
    _check_resume_architecture(loaded, config)
    with pytest.raises(ValueError, match="Resume architecture"):
        _check_resume_architecture(loaded, replace(config, architecture=replace(spec, pool_size=5)))
    np.testing.assert_array_equal(before, loaded.predict(obs, deterministic=True)[0])


def test_same_convolution_matches_padding_one_at_stride_one():
    first = torch.nn.Conv2d(7, 8, 3, padding="same")
    second = torch.nn.Conv2d(7, 8, 3, padding=1)
    second.load_state_dict(first.state_dict())
    inputs = torch.randn(2, 7, 12, 12)
    torch.testing.assert_close(first(inputs), second(inputs))


def test_adaptive_pooling_keeps_weight_shapes_when_board_size_changes():
    small = SnakeEnv(EnvConfig(width=8, height=8, grid_size=8))
    large = SnakeEnv(EnvConfig(width=20, height=20, grid_size=20))
    first = CompactGridFeaturesExtractor(small.observation_space["grid"])
    second = CompactGridFeaturesExtractor(large.observation_space["grid"])
    second.load_state_dict(first.state_dict(), strict=True)
    assert second(torch.zeros(1, 7, 20, 20)).shape == (1, 128)
