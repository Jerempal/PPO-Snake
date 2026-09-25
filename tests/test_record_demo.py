import json
from types import SimpleNamespace

import numpy as np
from PIL import Image

from snake_rl.config import load_config
from snake_rl.record_demo import record_demo


def test_demo_discovers_run_config_and_saves_gif(tmp_path, monkeypatch):
    import stable_baselines3

    config = load_config("configs/ppo_8.toml")
    (tmp_path / "resolved_config.json").write_text(json.dumps(config.to_dict()))
    model_path = tmp_path / "best_model/best_model.zip"

    def predict(obs, **kwargs):
        assert obs["grid"].shape == (7, 8, 8)
        return np.asarray(1), None

    monkeypatch.setattr(
        stable_baselines3.PPO, "load", lambda path: SimpleNamespace(predict=predict)
    )
    target = tmp_path / "demo.gif"
    record_demo(
        model_path=model_path,
        config_path=None,
        output=target,
        seed=93000,
        steps=2,
        frame_interval=1,
    )
    with Image.open(target) as gif:
        assert gif.size == (644, 354)
        assert gif.n_frames == 2
