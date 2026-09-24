from collections import deque
from types import SimpleNamespace

from snake_rl.progress import TrainingProgressCallback


def test_progress_helpers_format_empty_and_populated_metrics() -> None:
    assert TrainingProgressCallback._mean(deque()) == "-"
    assert TrainingProgressCallback._mean(deque([1.0, 3.0])) == "2.00"
    assert TrainingProgressCallback._metric({}, "train/loss") == "-"
    assert TrainingProgressCallback._metric({"train/loss": 0.125}, "train/loss") == "0.125"


def test_progress_callback_tracks_completed_episodes(monkeypatch) -> None:
    class FakeProgress:
        def __init__(self) -> None:
            self.updated = 0
            self.postfix = {}
            self.closed = False
            self.description = ""

        def update(self, value: int) -> None:
            self.updated += value

        def set_postfix(self, values, refresh: bool) -> None:
            assert refresh
            self.postfix = values

        def close(self) -> None:
            self.closed = True

        def set_description(self, value: str, refresh: bool) -> None:
            assert refresh
            self.description = value

    class FakeLogger:
        def __init__(self) -> None:
            self.name_to_value = {}
            self.records = {}
            self.dumped_at = None

        def record(self, name, value) -> None:
            self.records[name] = value

        def dump(self, step: int) -> None:
            self.dumped_at = step

    progress = FakeProgress()
    monkeypatch.setattr("snake_rl.progress.tqdm", lambda **_: progress)
    callback = TrainingProgressCallback(total_timesteps=32, rollout_size=16, n_epochs=2)
    logger = FakeLogger()
    model = SimpleNamespace(num_timesteps=0, _n_updates=0, logger=logger)
    callback.model = model
    callback._on_training_start()
    callback._on_rollout_start()

    model.num_timesteps = 16
    callback.locals = {
        "dones": [True, False],
        "infos": [
            {
                "score": 4,
                "initial_length": 180,
                "looped": True,
                "collision": None,
                "episode": {"r": 1.5, "l": 20},
            },
            {},
        ],
    }
    assert callback._on_step()

    assert progress.updated == 16
    assert callback.episodes == 1
    assert progress.postfix["games"] == 1
    assert progress.postfix["score"] == "4.00"
    assert progress.postfix["loop"] == "1.00"

    model.logger.name_to_value = {
        "train/loss": 0.25,
        "train/approx_kl": 0.01,
    }
    model._n_updates = 2
    callback._on_rollout_end()
    assert progress.description == "PPO optimisation"
    callback._on_training_end()

    assert progress.postfix["loss"] == "0.25"
    assert progress.postfix["epochs_left"] == 2
    assert progress.closed
    assert logger.records["game/score_mean_100"] == 4
    assert logger.records["game/initial_length_mean_100"] == 180
    assert logger.records["game/loop_rate_100"] == 1
    assert logger.records["game/collision_rate_100"] == 0
    assert logger.dumped_at == 16
