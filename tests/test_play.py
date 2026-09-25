from types import SimpleNamespace

import pytest

from snake_rl.config import load_config
from snake_rl.env import RelativeAction, SnakeEnv
from snake_rl.play import play


@pytest.mark.parametrize("keyboard", [False, True])
@pytest.mark.parametrize("with_model", [False, True])
def test_play_handles_keyboard_and_idle_frames(monkeypatch, keyboard, with_model):
    import stable_baselines3

    events = iter(
        [
            [SimpleNamespace(type=2, key=10)] if keyboard else [],
            [SimpleNamespace(type=1)],
        ]
    )
    pygame = SimpleNamespace(
        K_UP=10,
        K_DOWN=11,
        K_LEFT=12,
        K_RIGHT=13,
        K_ESCAPE=14,
        QUIT=1,
        KEYDOWN=2,
        event=SimpleNamespace(get=lambda: next(events)),
    )
    monkeypatch.setitem(__import__("sys").modules, "pygame", pygame)
    actions = []

    class HeadlessEnv(SnakeEnv):
        def __init__(self, config, rewards, **kwargs):
            super().__init__(config, rewards)

        def step(self, action):
            actions.append(action)
            return super().step(action)

    monkeypatch.setattr("snake_rl.play.SnakeEnv", HeadlessEnv)
    model = SimpleNamespace(predict=lambda *args, **kwargs: (RelativeAction.RIGHT, None))
    monkeypatch.setattr(stable_baselines3.PPO, "load", lambda path: model)
    play(load_config("configs/ppo_8.toml"), "model.zip" if with_model else None, 42)
    expected = (
        RelativeAction.RIGHT
        if with_model
        else (RelativeAction.LEFT if keyboard else RelativeAction.STRAIGHT)
    )
    assert actions == [expected]
