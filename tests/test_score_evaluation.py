import numpy as np

from snake_rl.score_evaluation import ScoreEvalCallback


class FakeLogger:
    def __init__(self) -> None:
        self.records = {}

    def record(self, name, value) -> None:
        self.records[name] = value


class FakeModel:
    def __init__(self) -> None:
        self.logger = FakeLogger()
        self.saved = []

    def predict(self, observation, deterministic):
        del observation, deterministic
        return np.array([0]), None

    def save(self, path) -> None:
        self.saved.append(path)


class FakeVecEnv:
    def __init__(self) -> None:
        self.index = 0
        self.num_envs = 1

    def reset(self):
        return np.zeros((1, 1))

    def seed(self, seed):
        self.index = seed % 2

    def step(self, action):
        del action
        score = (3, 5)[self.index % 2]
        self.index += 1
        return (
            np.zeros((1, 1)),
            np.array([float(score)]),
            np.array([True]),
            [{"score": score, "initial_length": 3 if score == 3 else 20}],
        )


class UnevenVecEnv:
    """Worker zero finishes every step; worker one only every third step."""

    def __init__(self) -> None:
        self.num_envs = 2
        self.steps = 0

    def reset(self):
        self.steps = 0
        return np.zeros((2, 1))

    def seed(self, seed):
        del seed
        self.steps = 0

    def step(self, action):
        del action
        self.steps += 1
        return (
            np.zeros((2, 1)),
            np.array([1.0, 9.0]),
            np.array([True, self.steps % 3 == 0]),
            [
                {"score": 1, "initial_length": 3},
                {"score": 9, "initial_length": 20},
            ],
        )


class BatchFakeModel(FakeModel):
    def predict(self, observation, deterministic):
        del deterministic
        return np.zeros(len(observation), dtype=np.int64), None


def test_score_evaluation_selects_score_and_stops_after_plateau(tmp_path) -> None:
    callback = ScoreEvalCallback(
        eval_env=FakeVecEnv(),
        eval_freq=1,
        n_eval_episodes=2,
        best_model_dir=tmp_path / "best_model",
        deterministic=True,
        patience=2,
        min_score_improvement=0.1,
        seed=10_000,
    )
    model = FakeModel()
    callback.model = model
    callback.num_timesteps = 0

    callback._on_training_start()

    assert callback.best_mean_score == 4
    assert model.saved == [tmp_path / "best_model" / "best_model"]
    assert model.logger.records["eval/mean_score"] == 4

    assert callback.history[0]["timesteps"] == 0
    assert callback.evaluations_without_improvement == 0

    callback.num_timesteps = 100
    callback.n_calls = 1
    assert callback._on_step()
    assert callback.history[1]["mean_score"] == callback.history[0]["mean_score"]
    callback.n_calls = 2
    assert not callback._on_step()
    summary = (tmp_path / "best_model" / "evaluation_summary.json").read_text()
    assert '"stopped_early": true' in summary


def test_score_evaluation_evaluates_final_policy_once(tmp_path) -> None:
    callback = ScoreEvalCallback(
        eval_env=FakeVecEnv(),
        eval_freq=100,
        n_eval_episodes=2,
        best_model_dir=tmp_path / "best_model",
        deterministic=True,
        patience=2,
        min_score_improvement=0.0,
        seed=10_000,
    )
    callback.model = FakeModel()
    callback.num_timesteps = 0
    callback._on_training_start()

    callback.num_timesteps = 500
    callback._on_training_end()
    callback._on_training_end()

    assert [record["timesteps"] for record in callback.history] == [0, 500]


def test_vector_evaluation_uses_equal_worker_quotas(tmp_path) -> None:
    callback = ScoreEvalCallback(
        eval_env=UnevenVecEnv(),
        eval_freq=1,
        n_eval_episodes=4,
        best_model_dir=tmp_path / "best_model",
        deterministic=True,
        patience=2,
        min_score_improvement=0.0,
        seed=10_000,
    )
    callback.model = BatchFakeModel()

    callback._on_training_start()

    assert callback.history[0]["mean_score"] == 5


def test_final_evaluation_detects_updates_without_new_transitions(tmp_path, monkeypatch):
    callback = ScoreEvalCallback(FakeVecEnv(), 1, 2, tmp_path / "best_model", True, 2, 0.0, 10_000)
    callback.model = FakeModel()
    callback.model._n_updates = 0
    callback.num_timesteps = 128
    callback.n_calls = 1
    callback._on_step()
    assert callback.best_mean_score == 4

    # SB3 optimise après la collecte, sans incrémenter num_timesteps.
    callback.model._n_updates = 1
    monkeypatch.setattr(callback, "_evaluate", lambda: {"mean_score": 9.0})
    callback._on_training_end()
    callback._on_training_end()
    assert [row["timesteps"] for row in callback.history] == [128, 128]
    assert callback.best_mean_score == 9
    assert len(callback.model.saved) == 2
