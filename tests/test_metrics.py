import pytest

from snake_rl.metrics import bootstrap_mean_interval, summarize_episodes


def test_bootstrap_interval_is_reproducible_and_contains_mean() -> None:
    first = bootstrap_mean_interval([1, 2, 3, 4], seed=7, samples=200)
    second = bootstrap_mean_interval([1, 2, 3, 4], seed=7, samples=200)

    assert first == second
    assert first[0] <= 2.5 <= first[1]


def test_bootstrap_interval_validates_arguments() -> None:
    with pytest.raises(ValueError, match="vide"):
        bootstrap_mean_interval([], seed=1)
    with pytest.raises(ValueError, match="positif"):
        bootstrap_mean_interval([1], seed=1, samples=0)
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_mean_interval([1], seed=1, confidence=1.0)


def test_episode_summary_separates_standard_and_long_starts() -> None:
    summary = summarize_episodes(
        returns=[1.0, 2.0, 3.0],
        scores=[1, 2, 6],
        lengths=[10, 20, 30],
        initial_lengths=[3, 100, 120],
        termination_reasons=["horizon", "body", "win"],
        bootstrap_seed=10,
        longest_food_gaps=[8, 12, 20],
        scores_at_steps={1_000: [1, 2, 6]},
    )

    assert summary["score_mean"] == pytest.approx(3.0)
    assert summary["groups"]["standard"]["score_mean"] == pytest.approx(1.0)
    assert summary["groups"]["long"]["score_mean"] == pytest.approx(4.0)
    assert summary["collision_rate"] == pytest.approx(1 / 3)
    assert summary["horizon_completion_rate"] == pytest.approx(2 / 3)
    assert summary["longest_food_gap_mean"] == pytest.approx(40 / 3)
    assert summary["scores_at_steps"]["1000"]["mean"] == pytest.approx(3.0)
