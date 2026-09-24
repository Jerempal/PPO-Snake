"""Shared episode metrics for neural and deterministic policies."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def bootstrap_mean_interval(
    values: Sequence[int] | Sequence[float],
    *,
    seed: int,
    samples: int = 2_000,
    confidence: float = 0.95,
) -> list[float]:
    """Retourne un intervalle bootstrap déterministe pour la moyenne."""
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        raise ValueError("values ne peut pas être vide.")
    if samples < 1:
        raise ValueError("samples doit être positif.")
    if not 0 < confidence < 1:
        raise ValueError("confidence doit être dans (0, 1).")
    generator = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        means[index] = generator.choice(array, size=len(array), replace=True).mean()
    tail = (1.0 - confidence) / 2.0
    return [float(np.quantile(means, tail)), float(np.quantile(means, 1.0 - tail))]


def summarize_episodes(
    *,
    returns: Sequence[float],
    scores: Sequence[int],
    lengths: Sequence[int],
    initial_lengths: Sequence[int],
    termination_reasons: Sequence[str],
    bootstrap_seed: int,
    longest_food_gaps: Sequence[int] | None = None,
    scores_at_steps: dict[int, Sequence[int]] | None = None,
) -> dict[str, object]:
    """Build the common evaluation payload used by every policy family."""
    score_array = np.asarray(scores)
    length_array = np.asarray(lengths)
    initial_array = np.asarray(initial_lengths)
    reason_array = np.asarray(termination_reasons)

    def group_metrics(indices: np.ndarray, seed_offset: int) -> dict[str, object] | None:
        if not np.any(indices):
            return None
        group_scores = score_array[indices]
        group_reasons = reason_array[indices]
        return {
            "episodes": int(np.sum(indices)),
            "score_mean": float(np.mean(group_scores)),
            "score_mean_ci95": bootstrap_mean_interval(
                group_scores, seed=bootstrap_seed + seed_offset
            ),
            "score_median": float(np.median(group_scores)),
            "score_max": int(np.max(group_scores)),
            "episode_length_mean": float(np.mean(length_array[indices])),
            "termination_counts": {
                reason: int(np.sum(group_reasons == reason))
                for reason in sorted(set(group_reasons))
            },
        }

    standard_indices = initial_array == 3
    long_indices = ~standard_indices
    total_steps = int(np.sum(length_array))
    result: dict[str, object] = {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "score_mean": float(np.mean(score_array)),
        "score_mean_ci95": bootstrap_mean_interval(scores, seed=bootstrap_seed),
        "score_max": int(np.max(score_array)),
        "episode_length_mean": float(np.mean(length_array)),
        "apples_per_1000_steps": (
            1_000.0 * float(np.sum(score_array)) / total_steps if total_steps else 0.0
        ),
        "initial_length_mean": float(np.mean(initial_array)),
        "groups": {
            "standard": group_metrics(standard_indices, 1),
            "long": group_metrics(long_indices, 2),
        },
        "termination_counts": {
            reason: termination_reasons.count(reason) for reason in sorted(set(termination_reasons))
        },
        "returns": list(returns),
        "scores": list(scores),
        "episode_lengths": list(lengths),
        "initial_lengths": list(initial_lengths),
        "termination_reasons": list(termination_reasons),
    }
    result["collision_rate"] = float(np.mean(np.isin(reason_array, ["wall", "body"])))
    result["horizon_completion_rate"] = float(np.mean(np.isin(reason_array, ["horizon", "win"])))
    if longest_food_gaps is not None:
        gap_array = np.asarray(longest_food_gaps)
        result["longest_food_gap_mean"] = float(np.mean(gap_array))
        result["longest_food_gap_max"] = int(np.max(gap_array))
        result["longest_food_gaps"] = list(longest_food_gaps)
    if scores_at_steps is not None:
        result["scores_at_steps"] = {
            str(step): {
                "mean": float(np.mean(values)),
                "mean_ci95": bootstrap_mean_interval(
                    values,
                    seed=bootstrap_seed + 10_000 + index,
                ),
                "median": float(np.median(values)),
                "max": int(np.max(values)),
                "scores": list(values),
            }
            for index, (step, values) in enumerate(sorted(scores_at_steps.items()))
        }
    return result
