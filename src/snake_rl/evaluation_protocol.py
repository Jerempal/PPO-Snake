"""Règles explicites et partagées pour l'évaluation finale des politiques."""

from __future__ import annotations

from dataclasses import replace

from snake_rl.config import ExperimentConfig

FIXED_HORIZON = "fixed-horizon"
PROTOCOLS = (FIXED_HORIZON,)


def prepare_evaluation(
    config: ExperimentConfig,
    *,
    protocol: str,
    horizon: int,
) -> tuple[ExperimentConfig, int | None]:
    """Retourne une configuration d'évaluation équitable et son horizon d'actions."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"Protocole d'évaluation inconnu : {protocol!r}.")
    if horizon < 1:
        raise ValueError("L'horizon doit être positif.")

    environment = replace(
        config.environment,
        loop_max_state_visits=None,
    )
    return replace(config, environment=environment), horizon
