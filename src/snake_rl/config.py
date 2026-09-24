"""Configuration typée du projet PPO hybride maintenu."""

from __future__ import annotations

import json
import math
import tomllib
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, TypeVar


@dataclass(frozen=True)
class RewardConfig:
    step: float = -0.001
    food: float = 1.0
    death: float = -2.0
    win: float = 5.0
    distance_beta: float = 0.3
    loop: float = -0.25

    def __post_init__(self) -> None:
        if any(not math.isfinite(value) for value in asdict(self).values()):
            raise ValueError("Les coefficients de récompense doivent être finis.")
        if self.distance_beta < 0:
            raise ValueError("distance_beta doit être positif ou nul.")

    def distance_coefficient(self, width: int, height: int) -> float:
        return self.distance_beta / (width + height - 2)


@dataclass(frozen=True)
class EnvConfig:
    width: int = 12
    height: int = 12
    grid_size: int = 12
    loop_max_state_visits: int | None = None
    observation_border: int = 0

    def __post_init__(self) -> None:
        if self.observation_border not in (0, 1):
            raise ValueError(
                "observation_border doit valoir 0 (actuel) ou 1 (anciens checkpoints)."
            )
        if self.width < 5 or self.height < 5:
            raise ValueError("Les dimensions de la carte doivent être d'au moins 5.")
        if self.grid_size < max(self.width, self.height):
            raise ValueError("grid_size doit couvrir les dimensions de la carte.")
        if self.loop_max_state_visits is not None and self.loop_max_state_visits < 2:
            raise ValueError("loop_max_state_visits doit être au moins égal à 2.")


@dataclass(frozen=True)
class StartStateConfig:
    """Départs synthétiques pour l'évaluation ; l'entraînement doit garder ``enabled`` à false."""

    enabled: bool = False
    standard_probability: float = 1.0
    min_length_fraction: float = 0.1
    max_length_fraction: float = 0.8
    length_distribution: str = "uniform"
    mean_length_fraction: float = 0.45
    std_length_fraction: float = 0.15

    def __post_init__(self) -> None:
        if not 0 <= self.standard_probability <= 1:
            raise ValueError("standard_probability doit être dans [0, 1].")
        if not 0 < self.min_length_fraction <= self.max_length_fraction < 1:
            raise ValueError(
                "Les fractions de longueur doivent respecter 0 < min_length_fraction <= "
                "max_length_fraction < 1."
            )
        if self.length_distribution not in {"uniform", "normal"}:
            raise ValueError("length_distribution doit être 'uniform' ou 'normal'.")
        if self.length_distribution == "normal":
            mean_is_bounded = (
                self.min_length_fraction <= self.mean_length_fraction <= self.max_length_fraction
            )
            if not mean_is_bounded:
                raise ValueError("mean_length_fraction doit être entre les bornes configurées.")
            if self.std_length_fraction <= 0:
                raise ValueError("std_length_fraction doit être positif.")


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    total_timesteps: int = 500_000
    n_envs: int = 16
    env_backend: str = "subprocess"
    learning_rate: float = 0.001
    learning_rate_schedule: str = "linear"
    final_learning_rate: float | None = 0.0001
    n_steps: int = 512
    batch_size: int = 512
    n_epochs: int = 15
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.15
    target_kl: float | None = 0.02
    ent_coef: float = 0.01
    hidden_size: int = 128
    hidden_layers: int = 2
    vf_coef: float = 0.5
    device: str = "auto"

    def __post_init__(self) -> None:
        for name in (
            "total_timesteps",
            "n_envs",
            "n_steps",
            "batch_size",
            "n_epochs",
            "hidden_size",
            "hidden_layers",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} doit être positif.")
        if self.hidden_size < 2:
            raise ValueError("hidden_size doit être au moins égal à 2.")
        if self.n_envs * self.n_steps <= 1 or self.batch_size <= 1:
            raise ValueError("PPO nécessite au moins deux échantillons par collecte et par lot.")
        if (self.n_envs * self.n_steps) % self.batch_size:
            raise ValueError("batch_size doit diviser n_envs * n_steps.")
        if self.env_backend not in {"dummy", "subprocess"}:
            raise ValueError("env_backend doit être 'dummy' ou 'subprocess'.")
        if self.learning_rate_schedule not in {"constant", "linear"}:
            raise ValueError("learning_rate_schedule doit être 'constant' ou 'linear'.")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate doit être fini et positif.")
        if self.learning_rate_schedule == "linear":
            if self.final_learning_rate is None or not (
                math.isfinite(self.final_learning_rate) and self.final_learning_rate > 0
            ):
                raise ValueError("Un planning linéaire nécessite un final_learning_rate positif.")
        elif self.final_learning_rate is not None:
            raise ValueError("final_learning_rate nécessite un planning linéaire.")
        if not 0 < self.gamma <= 1 or not 0 < self.gae_lambda <= 1:
            raise ValueError("gamma et gae_lambda doivent être dans (0, 1].")
        if not 0 < self.clip_range < 1:
            raise ValueError("clip_range doit être dans (0, 1).")
        if self.target_kl is not None and not (
            math.isfinite(self.target_kl) and self.target_kl > 0
        ):
            raise ValueError("target_kl doit être fini et positif.")
        if any(not math.isfinite(v) or v < 0 for v in (self.ent_coef, self.vf_coef)):
            raise ValueError("ent_coef et vf_coef doivent être finis et positifs ou nuls.")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device doit être auto, cpu ou cuda.")


@dataclass(frozen=True)
class EvaluationConfig:
    frequency: int = 75_000
    episodes: int = 30
    n_envs: int = 8
    deterministic: bool = True
    patience: int = 80
    min_score_improvement: float = 0.0
    horizon: int = 10_000

    def __post_init__(self) -> None:
        for name in ("frequency", "episodes", "n_envs", "patience", "horizon"):
            if getattr(self, name) < 1:
                raise ValueError(f"Evaluation {name} doit être positif.")
        if not math.isfinite(self.min_score_improvement) or self.min_score_improvement < 0:
            raise ValueError("min_score_improvement doit être fini et positif ou nul.")


@dataclass(frozen=True)
class CheckpointConfig:
    frequency: int = 100_000

    def __post_init__(self) -> None:
        if self.frequency < 1:
            raise ValueError("La fréquence des checkpoints doit être positive.")


@dataclass(frozen=True)
class TrajectoryRestartConfig:
    """Départs optionnels capturés dans les trajectoires de la politique."""

    enabled: bool = False
    fractions: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75)
    standard_probability: float = 0.25
    promotion_window: int = 100
    promotion_success_rate: float = 0.25
    reservoir_size: int = 32

    def __post_init__(self) -> None:
        object.__setattr__(self, "fractions", tuple(self.fractions))
        if not self.fractions or any(not 0 < f < 1 for f in self.fractions):
            raise ValueError("Les fractions de trajectoire doivent être dans (0, 1).")
        if any(a >= b for a, b in zip(self.fractions, self.fractions[1:], strict=False)):
            raise ValueError("Les fractions de trajectoire doivent être strictement croissantes.")
        if not 0 < self.standard_probability < 1:
            raise ValueError("La probabilité de départ standard doit être dans (0, 1).")
        if not 0 < self.promotion_success_rate <= 1:
            raise ValueError("Le taux de promotion doit être dans (0, 1].")
        for name in ("promotion_window", "reservoir_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"Trajectory {name} doit être un entier positif.")

    def lengths(self, area: int) -> tuple[int, ...]:
        """L'occupation inclut le corps initial, pas seulement les pommes mangées."""
        lengths = tuple(math.ceil(fraction * area) for fraction in self.fractions)
        if lengths[0] <= 3 or lengths[-1] >= area or len(set(lengths)) != len(lengths):
            raise ValueError(
                "Les fractions doivent donner des longueurs distinctes entre 3 et area."
            )
        return lengths


@dataclass(frozen=True)
class ArchitectureConfig:
    """Construction du CNN ; les tailles des branches et des MLP sont dans training."""

    cnn_channels: tuple[int, ...] = (24, 32, 48)
    strides: tuple[int, ...] = (1, 1, 1)
    pool_size: int = 9
    residual_blocks: bool = True
    normalization: str = "group"
    groups: int = 4

    def __post_init__(self) -> None:
        object.__setattr__(self, "cnn_channels", tuple(self.cnn_channels))
        object.__setattr__(self, "strides", tuple(self.strides))
        if len(self.cnn_channels) != 3 or any(c < 1 for c in self.cnn_channels):
            raise ValueError("cnn_channels nécessite trois nombres de canaux positifs.")
        if len(self.strides) != 3 or any(s not in (1, 2) for s in self.strides):
            raise ValueError("strides nécessite trois valeurs, chacune égale à 1 ou 2.")
        if self.pool_size < 1 or self.groups < 1:
            raise ValueError("pool_size et groups doivent être positifs.")
        if self.normalization not in ("none", "group", "batch"):
            raise ValueError("normalization doit être none, group ou batch.")
        if self.normalization == "group" and any(c % self.groups for c in self.cnn_channels):
            raise ValueError("Chaque nombre de canaux CNN doit être divisible par groups.")


@dataclass(frozen=True)
class ExperimentConfig:
    environment: EnvConfig
    rewards: RewardConfig
    start_states: StartStateConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    checkpoint: CheckpointConfig
    trajectory_restart: TrajectoryRestartConfig = field(default_factory=TrajectoryRestartConfig)
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    map_sizes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "map_sizes", tuple(self.map_sizes))
        if self.map_sizes:
            if len(set(self.map_sizes)) != len(self.map_sizes) or len(self.map_sizes) < 2:
                raise ValueError("map_sizes doit contenir au moins deux tailles distinctes.")
            if self.training.n_envs % len(self.map_sizes):
                raise ValueError("n_envs doit être divisible par le nombre de map_sizes.")
            if self.evaluation.n_envs < len(self.map_sizes):
                raise ValueError("L'évaluation nécessite au moins un worker par taille de carte.")
            for size in self.map_sizes:
                replace(self.environment, width=size, height=size)
                if self.trajectory_restart.enabled:
                    self.trajectory_restart.lengths(size * size)
        if self.start_states.enabled:
            raise ValueError(
                "Structured starts are evaluation-only; use trajectory_restart "
                "for captured training starts."
            )
        if self.trajectory_restart.enabled:
            self.trajectory_restart.lengths(self.environment.width * self.environment.height)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["trajectory_restart"]["fractions"] = list(self.trajectory_restart.fractions)
        payload["architecture"]["cnn_channels"] = list(self.architecture.cnn_channels)
        payload["architecture"]["strides"] = list(self.architecture.strides)
        payload["map_sizes"] = list(self.map_sizes)
        return payload


ConfigType = TypeVar("ConfigType")


def _build_section(cls: type[ConfigType], values: dict[str, Any], section: str) -> ConfigType:
    unknown = sorted(set(values) - {field.name for field in fields(cls)})
    if unknown:
        raise ValueError(f"Unknown keys in [{section}]: {', '.join(unknown)}")
    return cls(**values)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a TOML configuration; unsupported experiment keys fail explicitly."""
    with Path(path).open("rb") as stream:
        return experiment_config_from_dict(tomllib.load(stream))


def load_resolved_config(path: str | Path) -> ExperimentConfig:
    """Load a current-schema JSON configuration saved alongside a model."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    # Historical JSON runs used a one-cell border. Never reinterpret their input.
    raw["environment"].setdefault("observation_border", 1)
    return experiment_config_from_dict(raw)


def experiment_config_from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    raw = dict(raw)
    map_sizes = raw.pop("map_sizes", ())
    sections = {
        "environment": EnvConfig,
        "rewards": RewardConfig,
        "start_states": StartStateConfig,
        "training": TrainingConfig,
        "evaluation": EvaluationConfig,
        "checkpoint": CheckpointConfig,
        "trajectory_restart": TrajectoryRestartConfig,
        "architecture": ArchitectureConfig,
    }
    required = set(sections) - {"start_states", "trajectory_restart", "architecture"}
    if set(raw) - set(sections):
        raise ValueError(
            f"Unknown configuration sections: {', '.join(sorted(set(raw) - set(sections)))}"
        )
    if required - set(raw):
        raise ValueError(
            f"Sections de configuration manquantes : {', '.join(sorted(required - set(raw)))}"
        )
    return ExperimentConfig(
        map_sizes=map_sizes,
        **{name: _build_section(cls, raw.get(name, {}), name) for name, cls in sections.items()},
    )


def evaluation_start_states(start_states: StartStateConfig, mode: str) -> StartStateConfig:
    """Resolve the evaluation cohort independently of the training distribution."""
    if mode == "standard":
        return StartStateConfig()
    if mode == "configured":
        return replace(start_states, enabled=True)
    if mode == "long":
        return replace(start_states, enabled=True, standard_probability=0.0)
    raise ValueError("start_state_mode doit être 'standard', 'configured' ou 'long'.")


def with_environment_overrides(
    config: ExperimentConfig,
    *,
    map_size: int | None = None,
    grid_size: int | None = None,
    width: int | None = None,
    height: int | None = None,
) -> ExperimentConfig:
    """Change la taille jouable indépendamment du canvas d'observation appris."""
    current = config.environment
    if map_size is not None and (width is not None or height is not None):
        raise ValueError("Utilisez map-size ou width/height, pas les deux.")
    return replace(
        config,
        environment=replace(
            current,
            width=map_size if map_size is not None else (current.width if width is None else width),
            height=(
                map_size if map_size is not None else (current.height if height is None else height)
            ),
            grid_size=current.grid_size if grid_size is None else grid_size,
        ),
    )
