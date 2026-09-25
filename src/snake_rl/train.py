"""Entraîne PPO et sauvegarde sa configuration, ses checkpoints et ses métriques."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from snake_rl.cli import add_map_arguments, apply_map_arguments
from snake_rl.config import ExperimentConfig, load_config
from snake_rl.env import SnakeEnv


def policy_spec(config: ExperimentConfig) -> tuple[str, dict[str, Any]]:
    from snake_rl.policy import CompactHybridFeaturesExtractor

    return "MultiInputPolicy", {
        "features_extractor_class": CompactHybridFeaturesExtractor,
        "features_extractor_kwargs": {
            "architecture": config.to_dict()["architecture"],
            "grid_features_dim": config.training.hidden_size,
            "vector_features_dim": config.training.hidden_size // 2,
        },
        "net_arch": [config.training.hidden_size] * config.training.hidden_layers,
        "normalize_images": False,
        "share_features_extractor": True,
    }


def _learning_rate(config: ExperimentConfig) -> float | Callable[[float], float]:
    if config.training.learning_rate_schedule == "constant":
        return config.training.learning_rate

    initial = config.training.learning_rate
    final = config.training.final_learning_rate

    def linear_schedule(progress_remaining: float) -> float:
        return final + progress_remaining * (initial - final)

    return linear_schedule


def _check_resume_architecture(model: Any, config: ExperimentConfig) -> None:
    """Do not write a configuration that misdescribes the loaded policy."""
    features = model.policy.features_extractor
    expected_mlp = [config.training.hidden_size] * config.training.hidden_layers
    if (
        features.grid_extractor.architecture != config.architecture
        or features.grid_extractor.features_dim != config.training.hidden_size
        or features.features_dim != config.training.hidden_size + config.training.hidden_size // 2
        or model.policy.net_arch != expected_mlp
    ):
        raise ValueError(
            "L'architecture chargée diffère de la configuration demandée. "
            "Utilisez la même architecture ou une initialisation explicite de la politique."
        )


def _write_run_manifest(
    run_dir: Path,
    config: ExperimentConfig,
) -> None:
    """Save the config needed to reload a model and a small lifecycle status."""
    (run_dir / "resolved_config.json").write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metadata: dict[str, Any] = {
        "status": "running",
        "created_at": datetime.now(UTC).isoformat(),
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _finalize_run_manifest(run_dir: Path, *, status: str) -> None:
    """Update only lifecycle information; checkpoints need no separate manifest."""
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "status": status,
            "completed_at": datetime.now(UTC).isoformat(),
        }
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def train(
    config: ExperimentConfig,
    output_dir: Path,
    run_name: str | None = None,
    resume_from: Path | None = None,
    initialize_policy_from: Path | None = None,
    initialize_only: bool = False,
) -> Path:
    """Run PPO training and return the experiment directory."""
    if resume_from is not None and initialize_policy_from is not None:
        raise ValueError("resume_from et initialize_policy_from sont exclusifs.")
    if initialize_only and initialize_policy_from is None:
        raise ValueError("initialize_only nécessite initialize_policy_from.")
    if resume_from is not None:
        resume_from = resume_from.resolve()
        if not resume_from.is_file():
            raise FileNotFoundError(f"Modèle parent introuvable : {resume_from}")
    if initialize_policy_from is not None:
        initialize_policy_from = initialize_policy_from.resolve()
        if not initialize_policy_from.is_file():
            raise FileNotFoundError(
                f"Modèle d'initialisation introuvable : {initialize_policy_from}"
            )
    try:
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import (
            CallbackList,
            CheckpointCallback,
        )
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.logger import configure
        from stable_baselines3.common.utils import set_random_seed
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except ImportError as exc:
        raise ImportError("L'entraînement nécessite : pip install 'snake-rl[train]'") from exc

    print(
        "Périphérique PyTorch : "
        f"demandé={config.training.device}, "
        f"cuda_disponible={torch.cuda.is_available()}, "
        f"torch_cuda={torch.version.cuda}, "
        f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}"
    )

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    resolved_name = run_name or f"ppo-{timestamp}-seed{config.training.seed}"
    run_dir = output_dir.resolve() / resolved_name
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_run_manifest(run_dir, config)

    training_env = None
    evaluation_env = None
    run_status = "failed"
    try:
        set_random_seed(config.training.seed)
        vec_env_class = (
            SubprocVecEnv if config.training.env_backend == "subprocess" else DummyVecEnv
        )
        restart_kwargs = {}
        if config.trajectory_restart.enabled:
            from snake_rl.training.trajectory_restart import TrajectoryRestartWrapper

            restart_kwargs = {
                "wrapper_class": TrajectoryRestartWrapper,
                "wrapper_kwargs": {"config": config.trajectory_restart},
            }
            if resume_from is not None or initialize_policy_from is not None:
                warnings.warn(
                    "Les réservoirs et paliers de trajectoires repartent de zéro dans ce run ; "
                    "ils ne sont pas restaurés depuis les checkpoints.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if config.map_sizes:
            from snake_rl.training.multimap import make_multimap_env

            training_env, map_assignment = make_multimap_env(config, run_dir)
        else:
            training_env = make_vec_env(
                SnakeEnv,
                n_envs=config.training.n_envs,
                seed=config.training.seed,
                env_kwargs={"env_config": config.environment, "reward_config": config.rewards},
                monitor_dir=str(run_dir / "monitor"),
                vec_env_cls=vec_env_class,
                **restart_kwargs,
            )
        from gymnasium.wrappers import TimeLimit

        from snake_rl.evaluation_protocol import FIXED_HORIZON, prepare_evaluation

        validation_config, horizon = prepare_evaluation(
            config,
            protocol=FIXED_HORIZON,
            horizon=config.evaluation.horizon,
        )
        evaluation_maps = None
        if config.map_sizes:
            evaluation_env, evaluation_maps = make_multimap_env(config, run_dir, evaluation=True)
        else:
            evaluation_env = make_vec_env(
                SnakeEnv,
                n_envs=min(config.evaluation.n_envs, config.evaluation.episodes),
                seed=config.training.seed + 10_000,
                env_kwargs={
                    "env_config": validation_config.environment,
                    "reward_config": config.rewards,
                },
                wrapper_class=TimeLimit,
                wrapper_kwargs={"max_episode_steps": horizon},
                vec_env_cls=vec_env_class,
            )

        vector_steps = config.training.n_envs
        checkpoint_callback = CheckpointCallback(
            save_freq=max(config.checkpoint.frequency // vector_steps, 1),
            save_path=str(run_dir / "checkpoints"),
            name_prefix="ppo_snake",
        )
        from snake_rl.score_evaluation import ScoreEvalCallback

        evaluation_callback = ScoreEvalCallback(
            eval_env=evaluation_env,
            best_model_dir=run_dir / "best_model",
            eval_freq=max(config.evaluation.frequency // vector_steps, 1),
            n_eval_episodes=config.evaluation.episodes * (len(config.map_sizes) or 1),
            map_sizes=evaluation_maps,
            deterministic=config.evaluation.deterministic,
            patience=config.evaluation.patience,
            min_score_improvement=config.evaluation.min_score_improvement,
            seed=config.training.seed + 10_000,
        )

        from snake_rl.progress import TrainingProgressCallback

        metrics_dir = run_dir / "metrics"
        progress_callback = TrainingProgressCallback(
            total_timesteps=config.training.total_timesteps,
            rollout_size=config.training.n_envs * config.training.n_steps,
            n_epochs=config.training.n_epochs,
        )
        training_callbacks = [
            checkpoint_callback,
            evaluation_callback,
            progress_callback,
        ]
        if config.trajectory_restart.enabled:
            from snake_rl.training.trajectory_restart import TrajectoryRestartCallback

            if config.map_sizes:
                for size in config.map_sizes:
                    indices = [i for i, value in enumerate(map_assignment) if value == size]
                    training_callbacks.append(
                        TrajectoryRestartCallback(
                            config.trajectory_restart,
                            metrics_dir / f"trajectory_restart_{size}.json",
                            indices=indices,
                            prefix=f"trajectory/{size}",
                        )
                    )
            else:
                training_callbacks.append(
                    TrajectoryRestartCallback(
                        config.trajectory_restart, metrics_dir / "trajectory_restart.json"
                    )
                )
        if config.map_sizes:
            from snake_rl.training.multimap import MapExposureCallback

            training_callbacks.append(
                MapExposureCallback(map_assignment, metrics_dir / "map_exposure.json")
            )
        if resume_from is None:
            policy_name, policy_kwargs = policy_spec(config)
            model = PPO(
                policy_name,
                training_env,
                seed=config.training.seed,
                learning_rate=_learning_rate(config),
                n_steps=config.training.n_steps,
                batch_size=config.training.batch_size,
                n_epochs=config.training.n_epochs,
                gamma=config.training.gamma,
                gae_lambda=config.training.gae_lambda,
                clip_range=config.training.clip_range,
                target_kl=config.training.target_kl,
                ent_coef=config.training.ent_coef,
                vf_coef=config.training.vf_coef,
                policy_kwargs=policy_kwargs,
                device=config.training.device,
                tensorboard_log=str(run_dir / "tensorboard"),
                verbose=0,
            )
            if initialize_policy_from is not None:
                source_model = PPO.load(
                    initialize_policy_from,
                    device=config.training.device,
                )
                try:
                    model.policy.load_state_dict(
                        source_model.policy.state_dict(),
                        strict=True,
                    )
                except RuntimeError as exc:
                    raise ValueError(
                        "Le modèle d'initialisation est incompatible avec l'architecture cible. "
                        "Les dimensions des observations peuvent différer uniquement "
                        "si chaque tenseur appris "
                        "conserve la même forme."
                    ) from exc
                model.save(run_dir / "initialized_model")
        else:
            model = PPO.load(
                resume_from,
                env=training_env,
                device=config.training.device,
                tensorboard_log=str(run_dir / "tensorboard"),
                learning_rate=_learning_rate(config),
                n_steps=config.training.n_steps,
                batch_size=config.training.batch_size,
                n_epochs=config.training.n_epochs,
                gamma=config.training.gamma,
                gae_lambda=config.training.gae_lambda,
                clip_range=config.training.clip_range,
                ent_coef=config.training.ent_coef,
                vf_coef=config.training.vf_coef,
                target_kl=config.training.target_kl,
                verbose=0,
            )
            _check_resume_architecture(model, config)

        if not initialize_only:
            model.set_logger(configure(str(metrics_dir), ["csv", "tensorboard"]))
            model.learn(
                total_timesteps=config.training.total_timesteps,
                callback=CallbackList(training_callbacks),
                tb_log_name="ppo",
                reset_num_timesteps=True,
            )
            model.save(run_dir / "final_model")
        run_status = "complete"
    finally:
        active_error = sys.exc_info()[0] is not None
        close_errors = []
        for environment in (training_env, evaluation_env):
            if environment is None:
                continue
            try:
                environment.close()
            except Exception as exc:  # pragma: no cover - third-party cleanup failure
                close_errors.append(exc)
        try:
            _finalize_run_manifest(run_dir, status=run_status)
        except Exception as exc:
            if not active_error:
                raise
            warnings.warn(
                f"Impossible de finaliser les métadonnées du run échoué : {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
        if close_errors:
            if not active_error:
                raise close_errors[0]
            warnings.warn(
                f"Impossible de fermer un environnement après un échec : {close_errors[0]}",
                RuntimeWarning,
                stacklevel=2,
            )

    outcome = (
        "Initialisation de la politique terminée" if initialize_only else "Entraînement terminé"
    )
    print(f"{outcome}. Fichiers : {run_dir}")
    if not initialize_only:
        from snake_rl.report import generate_reports

        try:
            generate_reports([run_dir])
        except Exception as exc:
            # A plotting error must not invalidate a successfully saved model.
            warnings.warn(
                f"Entraînement sauvegardé, mais le rapport automatique a échoué : {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ppo.toml"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--run-name", help="Nom facultatif et stable du run.")
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Reprendre l'entraînement depuis un checkpoint PPO dans un nouveau dossier.",
    )
    parser.add_argument(
        "--initialize-policy-from",
        type=Path,
        help=(
            "Créer un modèle PPO cible et copier seulement les poids de cette politique. "
            "L'état de l'optimiseur et le calendrier ne sont pas repris."
        ),
    )
    parser.add_argument(
        "--initialize-only",
        action="store_true",
        help="Sauvegarder le modèle initialisé sans collecter de nouvelles transitions.",
    )
    parser.add_argument(
        "--timesteps", type=int, help="Remplacer le nombre total de transitions de ce run."
    )
    parser.add_argument("--seed", type=int, help="Remplacer la graine d'entraînement de ce run.")
    parser.add_argument(
        "--n-epochs", type=int, help="Remplacer le nombre d'époques d'optimisation PPO."
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        help="Remplacer le périphérique PyTorch ; CUDA échoue si aucun GPU n'est disponible.",
    )
    parser.add_argument(
        "--distance-beta", type=float, help="Coefficient de guidage par distance normalisée."
    )
    parser.add_argument("--food-reward", type=float, help="Récompense par pomme.")
    parser.add_argument(
        "--trajectory-restart",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Activer/désactiver les reprises de trajectoires (entraînement uniquement).",
    )
    parser.add_argument(
        "--report-evaluate",
        action="store_true",
        help="Évaluer aussi les modèles après l'entraînement (30 parties chacun).",
    )
    add_map_arguments(parser)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.resume_from is not None and args.initialize_policy_from is not None:
        parser.error("--resume-from et --initialize-policy-from sont exclusifs.")
    if args.initialize_only and args.initialize_policy_from is None:
        parser.error("--initialize-only nécessite --initialize-policy-from.")
    config = apply_map_arguments(load_config(args.config), args)
    if args.timesteps is not None:
        if args.timesteps < 1:
            raise ValueError("--timesteps doit être positif.")
        config = replace(config, training=replace(config.training, total_timesteps=args.timesteps))
    if args.seed is not None:
        config = replace(config, training=replace(config.training, seed=args.seed))
    if args.n_epochs is not None:
        if args.n_epochs < 1:
            raise ValueError("--n-epochs doit être positif.")
        config = replace(config, training=replace(config.training, n_epochs=args.n_epochs))
    if args.device is not None:
        config = replace(config, training=replace(config.training, device=args.device))
    if args.trajectory_restart is not None:
        config = replace(
            config,
            trajectory_restart=replace(config.trajectory_restart, enabled=args.trajectory_restart),
        )
    reward_overrides = {"food": args.food_reward, "distance_beta": args.distance_beta}
    configured_rewards = {
        name: value for name, value in reward_overrides.items() if value is not None
    }
    if configured_rewards:
        config = replace(config, rewards=replace(config.rewards, **configured_rewards))
    run_dir = train(
        config,
        args.output_dir,
        args.run_name,
        resume_from=args.resume_from,
        initialize_policy_from=args.initialize_policy_from,
        initialize_only=args.initialize_only,
    )
    if not args.initialize_only and args.report_evaluate:
        from snake_rl.report import generate_reports

        generate_reports([run_dir], evaluate_models=True)


if __name__ == "__main__":
    main()
