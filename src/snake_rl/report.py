"""Génère les graphiques et un résumé ; évalue éventuellement les modèles."""

import argparse
import json
from html import escape
from pathlib import Path

from snake_rl.config import load_resolved_config
from snake_rl.evaluate import evaluate
from snake_rl.evaluation_protocol import FIXED_HORIZON, prepare_evaluation
from snake_rl.run_report import _evaluation_points, render_run_report


def generate_reports(
    run_dirs: list[Path],
    *,
    output_dir: Path | None = None,
    evaluate_models: bool = False,
    episodes: int = 30,
    seed: int = 71_000,
    horizon: int = 10_000,
    n_envs: int = 16,
) -> list[Path]:
    """Lit les journaux existants ; --evaluate relance toujours les parties."""
    if not run_dirs or min(episodes, horizon, n_envs) < 1:
        raise ValueError(
            "Les runs et les nombres de parties, d'actions et de workers doivent être positifs."
        )
    if len({path.resolve() for path in run_dirs}) != len(run_dirs):
        raise ValueError("Les dossiers de runs sont dupliqués.")
    if output_dir and len({path.name for path in run_dirs}) != len(run_dirs):
        raise ValueError("Les noms de runs doivent être uniques dans le dossier de sortie.")
    outputs = []
    for run in run_dirs:
        config = load_resolved_config(run / "resolved_config.json")
        if config.map_sizes and evaluate_models:
            raise ValueError(
                "Multi-cartes : évaluez chaque taille avec snake-evaluate --map-size N."
            )
        destination = output_dir / run.name if output_dir else run / "report"
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "training.svg").write_text(render_run_report([run]), encoding="utf-8")
        metric = "selection_score" if config.map_sizes else "mean_score"
        history = _evaluation_points(run, metric)
        lines = [
            f"# {run.name}",
            "",
            f"Cartes : {config.map_sizes or (config.environment.width,)} ; "
            f"canvas {config.environment.grid_size} ; graine {config.training.seed}.",
            "",
        ]
        if history:
            step, score = max(history, key=lambda point: point[1])
            unit = "normalisé (poids égaux)" if config.map_sizes else "pommes"
            lines += [
                f"Meilleure validation : {score:.4f} {unit} à {int(step):,} transitions.",
                f"Validation finale : {history[-1][1]:.4f}.",
                "",
            ]
        summary = run / "best_model/evaluation_summary.json"
        if config.map_sizes and summary.exists():
            records = json.loads(summary.read_text(encoding="utf-8")).get("history", [])
            if records:
                best = max(records, key=lambda row: row["selection_score"])
                lines += [
                    "| Carte | Pommes au meilleur checkpoint | Victoires | Pommes finales |",
                    "| --- | ---: | ---: | ---: |",
                ]
                for size in config.map_sizes:
                    lines.append(
                        f"| {size}x{size} | {best[f'map_{size}_mean_score']:.2f} | "
                        f"{best[f'map_{size}_win_rate']:.1%} | "
                        f"{records[-1][f'map_{size}_mean_score']:.2f} |"
                    )
                lines.append("")
        if evaluate_models:
            effective, limit = prepare_evaluation(config, protocol=FIXED_HORIZON, horizon=horizon)
            lines += [
                f"Test depuis le départ standard : {episodes} parties, "
                f"graine {seed}, horizon {horizon}.",
                "",
                "| Checkpoint | Pommes moyennes | Victoires |",
                "| --- | ---: | ---: |",
            ]
            for label, model in (
                ("best", run / "best_model/best_model.zip"),
                ("final", run / "final_model.zip"),
            ):
                if not model.is_file():
                    lines.append(f"| {label} | Indisponible | — |")
                    continue
                metrics = evaluate(
                    model,
                    effective,
                    episodes,
                    seed,
                    True,
                    n_envs=n_envs,
                    max_episode_steps=limit,
                    protocol=FIXED_HORIZON,
                )
                target = destination / f"{label}-evaluation.json"
                target.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
                lines.append(
                    f"| [{label}]({target.name}) | {metrics['score_mean']:.2f} | "
                    f"{metrics['termination_counts'].get('win', 0)}/{episodes} |"
                )
        lines += [
            "",
            "[Diagnostics d'entraînement](training.svg)",
            "",
            "La validation sélectionne les checkpoints ; elle ne remplace pas un test indépendant. "
            "Les scores comptent les pommes, pas les récompenses.",
            "",
        ]
        target = destination / "report.md"
        target.write_text("\n".join(lines), encoding="utf-8")
        (destination / "index.html").write_text(
            '<!doctype html><meta charset="utf-8"><title>Snake training</title>'
            "<style>body{max-width:1200px;margin:24px auto;font:16px system-ui}"
            "img{width:100%;height:auto}</style>"
            f'<h1>{escape(run.name)}</h1><a href="report.md">Résumé</a>'
            '<img src="training.svg" alt="Diagnostics d entrainement">',
            encoding="utf-8",
        )
        outputs.append(target)
        print(f"Rapport : {target}")
    if output_dir:
        links = "".join(
            f'<li><a href="{escape(p.parent.name)}/index.html">{escape(p.parent.name)}</a></li>'
            for p in outputs
        )
        (output_dir / "index.html").write_text(
            '<!doctype html><meta charset="utf-8"><title>Snake reports</title><ul>'
            + links
            + "</ul>",
            encoding="utf-8",
        )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--evaluate", action="store_true", help="Relancer les évaluations des modèles."
    )
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=71_000)
    parser.add_argument("--horizon", type=int, default=10_000)
    parser.add_argument("--n-envs", type=int, default=16)
    args = parser.parse_args()
    if args.evaluate:
        import torch

        torch.set_num_threads(1)
    generate_reports(
        args.runs,
        output_dir=args.output_dir,
        evaluate_models=args.evaluate,
        episodes=args.episodes,
        seed=args.seed,
        horizon=args.horizon,
        n_envs=args.n_envs,
    )


if __name__ == "__main__":
    main()
