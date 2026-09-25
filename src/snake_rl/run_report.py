"""Génère un rapport SVG à partir des métriques d'un ou plusieurs runs."""

from __future__ import annotations

import csv
import json
import math
from html import escape
from pathlib import Path

COLORS = (
    "#2563eb",
    "#dc2626",
    "#059669",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#4f46e5",
    "#be123c",
    "#047857",
    "#7e22ce",
    "#c2410c",
    "#0e7490",
)


def _number(value: float) -> str:
    if 0 < abs(value) < 1:
        return f"{value:.3g}"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _evaluation_points(run_dir: Path, metric: str) -> list[tuple[float, float]]:
    path = run_dir / "best_model" / "evaluation_summary.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    points = []
    for record in payload.get("history", []):
        if metric in record:
            points.append((float(record["timesteps"]), float(record[metric])))
    return points


def _progress_points(run_dir: Path, metric: str) -> list[tuple[float, float]]:
    path = run_dir / "metrics" / "progress.csv"
    if not path.is_file():
        return []
    points = []
    current_step: float | None = None
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            raw_step = row.get("time/total_timesteps", "").strip()
            if raw_step:
                current_step = float(raw_step)
            raw_value = row.get(metric, "").strip()
            if current_step is not None and raw_value:
                value = float(raw_value)
                if math.isfinite(value):
                    points.append((current_step, value))
    return points


def _render_chart(
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    title: str,
    series: list[tuple[str, list[tuple[float, float]], str, bool]],
    x_label: str = "transitions",
    show_legend: bool = True,
) -> list[str]:
    available = [
        (label, points, color, dashed) for label, points, color, dashed in series if points
    ]
    lines = [
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="10" '
        'fill="#ffffff" stroke="#d8dee9"/>',
        f'<text x="{x + 18}" y="{y + 28}" font-family="Arial, sans-serif" '
        f'font-size="17" font-weight="700" fill="#17202a">{escape(title)}</text>',
    ]
    if not available:
        lines.append(
            f'<text x="{x + width / 2:.1f}" y="{y + height / 2:.1f}" text-anchor="middle" '
            'font-family="Arial, sans-serif" font-size="14" fill="#6b7280">Aucune donnée</text>'
        )
        return lines

    all_points = [point for _, points, _, _ in available for point in points]
    x_min = min(point[0] for point in all_points)
    x_max = max(point[0] for point in all_points)
    y_min = min(0.0, min(point[1] for point in all_points))
    y_max = max(point[1] for point in all_points)
    if x_max == x_min:
        x_max = x_min + 1.0
    if y_max == y_min:
        y_max = y_min + 1.0

    legend_rows = math.ceil(len(available) / 2) if show_legend else 0
    left, right, top, bottom = 58, 18, max(54, 54 + legend_rows * 17), 42
    plot_x = x + left
    plot_y = y + top
    plot_width = width - left - right
    plot_height = height - top - bottom

    def sx(value: float) -> float:
        return plot_x + plot_width * (value - x_min) / (x_max - x_min)

    def sy(value: float) -> float:
        return plot_y + plot_height * (1.0 - (value - y_min) / (y_max - y_min))

    for index in range(5):
        ratio = index / 4
        value = y_max - ratio * (y_max - y_min)
        position = plot_y + ratio * plot_height
        lines.extend(
            [
                f'<line x1="{plot_x}" y1="{position:.1f}" x2="{plot_x + plot_width}" '
                f'y2="{position:.1f}" stroke="#edf1f5"/>',
                f'<text x="{plot_x - 8}" y="{position + 4:.1f}" text-anchor="end" '
                f'font-family="Arial, sans-serif" font-size="11" fill="#64748b">'
                f"{_number(value)}</text>",
            ]
        )
    lines.append(
        f'<text x="{plot_x}" y="{plot_y + plot_height + 28}" font-family="Arial, sans-serif" '
        f'font-size="11" fill="#64748b">{_number(x_min)} {escape(x_label)}</text>'
    )
    lines.append(
        f'<text x="{plot_x + plot_width}" y="{plot_y + plot_height + 28}" text-anchor="end" '
        f'font-family="Arial, sans-serif" font-size="11" fill="#64748b">'
        f"{_number(x_max)} {escape(x_label)}</text>"
    )

    for _, points, color, dashed in available:
        coordinates = " ".join(f"{sx(px):.1f},{sy(py):.1f}" for px, py in points)
        dash = ' stroke-dasharray="7 5"' if dashed else ""
        lines.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" '
            f'stroke-width="2.2" stroke-linejoin="round"{dash}/>'
        )

    if not show_legend:
        return lines

    legend_column_width = (width - 36) / 2
    for index, (label, _, color, dashed) in enumerate(available):
        legend_x = x + 18 + (index % 2) * legend_column_width
        legend_y = y + 50 + (index // 2) * 17
        dash = ' stroke-dasharray="7 5"' if dashed else ""
        lines.extend(
            [
                f'<line x1="{legend_x:.1f}" y1="{legend_y - 4}" '
                f'x2="{legend_x + 20:.1f}" '
                f'y2="{legend_y - 4}" stroke="{color}" stroke-width="2"{dash}/>',
                f'<text x="{legend_x + 26:.1f}" y="{legend_y}" '
                f'font-family="Arial, sans-serif" font-size="10" fill="#475569">'
                f"{escape(label)}</text>",
            ]
        )
    return lines


def render_run_report(run_dirs: list[Path]) -> str:
    """Return a deterministic SVG report for local training artifacts."""
    if not run_dirs:
        raise ValueError("Au moins un dossier de run est nécessaire.")
    resolved = [path.resolve() for path in run_dirs]
    summaries = []
    for run_dir in resolved:
        evaluation_path = run_dir / "best_model" / "evaluation_summary.json"
        config_path = run_dir / "resolved_config.json"
        metadata_path = run_dir / "metadata.json"
        evaluation = (
            json.loads(evaluation_path.read_text(encoding="utf-8"))
            if evaluation_path.is_file()
            else {"history": []}
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        )
        history = evaluation.get("history", [])
        selection_metric = config["evaluation"].get("selection_metric", "mean_score")
        if config.get("map_sizes"):
            selection_metric = "normalized_mean_score"
        selection_label = "normalisé" if config.get("map_sizes") else "standard"
        best = (
            max(history, key=lambda row: float(row.get("selection_score", float("-inf"))))
            if history
            else {}
        )
        final = history[-1] if history else {}
        summaries.append(
            {
                "name": run_dir.name,
                "status": {"complete": "terminé", "running": "en cours", "failed": "échoué"}.get(
                    metadata.get("status"), "inconnu"
                ),
                "best": float(best.get("selection_score", float("nan"))),
                "best_step": int(best.get("timesteps", 0)),
                "final": float(final.get("selection_score", float("nan"))),
                "n_steps": int(config["training"]["n_steps"]),
                "selection_metric": selection_metric,
                "selection_label": selection_label,
            }
        )

    width = 1200
    chart_top = 120 + len(summaries) * 24
    chart_width, chart_height, gap = 550, 300, 24
    height = chart_top + 5 * chart_height + 4 * gap + 38
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description">',
        '<title id="title">Diagnostics de l entrainement Snake</title>',
        '<desc id="description">Courbes de validation et diagnostics d optimisation '
        "produits à partir des journaux locaux.</desc>",
        f'<rect width="{width}" height="{height}" fill="#f5f7fa"/>',
        '<text x="40" y="43" font-family="Arial, sans-serif" font-size="26" '
        'font-weight="700" fill="#17202a">Diagnostics de l entrainement Snake</text>',
        '<text x="40" y="69" font-family="Arial, sans-serif" font-size="13" '
        'fill="#64748b">Les courbes de validation servent a selectionner le modele, '
        "pas à estimer le test final.</text>",
    ]

    for index, summary in enumerate(summaries):
        color = COLORS[index % len(COLORS)]
        y = 96 + index * 24
        text = (
            f"{summary['name']} : {summary['status']} · meilleur {summary['selection_label']}="
            f"{_number(summary['best'])} "
            f"à {_number(summary['best_step'])} · final={_number(summary['final'])} · "
            f"taille_collecte={summary['n_steps']}"
        )
        lines.extend(
            [
                f'<circle cx="48" cy="{y - 4}" r="5" fill="{color}"/>',
                f'<text x="62" y="{y}" font-family="Arial, sans-serif" font-size="13" '
                f'fill="#334155">{escape(text)}</text>',
            ]
        )

    def series(metric: str, *, evaluation: bool = False, dashed: bool = False):
        return [
            (
                run_dir.name,
                (
                    _evaluation_points(run_dir, metric)
                    if evaluation
                    else _progress_points(run_dir, metric)
                ),
                COLORS[index % len(COLORS)],
                dashed,
            )
            for index, run_dir in enumerate(resolved)
        ]

    selection_series = [
        (
            run_dir.name,
            _evaluation_points(run_dir, str(summary["selection_metric"])),
            COLORS[index % len(COLORS)],
            False,
        )
        for index, (run_dir, summary) in enumerate(zip(resolved, summaries, strict=True))
    ]
    panels = (
        ("Score de sélection en validation", selection_series),
        ("Score d entrainement (100 dernieres parties)", series("game/score_mean_100")),
        ("Perte totale (dernier mini-lot)", series("train/loss")),
        ("Perte de valeur (moyenne du mini-lot)", series("train/value_loss")),
        ("Divergence KL approximative de PPO", series("train/approx_kl")),
        ("Variance expliquée", series("train/explained_variance")),
        ("Perte du gradient de politique", series("train/policy_gradient_loss")),
        ("Perte d entropie (entropie negative)", series("train/entropy_loss")),
        ("Fraction de découpage PPO", series("train/clip_fraction")),
        ("Taux d apprentissage", series("train/learning_rate")),
    )
    for index, (title, data) in enumerate(panels):
        lines.extend(
            _render_chart(
                x=38 + (index % 2) * (chart_width + gap),
                y=chart_top + (index // 2) * (chart_height + gap),
                width=chart_width,
                height=chart_height,
                title=title,
                series=data,
                show_legend=False,
            )
        )
    lines.append("</svg>")
    lines.append("")
    return "\n".join(lines)
