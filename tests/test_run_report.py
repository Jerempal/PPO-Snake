import json
from pathlib import Path

from snake_rl.config import load_config
from snake_rl.run_report import render_run_report


def _make_report_run(tmp_path: Path, name: str, score: float) -> Path:
    run_dir = tmp_path / name
    (run_dir / "best_model").mkdir(parents=True)
    (run_dir / "metrics").mkdir()
    config = load_config("configs/ppo.toml")
    (run_dir / "resolved_config.json").write_text(json.dumps(config.to_dict()), encoding="utf-8")
    (run_dir / "metadata.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    (run_dir / "best_model" / "evaluation_summary.json").write_text(
        json.dumps(
            {
                "history": [
                    {
                        "timesteps": 0,
                        "mean_score": 0.1,
                        "long_mean_score": 0.1,
                        "selection_score": 0.1,
                    },
                    {
                        "timesteps": 100_000,
                        "mean_score": score,
                        "long_mean_score": score,
                        "selection_score": score,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "metrics" / "progress.csv").write_text(
        "time/total_timesteps,game/score_mean_100,train/approx_kl,"
        "train/explained_variance\n"
        "8192,1.0,0.01,0.2\n"
        "16384,2.0,,\n"
        ",,0.02,0.8\n",
        encoding="utf-8",
    )
    return run_dir


def test_run_report_renders_multiple_sparse_runs_deterministically(tmp_path: Path) -> None:
    first = _make_report_run(tmp_path, "first", 2.0)
    second = _make_report_run(tmp_path, "second", 3.0)

    rendered = render_run_report([first, second])

    assert rendered == render_run_report([first, second])
    assert "Diagnostics de l entrainement Snake" in rendered
    assert "first : terminé" in rendered
    assert "second : terminé" in rendered
    assert "Score de sélection en validation" in rendered
    assert "Divergence KL approximative de PPO" in rendered
    assert "Perte de valeur (moyenne du mini-lot)" in rendered
    assert "Double DQN" not in rendered
    assert "Perte d entropie" in rendered
    assert rendered.endswith("</svg>\n")


def test_run_report_expands_to_fit_many_runs_without_chart_legends(tmp_path: Path) -> None:
    runs = [_make_report_run(tmp_path, f"run-{index:02d}", float(index)) for index in range(12)]

    rendered = render_run_report(runs)

    assert 'height="2042"' in rendered
    assert rendered.count("run-00") == 1
    assert rendered.count("run-11") == 1
