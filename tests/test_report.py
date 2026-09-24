import json
from pathlib import Path

import pytest

from snake_rl.config import load_config
from snake_rl.report import generate_reports, main


def make_run(tmp_path):
    run = tmp_path / "run"
    (run / "best_model").mkdir(parents=True)
    (run / "resolved_config.json").write_text(
        json.dumps(load_config("configs/smoke.toml").to_dict())
    )
    (run / "best_model/best_model.zip").write_bytes(b"best")
    return run


def test_report_fresh_evaluation_and_missing_final(tmp_path, monkeypatch):
    run = make_run(tmp_path)
    calls = []

    def evaluate(model, *args, **kwargs):
        calls.append(model)
        return {"score_mean": 2, "termination_counts": {"win": 1}}

    monkeypatch.setattr("snake_rl.report.evaluate", evaluate)
    options = dict(output_dir=tmp_path / "reports", evaluate_models=True, episodes=2, horizon=10)
    outputs = generate_reports([run], **options)
    assert "Indisponible" in outputs[0].read_text(encoding="utf-8")
    assert (tmp_path / "reports/index.html").exists()
    generate_reports([run], **options)
    assert len(calls) == 2  # Explicit evaluations are fresh, never hidden cache hits.


def test_lightweight_report_does_not_evaluate(tmp_path, monkeypatch):
    run = make_run(tmp_path)
    monkeypatch.setattr("snake_rl.report.evaluate", lambda *a, **k: pytest.fail("unexpected games"))
    monkeypatch.setattr("sys.argv", ["snake-report", str(run)])
    main()
    assert (run / "report/index.html").exists()
    assert not (run / "report/best-evaluation.json").exists()


def test_invalid_inputs(tmp_path):
    with pytest.raises(ValueError):
        generate_reports([])
    with pytest.raises(ValueError):
        generate_reports([tmp_path, tmp_path])
    with pytest.raises(ValueError):
        generate_reports([Path("a/run"), Path("b/run")], output_dir=tmp_path)
