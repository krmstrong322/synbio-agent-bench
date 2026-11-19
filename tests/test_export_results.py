"""scripts/export_results.py on a small synthetic runs/ tree in the real layout."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("export_results", ROOT / "scripts/export_results.py")
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)

TASKS = list(ex.TASKS)


def run(root, src, task, model, overall, cost, seeds, runner="modal", skin=0.1):
    d = root / src / task.replace("/", "__") / model / "rep0"
    g = {"overall": overall, "valid": True, "spec_score": 1.0, "sim_score": 0.99, "shape_score": 0.8,
         "sim_runner": runner, "verlet_skin": skin}
    (d / "eval").mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({"task": task, "model": model, "cost_usd": cost, "n_model_calls": 9,
                                               "tokens": {}, "termination": "completed", "submitted": {}, "grade": g}))
    for k, v in enumerate(seeds, 1):
        key, sub = ("wireframe_fit", "f1") if "wireframe" in task else ("shape_unrolled", "iou")
        (d / "eval" / f"seed_{k}").mkdir()
        (d / "eval" / f"seed_{k}" / "result.json").write_text(json.dumps({"metrics": {key: {sub: v}}}))


def test_export(tmp_path, monkeypatch, capsys):
    runs = tmp_path / "runs"
    for model, base, cost in (("gpt-6.1-sol", 0.95, 0.1), ("claude-opus-5-5", 0.94, 0.5), ("glm-5.3", 0.9, 0.06)):
        for i, task in enumerate(TASKS):
            seeds = [0.8, 0.7, 0.75] if "wireframe" in task else [0.93]
            run(runs, "exp", task, model, base - i * 0.01, cost, seeds)
    run(runs, "exp", TASKS[3], "reference", 0.93, None, [0.8, 0.7, 0.8])
    run(runs, "old", TASKS[0], "claude-haiku-5-5", 0.96, 0.05, [0.9], runner="wsl", skin=None)  # graded differently
    out = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", ["x", "--source", str(runs / "exp"), "--source", str(runs / "old"),
                                      "--out", str(out)])
    ex.main()
    summary = (out / "summary.csv").read_text()
    assert summary.splitlines()[1].startswith("gpt-6.1-sol")  # best mean first
    assert "claude-haiku-5-5" not in summary  # excluded: not graded on Modal at skin 0.1
    readme = (out / "README.md").read_text()
    assert "| GPT-6.1 Sol | OpenAI | 0.950 | 0.940 | 0.930 | 0.920 | **0.935** | $0.40 |" in readme
    assert "| Expert reference | 0.80 | 0.70 | 0.80 | 0.77 |" in readme
    assert "old/" in readme and "runner=wsl" in readme
    for f in ("score_vs_cost", "scores_by_design", "tetrahedron_seeds"):
        assert (out / "figures" / f"{f}.png").stat().st_size > 10_000
    runs_csv = (out / "runs.csv").read_text()
    assert "[0.8, 0.7, 0.75]" in runs_csv
