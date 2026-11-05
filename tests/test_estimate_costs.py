"""scripts/estimate_costs.py on synthetic baseline runs written in the real transcript format."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml
from langchain_core.messages import AIMessage, HumanMessage, messages_to_dict

from synbio_bench.cost import estimate_cost

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("estimate_costs", ROOT / "scripts/estimate_costs.py")
ec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ec)

HAIKU = {"input": 0.10, "output": 0.50, "cache_read": 0.01, "cache_write_5m": 0.125}
OPUS = {"input": 4.0, "output": 20.0, "cache_read": 0.2, "cache_write_5m": 5.0}


def ai(prompt, read, write, out):
    return AIMessage(content="x", usage_metadata={
        "input_tokens": prompt, "output_tokens": out, "total_tokens": prompt + out,
        "input_token_details": {"cache_read": read, "ephemeral_5m_input_tokens": write}})


def write_run(root: Path, task: str, model: str, msgs: list, price: dict) -> None:
    d = root / task.replace("/", "__") / model / "rep0"
    d.mkdir(parents=True)
    (d / "transcript.json").write_text(json.dumps(messages_to_dict([HumanMessage("go"), *msgs]), default=str))
    usage = {}
    for m in msgs:  # what the usage callback aggregates
        u = usage.setdefault("m", {"input_tokens": 0, "output_tokens": 0, "input_token_details": {
            "cache_read": 0, "ephemeral_5m_input_tokens": 0}})
        u["input_tokens"] += m.usage_metadata["input_tokens"]
        u["output_tokens"] += m.usage_metadata["output_tokens"]
        for k in ("cache_read", "ephemeral_5m_input_tokens"):
            u["input_token_details"][k] += m.usage_metadata["input_token_details"][k]
    (d / "agent.json").write_text(json.dumps({"cost_usd": estimate_cost(usage, price)["total_usd"]}))


def test_estimates_from_claude_runs(tmp_path, monkeypatch, capsys):
    runs = tmp_path / "runs" / "trio"
    small = [ai(20_000, 0, 20_000, 2_000), ai(40_000, 20_000, 20_000, 3_000)]
    big = [ai(100_000, 0, 100_000, 10_000), ai(300_000, 100_000, 200_000, 20_000)]  # 2nd call past 272k
    write_run(runs, "origami/6hb", "claude-haiku-5-5", small, HAIKU)
    write_run(runs, "origami/6hb", "claude-opus-5-5", big, OPUS)

    models = tmp_path / "models.yaml"
    sol = {"input": 2.0, "output": 10.0, "cache_read": 0.1, "cache_write_5m": 2.5,
           "long_prompt": {"min_prompt_tokens": 272000, "input": 4.0, "output": 15.0, "cache_read": 0.2,
                           "cache_write_5m": 5.0}}
    models.write_text(yaml.safe_dump({"models": [
        {"id": "claude-haiku-5-5", "model": "a:h", "price_per_mtok": HAIKU},
        {"id": "claude-opus-5-5", "model": "a:o", "price_per_mtok": OPUS},
        {"id": "sol", "model": "openai:openai/sol", "price_per_mtok": sol}]}))
    tasks = tmp_path / "tasks" / "origami"
    tasks.mkdir(parents=True)
    (tasks / "6hb.yaml").write_text("id: origami/6hb\nseeds: [1, 2, 3]\n")
    exp = tmp_path / "exp.yaml"
    exp.write_text(yaml.safe_dump({"name": "est", "tasks": ["origami/6hb"], "models": ["sol"], "repeats": 1}))
    monkeypatch.setattr(ec, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["est", "--experiment", str(exp), "--models-yaml", str(models),
                                      "--baseline-runs", str(runs), "--baseline-models", "claude-haiku-5-5",
                                      "claude-opus-5-5"])
    ec.main()
    out = capsys.readouterr().out
    assert out.count("ok  ") == 2 and "OFF" not in out  # replaying at Claude prices reproduces agent.json

    # As-Haiku: (0 + 20k*2.5 + 2k*10)/1e6 + (0 + 20k*0.1 + 20k*2.5 + 3k*10)/1e6
    as_haiku = (20_000 * 2.5 + 2_000 * 10 + 20_000 * 0.1 + 20_000 * 2.5 + 3_000 * 10) / 1e6
    # As-Opus: first call base rates; second call (300k prompt) at the long-prompt tier.
    as_opus = (100_000 * 2.5 + 10_000 * 10) / 1e6 + (100_000 * 0.2 + 200_000 * 5.0 + 20_000 * 15) / 1e6
    md = (tmp_path / "runs" / "est" / "cost_estimate.md").read_text()
    assert f"${as_haiku:.2f}" in md and f"${as_opus:.2f}" in md
    assert "**reached**" in md
    rows, totals = ec.estimate(["origami/6hb"], {"origami/6hb": 3}, {"sol": sol},
                               {"origami/6hb": ec.find_baselines("origami/6hb", [runs],
                                                                 ["claude-haiku-5-5", "claude-opus-5-5"])}, 1)
    assert totals[0]["low"] == pytest.approx(as_haiku) and totals[0]["grading"] == pytest.approx(3 * 0.045)
    no_cache_opus = (100_000 * 2.0 + 10_000 * 10 + 300_000 * 4.0 + 20_000 * 15) / 1e6
    assert totals[0]["high"] == pytest.approx(no_cache_opus)
