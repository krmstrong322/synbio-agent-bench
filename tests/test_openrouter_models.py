"""scripts/openrouter_models.py without the network: entries from OpenRouter's model records."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("openrouter_models", ROOT / "scripts/openrouter_models.py")
orm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(orm)


def record(slug="qwen/qwen4-max", tools=True, **over):
    m = {"id": slug, "name": "Qwen4 Max", "created": 1790000000,
         "supported_parameters": ["tools", "reasoning"] if tools else ["temperature"],
         "pricing": {"prompt": "0.0000012", "completion": "0.000006", "input_cache_read": "0.00000024"},
         "top_provider": {"max_completion_tokens": 128000}}
    m.update(over)
    return m


def test_entry_prices_and_limits():
    entry, warnings = orm.entry_for(record())
    assert entry["id"] == "qwen4-max" and entry["model"] == "openai:qwen/qwen4-max"
    assert entry["price_per_mtok"] == {"input": 1.2, "output": 6.0, "cache_read": 0.24}
    assert entry["kwargs"]["max_tokens"] == 64000 and entry["kwargs"]["api_key_env"] == "OPENROUTER_API_KEY"
    assert "extra_body" not in entry["kwargs"] and warnings == []

    entry, warnings = orm.entry_for(record(top_provider={"max_completion_tokens": 16384}), reasoning_effort="high")
    assert entry["kwargs"]["max_tokens"] == 16384 and entry["kwargs"]["extra_body"] == {"reasoning": {"effort": "high"}}
    assert any("max output is 16384" in w for w in warnings)

    _, warnings = orm.entry_for(record(tools=False))
    assert any("tool calling" in w for w in warnings)
    _, warnings = orm.entry_for(record(pricing={"prompt": "0.000001", "completion": "0.000002",
                                                "overrides": [{"utc_days": ["monday"], "prompt": "0.000002"}]}))
    assert any("time of day" in w for w in warnings)
    entry, warnings = orm.entry_for(record(pricing={
        "prompt": "0.000002", "completion": "0.00001", "input_cache_read": "0.0000001",
        "overrides": [{"min_prompt_tokens": 272000, "prompt": "0.000004", "completion": "0.000015",
                       "input_cache_read": "0.0000002", "input_cache_write": "0.000005"}]}))
    assert entry["price_per_mtok"]["long_prompt"] == {"min_prompt_tokens": 272000, "input": 4.0, "output": 15.0,
                                                      "cache_read": 0.2, "cache_write_5m": 5.0}
    assert any("272,000+" in w for w in warnings) and not any("time of day" in w for w in warnings)


def test_ids():
    assert orm.model_id("deepseek/deepseek-v4.1-flash-1017") == "deepseek-v4.1-flash-1017"
    assert orm.model_id("meta-llama/llama-5-405b:free") == "llama-5-405b"
    assert orm.unique_id("gpt-5.5", {"gpt-5.5"}) == "gpt-5.5-or"
    assert orm.unique_id("gpt-5.5", {"gpt-5.5", "gpt-5.5-or"}) == "gpt-5.5-or2"


def test_add_appends_parseable_entries(tmp_path, monkeypatch, capsys):
    models_yaml = tmp_path / "models.yaml"
    models_yaml.write_text("models:\n  - id: gpt-5.5\n    model: openai:gpt-5.5\n"
                           "  - id: qwen-old\n    model: openai:qwen/qwen3-max\n  # - id: example\n")
    monkeypatch.setattr(orm, "MODELS_YAML", models_yaml)
    monkeypatch.setattr(orm, "fetch_models", lambda: [
        record(), record("openai/gpt-5.5"), record("qwen/qwen3-max"), record("x/no-tools", tools=False)])
    monkeypatch.setattr(sys, "argv", ["orm", "qwen/qwen4-max", "openai/gpt-5.5", "qwen/qwen3-max", "x/no-tools",
                                      "--add"])
    with pytest.raises(SystemExit) as e:
        orm.main()
    assert e.value.code == 1  # the model without tools is refused
    ids = [m["id"] for m in yaml.safe_load(models_yaml.read_text())["models"]]
    assert ids == ["gpt-5.5", "qwen-old", "qwen4-max", "gpt-5.5-or"]  # existing qwen3 entry reused, not duplicated
    err = capsys.readouterr().err
    assert "already in models.yaml as id 'qwen-old'" in err and "models: [qwen4-max, gpt-5.5-or, qwen-old]" in err
