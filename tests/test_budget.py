"""Budget notices and the last chance, driven through the real deep agent with a scripted fake model."""

from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage

from origami_agent.agent import build_agent
from origami_agent.tools import Workspace
from synbio_bench.budget import LAST_CHANCE_TOOLS, Budget, BudgetMiddleware
from synbio_bench.origami.harness import SYSTEM_PROMPT, Submission, make_submit_tool
from synbio_bench.runner import drive_agent


class ScriptedModel(GenericFakeChatModel):
    """Replies from a fixed script and records which tools it was offered on each call."""

    offered: list[list[str]] = []

    def bind_tools(self, tools: Any, **kwargs: Any):
        self.offered.append(sorted(t["name"] if isinstance(t, dict) else t.name for t in tools))
        return self


def _call(name: str, i: int, **args: Any) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{i}"}])


def _run(tmp_path, script: list[AIMessage], max_calls: int, dropped: list | None = None):
    ws = Workspace(root=tmp_path / "workspace")
    (ws.root / "designs").mkdir(parents=True)
    (ws.root / "designs" / "v1.sc").write_text("{}")
    model = ScriptedModel(messages=iter(script), offered=[])
    sub = Submission(spec={})
    budget = Budget(max_model_calls=max_calls, wall_timeout_s=3600, submitted_fn=lambda: sub.record is not None)
    agent = build_agent(
        model, ws, extra_tools=[make_submit_tool(ws, sub)], system_prompt=SYSTEM_PROMPT,
        middleware=[BudgetMiddleware(budget)],
    )
    messages, ended = drive_agent(agent, "design something", budget, {"last_chance_calls": 3}, dropped=dropped)
    return sub, model, budget, messages, ended


def test_step_limit_then_last_chance_submission(tmp_path):
    script = [_call("ls", i, path="/") for i in range(4)]
    script += [_call("submit_design", 9, design_path="/designs/v1.sc", rationale="best so far"), AIMessage("done")]
    sub, model, budget, messages, ended = _run(tmp_path, script, max_calls=4)

    assert ended["termination"] == "step_limit"
    assert ended["last_chance"] == {"termination": "completed", "submitted": True}
    assert sub.record is not None
    assert budget.model_calls == 6
    # full toolset during the run, only the last-chance tools afterwards
    assert "write_file" in model.offered[0] and "run_python" in model.offered[0]
    assert set(model.offered[-1]) == set(LAST_CHANCE_TOOLS)
    # notices rode along on tool results; the 4th call used the whole budget and ended the run
    notes = [m.content for m in messages if isinstance(m, ToolMessage) and "[harness] Budget" in str(m.content)]
    assert [n.split("[harness] Budget ")[1].split("%")[0] for n in notes] == ["50", "75", "100"]
    assert "No design has been submitted yet." in notes[0]


def test_stopping_without_submitting_gets_one_reminder(tmp_path):
    script = [AIMessage("I think I am done."), AIMessage("Still not submitting.")]
    _, _, budget, _, ended = _run(tmp_path, script, max_calls=50)
    assert ended["termination"] == "completed"
    assert ended["last_chance"] == {"termination": "completed", "submitted": False}
    assert budget.model_calls == 2  # one reminder, not a loop


def test_cut_off_reply_is_dropped_and_the_run_carries_on(tmp_path):
    cut = AIMessage("Planning at great length", response_metadata={"stop_reason": "max_tokens"})
    script = [cut, _call("submit_design", 1, design_path="designs/v1.sc", rationale="r"), AIMessage("done")]
    dropped: list = []
    sub, _, budget, messages, ended = _run(tmp_path, script, max_calls=50, dropped=dropped)
    assert ended == {"termination": "completed", "cut_off_replies": 1}
    assert [m.content for m in dropped] == ["Planning at great length"]  # kept for token accounting
    assert sub.record is not None and budget.model_calls == 3
    assert not any(m.content == "Planning at great length" for m in messages)
    assert any("[harness] Your last reply hit the output token limit" in str(m.content) for m in messages)


def test_submitted_runs_get_no_last_chance(tmp_path):
    rationale = "raster H0 → H28, 29 rows × ~138 bp"  # non-cp1252 text must not break submission on Windows
    script = [_call("submit_design", 1, design_path="designs/v1.sc", rationale=rationale), AIMessage("done")]
    _, _, budget, _, ended = _run(tmp_path, script, max_calls=50)
    assert ended == {"termination": "completed"}
    assert budget.model_calls == 2


def test_cost_cap_counts_towards_budget():
    cost = {"usd": 0.0}
    b = Budget(max_model_calls=100, wall_timeout_s=3600, cost_cap_usd=2.0, cost_fn=lambda: cost["usd"])
    assert b.exceeded() is None and b.notice() is None
    cost["usd"] = 1.6
    assert b.notice().startswith("[harness] Budget 80% used") and "$1.60 of the $2.00 cost cap" in b.notices[-1]
    assert b.notice() is None  # each threshold is announced once
    cost["usd"] = 2.0
    assert b.exceeded() == "cost_cap"


def test_snapshot_keeps_models_from_earlier_runs(tmp_path):
    from synbio_bench.origami.oxdna import OxDNAConfig
    from synbio_bench.runner import _write_snapshot

    snap = tmp_path / "experiment.snapshot.yaml"
    exp = {"name": "e", "models": ["a", "b"]}
    _write_snapshot(snap, exp, {"a": {"id": "a", "price_per_mtok": {"input": 1}}}, OxDNAConfig(mode="mock"))
    _write_snapshot(snap, exp, {"b": {"id": "b"}}, OxDNAConfig(mode="mock"))
    _write_snapshot(snap, exp, {"a": {"id": "a", "price_per_mtok": {"input": 2}}}, OxDNAConfig(mode="mock"))
    import yaml

    models = yaml.safe_load(snap.read_text())["models"]
    assert set(models) == {"a", "b"} and models["a"]["price_per_mtok"]["input"] == 2


def test_costs_command_lists_runs_per_model(tmp_path):
    import json

    from synbio_bench.runner import agent_costs

    runs = (("luna", "t1", 0.01, {"submitted": {"path": "x"}}),
            ("luna", "t2", 0.02, {"submitted": {"path": "x"}, "last_chance": {"submitted": True}}),
            ("kimi", "t1", 0.5, {"submitted": None}))
    for model, task, cost, extra in runs:
        d = tmp_path / task / model / "rep0"
        d.mkdir(parents=True)
        (d / "agent.json").write_text(json.dumps({"model": model, "termination": "completed", "n_model_calls": 7,
                                                  "tokens": {"input_tokens": 50000, "cache_read_tokens": 40000,
                                                             "output_tokens": 3000}, "cost_usd": cost, **extra}))
    out = agent_costs(tmp_path, ["luna"])
    assert "| luna | t1/rep0 | completed | yes | 7 | 50 (40) | 3 | 0.010 |" in out
    assert "| luna | t2/rep0 | completed | at last chance |" in out
    assert "| kimi | t1/rep0 | completed | no |" in agent_costs(tmp_path)
    assert "| luna | 2 | **0.030** |" in out and "kimi" not in out
    assert "| kimi | 1 | **0.500** |" in agent_costs(tmp_path)


def test_api_errors_are_not_results():
    from synbio_bench.runner import _infra_failure

    assert _infra_failure({"termination": "error", "n_model_calls": 0})
    assert _infra_failure({"termination": "error", "n_model_calls": 11, "submitted": {"path": "x"}})
    assert _infra_failure({"termination": "step_limit", "last_chance": {"termination": "error"}, "submitted": None})
    assert not _infra_failure({"termination": "step_limit", "last_chance": {"termination": "error"},
                               "submitted": {"path": "x"}})
    assert not _infra_failure({"termination": "completed"})


def test_run_skips_finished_runs_and_redoes_api_errors(tmp_path, monkeypatch, capsys):
    import json

    from synbio_bench import runner
    from synbio_bench.origami.oxdna import OxDNAConfig

    calls = []

    def fake_agent(task, model_cfg, agent_cfg, run_dir, seed):
        calls.append(run_dir.parent.parent.name)
        return {"task": task["id"], "model": model_cfg["id"], "termination": "completed", "n_model_calls": 3}

    monkeypatch.setattr(runner, "run_agent", fake_agent)
    monkeypatch.setattr(runner, "load_task", lambda t: {"id": t, "spec": {}})
    done = tmp_path / "t__done" / "m" / "rep0"
    failed = tmp_path / "t__failed" / "m" / "rep0"
    for d, info in ((done, {"termination": "completed"}),
                    (failed, {"termination": "error", "error": "APIStatusError: 402 insufficient credits"})):
        d.mkdir(parents=True)
        (d / "agent.json").write_text(json.dumps(info))
    (failed / "result.json").write_text("{}")  # graded as invalid before errors were excluded
    for t in ("t/done", "t/failed", "t/new"):
        runner.run_one({}, t, {"id": "m"}, 0, OxDNAConfig(mode="mock"), tmp_path, do_eval=False)
    assert calls == ["t__failed", "t__new"]  # the finished, ungraded run is not repeated
    out = capsys.readouterr().out
    assert "agent done, not graded yet" in out and "402 insufficient credits" in out
    archived = list((tmp_path / "t__failed" / "m").glob("_failed_rep0_*"))
    assert len(archived) == 1 and (archived[0] / "result.json").exists()
    assert json.loads((failed / "agent.json").read_text())["termination"] == "completed"
    assert list((tmp_path / "t__failed" / "m").glob("rep*")) == [failed]  # the archive isn't seen as a run


def test_costs_lists_api_errors(tmp_path):
    import json

    from synbio_bench.runner import agent_costs

    d = tmp_path / "t1" / "sol" / "rep0"
    d.mkdir(parents=True)
    (d / "agent.json").write_text(json.dumps({"model": "sol", "termination": "error", "n_model_calls": 9,
                                              "error": "BadRequestError: Error code: 400 - reasoning items missing\nmore",
                                              "tokens": {}, "cost_usd": 0.05}))
    out = agent_costs(tmp_path)
    assert "- sol t1/rep0: BadRequestError: Error code: 400 - reasoning items missing" in out and "more" not in out
