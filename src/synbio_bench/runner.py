"""Experiment runner: tasks x models x repeats -> agent run -> graded result.

Layout: runs/<experiment>/<task>/<model>/rep<k>/
    workspace/        the agent's filesystem (designs, scripts, sims, final/)
    transcript.json   full message history
    agent.json        tokens, tool calls, timing, termination reason
    eval/             grader simulations
    result.json       agent.json + grading, the unit of analysis
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.messages import AIMessage, messages_to_dict
from langgraph.errors import GraphRecursionError

from .origami.agent import build_agent, make_model
from .origami.oxdna import OxDNAConfig, config_from_env
from .origami.scoring import evaluate_submission
from .origami.tools import Workspace

ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def load_task(task_id: str) -> dict[str, Any]:
    return load_yaml(ROOT / "tasks" / f"{task_id}.yaml")


def load_models(path: Path) -> dict[str, dict[str, Any]]:
    return {m["id"]: m for m in load_yaml(path)["models"]}


def load_oxdna(path: Path) -> OxDNAConfig:
    return config_from_env(OxDNAConfig.from_dict(load_yaml(path)))


def _usage_totals(usage: dict[str, Any]) -> dict[str, int]:
    tot = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cache_read_tokens": 0}
    for u in usage.values():
        tot["input_tokens"] += u.get("input_tokens", 0)
        tot["output_tokens"] += u.get("output_tokens", 0)
        tot["total_tokens"] += u.get("total_tokens", 0)
        tot["cache_read_tokens"] += (u.get("input_token_details") or {}).get("cache_read", 0)
    return tot


def run_agent(task: dict, model_cfg: dict, oxdna: OxDNAConfig, agent_cfg: dict, run_dir: Path, seed: int) -> dict:
    ws = Workspace(
        root=run_dir / "workspace",
        spec=task["spec"],
        oxdna=oxdna,
        max_simulations=agent_cfg.get("max_simulations", 4),
        script_timeout_s=agent_cfg.get("script_timeout_s", 180),
    )
    (ws.root / "designs").mkdir(parents=True, exist_ok=True)
    model = make_model(model_cfg["model"], **model_cfg.get("kwargs", {}))
    agent = build_agent(model, ws)

    info: dict[str, Any] = {"task": task["id"], "model": model_cfg["id"], "seed": seed, "termination": "completed"}
    state: dict[str, Any] = {"messages": []}
    deadline = time.time() + agent_cfg.get("wall_timeout_s", 3600)
    t0 = time.time()
    with get_usage_metadata_callback() as cb:
        try:
            for state in agent.stream(
                {"messages": [{"role": "user", "content": task["prompt"]}]},
                config={"recursion_limit": agent_cfg.get("recursion_limit", 400)},
                stream_mode="values",
            ):
                if time.time() > deadline:
                    info["termination"] = "wall_timeout"
                    break
        except GraphRecursionError:
            info["termination"] = "recursion_limit"
        except Exception as e:  # noqa: BLE001 - provider/API errors are a result, not a crash
            info["termination"] = "error"
            info["error"] = f"{type(e).__name__}: {e}"
            info["traceback"] = traceback.format_exc()
        usage = dict(cb.usage_metadata)

    messages = state.get("messages", [])
    if info["termination"] == "completed" and messages and _hit_output_limit(messages[-1]):
        info["termination"] = "output_limit"  # the model's last turn was cut off, not a deliberate stop
    (run_dir / "transcript.json").write_text(json.dumps(messages_to_dict(messages), indent=1, default=str))
    ai = [m for m in messages if isinstance(m, AIMessage)]
    info.update(
        wall_time_s=round(time.time() - t0, 1),
        n_messages=len(messages),
        n_model_calls=len(ai),
        tool_calls=_count_tool_calls(ai),
        simulations_used=ws.sim_count,
        submitted=ws.submitted,
        usage=usage,
        tokens=_usage_totals(usage),
        final_message=(ai[-1].text if ai else "")[:4000],
    )
    if price := model_cfg.get("price_per_mtok"):
        t = info["tokens"]
        info["cost_usd"] = round((t["input_tokens"] * price["input"] + t["output_tokens"] * price["output"]) / 1e6, 4)
    return info


def _hit_output_limit(msg: Any) -> bool:
    meta = getattr(msg, "response_metadata", {}) or {}
    return (meta.get("finish_reason") or meta.get("stop_reason")) in ("length", "max_tokens")


def _count_tool_calls(ai: list[AIMessage]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for m in ai:
        for tc in m.tool_calls:
            counts[tc["name"]] = counts.get(tc["name"], 0) + 1
    return counts


def grade(run_dir: Path, task: dict, oxdna: OxDNAConfig, eval_cfg: dict, reanalyze: bool = False) -> dict:
    return evaluate_submission(
        run_dir / "workspace" / "final" / "design.sc",
        task["spec"],
        oxdna,
        run_dir / "eval",
        protocol=eval_cfg.get("protocol", "standard"),
        seeds=eval_cfg.get("seeds", [1]),
        weights=task.get("weights") or eval_cfg.get("weights"),
        reanalyze=reanalyze,
    )


def run_one(exp: dict, task_id: str, model_cfg: dict, rep: int, oxdna: OxDNAConfig, out_root: Path, do_eval: bool) -> Path:
    run_dir = out_root / task_id.replace("/", "__") / model_cfg["id"] / f"rep{rep}"
    if (run_dir / "result.json").exists():
        print(f"[skip] {run_dir} already has result.json")
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    task = load_task(task_id)
    print(f"[run ] {task_id} | {model_cfg['id']} | rep{rep}")
    info = run_agent(task, model_cfg, oxdna, exp.get("agent", {}), run_dir, seed=rep)
    (run_dir / "agent.json").write_text(json.dumps(info, indent=1, default=str))
    if _infra_failure(info):
        print(f"[fail] {task_id} | {model_cfg['id']} | rep{rep}: {info['error']} (not graded; rerun to retry)")
        return run_dir
    if do_eval:
        info["grade"] = grade(run_dir, task, oxdna, exp.get("evaluation", {}))
        (run_dir / "result.json").write_text(json.dumps(info, indent=1, default=str))
        _figures(run_dir, task, model_cfg["id"])
    print(f"[done] {task_id} | {model_cfg['id']} | rep{rep} -> {info['termination']}, "
          f"overall={info.get('grade', {}).get('overall', 'not graded')}")
    return run_dir


def _figures(run_dir: Path, task: dict, model: str) -> None:
    """Render design/structure figures; a plotting failure must never lose a graded run."""
    from .origami.viz import figures_for_run

    try:
        figures_for_run(run_dir, task["spec"], f"{task['id']} · {model} · {run_dir.name}", force=True)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] figures failed for {run_dir}: {type(e).__name__}: {e}")


def _infra_failure(info: dict) -> bool:
    """The model never answered (bad key, network, unknown model): not a result, so don't grade it."""
    return info["termination"] == "error" and info.get("n_model_calls", 0) == 0


def _preload_providers() -> None:
    """Import provider integrations up front: concurrent first imports from worker threads can deadlock or
    see half-initialised modules (seen with langchain_anthropic)."""
    import importlib

    for mod in ("langchain_anthropic.chat_models", "langchain_openai.chat_models", "langchain_google_genai"):
        try:
            importlib.import_module(mod)
        except ImportError:
            pass


def cmd_run(args: argparse.Namespace) -> None:
    exp = load_yaml(Path(args.experiment))
    models = load_models(Path(args.models))
    oxdna = load_oxdna(Path(args.oxdna))
    model_ids = args.model or exp["models"]
    task_ids = args.task or exp["tasks"]
    repeats = args.repeats or exp.get("repeats", 1)
    out_root = ROOT / "runs" / exp["name"]
    (out_root).mkdir(parents=True, exist_ok=True)
    (out_root / "experiment.snapshot.yaml").write_text(
        yaml.safe_dump({"experiment": exp, "models": {m: models[m] for m in model_ids}, "oxdna": oxdna.__dict__})
    )
    jobs = [(t, models[m], r) for t in task_ids for m in model_ids for r in range(repeats)]
    _preload_providers()
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        for f in [pool.submit(run_one, exp, t, m, r, oxdna, out_root, not args.no_eval) for t, m, r in jobs]:
            f.result()


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Grade agent runs that have agent.json but no result.json (e.g. after `run --no-eval`)."""
    exp = load_yaml(Path(args.experiment))
    oxdna = load_oxdna(Path(args.oxdna))
    for agent_json in sorted((ROOT / "runs" / exp["name"]).glob("*/*/rep*/agent.json")):
        run_dir = agent_json.parent
        if (run_dir / "result.json").exists() and not (args.force or args.reanalyze):
            continue
        info = json.loads(agent_json.read_text())
        if _infra_failure(info):
            continue
        print(f"[eval] {run_dir.relative_to(ROOT)}")
        task = load_task(info["task"])
        info["grade"] = grade(run_dir, task, oxdna, exp.get("evaluation", {}), args.reanalyze)
        (run_dir / "result.json").write_text(json.dumps(info, indent=1, default=str))
        _figures(run_dir, task, info["model"])


def cmd_baseline(args: argparse.Namespace) -> None:
    """Build each task's reference design (if it has one) and grade it as the `reference` arm."""
    exp = load_yaml(Path(args.experiment))
    oxdna = load_oxdna(Path(args.oxdna))
    for task_id in args.task or exp["tasks"]:
        task = load_task(task_id)
        if not task.get("reference"):
            print(f"[skip] {task_id}: no reference design")
            continue
        run_dir = ROOT / "runs" / exp["name"] / task_id.replace("/", "__") / "reference" / "rep0"
        final = run_dir / "workspace" / "final"
        final.mkdir(parents=True, exist_ok=True)
        script, *script_args = task["reference"].split()  # e.g. "reference_designs/rothemund.py rect"
        proc = subprocess.run([sys.executable, str(ROOT / script), *script_args, str(final / "design.sc")],
                              capture_output=True, text=True, cwd=ROOT / Path(script).parent)
        info = {"task": task_id, "model": "reference", "seed": 0, "termination": "reference",
                "build_log": proc.stdout + proc.stderr}
        info["grade"] = grade(run_dir, task, oxdna, exp.get("evaluation", {}))
        (run_dir / "result.json").write_text(json.dumps(info, indent=1, default=str))
        _figures(run_dir, task, "reference")
        print(f"[ref ] {task_id}: overall={info['grade']['overall']} valid={info['grade']['valid']}")


def main() -> None:
    load_dotenv(ROOT / ".env")
    p = argparse.ArgumentParser(prog="synbio-bench")
    p.add_argument("--experiment", default=str(ROOT / "configs" / "experiment.yaml"))
    p.add_argument("--models", default=str(ROOT / "configs" / "models.yaml"))
    p.add_argument("--oxdna", default=str(ROOT / "configs" / "oxdna.yaml"))
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run agents (and grade unless --no-eval)")
    r.add_argument("--model", action="append", help="model id (repeatable); default: experiment list")
    r.add_argument("--task", action="append", help="task id (repeatable); default: experiment list")
    r.add_argument("--repeats", type=int)
    r.add_argument("--parallel", type=int, default=1, help="concurrent agent runs (oxDNA runs stay serialized)")
    r.add_argument("--no-eval", action="store_true", help="skip grading; run `evaluate` later")
    r.set_defaults(func=cmd_run)

    e = sub.add_parser("evaluate", help="grade runs that have not been graded yet")
    e.add_argument("--force", action="store_true", help="re-grade runs that already have result.json")
    e.add_argument("--reanalyze", action="store_true",
                   help="recompute metrics from existing grading simulations instead of re-simulating")
    e.set_defaults(func=cmd_evaluate)

    b = sub.add_parser("baseline", help="grade reference designs as a positive control")
    b.add_argument("--task", action="append")
    b.set_defaults(func=cmd_baseline)

    from .report import add_report_parser
    add_report_parser(sub)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
