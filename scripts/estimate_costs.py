"""Estimate what models would cost on an experiment, from the token usage of runs you already have.

Each baseline run's transcript records every model call's tokens (prompt, cache reads/writes, output). This
replays those calls at another model's prices from configs/models.yaml, call by call, so long-prompt price
tiers (e.g. OpenAI's 272k+ prompts) apply where a call's prompt was that long. It assumes the new model would
use about as many tokens as the baseline did on the same task, which is the main uncertainty: the range over
the baseline models shows how much that alone moves the cost.

    uv run python scripts/estimate_costs.py --experiment configs/open-models.yaml

Baselines default to the Claude runs in runs/origami-trio and runs/wireframe-tetrahedron. Before estimating,
it re-prices each baseline run at its own model's prices and compares with the cost recorded in agent.json,
to check the token accounting.

Two cache scenarios bound the input cost: "cached" assumes the new model caches as much of its prompts as
Claude did (OpenAI caches automatically; for others it depends on the OpenRouter provider), "no cache" bills
every prompt token at the full input price.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
GRADING_USD_PER_SEED = 0.045  # Modal, shape protocol, 3 sims per GPU (docs/modal.md), incl. ~30% overhead
BASELINE_MODELS = ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"]
BASELINE_RUNS = ["runs/origami-trio", "runs/wireframe-tetrahedron"]


def calls_from_transcript(path: Path) -> list[dict[str, int]]:
    """Per model call: prompt tokens (incl. cached), cache read/write, output (incl. reasoning)."""
    calls = []
    for msg in json.loads(path.read_text()):
        if msg.get("type") != "ai":
            continue
        u = (msg.get("data") or {}).get("usage_metadata") or {}
        if not u:
            continue
        d = u.get("input_token_details") or {}
        writes = (d.get("ephemeral_5m_input_tokens") or 0) + (d.get("ephemeral_1h_input_tokens") or 0)
        writes = writes or (d.get("cache_creation") or 0)
        calls.append({"prompt": u.get("input_tokens") or 0, "cache_read": d.get("cache_read") or 0,
                      "cache_write": writes, "output": u.get("output_tokens") or 0})
    return calls


def call_cost(call: dict[str, int], price: dict[str, Any], cached: bool = True) -> float:
    rates = dict(price)
    tier = price.get("long_prompt")
    if tier and call["prompt"] >= tier.get("min_prompt_tokens", float("inf")):
        rates.update({k: v for k, v in tier.items() if k != "min_prompt_tokens"})
    if not cached:
        return (call["prompt"] * rates["input"] + call["output"] * rates["output"]) / 1e6
    uncached = max(0, call["prompt"] - call["cache_read"] - call["cache_write"])
    return (uncached * rates["input"]
            + call["cache_read"] * rates.get("cache_read", rates["input"])
            + call["cache_write"] * rates.get("cache_write_5m", rates["input"])
            + call["output"] * rates["output"]) / 1e6


def run_cost(calls: list[dict[str, int]], price: dict[str, Any], cached: bool = True) -> float:
    return sum(call_cost(c, price, cached) for c in calls)


def find_baselines(task_id: str, run_roots: list[Path], models: list[str]) -> dict[str, list[list[dict]]]:
    """model -> list of runs (each a list of calls) for this task across the baseline experiments."""
    out: dict[str, list[list[dict]]] = {}
    for root in run_roots:
        for m in models:
            for t in sorted((root / task_id.replace("/", "__") / m).glob("rep*/transcript.json")):
                calls = calls_from_transcript(t)
                if calls:
                    out.setdefault(m, []).append(calls)
    return out


def check_baselines(run_roots: list[Path], models: list[str], prices: dict[str, dict]) -> list[str]:
    """Re-price baseline runs at their own prices and compare with agent.json's recorded cost."""
    lines = []
    for root in run_roots:
        for m in models:
            if m not in prices:
                continue
            for agent_json in sorted(root.glob(f"*/{m}/rep*/agent.json")):
                t = agent_json.with_name("transcript.json")
                recorded = json.loads(agent_json.read_text()).get("cost_usd")
                if recorded is None or not t.exists():
                    continue
                replay = run_cost(calls_from_transcript(t), prices[m])
                ok = abs(replay - recorded) <= max(0.005, 0.03 * recorded)
                lines.append(f"{'ok ' if ok else 'OFF'} {agent_json.parent.relative_to(ROOT) if agent_json.is_relative_to(ROOT) else agent_json.parent}: "
                             f"recorded ${recorded:.3f}, replayed ${replay:.3f}")
    return lines


def estimate(tasks: list[str], seeds_of: dict[str, int], candidates: dict[str, dict], baselines: dict[str, dict],
             repeats: int) -> tuple[list[dict], list[dict]]:
    rows, totals = [], []
    for cid, price in candidates.items():
        tot = {"model": cid, "low": 0.0, "typical": 0.0, "high": 0.0, "grading": 0.0, "missing": []}
        for task in tasks:
            runs = baselines.get(task, {})
            if not runs:
                tot["missing"].append(task)
                continue
            per_baseline = {b: statistics.mean(run_cost(r, price) for r in rs) for b, rs in runs.items()}
            uncached = {b: statistics.mean(run_cost(r, price, cached=False) for r in rs) for b, rs in runs.items()}
            max_prompt = max(c["prompt"] for rs in runs.values() for r in rs for c in r)
            row = {"model": cid, "task": task, **{f"as_{b}": round(v, 3) for b, v in per_baseline.items()},
                   "low": min(per_baseline.values()), "typical": statistics.median(per_baseline.values()),
                   "high": max(uncached.values()), "max_prompt": max_prompt}
            rows.append(row)
            tot["low"] += row["low"] * repeats
            tot["typical"] += row["typical"] * repeats
            tot["high"] += row["high"] * repeats
            tot["grading"] += GRADING_USD_PER_SEED * seeds_of[task] * repeats
        totals.append(tot)
    return rows, totals


def short(b: str) -> str:
    return b.removeprefix("claude-").removesuffix("-5-5")


def report(rows: list[dict], totals: list[dict], baseline_models: list[str], candidates: dict[str, dict],
           repeats: int) -> str:
    lines = ["# Cost estimate per model", "",
             "Agent API cost if each model used about as many tokens as a Claude model did on the same task.",
             "typical = median over the Claude baselines with Claude-like caching; low = cheapest baseline;",
             "high = the most token-hungry baseline with no prompt caching at all.", ""]
    head = "| model | task | " + " | ".join(f"as {short(b)}" for b in baseline_models) + " | max prompt | long-prompt tier |"
    lines += [head, "|" + "---|" * (4 + len(baseline_models))]
    for r in rows:
        tier = candidates[r["model"]].get("long_prompt")
        hit = "–" if not tier else ("**reached**" if r["max_prompt"] >= tier["min_prompt_tokens"]
                                    else f"not reached (<{tier['min_prompt_tokens'] // 1000}k)")
        cells = " | ".join(f"${r[f'as_{b}']:.2f}" if f"as_{b}" in r else "–" for b in baseline_models)
        lines.append(f"| {r['model']} | {r['task']} | {cells} | {r['max_prompt'] // 1000}k | {hit} |")
    lines += ["", f"## Whole experiment ({repeats} repeat(s) per task)", "",
              "| model | in $/M | out $/M | agents: low | typical | high | Modal grading | total (typical) |",
              "|---|---|---|---|---|---|---|---|"]
    grand = 0.0
    for t in sorted(totals, key=lambda t: t["typical"]):
        p = candidates[t["model"]]
        total = t["typical"] + t["grading"]
        grand += total
        miss = f" (no baseline for {', '.join(t['missing'])})" if t["missing"] else ""
        lines.append(f"| {t['model']}{miss} | {p['input']} | {p['output']} | ${t['low']:.2f} | ${t['typical']:.2f} | "
                     f"${t['high']:.2f} | ${t['grading']:.2f} | **${total:.2f}** |")
    lines += ["", f"All models, typical: **${grand:.2f}** (agents + grading).", "",
              "Caveats: models differ in how many calls they make and how much they reason, which this cannot",
              "predict; the agent budget caps each run (max_model_calls, cost_cap_usd), so the cost cap bounds",
              "the worst case per run. OpenRouter time-of-day surcharges are not included."]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", default=str(ROOT / "configs" / "open-models.yaml"))
    p.add_argument("--models", nargs="*", help="model ids to estimate (default: the experiment's models)")
    p.add_argument("--baseline-runs", nargs="*", default=BASELINE_RUNS, help="experiment run dirs to learn from")
    p.add_argument("--baseline-models", nargs="*", default=BASELINE_MODELS)
    p.add_argument("--models-yaml", default=str(ROOT / "configs" / "models.yaml"))
    args = p.parse_args()

    exp = yaml.safe_load(Path(args.experiment).read_text())
    all_models = {m["id"]: m for m in yaml.safe_load(Path(args.models_yaml).read_text())["models"]}
    ids = args.models or exp.get("models") or []
    if not ids:
        raise SystemExit("no models: list them in the experiment config or pass --models")
    missing = [i for i in ids if i not in all_models or not all_models[i].get("price_per_mtok")]
    if missing:
        raise SystemExit(f"no price in models.yaml for: {', '.join(missing)} (add with scripts/openrouter_models.py)")
    candidates = {i: all_models[i]["price_per_mtok"] for i in ids}
    roots = [Path(r) if Path(r).is_absolute() else ROOT / r for r in args.baseline_runs]
    tasks = exp["tasks"]
    seeds_of = {}
    for t in tasks:
        task = yaml.safe_load((ROOT / "tasks" / f"{t}.yaml").read_text())
        seeds_of[t] = len(task.get("seeds") or (exp.get("evaluation") or {}).get("seeds") or [1])

    own_prices = {m: all_models[m]["price_per_mtok"] for m in args.baseline_models
                  if m in all_models and all_models[m].get("price_per_mtok")}
    checks = check_baselines(roots, args.baseline_models, own_prices)
    print(f"Token accounting check ({len(checks)} baseline runs re-priced at their own prices):")
    print("\n".join(f"  {c}" for c in checks) or "  no baseline runs with recorded costs found")
    baselines = {t: find_baselines(t, roots, args.baseline_models) for t in tasks}
    for t in tasks:
        n = {short(b): len(rs) for b, rs in baselines[t].items()}
        print(f"baseline runs for {t}: {n or 'none'}")

    rows, totals = estimate(tasks, seeds_of, candidates, baselines, int(exp.get("repeats", 1)))
    md = report(rows, totals, args.baseline_models, candidates, int(exp.get("repeats", 1)))
    out = ROOT / "runs" / exp["name"] / "cost_estimate.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print()
    print(md)
    print(f"(written to {out})")


if __name__ == "__main__":
    main()
