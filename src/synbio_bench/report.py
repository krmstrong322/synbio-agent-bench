"""Aggregate result.json files into a per-run CSV and a model x task summary."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import yaml

from .cost import estimate_cost

ROOT = Path(__file__).resolve().parents[2]

COLUMNS = [
    "task", "model", "seed", "termination", "submitted", "valid", "overall", "spec_score", "sim_score",
    "shape_score", "shape_projected_iou", "flatness", "bp_fraction_mean", "rmsf_mean_nm", "rmsd_to_design_nm", "initial_stretched_bonds",
    "simulations_used", "n_model_calls", "input_tokens", "output_tokens", "cost_usd", "wall_time_s", "sim_runner",
]


def with_cost(r: dict, prices: dict[str, dict]) -> dict:
    """Price a run with the current models.yaml rates (so older runs and price changes are covered)."""
    if cost := estimate_cost(r.get("usage") or {}, prices.get(r["model"])):
        r["cost"], r["cost_usd"] = cost, cost["total_usd"]
    return r


def load_prices(models_path: str | Path) -> dict[str, dict]:
    models = yaml.safe_load(Path(models_path).read_text())["models"]
    return {m["id"]: m["price_per_mtok"] for m in models if m.get("price_per_mtok")}


def flatten(r: dict) -> dict:
    g = r.get("grade", {})
    t = r.get("tokens", {})
    return {
        "task": r["task"], "model": r["model"], "seed": r.get("seed"), "termination": r.get("termination"),
        "submitted": g.get("submitted"), "valid": g.get("valid"), "overall": g.get("overall"),
        "spec_score": g.get("spec_score"), "sim_score": g.get("sim_score"), "shape_score": g.get("shape_score"),
        "shape_projected_iou": g.get("shape_projected_iou"), "flatness": g.get("flatness"),
        "bp_fraction_mean": g.get("bp_fraction_mean"), "rmsf_mean_nm": g.get("rmsf_mean_nm"),
        "rmsd_to_design_nm": g.get("rmsd_to_design_nm"), "initial_stretched_bonds": g.get("initial_stretched_bonds"),
        "simulations_used": r.get("simulations_used"), "n_model_calls": r.get("n_model_calls"),
        "input_tokens": t.get("input_tokens"), "output_tokens": t.get("output_tokens"),
        "cost_usd": r.get("cost_usd"), "wall_time_s": r.get("wall_time_s"), "sim_runner": g.get("sim_runner"),
    }


def _ms(vals: list[float | None], digits: int = 2) -> str:
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.{digits}f}"
    return f"{statistics.mean(vals):.{digits}f} ± {statistics.stdev(vals):.{digits}f}"


def _k(v: float | None) -> float | None:
    return v / 1000 if v is not None else None


def _ended(rs: list[dict]) -> str:
    """How runs terminated, e.g. 'completed' or 'completed 2, recursion_limit 1'."""
    counts: dict[str, int] = defaultdict(int)
    for r in rs:
        counts[r["termination"] or "?"] += 1
    if len(counts) == 1 and len(rs) == 1:
        return next(iter(counts))
    return ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))


def cmd_report(args: argparse.Namespace) -> None:
    exp = yaml.safe_load(Path(args.experiment).read_text())
    root = ROOT / "runs" / exp["name"]
    prices = load_prices(args.models)
    rows = [flatten(with_cost(json.loads(p.read_text()), prices)) for p in sorted(root.glob("*/*/rep*/result.json"))]
    if not rows:
        print(f"no graded runs under {root}")
        return
    with open(root / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["task"], r["model"])].append(r)
    lines = [
        f"# {exp['name']}: results", "",
        "| task | model | n | ended | valid | overall | spec | bp formed | shape IoU | RMSF nm | tokens in/out (k) | cost $ | time min |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (task, model), rs in sorted(groups.items()):
        lines.append(
            f"| {task} | {model} | {len(rs)} | {_ended(rs)} | {sum(bool(r['valid']) for r in rs)}/{len(rs)} "
            f"| {_ms([r['overall'] for r in rs])} | {_ms([r['spec_score'] for r in rs])} "
            f"| {_ms([r['bp_fraction_mean'] for r in rs])} | {_ms([r['shape_score'] for r in rs])} "
            f"| {_ms([r['rmsf_mean_nm'] for r in rs])} "
            f"| {_ms([_k(r['input_tokens']) for r in rs], 0)} / {_ms([_k(r['output_tokens']) for r in rs], 0)} "
            f"| {_ms([r['cost_usd'] for r in rs], 3)} "
            f"| {_ms([r['wall_time_s'] / 60 if r['wall_time_s'] else None for r in rs], 1)} |"
        )
    costs = [r["cost_usd"] for r in rows if r["cost_usd"] is not None]
    if costs:
        lines += ["", f"Estimated API cost across these runs: **${sum(costs):.3f}** (cache-aware, from configs/models.yaml prices)."]
    if any(r["sim_runner"] == "mock" for r in rows):
        lines += ["", "**Warning:** some runs were graded with the mock oxDNA runner; their sim scores are synthetic."]
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {root / 'results.csv'} and {root / 'summary.md'}")
    if not args.no_html:
        from .html_report import write_html

        out = write_html(root, exp, lines, embed=args.embed, figures=not args.no_figures, prices=prices)
        print(f"wrote {out}")


def add_report_parser(sub) -> None:
    r = sub.add_parser("report", help="aggregate graded runs into results.csv, summary.md and report.html")
    r.add_argument("--no-html", action="store_true", help="skip report.html")
    r.add_argument("--no-figures", action="store_true", help="HTML report without rendering figures")
    r.add_argument("--embed", action="store_true", help="embed images so the HTML file stands alone (larger)")
    r.set_defaults(func=cmd_report)
