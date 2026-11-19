"""Collect graded runs from several experiments into a publishable results/ folder: tables and figures.

    uv run python scripts/export_results.py
    uv run python scripts/export_results.py --source runs/open-models --source runs/origami-trio-skin01 ...

Only runs graded the same way are included (Modal, the same verlet_skin), so every model is compared on
equal terms; anything else is listed as excluded. Writes:

    results/runs.csv          one row per graded run (scores, per-seed shape scores, cost, calls, tokens)
    results/summary.csv       one row per model: score per task, mean, agent cost
    results/README.md         the summary table, figures and how to read them
    results/figures/*.png|svg score vs cost, scores by task, tetrahedron per-seed shape scores

runs/ stays local (it holds trajectories); results/ is small and meant to be committed.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = ["runs/open-models", "runs/origami-trio-skin01", "runs/wireframe-tetrahedron-modal"]
TASKS = {  # task id -> short label, in display order
    "origami-shapes/rectangle-4000": "Rectangle",
    "origami-shapes/triangle-4000": "Triangle",
    "origami/6hb-honeycomb-3k": "6-helix bundle",
    "origami-wireframe/tetrahedron": "Tetrahedron",
}
NAMES = {
    "claude-haiku-5-5": "Claude Haiku 5.5", "claude-sonnet-5-5": "Claude Sonnet 5.5",
    "claude-opus-5-5": "Claude Opus 5.5", "gpt-6-luna": "GPT-6 Luna", "gpt-6.1-terra": "GPT-6.1 Terra",
    "gpt-6.1-sol": "GPT-6.1 Sol", "glm-5.3": "GLM 5.3", "qwen3.8-max-0902": "Qwen3.8 Max",
    "kimi-k3.1-pro": "Kimi K3.1 Pro", "reference": "Expert reference",
}
# Validated categorical slots 1-3 (dataviz reference palette; all-pairs CVD-safe), one per model family.
FAMILY_COLOR = {"Anthropic": "#2a78d6", "OpenAI": "#eb6834", "Open-weight": "#1baf7a", "Reference": "#52514e"}
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def family(model: str) -> str:
    if model == "reference":
        return "Reference"
    if model.startswith("claude-"):
        return "Anthropic"
    if model.startswith(("gpt-", "o1", "o3", "o4")):
        return "OpenAI"
    return "Open-weight"


def name(model: str) -> str:
    return NAMES.get(model, model)


def _seed_scores(run_dir: Path, task: str) -> list[float]:
    """Per-seed shape score: wireframe F1 for wireframe tasks, unrolled-sheet IoU for flat shapes."""
    out = []
    for r in sorted((run_dir / "eval").glob("seed_*/result.json"), key=lambda p: int(p.parent.name.split("_")[1])):
        m = (json.loads(r.read_text()).get("metrics") or {})
        key, sub = ("wireframe_fit", "f1") if "wireframe" in task else ("shape_unrolled", "iou")
        v = (m.get(key) or {}).get(sub)
        if v is not None:
            out.append(round(float(v), 4))
    return out


def collect(sources: list[Path], runner: str, skin: float) -> tuple[list[dict], list[str]]:
    rows, excluded, seen = [], [], {}
    for src in sources:
        for res in sorted(src.glob("*/*/rep*/result.json")):
            run_dir = res.parent
            info = json.loads(res.read_text())
            g = info.get("grade") or {}
            task, model = info.get("task"), info.get("model")
            if task not in TASKS:
                continue
            where = f"{src.name}/{run_dir.relative_to(src).as_posix()}"
            if g.get("sim_runner") != runner or g.get("verlet_skin") != skin:
                if g.get("valid") is not False:  # invalid designs are never simulated, so they carry no runner
                    excluded.append(f"{where}: graded with runner={g.get('sim_runner')} skin={g.get('verlet_skin')}")
                    continue
            key = (task, model, run_dir.name)
            if key in seen:
                excluded.append(f"{where}: duplicate of {seen[key]}")
                continue
            seen[key] = where
            tokens = info.get("tokens") or {}
            rows.append({
                "task": task, "task_label": TASKS[task], "model": model, "model_name": name(model),
                "family": family(model), "rep": run_dir.name, "source": src.name,
                "overall": g.get("overall"), "valid": g.get("valid"), "spec_score": g.get("spec_score"),
                "bp_formed": g.get("sim_score"), "shape_score": g.get("shape_score"),
                "seed_shape_scores": _seed_scores(run_dir, task),
                "cost_usd": info.get("cost_usd"), "model_calls": info.get("n_model_calls"),
                "input_tokens": tokens.get("input_tokens"), "cached_tokens": tokens.get("cache_read_tokens"),
                "output_tokens": tokens.get("output_tokens"), "wall_time_min": round((info.get("wall_time_s") or 0) / 60, 1),
                "termination": info.get("termination"), "submitted": bool(info.get("submitted")) or model == "reference",
                "sim_runner": g.get("sim_runner"), "verlet_skin": g.get("verlet_skin"),
            })
    return rows, excluded


def summarize(rows: list[dict]) -> list[dict]:
    models: dict[str, dict] = {}
    for r in rows:
        m = models.setdefault(r["model"], {"model": r["model"], "model_name": r["model_name"], "family": r["family"],
                                            "cost_usd": 0.0, "runs": 0, "_scores": {}})
        m["_scores"].setdefault(r["task"], []).append(r["overall"] or 0.0)
        m["cost_usd"] += r["cost_usd"] or 0.0
        m["runs"] += 1
    out = []
    for m in models.values():
        row = {k: v for k, v in m.items() if not k.startswith("_")}
        per_task = {t: statistics.mean(v) for t, v in m["_scores"].items()}
        for t, label in TASKS.items():
            row[label] = round(per_task[t], 4) if t in per_task else None
        complete = all(t in per_task for t in TASKS)
        row["mean"] = round(statistics.mean(per_task.values()), 4) if complete else None
        row["cost_usd"] = round(row["cost_usd"], 3)
        out.append(row)
    return sorted(out, key=lambda r: (r["model"] == "reference", -(r["mean"] or 0)))


# ---------------------------------------------------------------------------
# Figures (static: every figure has a table in README.md / the post)
# ---------------------------------------------------------------------------


def _style(ax, grid_axis: str = "x") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _header(fig, title: str, families: list[str]) -> None:
    """Title on the first line, legend on the second (left-aligned), then lay the axes out below them."""
    import matplotlib.lines as mlines

    h = fig.get_figheight()
    fig.suptitle(title, x=0.012, y=1 - 0.10 / h, ha="left", va="top", fontsize=12, color=INK, fontweight="bold")
    handles = [mlines.Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor=FAMILY_COLOR[f],
                             markeredgecolor=SURFACE, label=f) for f in families]
    fig.legend(handles=handles, loc="upper left", frameon=False, fontsize=9, labelcolor=INK2, ncol=len(handles),
               handletextpad=0.3, columnspacing=1.2, bbox_to_anchor=(0.0, 1 - 0.36 / h), borderaxespad=0)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.50 / h), pad=0.6)


def _save(fig, out: Path, stem: str) -> None:
    for ext in ("png", "svg"):
        fig.savefig(out / f"{stem}.{ext}", dpi=200, facecolor=SURFACE, bbox_inches="tight")


def _place_labels(fig, ax, xs: list[float], ys: list[float], texts: list[str], point_px: float = 7.0) -> None:
    """Direct-label each point at the first nearby spot that overlaps no other label or point (checked in pixels);
    labels set further out get a leader line back to their point."""
    from matplotlib.transforms import Bbox

    renderer = fig.canvas.get_renderer()
    to_px = ax.transData.transform
    pts_px = [to_px((x, y)) for x, y in zip(xs, ys)]
    point_boxes = [Bbox.from_extents(px - point_px, py - point_px, px + point_px, py + point_px) for px, py in pts_px]
    axes_box = ax.get_window_extent(renderer)
    near = [(9, 0, "left", "center"), (-9, 0, "right", "center"), (0, 11, "center", "bottom"),
            (0, -11, "center", "top"), (9, 9, "left", "bottom"), (9, -9, "left", "top"),
            (-9, 9, "right", "bottom"), (-9, -9, "right", "top")]
    far = [(dx * 2.2, dy * 2.2, ha, va) for dx, dy, ha, va in near if (dx, dy) != (0, 0)] + \
          [(dx * 3.4, dy * 3.4, ha, va) for dx, dy, ha, va in near]
    placed: list = []
    # Crowded points first: they have the fewest free spots.
    crowd = [sum(abs(a[0] - b[0]) < 120 and abs(a[1] - b[1]) < 40 for b in pts_px) for a in pts_px]
    for i in sorted(range(len(xs)), key=lambda i: -crowd[i]):
        best = None
        for k, (dx, dy, ha, va) in enumerate(near + far):
            t = ax.annotate(texts[i], (xs[i], ys[i]), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                            fontsize=8.5, color=INK, zorder=5)
            bb = t.get_window_extent(renderer).expanded(1.06, 1.15)
            clash = any(bb.overlaps(o) for o in placed) or any(
                bb.overlaps(pb) for j, pb in enumerate(point_boxes) if j != i)
            inside = axes_box.contains(bb.x0, bb.y0) and axes_box.contains(bb.x1, bb.y1)
            if not clash and inside:
                best = (t, bb, k >= len(near))
                break
            t.remove()
        if best is None:  # nothing free: furthest right spot with a leader line
            t = ax.annotate(texts[i], (xs[i], ys[i]), xytext=(30, 0), textcoords="offset points", ha="left",
                            va="center", fontsize=8.5, color=INK, zorder=5)
            best = (t, t.get_window_extent(renderer), True)
        t, bb, is_far = best
        if is_far:
            t.arrow_patch = None
            t.set_annotation_clip(False)
            t.arrowprops = {"arrowstyle": "-", "color": MUTED, "linewidth": 0.6, "shrinkA": 1, "shrinkB": 5}
            t._arrow_relpos = (0.5, 0.5)
            t.update_positions(renderer)
        placed.append(bb)


def fig_score_vs_cost(summary: list[dict], out: Path) -> None:
    import matplotlib.pyplot as plt

    pts = [s for s in summary if s["model"] != "reference" and s["mean"] is not None and s["cost_usd"] > 0]
    fig, ax = plt.subplots(figsize=(7.2, 4.8), facecolor=SURFACE)
    _style(ax, grid_axis="both")
    for s in pts:
        ax.scatter(s["cost_usd"], s["mean"], s=72, color=FAMILY_COLOR[s["family"]], edgecolor=SURFACE,
                   linewidth=1.5, zorder=3)
    ax.set_xscale("log")
    ys = [s["mean"] for s in pts]
    span = (max(ys) - min(ys)) or 0.01
    ax.set_xlim(min(s["cost_usd"] for s in pts) / 2.2, max(s["cost_usd"] for s in pts) * 3.5)
    ax.set_ylim(min(ys) - span * 0.15, max(ys) + span * 0.15)
    ax.set_xlabel("Agent API cost for the four designs (USD, log scale)", color=INK2, fontsize=9)
    ax.set_ylabel("Mean score (0-1)", color=INK2, fontsize=9)
    from matplotlib.ticker import FuncFormatter

    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
    _header(fig, "Score against cost", [f for f in ("Anthropic", "OpenAI", "Open-weight") if any(s["family"] == f for s in pts)])
    _place_labels(fig, ax, [s["cost_usd"] for s in pts], ys, [s["model_name"] for s in pts])
    _save(fig, out, "score_vs_cost")
    plt.close(fig)


def fig_scores_by_task(rows: list[dict], summary: list[dict], out: Path) -> None:
    import matplotlib.pyplot as plt

    order = [s["model"] for s in summary if s["model"] != "reference"][::-1]  # best at top
    fig, axes = plt.subplots(1, len(TASKS), figsize=(10, 0.36 * len(order) + 1.6), sharey=True, facecolor=SURFACE)
    for ax, (task, label) in zip(axes, TASKS.items()):
        _style(ax, grid_axis="x")
        vals = {r["model"]: r["overall"] for r in rows if r["task"] == task and r["model"] in order}
        for y, m in enumerate(order):
            if m in vals and vals[m] is not None:
                ax.scatter(vals[m], y, s=56, color=FAMILY_COLOR[family(m)], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ref = [r["overall"] for r in rows if r["task"] == task and r["model"] == "reference"]
        if ref:
            ax.axvline(ref[0], color=MUTED, linewidth=1, linestyle=(0, (3, 3)), zorder=2)
            ax.text(ref[0], len(order) - 0.4, " reference", fontsize=7.5, color=INK2, va="bottom")
        present = [v for v in vals.values() if v is not None] + ref
        lo, hi = min(present), max(present)
        pad = max((hi - lo) * 0.15, 0.005)
        ax.set_xlim(lo - pad, min(1.0, hi + pad) if hi < 1 else hi + pad)
        ax.set_title(label, fontsize=10, color=INK, loc="left")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([name(m) for m in order], fontsize=9, color=INK)
        ax.tick_params(axis="x", labelsize=8)
    fig.supxlabel("Score (0-1); each panel has its own range", color=INK2, fontsize=9)
    _header(fig, "Score by design", [f for f in ("Anthropic", "OpenAI", "Open-weight") if any(family(m) == f for m in order)])
    _save(fig, out, "scores_by_design")
    plt.close(fig)


def fig_tetra_seeds(rows: list[dict], out: Path) -> None:
    import matplotlib.pyplot as plt

    tet = [r for r in rows if r["task"] == "origami-wireframe/tetrahedron" and r["seed_shape_scores"]]
    if not tet:
        return
    tet.sort(key=lambda r: statistics.mean(r["seed_shape_scores"]))
    fig, ax = plt.subplots(figsize=(7.2, 0.4 * len(tet) + 1.5), facecolor=SURFACE)
    _style(ax, grid_axis="x")
    for y, r in enumerate(tet):
        c = FAMILY_COLOR[r["family"]]
        seeds = r["seed_shape_scores"]
        ax.plot([min(seeds), max(seeds)], [y, y], color=c, linewidth=2, solid_capstyle="round", zorder=2)
        ax.scatter(seeds, [y] * len(seeds), s=40, color=c, edgecolor=SURFACE, linewidth=1.2, zorder=3)
        ax.scatter([statistics.mean(seeds)], [y], s=90, facecolor=SURFACE, edgecolor=c, linewidth=2, zorder=4)
    ax.set_yticks(range(len(tet)))
    ax.set_yticklabels([r["model_name"] for r in tet], fontsize=9, color=INK)
    lowest = min(v for r in tet for v in r["seed_shape_scores"])
    ax.set_xlim(max(0.0, round(lowest - 0.08, 1)), 1.0)
    ax.set_xlabel("Wireframe F1 per oxDNA seed (dots), mean (ring); 1 = traces the target exactly",
                  color=INK2, fontsize=9)
    fams = [f for f in ("Anthropic", "OpenAI", "Open-weight", "Reference") if any(r["family"] == f for r in tet)]
    _header(fig, "Tetrahedron: shape score per simulation seed", fams)
    _save(fig, out, "tetrahedron_seeds")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()})


def summary_table(summary: list[dict]) -> str:
    head = "| Model | Family | " + " | ".join(TASKS.values()) + " | Mean | Agent cost |"
    lines = [head, "|---|---|" + "---|" * (len(TASKS) + 2)]
    for s in summary:
        cells = " | ".join("–" if s[label] is None else f"{s[label]:.3f}" for label in TASKS.values())
        mean = "–" if s["mean"] is None else f"**{s['mean']:.3f}**"
        cost = "–" if s["model"] == "reference" else f"${s['cost_usd']:.2f}"
        lines.append(f"| {s['model_name']} | {s['family']} | {cells} | {mean} | {cost} |")
    return "\n".join(lines)


def tetra_table(rows: list[dict]) -> str:
    tet = sorted((r for r in rows if r["task"] == "origami-wireframe/tetrahedron" and r["seed_shape_scores"]),
                 key=lambda r: -statistics.mean(r["seed_shape_scores"]))
    lines = ["| Model | Seed 1 | Seed 2 | Seed 3 | Mean |", "|---|---|---|---|---|"]
    for r in tet:
        s = r["seed_shape_scores"] + [None] * (3 - len(r["seed_shape_scores"]))
        cells = " | ".join("–" if v is None else f"{v:.2f}" for v in s[:3])
        lines.append(f"| {r['model_name']} | {cells} | {statistics.mean(r['seed_shape_scores']):.2f} |")
    return "\n".join(lines)


def readme(summary: list[dict], rows: list[dict], excluded: list[str], runner: str, skin: float) -> str:
    n_models = len([s for s in summary if s["model"] != "reference"])
    return f"""# Results

{n_models} models designed four DNA origami structures as autonomous agents; each design was then graded with
spec checks and coarse-grained oxDNA simulation. Generated by `scripts/export_results.py` from the local `runs/`.

All runs here were graded the same way: oxDNA on Modal GPUs (`{runner}`), `verlet_skin` {skin}, the `shape`
protocol, one oxDNA seed for the flat designs and three for the tetrahedron. One agent run per model and design.

## Overall score (0-1)

{summary_table(summary)}

Overall score: the spec-check pass rate, the fraction of designed base pairs formed in simulation and, for shaped
designs, the shape score (unrolled-sheet IoU for the flat shapes, wireframe F1 for the tetrahedron), weighted
equally. Agent cost is the API cost of the four agent runs, from recorded token use at list prices.

![Score against cost](figures/score_vs_cost.png)

![Score by design](figures/scores_by_design.png)

## Tetrahedron: shape score per simulation seed

{tetra_table(rows)}

![Tetrahedron per-seed shape score](figures/tetrahedron_seeds.png)

## How to read this

- **One agent run per model and design.** Run-to-run variation is large: one model's two tetrahedron attempts
  scored 0.974 and 0.928. Differences of a few hundredths are within that noise; treat them as ties.
- The 6-helix bundle barely separates models (almost every design keeps nearly all its base pairs). The
  tetrahedron separates them most.
- Simulation seeds add their own noise on the tetrahedron: the expert reference design ranges widely across seeds.

Files: `runs.csv` (one row per run), `summary.csv` (one row per model), `figures/` (PNG and SVG).
""" + ("\nExcluded (graded differently, or duplicates):\n\n" + "\n".join(f"- {e}" for e in excluded) + "\n"
       if excluded else "")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", action="append", help="experiment run dirs (default: open-models, the Modal "
                                                       "re-grades of origami-trio and wireframe-tetrahedron)")
    p.add_argument("--runner", default="modal")
    p.add_argument("--skin", type=float, default=0.1)
    p.add_argument("--out", default=str(ROOT / "results"))
    args = p.parse_args()

    sources = [Path(s) if Path(s).is_absolute() else ROOT / s for s in (args.source or DEFAULT_SOURCES)]
    missing = [str(s) for s in sources if not s.is_dir()]
    if missing:
        raise SystemExit(f"missing run dirs: {', '.join(missing)}")
    rows, excluded = collect(sources, args.runner, args.skin)
    if not rows:
        raise SystemExit("no graded runs found")
    summary = summarize(rows)
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    write_csv(out / "runs.csv", rows)
    write_csv(out / "summary.csv", summary)
    import matplotlib

    matplotlib.use("Agg")
    fig_score_vs_cost(summary, out / "figures")
    fig_scores_by_task(rows, summary, out / "figures")
    fig_tetra_seeds(rows, out / "figures")
    (out / "README.md").write_text(readme(summary, rows, excluded, args.runner, args.skin), encoding="utf-8")

    print(summary_table(summary))
    incomplete = [s["model_name"] for s in summary if s["mean"] is None and s["model"] != "reference"]
    if incomplete:
        print(f"\n[note] no mean for {', '.join(incomplete)}: a design is missing")
    if excluded:
        print(f"\n[note] {len(excluded)} run(s) excluded (graded differently or duplicates); listed in results/README.md")
    print(f"\nwrote {out}/ (runs.csv, summary.csv, README.md, figures/)")


if __name__ == "__main__":
    main()
