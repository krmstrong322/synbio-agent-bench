"""Figures for runs: a scadnano-style design diagram and views of the simulated structure.

- design.svg: every helix as a forward/reverse track pair, strands as paths
  (scaffold thick blue, staples coloured), 5' ends as squares and 3' ends as
  arrows. Red shading marks designed base pairs that failed to form in simulation.
- structure_occupancy.png / structure_strands.png: the relaxed oxDNA mean
  structure from three orthogonal directions (its principal axes), coloured by
  base-pair occupancy or by strand. For shape tasks a fourth panel shows the sheet
  unrolled along its surface (what the shape score grades), with the fitted target
  outline; the top view shows the projected outline.

Everything is rendered from local data; no browser or online viewer is involved.
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import scadnano as sc

from origami_agent.design import load_design

from . import shapes

SCAFFOLD_COLOR = "#2c5aa0"
STAPLE_COLORS = ["#e15759", "#f28e2b", "#59a14f", "#b07aa1", "#9c755f", "#edc948", "#76b7b2", "#ff9da7", "#4e9a06", "#d37295"]


# ---------------------------------------------------------------------------
# Design diagram
# ---------------------------------------------------------------------------


def render_design_svg(
    design: sc.Design,
    out: Path,
    occupancy: dict[tuple[int, int, int], float] | None = None,
    title: str = "",
) -> Path:
    """Write a path-view SVG of `design`. `occupancy` maps (helix, offset, insertion) -> fraction formed."""
    helices = sorted(design.helices)
    row = {h: i for i, h in enumerate(helices)}
    doms = [d for s in design.strands for d in s.domains if isinstance(d, sc.Domain)]
    if not doms:
        raise ValueError("design has no domains")
    lo, hi = min(d.start for d in doms), max(d.end for d in doms)
    bp = float(np.clip(1500 / max(hi - lo, 1), 1.5, 8.0))
    row_h, track, left, top = 26, 9, 44, 34 if title else 12
    width, height = left + (hi - lo) * bp + 16, top + len(helices) * row_h + 8

    def x(o: float) -> float:
        return left + (o - lo + 0.5) * bp

    def y(h: int, fwd: bool) -> float:
        return top + row[h] * row_h + (0 if fwd else track)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}" font-family="sans-serif">',
        f'<rect width="100%" height="100%" fill="#ffffff"/>',
    ]
    if title:
        parts.append(f'<text x="{left}" y="20" font-size="14" fill="#222">{escape(title)}</text>')

    # helix bands and labels
    for h in helices:
        hd = [d for d in doms if d.helix == h]
        if hd:
            a, b = min(d.start for d in hd), max(d.end for d in hd)
            parts.append(
                f'<rect x="{x(a) - bp / 2:.1f}" y="{y(h, True) - 4:.1f}" width="{(b - a) * bp:.1f}" '
                f'height="{track + 8}" fill="#f1f3f5"/>'
            )
        parts.append(
            f'<text x="{left - 8}" y="{y(h, True) + track / 2 + 4:.1f}" font-size="11" fill="#555" '
            f'text-anchor="end">{h}</text>'
        )

    # failed base pairs
    if occupancy:
        for (h, o, _ins), occ in occupancy.items():
            if occ < 0.95 and h in row:
                parts.append(
                    f'<rect x="{x(o) - bp / 2:.1f}" y="{y(h, True) - 4:.1f}" width="{bp:.1f}" height="{track + 8}" '
                    f'fill="#d62728" fill-opacity="{0.15 + 0.7 * (1 - occ):.2f}"/>'
                )

    # deletions and insertions
    for d in doms:
        for o in d.deletions:
            cx, cy = x(o), y(d.helix, d.forward)
            parts.append(f'<path d="M{cx - 2},{cy - 2}L{cx + 2},{cy + 2}M{cx - 2},{cy + 2}L{cx + 2},{cy - 2}" stroke="#888" stroke-width="1"/>')
        for o, n in d.insertions:
            parts.append(f'<circle cx="{x(o):.1f}" cy="{y(d.helix, d.forward) - 5:.1f}" r="2.5" fill="none" stroke="#888"/>')

    # strands
    staple_i = 0
    for strand in design.strands:
        if strand.is_scaffold:
            color, w = SCAFFOLD_COLOR, 2.4
        else:
            color, w = STAPLE_COLORS[staple_i % len(STAPLE_COLORS)], 1.5
            staple_i += 1
        pts: list[tuple[float, float]] = []
        dashed_after: set[int] = set()
        for d in strand.domains:
            if isinstance(d, sc.Domain):
                first, last = (d.start, d.end - 1) if d.forward else (d.end - 1, d.start)
                pts += [(x(first), y(d.helix, d.forward)), (x(last), y(d.helix, d.forward))]
            elif isinstance(d, sc.Loopout) and pts:
                dashed_after.add(len(pts) - 1)
        if not pts:
            continue
        path = f"M{pts[0][0]:.1f},{pts[0][1]:.1f}" + "".join(f"L{px:.1f},{py:.1f}" for px, py in pts[1:])
        parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{w}" stroke-linejoin="round"/>')
        for k in dashed_after:
            (ax, ay), (bx, by) = pts[k], pts[k + 1]
            parts.append(f'<path d="M{ax:.1f},{ay:.1f}L{bx:.1f},{by:.1f}" stroke="#fff" stroke-width="{w}" stroke-dasharray="3,3"/>')
        sx, sy = pts[0]
        parts.append(f'<rect x="{sx - 2.5:.1f}" y="{sy - 2.5:.1f}" width="5" height="5" fill="{color}"/>')
        (ax, ay), (ex, ey) = pts[-2] if len(pts) > 1 else pts[0], pts[-1]
        direction = 1 if ex >= ax else -1
        parts.append(
            f'<path d="M{ex + 4 * direction:.1f},{ey:.1f}L{ex - 2 * direction:.1f},{ey - 3.5:.1f}'
            f'L{ex - 2 * direction:.1f},{ey + 3.5:.1f}Z" fill="{color}"/>'
        )

    parts.append("</svg>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


def occupancy_by_slot(sim_dir: Path) -> dict[tuple[int, int, int], float]:
    data = np.load(sim_dir / "analysis.npz")
    slots = json.loads((sim_dir / "index_map.json").read_text())["slots"]
    out = {}
    for (i, _), occ in zip(data["pairs"], data["pair_occupancy"]):
        h, o, ins, _ = slots[i]
        out[(h, o, ins)] = float(occ)
    return out


# ---------------------------------------------------------------------------
# Structure views
# ---------------------------------------------------------------------------


def render_structure_png(
    sim_dir: Path,
    out: Path,
    color_by: str = "occupancy",
    shape: dict[str, Any] | None = None,
    title: str = "",
) -> Path:
    """Three orthogonal views of the simulated mean structure, coloured by `occupancy`, `strands` or `rmsf`."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    data = np.load(sim_dir / "analysis.npz")
    imap = json.loads((sim_dir / "index_map.json").read_text())
    pts = data["mean_nm"]
    proj = shapes.project_to_plane(pts)
    strand_of = np.asarray(imap["strand_of"])
    scaffolds = set(imap.get("scaffold_strands") or [int(np.bincount(strand_of).argmax())])
    n = len(pts)

    cbar_label = None
    if color_by == "occupancy":
        values = np.full(n, np.nan)
        for (i, j), occ in zip(data["pairs"], data["pair_occupancy"]):
            values[i] = values[j] = occ
        cmap, vmin, vmax, cbar_label = plt.get_cmap("RdYlGn"), 0.0, 1.0, "fraction of time base-paired"
    elif color_by == "rmsf":
        values = data["rmsf_nm"]
        cmap, vmin, vmax, cbar_label = plt.get_cmap("viridis"), 0.0, float(np.percentile(values, 98)), "RMSF (nm)"
    elif color_by == "strands":
        staples = sorted(set(strand_of.tolist()) - scaffolds)
        index = {s: k + 1 for k, s in enumerate(staples)}
        values = np.array([0 if s in scaffolds else 1 + (index[s] - 1) % len(STAPLE_COLORS) for s in strand_of], float)
        cmap, vmin, vmax = ListedColormap(["#b8c2cc"] + STAPLE_COLORS), 0, len(STAPLE_COLORS)
    else:
        raise ValueError(f"unknown color_by {color_by!r}")

    ext = np.ptp(proj, axis=0) + 4
    views = [("top", 0, 1, 2), ("side", 0, 2, 1), ("end", 1, 2, 0)]
    ratios = [ext[0], ext[0], max(ext[1], 0.35 * ext[0])]
    unrolled = data["unrolled_xy"] if shape and "unrolled_xy" in data.files and len(data["unrolled_xy"]) else None
    if unrolled is not None:
        unrolled = shapes.project_to_plane(np.c_[unrolled, np.zeros(len(unrolled))])[:, :2]
        ratios.append(np.ptp(unrolled[:, 0]) + 4)
    height = float(np.clip(13 * max(ext[1], ext[2]) / sum(ratios[:3]) + 1.3, 2.8, 6.0))
    fig, axes = plt.subplots(
        1, len(ratios), figsize=(15 if unrolled is None else 19, height), layout="constrained",
        gridspec_kw={"width_ratios": ratios},
    )
    size = float(np.clip(25000 / n, 1.2, 8))
    sc_handle = None
    for ax, (name, a, b, depth) in zip(axes, views):
        order = np.argsort(proj[:, depth])
        v = values[order]
        nan = np.isnan(v)
        if nan.any():
            ax.scatter(proj[order][nan, a], proj[order][nan, b], s=size, c="#cfd4da", linewidths=0)
        sc_handle = ax.scatter(
            proj[order][~nan, a], proj[order][~nan, b], s=size, c=v[~nan], cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0
        )
        ax.set_aspect("equal")
        ax.set_title(f"{name} view", fontsize=10)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("nm", fontsize=8)
    if shape:
        cells, flatness = shapes.outline_cells(pts)
        fit = shapes.best_iou(cells, shape)
        poly = np.array(fit["polygon"] + fit["polygon"][:1])
        axes[0].plot(poly[:, 0], poly[:, 1], "k--", lw=1.2)
        axes[0].set_title(f"top view — projected IoU {fit['iou']:.2f}, flatness {flatness:.2f}", fontsize=10)
    if unrolled is not None:
        ufit = shapes.best_iou(shapes._dilated_cells(unrolled), shape)
        upoly = np.array(ufit["polygon"] + ufit["polygon"][:1])
        ax = axes[3]
        ax.scatter(unrolled[:, 0], unrolled[:, 1], s=max(size, 2.0), c="#4e79a7", linewidths=0)
        ax.plot(upoly[:, 0], upoly[:, 1], "k--", lw=1.2)
        ax.set_aspect("equal")
        ax.set_title(f"unrolled sheet — target {shape['type']} (graded IoU {ufit['iou']:.2f})", fontsize=10)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("nm", fontsize=8)
    if cbar_label and sc_handle is not None:
        fig.colorbar(sc_handle, ax=axes, shrink=0.7, pad=0.01, label=cbar_label)
    if title:
        fig.suptitle(title, fontsize=12)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Per-run figures
# ---------------------------------------------------------------------------


def best_sim_dir(run_dir: Path) -> Path | None:
    """Prefer the grader's simulation; fall back to the agent's latest one."""
    graded = sorted((run_dir / "eval").glob("seed_*/analysis.npz"))
    if graded:
        return graded[0].parent
    agent = sorted((run_dir / "workspace" / "sims").glob("sim_*/analysis.npz"))
    return agent[-1].parent if agent else None


def figures_for_run(run_dir: Path, spec: dict[str, Any], title: str = "", force: bool = False) -> dict[str, str]:
    """Render the figures for one run into run_dir/figures; returns {name: path relative to run_dir}."""
    figs: dict[str, str] = {}
    out = run_dir / "figures"
    design_path = run_dir / "workspace" / "final" / "design.sc"
    sim = best_sim_dir(run_dir)
    if design_path.is_file():
        target = out / "design.svg"
        if force or not target.exists():
            occ = occupancy_by_slot(sim) if sim and sim.parent.name == "eval" else None
            render_design_svg(load_design(design_path), target, occ, title)
        figs["design"] = target.relative_to(run_dir).as_posix()
    if sim:
        for mode in ("occupancy", "strands"):
            target = out / f"structure_{mode}.png"
            if force or not target.exists():
                render_structure_png(sim, target, mode, spec.get("shape"), title)
            figs[f"structure_{mode}"] = target.relative_to(run_dir).as_posix()
        figs["sim_dir"] = sim.relative_to(run_dir).as_posix()
    return figs
