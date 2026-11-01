"""Lightweight oxDNA trajectory analysis (numpy only; no oxpy needed).

Metrics:
- designed base-pair occupancy from the `hb_list` observable
- RMSF (fluctuation about the mean structure)
- RMSD of the mean structure to the idealised scadnano geometry
- extents / radius of gyration of the mean structure
- backbone bond lengths of the exported, unrelaxed configuration (a crossover
  placed at the wrong offset shows up as an unusually long bond)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np

OX_TO_NM = 0.8518
STRETCHED_BOND_NM = 2.6  # scadnano exports helix axes 3 nm apart, so a well-placed crossover is ~2.15 nm before relaxation


# ---------------------------------------------------------------------------
# File IO
# ---------------------------------------------------------------------------


def read_conf(path: str | Path) -> tuple[list[str], np.ndarray]:
    lines = Path(path).read_text().splitlines()
    return lines[:3], np.array([[float(x) for x in ln.split()] for ln in lines[3:] if ln.strip()])


def format_conf(header: list[str], body: np.ndarray) -> str:
    return "\n".join(header) + "\n" + "\n".join(" ".join(f"{x:.6f}" for x in row) for row in body) + "\n"


def write_conf(path: str | Path, header: list[str], body: np.ndarray) -> None:
    Path(path).write_text(format_conf(header, body))


def box_of(header: list[str]) -> np.ndarray:
    return np.array([float(x) for x in header[1].split("=")[1].split()])


def iter_trajectory(path: str | Path, n: int) -> Iterator[tuple[list[str], np.ndarray]]:
    with open(path) as f:
        while True:
            header = [f.readline().strip() for _ in range(3)]
            if not header[0]:
                return
            rows = [f.readline() for _ in range(n)]
            if not rows[-1]:
                return  # truncated final frame
            yield header, np.array([[float(x) for x in r.split()[:3]] for r in rows])


def read_topology(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (strand_id, n3) arrays from an old-style oxDNA topology."""
    lines = Path(path).read_text().split("\n")
    n = int(lines[0].split()[0])
    rows = [ln.split() for ln in lines[1 : n + 1]]
    strand = np.array([int(r[0]) for r in rows])
    n3 = np.array([int(r[2]) for r in rows])
    return strand, n3


def read_hb_list(path: str | Path) -> list[set[tuple[int, int]]]:
    frames: list[set[tuple[int, int]]] = []
    if not Path(path).exists():
        return frames
    for ln in Path(path).read_text().splitlines():
        if ln.startswith("#"):
            frames.append(set())
        elif ln.strip() and frames:
            a, b = (int(x) for x in ln.split()[:2])
            frames[-1].add((min(a, b), max(a, b)))
    return frames


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def unwrap_strands(pos: np.ndarray, strand: np.ndarray, box: np.ndarray, links: np.ndarray) -> np.ndarray:
    """Put every strand in the periodic image that keeps the structure whole.

    oxDNA writes each strand contiguously but wraps strands into the box
    independently, so after the structure drifts, partners can sit a box length
    apart. Starting from the longest strand, walk the designed base pairs
    (`links`, an (n, 2) index array) and move each newly reached strand to the
    image nearest its already-placed partner. This is exact however far the
    structure has drifted or rotated. Strands with no pairs go to the image
    nearest the structure's centre.
    """
    out = pos.copy()
    ids, counts = np.unique(strand, return_counts=True)
    placed = {int(ids[np.argmax(counts)])}
    frontier = list(placed)
    while frontier:
        s = frontier.pop()
        for i, j in links:
            for a, b in ((i, j), (j, i)):
                t = int(strand[b])
                if strand[a] == s and t not in placed:
                    out[strand == t] -= np.round((out[b] - out[a]) / box) * box
                    placed.add(t)
                    frontier.append(t)
    centre = out[np.isin(strand, list(placed))].mean(0)
    for t in set(map(int, ids)) - placed:
        m = strand == t
        out[m] -= np.round((out[m].mean(0) - centre) / box) * box
    return out


def kabsch_align(mobile: np.ndarray, target: np.ndarray) -> np.ndarray:
    mc, tc = mobile.mean(0), target.mean(0)
    a, b = mobile - mc, target - tc
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1, 1, d]) @ vt
    return a @ rot + tc


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum(1).mean()))


def principal_extents(pos: np.ndarray) -> list[float]:
    c = pos - pos.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    proj = c @ vt.T
    return sorted((float(x) * OX_TO_NM for x in proj.max(0) - proj.min(0)), reverse=True)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def _slot_label(slot: list | None) -> str:
    if slot is None:
        return "ss-loop/extension"
    h, off, ins, fwd = slot
    return f"helix {h} offset {off}{'' if ins < 0 else f' ins{ins}'} {'fwd' if fwd else 'rev'}"


def backbone_report(top: Path, conf: Path, index_map: Path | None = None) -> dict[str, Any]:
    _, n3 = read_topology(top)
    _, body = read_conf(conf)
    pos = body[:, :3]
    i = np.nonzero(n3 >= 0)[0]
    d = np.linalg.norm(pos[i] - pos[n3[i]], axis=1) * OX_TO_NM
    slots = json.loads(Path(index_map).read_text())["slots"] if index_map else None
    worst = np.argsort(d)[::-1][:10]
    stretched = int((d > STRETCHED_BOND_NM).sum())
    return {
        "median_bond_nm": round(float(np.median(d)), 3),
        "max_bond_nm": round(float(d.max()), 3),
        "num_stretched_bonds": stretched,
        "worst_bonds": [
            {
                "nm": round(float(d[k]), 2),
                "from": _slot_label(slots[i[k]]) if slots else int(i[k]),
                "to": _slot_label(slots[n3[i[k]]]) if slots else int(n3[i[k]]),
            }
            for k in worst
            if d[k] > STRETCHED_BOND_NM
        ],
    }


def _low_occupancy_regions(pairs: list[tuple[int, int]], occ: np.ndarray, slots: list, threshold: float = 0.5) -> list[dict]:
    """Group poorly formed designed pairs into contiguous (helix, offset) runs."""
    by_helix: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for (i, _), o in zip(pairs, occ):
        if o < threshold:
            h, off, _, _ = slots[i]
            by_helix[h].append((off, float(o)))
    regions = []
    for h, items in sorted(by_helix.items()):
        items.sort()
        start, prev, vals = items[0][0], items[0][0], [items[0][1]]
        for off, o in items[1:] + [(None, None)]:
            if off is not None and off <= prev + 1:
                prev = off
                vals.append(o)
                continue
            regions.append({"helix": h, "offsets": [start, prev], "n_pairs": len(vals), "mean_occupancy": round(float(np.mean(vals)), 2)})
            if off is not None:
                start, prev, vals = off, off, [o]
    return sorted(regions, key=lambda r: -r["n_pairs"])[:15]


def analyze_run(work: str | Path, shape: dict[str, Any] | None = None) -> dict[str, Any]:
    """Metrics for a finished production run; `shape` adds flat-shape fidelity of the mean structure."""
    work = Path(work)
    strand, n3 = read_topology(work / "initial.top")
    n = len(strand)
    imap = json.loads((work / "index_map.json").read_text())
    pairs = [tuple(p) for p in imap["pairs"]]
    slots = imap["slots"]

    init_header, init_body = read_conf(work / "initial.dat")
    ref = init_body[:, :3]

    # one designed pair per pair of strands is enough to stitch the structure back together
    links = np.array(list({(strand[i], strand[j]): (i, j) for i, j in pairs if strand[i] != strand[j]}.values()))
    frames = [unwrap_strands(p, strand, box_of(h), links) for h, p in iter_trajectory(work / "trajectory.dat", n)]
    if not frames:
        raise RuntimeError("production trajectory has no complete frames")
    window = frames[len(frames) // 2 :]

    # iterative mean structure: align to first frame, average, re-align to mean
    mean = window[0]
    for _ in range(3):
        aligned = np.array([kabsch_align(f, mean) for f in window])
        mean = aligned.mean(0)
    rmsf = np.sqrt(((aligned - mean) ** 2).sum(2).mean(0)) * OX_TO_NM
    mean_vs_design = kabsch_align(mean, ref)

    metrics: dict[str, Any] = {
        "n_frames": len(frames),
        "n_frames_analyzed": len(window),
        "rmsf_mean_nm": round(float(rmsf.mean()), 3),
        "rmsf_p95_nm": round(float(np.percentile(rmsf, 95)), 3),
        "rmsd_mean_structure_to_design_nm": round(rmsd(mean_vs_design, ref) * OX_TO_NM, 3),
        "radius_of_gyration_nm": round(float(np.sqrt(((mean - mean.mean(0)) ** 2).sum(1).mean())) * OX_TO_NM, 2),
        "extents_nm": [round(x, 1) for x in principal_extents(mean)],
        "design_extents_nm": [round(x, 1) for x in principal_extents(ref)],
    }

    occ = np.zeros(0)
    hb_frames = read_hb_list(work / "hb_list.txt")
    if hb_frames and pairs:
        hb_window = hb_frames[len(hb_frames) // 2 :]
        occ = np.array([sum(p in f for f in hb_window) / len(hb_window) for p in pairs])
        per_helix: dict[int, list[float]] = defaultdict(list)
        for (i, _), o in zip(pairs, occ):
            per_helix[slots[i][0]].append(o)
        metrics.update(
            {
                "bp_fraction_mean": round(float(occ.mean()), 4),
                "bp_fraction_final_frame": round(len(hb_frames[-1] & set(pairs)) / len(pairs), 4),
                "bp_fraction_per_helix": {h: round(float(np.mean(v)), 3) for h, v in sorted(per_helix.items())},
                "pairs_below_50pct": int((occ < 0.5).sum()),
                "low_occupancy_regions": _low_occupancy_regions(pairs, occ, slots),
            }
        )
    else:
        metrics["bp_fraction_mean"] = None
        metrics["note"] = "no hb_list output found; base-pair occupancy unavailable"

    energy_file = work / "energy.dat"
    if energy_file.exists():
        rows = [ln.split() for ln in energy_file.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
        if rows:
            metrics["final_potential_energy_per_nt"] = float(rows[-1][1])

    unrolled_xy = np.zeros((0, 2))
    if shape:
        from .shapes import simulated_shape_iou, unrolled_shape_iou

        # graded: the sheet flattened along its own surface (rolling/bending don't count, tearing does)
        metrics["shape_unrolled"], unrolled = unrolled_shape_iou(mean * OX_TO_NM, slots, n3, shape)
        unrolled_xy = unrolled.get("xy", unrolled_xy)
        # descriptive: how the sheet sits in solution
        metrics["shape_fidelity"] = simulated_shape_iou(mean * OX_TO_NM, shape)

    # per-nucleotide data for figures (see viz.py)
    np.savez_compressed(
        work / "analysis.npz",
        mean_nm=mean * OX_TO_NM,
        rmsf_nm=rmsf,
        pairs=np.array(pairs, dtype=int).reshape(-1, 2),
        pair_occupancy=occ,
        strand=strand,
        unrolled_xy=unrolled_xy,
    )
    return metrics
