"""Grade a submitted design: spec compliance plus oxDNA structural stability.

overall = 0                                                       if the design is invalid
        = w_spec * spec_score + w_sim * sim_score [+ w_shape * shape_score]   otherwise

valid      : loads, legal, has a scaffold, every base sequenced, no mismatched designed pairs
spec_score : mean pass fraction over validation checks (lattice, helix count, lengths, coverage, ...)
sim_score  : mean fraction of designed base pairs formed in the evaluation simulation(s)
shape_score: (shape tasks) IoU of the simulated sheet, unrolled along its surface, with the target shape
             (rolling/bending in solution doesn't count; tearing does). The projected outline and
             flatness are reported separately.
             (wireframe tasks) F1 of the simulated mean structure traced onto the target polyhedron's edges
             (see wireframe.py).
All raw metrics are kept so other weightings can be computed later.
"""

from __future__ import annotations

import statistics
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .spec import validate_design
from .oxdna import OxDNAConfig, SimJob, simulate_many

HARD_ERROR_PREFIXES = ("could not load", "illegal", "two nucleotides", "no strand is marked")


@dataclass
class Submission:
    """One design to grade: the submitted file, its task spec, where its grading runs go, and score weights."""

    design_path: Path
    spec: dict[str, Any]
    out_dir: Path
    weights: dict[str, float] | None = None


def evaluate_submission(
    design_path: Path,
    spec: dict[str, Any],
    oxdna: OxDNAConfig,
    out_dir: Path,
    protocol: str = "standard",
    seeds: list[int] | tuple[int, ...] = (1,),
    weights: dict[str, float] | None = None,
    reanalyze: bool = False,
) -> dict[str, Any]:
    return evaluate_many([Submission(design_path, spec, out_dir, weights)], oxdna, protocol, seeds, reanalyze)[0]


def evaluate_many(
    subs: list[Submission],
    oxdna: OxDNAConfig,
    protocol: str = "standard",
    seeds: list[int] | tuple[int, ...] = (1,),
    reanalyze: bool = False,
    on_done: Callable[[int, dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Grade several submissions, pooling all their simulations.

    On Modal, simulations from different designs share GPUs (modal.sims_per_gpu at a time, similar sizes
    together); locally they run one after another as before. `on_done(index, grade)` is called as soon as a
    submission's grade is complete (from a worker thread on Modal; invalid designs right away), so callers
    can save results as they come.
    """
    preps = [_check(sub) for sub in subs]
    grades: list[dict[str, Any] | None] = [None] * len(subs)
    jobs: list[SimJob] = []
    owner: list[int] = []
    for i, (sub, prep) in enumerate(zip(subs, preps)):
        if prep["valid"]:
            for seed in seeds:
                jobs.append(SimJob(sub.design_path, sub.out_dir / f"seed_{seed}", seed,
                                   sub.spec.get("shape"), sub.spec.get("wireframe")))
                owner.append(i)
        else:
            grades[i] = _score(sub, prep, [], oxdna, protocol, seeds)
            if on_done:
                on_done(i, grades[i])

    lock = threading.Lock()
    got: dict[int, list[tuple[int, dict[str, Any]]]] = {i: [] for i in set(owner)}
    remaining = Counter(owner)

    def on_result(j: int, result: dict[str, Any]) -> None:
        i = owner[j]
        with lock:
            got[i].append((j, result))
            remaining[i] -= 1
            complete = remaining[i] == 0
        if complete:
            sims = [r for _, r in sorted(got[i], key=lambda t: t[0])]  # seed order
            grades[i] = _score(subs[i], preps[i], sims, oxdna, protocol, seeds)
            if on_done:
                on_done(i, grades[i])

    simulate_many(jobs, oxdna, protocol=protocol, reanalyze=reanalyze, on_result=on_result)
    return grades  # type: ignore[return-value]


def _check(sub: Submission) -> dict[str, Any]:
    """Spec checks and validity; no simulation."""
    if not sub.design_path.is_file():
        return {"submitted": False, "valid": False}
    v = validate_design(sub.design_path, sub.spec)
    summary = v.get("summary", {})
    valid = (
        bool(summary)
        and not any(e.startswith(HARD_ERROR_PREFIXES) for e in v["errors"])
        and summary.get("unassigned_bases", 1) == 0
        and summary.get("mismatched_pairs", 1) == 0
    )
    return {"submitted": True, "valid": valid, "v": v, "summary": summary,
            "spec_score": statistics.mean(v["checks"].values()) if v["checks"] else 0.0}


def _score(sub: Submission, prep: dict[str, Any], sims: list[dict[str, Any]], oxdna: OxDNAConfig,
           protocol: str, seeds: list[int] | tuple[int, ...]) -> dict[str, Any]:
    if not prep["submitted"]:
        return {"submitted": False, "valid": False, "overall": 0.0, "spec_score": 0.0, "sim_score": 0.0}
    spec, v, summary, valid, spec_score = sub.spec, prep["v"], prep["summary"], prep["valid"], prep["spec_score"]
    weights = sub.weights or {"spec": 0.5, "sim": 0.5}
    shaped = "shape" in spec or "wireframe" in spec
    if shaped and "shape" not in weights:
        weights = {"spec": 1 / 3, "sim": 1 / 3, "shape": 1 / 3}

    bp, shape_iou = [], []
    for r in sims:
        m = r.get("metrics", {}) if r.get("ok") else {}
        bp.append(m.get("bp_fraction_mean") or 0.0)
        if "wireframe" in spec:
            shape_iou.append((m.get("wireframe_fit") or {}).get("f1", 0.0))
        else:
            shape_iou.append((m.get("shape_unrolled") or {}).get("iou", 0.0))
    sim_score = statistics.mean(bp) if bp else 0.0
    shape_score = statistics.mean(shape_iou) if shape_iou else 0.0
    overall = weights["spec"] * spec_score + weights["sim"] * sim_score
    if shaped:
        overall += weights.get("shape", 0.0) * shape_score

    def mean_metric(key: str, sub: str | None = None) -> float | None:
        vals = [s["metrics"].get(key) for s in sims if s.get("ok")]
        if sub:
            vals = [v.get(sub) if isinstance(v, dict) else None for v in vals]
        vals = [v for v in vals if v is not None]
        return round(statistics.mean(vals), 4) if vals else None

    return {
        "submitted": True,
        "valid": valid,
        "spec_score": round(spec_score, 4),
        "sim_score": round(sim_score, 4),
        "shape_score": round(shape_score, 4) if shaped else None,
        "overall": round(overall, 4) if valid else 0.0,
        "errors": v["errors"],
        "warnings": v["warnings"],
        "checks": v["checks"],
        "design_summary": summary,
        "sim_ok": [bool(s.get("ok")) for s in sims],
        "sim_failures": [{k: s.get(k) for k in ("failed_stage", "error", "log_tail")} for s in sims if not s.get("ok")],
        "bp_fraction_mean": mean_metric("bp_fraction_mean"),
        "rmsf_mean_nm": mean_metric("rmsf_mean_nm"),
        "rmsd_to_design_nm": mean_metric("rmsd_mean_structure_to_design_nm"),
        "shape_projected_iou": mean_metric("shape_fidelity", "iou"),
        "flatness": mean_metric("shape_fidelity", "flatness"),
        "wireframe_precision": mean_metric("wireframe_fit", "precision"),
        "wireframe_recall": mean_metric("wireframe_fit", "recall"),
        "initial_stretched_bonds": sims[0]["initial_backbone"]["num_stretched_bonds"] if sims and "initial_backbone" in sims[0] else None,
        "sim_runner": oxdna.mode,
        "protocol": protocol,
        "verlet_skin": oxdna.verlet_skin,
        "seeds": list(seeds),
        "sims_sharing_gpu": [1 + s.get("shared_gpu_with", 0) for s in sims],  # per seed, 1 = had a GPU to itself
    }
