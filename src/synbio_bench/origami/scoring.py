"""Grade a submitted design: spec compliance plus oxDNA structural stability.

overall = 0                                                       if the design is invalid
        = w_spec * spec_score + w_sim * sim_score [+ w_shape * shape_score]   otherwise

valid      : loads, legal, has a scaffold, every base sequenced, no mismatched designed pairs
spec_score : mean pass fraction over validation checks (lattice, helix count, lengths, coverage, ...)
sim_score  : mean fraction of designed base pairs formed in the evaluation simulation(s)
shape_score: (shape tasks) IoU of the simulated mean structure's outline with the target shape
All raw metrics are kept so other weightings can be computed later.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from .design import validate_design
from .oxdna import OxDNAConfig, simulate

HARD_ERROR_PREFIXES = ("could not load", "illegal", "two nucleotides", "no strand is marked")


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
    weights = weights or {"spec": 0.5, "sim": 0.5}
    if "shape" in spec and "shape" not in weights:
        weights = {"spec": 1 / 3, "sim": 1 / 3, "shape": 1 / 3}
    if not design_path.is_file():
        return {"submitted": False, "valid": False, "overall": 0.0, "spec_score": 0.0, "sim_score": 0.0}

    v = validate_design(design_path, spec)
    summary = v.get("summary", {})
    valid = (
        bool(summary)
        and not any(e.startswith(HARD_ERROR_PREFIXES) for e in v["errors"])
        and summary.get("unassigned_bases", 1) == 0
        and summary.get("mismatched_pairs", 1) == 0
    )
    spec_score = statistics.mean(v["checks"].values()) if v["checks"] else 0.0

    sims, bp, shape_iou = [], [], []
    if valid:
        for seed in seeds:
            r = simulate(design_path, out_dir / f"seed_{seed}", oxdna, protocol=protocol, seed=seed,
                         reanalyze=reanalyze, shape=spec.get("shape"))
            sims.append(r)
            m = r.get("metrics", {}) if r.get("ok") else {}
            bp.append(m.get("bp_fraction_mean") or 0.0)
            shape_iou.append((m.get("shape_fidelity") or {}).get("iou", 0.0))
    sim_score = statistics.mean(bp) if bp else 0.0
    shape_score = statistics.mean(shape_iou) if shape_iou else 0.0
    overall = weights["spec"] * spec_score + weights["sim"] * sim_score
    if "shape" in spec:
        overall += weights.get("shape", 0.0) * shape_score

    def mean_metric(key: str) -> float | None:
        vals = [s["metrics"][key] for s in sims if s.get("ok") and s["metrics"].get(key) is not None]
        return round(statistics.mean(vals), 4) if vals else None

    return {
        "submitted": True,
        "valid": valid,
        "spec_score": round(spec_score, 4),
        "sim_score": round(sim_score, 4),
        "shape_score": round(shape_score, 4) if "shape" in spec else None,
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
        "initial_stretched_bonds": sims[0]["initial_backbone"]["num_stretched_bonds"] if sims and "initial_backbone" in sims[0] else None,
        "sim_runner": oxdna.mode,
        "protocol": protocol,
        "seeds": list(seeds),
    }
