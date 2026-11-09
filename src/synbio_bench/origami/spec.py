"""Task-spec checks for the benchmark grader.

The shipped agent's structural check (`origami_agent.design.check_design`) is
the base; this adds the task specification (lattice, helix count, lengths,
M13 source, coverage, shape footprint, ...) that grading scores.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import scadnano as sc

from origami_agent.design import _grid_neighbors, _scaffolds, check_design, load_design

def _is_m13_fragment(seq: str) -> bool:
    seq = seq.upper()
    for variant in sc.M13Variant:
        full = sc.m13(rotation=0, variant=variant)
        if seq in full + full:
            return True
    return False


def _as_dna(seq: str) -> str:
    """Upper-case and map RNA U to T, so a provided mRNA matches the DNA scaffold assigned from it."""
    return (seq or "").upper().replace("U", "T")


def provided_scaffold_sequence(spec: dict[str, Any], base_dir: Path | None = None) -> str | None:
    """The sequence a `source: provided` task folds: `scaffold.sequence`, or the text of `scaffold.sequence_file`
    (a FASTA or plain file, path relative to `base_dir` or the repo root). Returns DNA (U->T), or None."""
    scaf = spec.get("scaffold", {})
    if scaf.get("sequence"):
        return _as_dna(scaf["sequence"])
    if scaf.get("sequence_file"):
        root = base_dir or Path(__file__).resolve().parents[3]
        text = (root / scaf["sequence_file"]).read_text()
        seq = "".join(ln.strip() for ln in text.splitlines() if not ln.startswith(">"))
        return _as_dna(seq)
    return None


def validate_design(design_path: str | Path, spec: dict[str, Any] | None = None) -> dict[str, Any]:
    """The structural check plus (optionally) a task spec.

    Returns {"ok", "errors", "warnings", "checks", "summary"}. `checks` maps a
    check name to a pass fraction in [0, 1] and is what the grader averages.
    """
    result = check_design(design_path)
    if spec and result.get("summary"):
        design = load_design(design_path)
        _check_spec(design, result["summary"], spec, result["errors"], result["warnings"], result["checks"])
        result["ok"] = not result["errors"]
    return result


def _in_range(x: float, rng: list[float] | tuple[float, float]) -> bool:
    return rng[0] <= x <= rng[1]


def _check_spec(
    design: sc.Design,
    summary: dict[str, Any],
    spec: dict[str, Any],
    errors: list[str],
    warnings: list[str],
    checks: dict[str, float],
) -> None:
    if "grid" in spec:
        ok = summary["grid"] == spec["grid"]
        checks["grid"] = float(ok)
        if not ok:
            errors.append(f"grid is {summary['grid']}, spec requires {spec['grid']}")

    if "num_helices" in spec:
        ok = summary["num_helices_used"] == spec["num_helices"]
        checks["num_helices"] = float(ok)
        if not ok:
            errors.append(f"design uses {summary['num_helices_used']} helices, spec requires {spec['num_helices']}")

    if "helix_bp" in spec:
        bp = summary["bp_per_helix"]
        good = [h for h, n in bp.items() if _in_range(n, spec["helix_bp"])]
        checks["helix_bp"] = len(good) / len(bp) if bp else 0.0
        bad = {h: n for h, n in bp.items() if h not in good}
        if bad:
            errors.append(f"helices with base-pair count outside {spec['helix_bp']}: {bad}")

    scaf_spec = spec.get("scaffold", {})
    if scaf_spec:
        scaffolds = _scaffolds(design)
        count_ok = len(scaffolds) == scaf_spec.get("count", 1)
        checks["scaffold_count"] = float(count_ok)
        if not count_ok:
            errors.append(f"design has {len(scaffolds)} scaffold strands, spec requires {scaf_spec.get('count', 1)}")
        if "length" in scaf_spec and scaffolds:
            ok = all(_in_range(s.dna_length(), scaf_spec["length"]) for s in scaffolds)
            checks["scaffold_length"] = float(ok)
            if not ok:
                errors.append(f"scaffold length {summary['scaffold_lengths']} outside {scaf_spec['length']}")
        if scaf_spec.get("source") == "m13" and scaffolds:
            ok = all(s.dna_sequence and _is_m13_fragment(s.dna_sequence) for s in scaffolds)
            checks["scaffold_is_m13"] = float(ok)
            if not ok:
                errors.append("scaffold sequence is not a contiguous fragment of an M13 variant")
        if scaf_spec.get("source") == "provided" and scaffolds:
            want = provided_scaffold_sequence(spec)
            got = _as_dna(scaffolds[0].dna_sequence) if scaffolds[0].dna_sequence else ""
            # The whole provided molecule must be the scaffold: exact match (orientation-insensitive).
            ok = bool(want) and len(scaffolds) == 1 and got == want
            checks["scaffold_is_provided"] = float(ok)
            if not ok:
                if not want:
                    errors.append("task gives source: provided but no scaffold.sequence or scaffold.sequence_file")
                else:
                    errors.append(f"scaffold is not the provided sequence (got {len(got)} nt, provided {len(want)} nt)")
        if "min_paired_fraction" in scaf_spec:
            frac = summary["scaffold_paired_fraction"]
            checks["scaffold_coverage"] = min(1.0, frac / scaf_spec["min_paired_fraction"])
            if frac < scaf_spec["min_paired_fraction"]:
                errors.append(f"only {frac:.1%} of scaffold is paired, spec requires {scaf_spec['min_paired_fraction']:.0%}")

    if "staple_length" in spec:
        lengths = [s.dna_length() for s in design.strands if not s.is_scaffold]
        good = [n for n in lengths if _in_range(n, spec["staple_length"])]
        checks["staple_length"] = len(good) / len(lengths) if lengths else 0.0
        if len(good) != len(lengths):
            errors.append(f"{len(lengths) - len(good)} of {len(lengths)} staples have length outside {spec['staple_length']}")

    if spec.get("planar"):
        from .shapes import planar_sheet

        ok = planar_sheet(design)
        checks["planar"] = float(ok)
        if not ok:
            errors.append("spec requires a flat sheet: all helices must lie in one plane, one layer thick")

    if "min_neighbors_per_helix" in spec:
        used = [h for h in design.helices if summary["bp_per_helix"].get(h)]
        n_nb = {h: sum(bool(_grid_neighbors(design, h, o)) for o in used if o != h) for h in used}
        good = [h for h, k in n_nb.items() if k >= spec["min_neighbors_per_helix"]]
        checks["compact"] = len(good) / len(used) if used else 0.0
        if len(good) < len(used):
            errors.append(
                f"helices with fewer than {spec['min_neighbors_per_helix']} lattice neighbours: "
                f"{sorted(set(used) - set(good))}"
            )

    if "shape" in spec:
        from .shapes import design_shape_iou

        fit = design_shape_iou(design, spec["shape"])
        summary["shape_fit"] = fit
        checks["shape_footprint"] = fit["iou"]
        min_iou = spec["shape"].get("min_iou", 0.9)
        if fit.get("error"):
            errors.append(f"shape: {fit['error']}")
        elif fit["iou"] < min_iou:
            errors.append(
                f"footprint matches the target {spec['shape']['type']} with IoU {fit['iou']:.2f}; "
                f"spec requires >= {min_iou}"
            )

    if "wireframe" in spec:
        from .wireframe import design_points_nm, fit_wireframe

        fit = fit_wireframe(design_points_nm(design), spec["wireframe"])
        summary["wireframe_fit"] = fit
        checks["wireframe_footprint"] = fit["f1"]
        min_f1 = spec["wireframe"].get("min_f1", 0.9)
        if fit["f1"] < min_f1:
            errors.append(
                f"design traces the target {spec['wireframe']['type']} (edges {spec['wireframe']['edge_nm']} nm) "
                f"with F1 {fit['f1']:.2f} (precision {fit['precision']:.2f}, recall {fit['recall']:.2f}); "
                f"spec requires >= {min_f1}"
            )

    if "max_staple_crossovers" in spec:
        too_many = 0
        for s in design.strands:
            if s.is_scaffold:
                continue
            doms = [d for d in s.domains if isinstance(d, sc.Domain)]
            if sum(a.helix != b.helix for a, b in zip(doms, doms[1:])) > spec["max_staple_crossovers"]:
                too_many += 1
        n_staples = summary["num_staples"] or 1
        checks["staple_crossovers"] = 1 - too_many / n_staples
        if too_many:
            warnings.append(f"{too_many} staples have more than {spec['max_staple_crossovers']} crossovers")
