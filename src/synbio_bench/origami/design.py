"""Load, inspect and validate scadnano designs.

The nucleotide ordering here mirrors scadnano's own oxDNA exporter
(`scadnano._convert_design_to_oxdna_system`): strands in design order, each
strand 5'->3', insertions expanded, deletions skipped. That lets us map every
(helix, offset, insertion, forward) slot to its oxDNA particle index and so
know exactly which base pairs the designer intended.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import scadnano as sc

# Slot key for a nucleotide on a helix: (helix, offset, insertion_index, forward).
# insertion_index is -1 for the base at the offset itself, 0..n-1 for inserted bases.
SlotKey = tuple[int, int, int, bool]

COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G"}


@dataclass
class NucleotideMap:
    """Flattened view of a design in oxDNA particle order."""

    slots: list[SlotKey | None] = field(default_factory=list)  # None for loopout/extension bases
    bases: list[str] = field(default_factory=list)
    strand_of: list[int] = field(default_factory=list)
    index_of: dict[SlotKey, int] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.bases)

    def designed_pairs(self) -> list[tuple[int, int]]:
        """Particle index pairs (i < j) the design intends to be hybridised."""
        pairs = []
        for (h, off, ins, fwd), i in self.index_of.items():
            if not fwd:
                continue
            j = self.index_of.get((h, off, ins, False))
            if j is not None:
                pairs.append((min(i, j), max(i, j)))
        return sorted(pairs)


def load_design(path: str | Path) -> sc.Design:
    return sc.Design.from_scadnano_file(str(path))


def _domain_slots(design: sc.Design, domain: sc.Domain) -> list[SlotKey]:
    """Slots of a domain in ascending-offset order (same order as scadnano's exporter)."""
    deletions = set(domain.deletions)
    insertions = dict(domain.insertions)
    slots: list[SlotKey] = []
    for offset in range(domain.start, domain.end):
        if offset in deletions:
            continue
        for j in range(insertions.get(offset, 0)):
            slots.append((domain.helix, offset, j, domain.forward))
        slots.append((domain.helix, offset, -1, domain.forward))
    return slots


def nucleotide_map(design: sc.Design) -> NucleotideMap:
    nm = NucleotideMap()
    for s_idx, strand in enumerate(design.strands):
        for domain in strand.domains:
            seq = domain.dna_sequence or ("?" * domain.dna_length())
            if isinstance(domain, sc.Domain):
                slots = _domain_slots(design, domain)
                seq_asc = seq if domain.forward else seq[::-1]
                items = list(zip(slots, seq_asc))
                if not domain.forward:
                    items.reverse()
            else:  # Loopout / Extension: single-stranded, unpaired by construction
                items = [(None, b) for b in seq]
            for slot, base in items:
                idx = len(nm.bases)
                nm.slots.append(slot)
                nm.bases.append(base.upper())
                nm.strand_of.append(s_idx)
                if slot is not None:
                    if slot in nm.index_of:
                        raise ValueError(f"two nucleotides occupy the same slot {slot}")
                    nm.index_of[slot] = idx
    return nm


def _scaffolds(design: sc.Design) -> list[sc.Strand]:
    return [s for s in design.strands if s.is_scaffold]


def _crossovers(design: sc.Design) -> list[tuple[int, int, int]]:
    """(helix_a, helix_b, offset_a) for each strand jump between consecutive Domains."""
    out = []
    for strand in design.strands:
        doms = [d for d in strand.domains if isinstance(d, sc.Domain)]
        for a, b in zip(doms, doms[1:]):
            if a.helix != b.helix:
                out.append((a.helix, b.helix, a.offset_3p()))
    return out


def _grid_neighbors(design: sc.Design, h1: int, h2: int) -> bool | None:
    """True if two helices sit one inter-helix distance apart, None if it can't be decided."""
    a, b = design.helices[h1], design.helices[h2]
    if a.group != b.group:
        return None
    group = design.groups[a.group]
    geometry = group.geometry or design.geometry
    pa = a.calculate_position(group.grid, geometry)
    pb = b.calculate_position(group.grid, geometry)
    d = ((pa.x - pb.x) ** 2 + (pa.y - pb.y) ** 2 + (pa.z - pb.z) ** 2) ** 0.5
    return d <= 1.2 * geometry.distance_between_helices()


def _connected_components(nodes: list[int], edges: list[tuple[int, int]]) -> int:
    parent = {n: n for n in nodes}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        parent[find(a)] = find(b)
    return len({find(n) for n in nodes})


def summarize_design(design: sc.Design, nm: NucleotideMap | None = None) -> dict[str, Any]:
    """Facts about a design that a designer (or a grader) would want to see."""
    nm = nm or nucleotide_map(design)
    scaffolds = _scaffolds(design)
    staples = [s for s in design.strands if not s.is_scaffold]
    staple_lengths = [s.dna_length() for s in staples]
    pairs = nm.designed_pairs()
    paired = {i for p in pairs for i in p}

    scaf_idx = {i for i, s in enumerate(nm.strand_of) if design.strands[s].is_scaffold}
    scaf_paired = len(scaf_idx & paired)

    mismatches = sum(
        1
        for i, j in pairs
        if nm.bases[i] in COMPLEMENT and nm.bases[j] in COMPLEMENT and COMPLEMENT[nm.bases[i]] != nm.bases[j]
    )
    unassigned = sum(1 for b in nm.bases if b not in COMPLEMENT)

    bp_per_helix: dict[int, int] = defaultdict(int)
    for i, _ in pairs:
        bp_per_helix[nm.slots[i][0]] += 1  # type: ignore[index]

    xovers = _crossovers(design)
    xover_pairs = Counter(tuple(sorted((a, b))) for a, b, _ in xovers)
    non_adjacent = [list(p) for p in xover_pairs if _grid_neighbors(design, *p) is False]
    helices = sorted(design.helices)
    used_helices = sorted({s[0] for s in nm.slots if s is not None})

    grids = {design.groups[h.group].grid.name for h in design.helices.values()}
    return {
        "grid": grids.pop() if len(grids) == 1 else sorted(grids),
        "num_helices": len(helices),
        "num_helices_used": len(used_helices),
        "helix_grid_positions": {
            h: list(design.helices[h].grid_position) if design.helices[h].grid_position else None for h in helices
        },
        "bp_per_helix": dict(sorted(bp_per_helix.items())),
        "num_strands": len(design.strands),
        "num_scaffolds": len(scaffolds),
        "scaffold_lengths": [s.dna_length() for s in scaffolds],
        "scaffold_is_circular": [s.circular for s in scaffolds],
        "num_staples": len(staples),
        "staple_length_min": min(staple_lengths) if staple_lengths else None,
        "staple_length_max": max(staple_lengths) if staple_lengths else None,
        "staple_length_mean": round(statistics.mean(staple_lengths), 1) if staple_lengths else None,
        "staple_length_histogram": dict(sorted(Counter(staple_lengths).items())),
        "total_nucleotides": nm.n,
        "designed_base_pairs": len(pairs),
        "scaffold_paired_fraction": round(scaf_paired / len(scaf_idx), 4) if scaf_idx else 0.0,
        "unpaired_nucleotides": nm.n - len(paired),
        "mismatched_pairs": mismatches,
        "unassigned_bases": unassigned,
        "num_crossovers": len(xovers),
        "crossovers_per_helix_pair": {f"{a}-{b}": c for (a, b), c in sorted(xover_pairs.items())},
        "crossovers_between_non_adjacent_helices": non_adjacent,
        "helix_connectivity_components": _connected_components(used_helices, list(xover_pairs)) if used_helices else 0,
        "num_insertions": sum(len(d.insertions) for s in design.strands for d in s.domains if isinstance(d, sc.Domain)),
        "num_deletions": sum(len(d.deletions) for s in design.strands for d in s.domains if isinstance(d, sc.Domain)),
        "num_loopouts": sum(1 for s in design.strands for d in s.domains if isinstance(d, sc.Loopout)),
    }


def _is_m13_fragment(seq: str) -> bool:
    seq = seq.upper()
    for variant in sc.M13Variant:
        full = sc.m13(rotation=0, variant=variant)
        if seq in full + full:
            return True
    return False


def validate_design(design_path: str | Path, spec: dict[str, Any] | None = None) -> dict[str, Any]:
    """Check legality, sequence consistency and (optionally) a task spec.

    Returns {"ok", "errors", "warnings", "checks", "summary"}. `checks` maps a
    check name to a pass fraction in [0, 1] and is what the grader averages.
    """
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, float] = {}

    try:
        design = load_design(design_path)
    except Exception as e:  # noqa: BLE001 - surface any parse failure to the agent
        return {"ok": False, "errors": [f"could not load design: {type(e).__name__}: {e}"], "warnings": [], "checks": {}}

    try:
        design._check_legal_design(warn_duplicate_strand_names=False)
    except Exception as e:  # noqa: BLE001
        errors.append(f"illegal design: {e}")

    try:
        nm = nucleotide_map(design)
    except ValueError as e:
        return {"ok": False, "errors": errors + [str(e)], "warnings": warnings, "checks": checks}
    summary = summarize_design(design, nm)

    if summary["num_scaffolds"] == 0:
        errors.append("no strand is marked as scaffold (call strand.set_scaffold())")
    if summary["unassigned_bases"]:
        errors.append(f"{summary['unassigned_bases']} bases have no DNA sequence assigned")
    if summary["mismatched_pairs"]:
        errors.append(f"{summary['mismatched_pairs']} designed base pairs are not Watson-Crick complementary")
    if summary["crossovers_between_non_adjacent_helices"]:
        errors.append(f"crossovers join non-adjacent helices: {summary['crossovers_between_non_adjacent_helices']}")
    if summary["helix_connectivity_components"] > 1:
        warnings.append(f"helices form {summary['helix_connectivity_components']} disconnected groups")

    checks["legal"] = 0.0 if any(e.startswith("illegal") for e in errors) else 1.0
    checks["sequences_complementary"] = (
        1.0 - summary["mismatched_pairs"] / summary["designed_base_pairs"] if summary["designed_base_pairs"] else 0.0
    )
    checks["helices_connected"] = 1.0 if summary["helix_connectivity_components"] == 1 else 0.0
    checks["crossovers_adjacent"] = 0.0 if summary["crossovers_between_non_adjacent_helices"] else 1.0

    if spec:
        _check_spec(design, summary, spec, errors, warnings, checks)

    return {"ok": not errors, "errors": errors, "warnings": warnings, "checks": checks, "summary": summary}


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
        positions = [design.helices[h].grid_position for h in design.helices]
        ok = all(p is not None for p in positions) and (
            len({p[0] for p in positions}) == 1 or len({p[1] for p in positions}) == 1
        )
        checks["planar"] = float(ok)
        if not ok:
            errors.append("spec requires a flat sheet: all helices must lie in one grid row or column")

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
