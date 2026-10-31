"""Geometric crossover finder.

Builds a probe design with the same helices (grid, positions, rolls, geometry)
fully double-stranded, exports it through scadnano's oxDNA converter, and
measures the distance between the backbone sites of antiparallel strands on
neighbouring helices at each offset. Good crossover offsets are where those
distances are smallest, so this works for any lattice or helix roll.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import scadnano as sc

from .analysis import OX_TO_NM
from .design import _grid_neighbors, nucleotide_map

# oxDNA2 backbone site offset along a1 (base direction) and a2 = a3 x a1, in oxDNA units.
_BACK_A1, _BACK_A2 = -0.34, 0.3408


def _probe(design: sc.Design) -> tuple[np.ndarray, dict]:
    probe = copy.deepcopy(design)
    for strand in list(probe.strands):
        probe.remove_strand(strand)
    for idx, h in probe.helices.items():
        lo, hi = h.min_offset, h.max_offset
        probe.draw_strand(idx, lo).move(hi - lo)
        probe.draw_strand(idx, hi).move(lo - hi)
    dat, _ = probe.to_oxdna_format(warn_duplicate_strand_names=False)
    rows = np.array([[float(x) for x in ln.split()] for ln in dat.splitlines()[3:] if ln.strip()])
    r, a1, a3 = rows[:, 0:3], rows[:, 3:6], rows[:, 6:9]
    a2 = np.cross(a3, a1)
    backbone = r + _BACK_A1 * a1 + _BACK_A2 * a2
    return backbone, nucleotide_map(probe).index_of


def crossover_positions(design: sc.Design, helix: int, helix2: int, tolerance_nm: float = 0.35) -> dict[str, Any]:
    """Find offsets where two helices' backbones face each other.

    For each strand direction on `helix` (the strand on `helix2` runs the other
    way), report:
      - half_crossover_offsets: offsets o where a strand can jump at base o
        (use with add_half_crossover(helix, helix2, offset=o, forward=...)).
      - full_crossover_offsets: offsets o where both bases o-1 and o are close,
        i.e. a valid add_full_crossover(helix, helix2, offset=o, forward=...).
    """
    neighbors = _grid_neighbors(design, helix, helix2)
    backbone, index = _probe(design)
    h1, h2 = design.helices[helix], design.helices[helix2]
    lo, hi = max(h1.min_offset, h2.min_offset), min(h1.max_offset, h2.max_offset)
    out: dict[str, Any] = {"helix": helix, "helix2": helix2, "adjacent": neighbors}
    for fwd in (True, False):
        dist = {}
        for o in range(lo, hi):
            i, j = index.get((helix, o, -1, fwd)), index.get((helix2, o, -1, not fwd))
            if i is not None and j is not None:
                dist[o] = float(np.linalg.norm(backbone[i] - backbone[j])) * OX_TO_NM
        if not dist:
            continue
        # scadnano's export puts helix axes 3 nm apart, so even ideal crossover
        # sites are ~1.4 nm apart before relaxation: rank relative to the closest.
        cutoff = min(dist.values()) + tolerance_nm
        half = [o for o, d in dist.items() if d <= cutoff]
        full = [o for o in range(lo + 1, hi) if o in dist and o - 1 in dist and max(dist[o], dist[o - 1]) <= cutoff]
        key = "forward" if fwd else "reverse"
        out[f"strand_on_helix_{key}"] = {
            "half_crossover_offsets": half,
            "full_crossover_offsets": full,
            "min_distance_nm": round(min(dist.values()), 2),
        }
    return out
