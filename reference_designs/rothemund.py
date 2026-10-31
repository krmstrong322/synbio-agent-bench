"""Rothemund (2006) rectangles from cadnano files, as positive controls.

Run: uv run python reference_designs/rothemund.py {rect|tallrect}[-twistfix] <out.sc>

The `-twistfix` variants add one deletion per helix every 48 bp, bringing the
average helical pitch from the design's 10.67 bp/turn to ~10.5, which removes
the global twist that 10.67 bp/turn square-lattice sheets show in solution.
"""

import json
import sys
from pathlib import Path

import scadnano as sc

FILES = {"rect": "Rothemund-rect.json", "tallrect": "Rothemund-tallrect.json"}

# The cadnano import keeps neither the square lattice's 10.67 bp/turn nor cadnano's helix phase, so the
# exported oxDNA start puts every crossover ~4 nm apart. These values (found by scanning roll for the
# fewest stretched backbone bonds) make every crossover line up (~2.05 nm, like a well-placed one).
BASES_PER_TURN = 10.67
HELIX_ROLL = 305
# With twist-correcting deletions the real pitch is ~10.5 bp/turn; this roll lines the crossovers up again.
TWISTFIX_BASES_PER_TURN = 10.5
TWISTFIX_HELIX_ROLL = 320


TWISTFIX_PERIOD = 48


def create_design(which: str) -> sc.Design:
    twistfix = which.endswith("-twistfix")
    path = Path(__file__).parent / "cadnano" / FILES[which.removesuffix("-twistfix")]
    design = sc.Design.from_cadnano_v2(json_dict=json.loads(path.read_text()))
    design.geometry = sc.Geometry(bases_per_turn=TWISTFIX_BASES_PER_TURN if twistfix else BASES_PER_TURN)
    for helix in design.helices.values():
        helix.roll = TWISTFIX_HELIX_ROLL if twistfix else HELIX_ROLL
    if twistfix:
        _add_twist_correction(design)
    design.assign_m13_to_scaffold()
    return design


def _add_twist_correction(design: sc.Design) -> None:
    """One deletion per helix every TWISTFIX_PERIOD bp, mid-way between the 16-bp crossover columns."""
    offsets = [o for s in design.strands for d in s.domains if isinstance(d, sc.Domain) for o in (d.start, d.end)]
    first, last = min(offsets), max(offsets)
    for helix in design.helices:
        for o in range(first + TWISTFIX_PERIOD // 2, last, TWISTFIX_PERIOD):
            if design.domains_at(helix, o):
                design.add_deletion(helix, o)


if __name__ == "__main__":
    which, out = sys.argv[1], sys.argv[2]
    create_design(which).write_scadnano_file(filename=out)
    print(f"wrote {out}")
