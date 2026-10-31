"""Reference 6-helix bundle on the honeycomb lattice (~126 bp per helix, ~760 nt M13 scaffold).

Run: uv run python reference_designs/hb6_honeycomb.py <out.sc>
"""

import sys

import scadnano as sc
from common import add_staple_crossovers, break_long_staples, crossover_positions, merge_short_staples, pick_near

LENGTH = 126
RING = [(1, 1), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1)]


def create_design() -> sc.Design:
    design = sc.Design(helices=[sc.Helix(grid_position=g, max_offset=LENGTH + 16) for g in RING], strands=[],
                       grid=sc.honeycomb)
    scaf_fwd = {h: h % 2 == 0 for h in range(6)}

    def half(a, b, fwd):
        return crossover_positions(design, a, b)[f"strand_on_helix_{'forward' if fwd else 'reverse'}"]["half_crossover_offsets"]

    # scaffold raster 0 -> 1 -> ... -> 5: right-end jumps on even helices, left-end jumps on odd ones
    right = {h: pick_near(half(h, h + 1, True), LENGTH + 4) for h in (0, 2, 4)}
    left = {h: pick_near(half(h, h + 1, False), 8) for h in (1, 3)}
    span = {0: (left[1], right[0] + 1), 1: (left[1], right[0] + 1), 2: (left[1], right[2] + 1),
            3: (left[3], right[2] + 1), 4: (left[3], right[4] + 1), 5: (left[3], right[4] + 1)}

    for h, (start, end) in span.items():
        if scaf_fwd[h]:
            design.draw_strand(h, start).move(end - start)
            design.draw_strand(h, end).move(start - end)
        else:
            design.draw_strand(h, end).move(start - end)
            design.draw_strand(h, start).move(end - start)
    for s in design.strands:
        d = s.first_domain()
        if d.forward == scaf_fwd[d.helix]:
            s.set_scaffold()

    for h in (0, 2, 4):
        design.add_half_crossover(helix=h, helix2=h + 1, offset=right[h], forward=True)
    for h in (1, 3):
        design.add_half_crossover(helix=h, helix2=h + 1, offset=left[h], forward=False)

    ring_pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    add_staple_crossovers(design, ring_pairs, scaf_fwd, period=21, margin=10)
    break_long_staples(design, max_len=49)
    merge_short_staples(design)
    design.assign_m13_to_scaffold()
    return design


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "hb6_honeycomb.sc"
    create_design().write_scadnano_file(filename=out)
    print(f"wrote {out}")
