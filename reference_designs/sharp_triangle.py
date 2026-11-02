"""Reference sharp triangle (~4000 nt): three trapezoids, after Rothemund (2006), Fig. 2f / Supp. Fig. S32.

Run: uv run python reference_designs/sharp_triangle.py <out.sc>

Each side of the triangle is a trapezoidal domain whose helices run parallel to that outer edge and get shorter
towards the open centre, so no helix is short. The trapezoids are helix groups rotated by 0, 120 and 240 degrees
in the sheet plane. Rows are joined in pairs at alternating ends, so each slanted edge is a 2-row staircase.
The scaffold rasters A (inner -> outer), crosses at the outer vertex into B (outer -> inner), then crosses at
the inner corner into C (inner -> outer). Every other row-to-row contact across a seam is bridged by a staple
with a short poly-T loop, like Rothemund's bridging staples.
"""

import math
import sys

import scadnano as sc
from common import add_staple_crossovers, break_long_staples, crossover_positions, merge_short_staples, pick_near

ROWS = 7  # helices per trapezoid (odd, so each trapezoid's raster ends on the opposite side it started)
SIDE_NM = 96.0  # outer side length of the triangle
GAP_NM = 1.0  # gap left across each seam (split between the two sides)
GEOMETRY = sc.Geometry(bases_per_turn=10.67, rise_per_base_pair=0.34, helix_radius=1.0, inter_helix_gap=0.6)  # 2.6 nm apart
PITCH = GEOMETRY.distance_between_helices()
RISE = GEOMETRY.rise_per_base_pair
SCAFFOLD_LOOP = 3  # unpaired scaffold bases where the scaffold crosses between trapezoids
NAMES = ("A", "B", "C")


def _ideal_ends(j: int) -> tuple[float, float]:
    """Row j's ends in bp along its helix, from the 30-degree seams that bisect the triangle's corners."""
    z = math.sqrt(3) * (j + 0.5) * PITCH + GAP_NM / 2
    return z / RISE, (SIDE_NM - z) / RISE


def _groups() -> dict[str, sc.HelixGroup]:
    h = SIDE_NM * math.sqrt(3) / 2
    vertices = [(0.0, 0.0), (SIDE_NM, 0.0), (SIDE_NM / 2, h)]  # (z, y) in the sheet plane
    groups = {}
    for k, (name, (vz, vy)) in enumerate(zip(NAMES, vertices)):
        a = math.radians(120 * k)
        inward = (-math.sin(a), math.cos(a))  # (z, y) direction of increasing row index
        pos = sc.Position3D(x=0, y=vy + inward[1] * PITCH / 2, z=vz + inward[0] * PITCH / 2)
        groups[name] = sc.HelixGroup(grid=sc.square, pitch=120 * k, position=pos, geometry=GEOMETRY)
    return groups


def create_design() -> sc.Design:
    max_offset = round(SIDE_NM / RISE) + 8
    helices = [sc.Helix(grid_position=(0, j), max_offset=max_offset, group=g) for g in NAMES for j in range(ROWS)]
    design = sc.Design(helices=helices, groups=_groups(), strands=[])
    hid = {(g, j): k * ROWS + j for k, g in enumerate(NAMES) for j in range(ROWS)}
    fwd = {h: (h % ROWS) % 2 == 0 for h in design.helices}  # scaffold direction: even rows forward

    def half(a: int, b: int) -> list[int]:
        return crossover_positions(design, a, b)[f"strand_on_helix_{'forward' if fwd[a] else 'reverse'}"]["half_crossover_offsets"]

    # Row ends: rows (j, j+1) share the end where the scaffold turns, snapped to a valid half crossover.
    # A and C start on the inner row: right-end turns for (6,5),(4,3),(2,1). B starts on the outer row:
    # right-end turns for (0,1),(2,3),(4,5).
    ends: dict[int, list[int]] = {}
    for g in NAMES:
        lo = {j: _ideal_ends(j)[0] for j in range(ROWS)}
        hi = {j: _ideal_ends(j)[1] for j in range(ROWS)}
        e = {j: [round(lo[j]), round(hi[j])] for j in range(ROWS)}
        right_pairs = [(j, j + 1) for j in range(0 if g == "B" else 1, ROWS - 1, 2)]
        left_pairs = [(j, j + 1) for j in range(1 if g == "B" else 0, ROWS - 1, 2)]
        for j, k in right_pairs:
            o = pick_near(half(hid[(g, j)], hid[(g, k)]), (hi[j] + hi[k]) / 2 - 0.5)
            e[j][1] = e[k][1] = o + 1
        for j, k in left_pairs:
            o = pick_near(half(hid[(g, j)], hid[(g, k)]), (lo[j] + lo[k]) / 2)
            e[j][0] = e[k][0] = o
        for j in range(ROWS):
            ends[hid[(g, j)]] = e[j]

    order = [hid[("A", j)] for j in reversed(range(ROWS))] + [hid[("B", j)] for j in range(ROWS)] \
        + [hid[("C", j)] for j in reversed(range(ROWS))]
    scaffold: list[sc.Domain | sc.Loopout] = []
    for prev, h in zip([None] + order, order):
        if prev is not None and design.helices[prev].group != design.helices[h].group:
            scaffold.append(sc.Loopout(length=SCAFFOLD_LOOP))
        scaffold.append(sc.Domain(helix=h, forward=fwd[h], start=ends[h][0], end=ends[h][1]))
    design.add_strand(sc.Strand(domains=scaffold, is_scaffold=True))

    for h, (start, end) in ends.items():  # one full-length staple per helix, crossed over below
        if fwd[h]:
            design.draw_strand(h, end).move(start - end)
        else:
            design.draw_strand(h, start).move(end - start)
    for g in NAMES:
        pairs = [(hid[(g, j)], hid[(g, j + 1)]) for j in range(ROWS - 1)]
        add_staple_crossovers(design, pairs, fwd, period=32, margin=6)

    # Bridges across the three seams: right ends of one trapezoid meet left ends of the next, row by row.
    # The scaffold already crosses A0->B0 (outer vertex) and B6->C6 (inner corner).
    for g, nxt in zip(NAMES, NAMES[1:] + NAMES[:1]):
        for j in range(ROWS):
            if (g, j) in (("A", 0), ("B", ROWS - 1)):
                continue
            _bridge(design, hid[(g, j)], hid[(nxt, j)], ends, fwd)

    break_long_staples(design, max_len=49)
    merge_short_staples(design)
    design.assign_m13_to_scaffold()
    for s in design.strands:  # poly-T in the bridging loops
        if not s.is_scaffold and s.dna_sequence and "?" in s.dna_sequence:
            s.set_dna_sequence(s.dna_sequence.replace("?", "T"))
    return design


def _staple_at(design: sc.Design, helix: int, offset: int, forward: bool) -> sc.Strand:
    return next(s for s in design.strands if not s.is_scaffold and any(
        isinstance(d, sc.Domain) and d.helix == helix and d.forward == forward and d.start <= offset < d.end
        for d in s.domains))


def _bridge(design: sc.Design, left_h: int, right_h: int, ends: dict[int, list[int]], fwd: dict[int, bool]) -> None:
    """Join the staple ending at left_h's right end to the staple starting at right_h's left end (or the reverse)."""
    staple_fwd = not fwd[left_h]
    a = _staple_at(design, left_h, ends[left_h][1] - 1, staple_fwd)
    b = _staple_at(design, right_h, ends[right_h][0], staple_fwd)
    if a is b:
        return
    first, second = (a, b) if staple_fwd else (b, a)  # 5' piece, 3' piece
    if first.domains[-1].helix not in (left_h, right_h) or second.domains[0].helix not in (left_h, right_h):
        return  # the staple doesn't end at the seam (its end there was crossed over); leave it
    loop = 2 if (left_h % ROWS) < ROWS // 2 else 1  # wider gaps towards the outer rows
    domains = [_copy(d) for d in first.domains] + [sc.Loopout(length=loop)] + [_copy(d) for d in second.domains]
    design.remove_strand(first)
    design.remove_strand(second)
    design.add_strand(sc.Strand(domains=domains))


def _copy(d):
    if isinstance(d, sc.Loopout):
        return sc.Loopout(length=d.length)
    return sc.Domain(helix=d.helix, forward=d.forward, start=d.start, end=d.end,
                     deletions=list(d.deletions), insertions=list(d.insertions))


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "sharp_triangle.sc"
    d = create_design()
    d.write_scadnano_file(filename=out)
    scaf = next(s for s in d.strands if s.is_scaffold)
    print(f"wrote {out}: scaffold {scaf.dna_length()} nt, {len(d.strands) - 1} staples")
