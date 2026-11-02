"""Reference wireframe tetrahedron (1044-nt scaffold) in the DAEDALUS style (Veneziano et al., Science 2016).

Run: uv run python reference_designs/wireframe_tetrahedron.py <out.sc>

Every edge is a two-helix (DX) bundle of 84 bp (8 turns at 10.5 bp/turn, ~29 nm) in its own helix group, oriented
with the group's yaw/pitch/roll so the edge runs along the tetrahedron edge and its two helices lie side by side
in the plane tangent to the surface. Each helix end is set back from the vertex so the three edges meeting there
don't collide; the scaffold and staples hop between edges there through 3-nt loops.

Scaffold route (DAEDALUS): pick a spanning tree of the edges. The scaffold walks around the tree, keeping to the
right-hand helix of each edge (seen from outside), and at every vertex turns onto the next edge counter-clockwise.
A tree edge is crossed twice, once on each helix. An edge outside the tree is entered from both ends: the scaffold
runs to the middle, crosses to the other helix and returns, so those edges have a scaffold crossover at mid-edge.
Staples pair each edge's two helices with crossovers every two turns, and vertex staples join the two edges on
either side of every scaffold turn at a vertex, with a short poly-T loop.

oxDNA (shape protocol, seed 1): 99.6% of base pairs formed, wireframe F1 0.90 against a 35-nm tetrahedron, corner
angles 52-70 degrees (ideal 60). A tighter 3-nm setback (1-nt loops) holds the corners closer (vertex RMSD 1.1 vs
1.8 nm) but crowds them, so the edges bow outward and the wireframe score drops to 0.87.
"""

import math
import sys

import numpy as np
import scadnano as sc
from common import crossover_positions, periodic, pick_near

EDGE_BP = 84  # 8 turns at 10.5 bp/turn
GEOMETRY = sc.Geometry(bases_per_turn=10.5, rise_per_base_pair=0.34, helix_radius=1.0, inter_helix_gap=0.6)
PITCH = GEOMETRY.distance_between_helices()  # 2.6 nm between the edge's two helix axes
RISE = GEOMETRY.rise_per_base_pair
SETBACK_NM = 4.5  # vertex point to each helix end, along the edge
STAPLE_XO_PERIOD = 21  # staple crossovers between an edge's helices every two turns
LOOP_NM = 0.65  # one unpaired base bridges about this much of a gap at a vertex
EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
TREE = {0, 1, 2}  # spanning tree: the three edges at vertex 0


def vertices() -> np.ndarray:
    edge_nm = EDGE_BP * RISE + 2 * SETBACK_NM
    return np.array([(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)], float) * edge_nm / (2 * math.sqrt(2))


def _rotate(v: np.ndarray, angle: float, axis: np.ndarray) -> np.ndarray:
    """scadnano's _OxdnaVector.rotate (degrees)."""
    u = axis / np.linalg.norm(axis)
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    return u * v.dot(u) + np.cross(c * np.cross(u, v), u) - s * np.cross(u, v)


def group_axes(yaw: float, pitch: float, roll: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A helix group's (x, y, z) axes after scadnano's yaw, pitch and roll; helices run along z."""
    x, y, z = np.eye(3)
    x, z = _rotate(x, -yaw, y), _rotate(z, -yaw, y)
    y, z = _rotate(y, pitch, x), _rotate(z, pitch, x)
    y, x = _rotate(y, -roll, z), _rotate(x, -roll, z)
    return x, y, z


def euler_for(forward: np.ndarray, side: np.ndarray) -> tuple[float, float, float]:
    """Yaw, pitch, roll that point a group's helices along `forward` with grid row 1 towards `side`.

    scadnano's rotations compose to z = (cos p sin y, sin p, cos p cos y); roll then turns row 1 about z."""
    yaw = math.atan2(forward[0], forward[2])
    pitch = math.asin(np.clip(forward[1], -1, 1))
    x0 = np.array([math.cos(yaw), 0, -math.sin(yaw)])
    y0 = np.array([-math.sin(pitch) * math.sin(yaw), math.cos(pitch), -math.sin(pitch) * math.cos(yaw)])
    roll = math.atan2(-side.dot(x0), side.dot(y0))
    yaw, pitch, roll = (math.degrees(a) for a in (yaw, pitch, roll))
    x, y, z = group_axes(yaw, pitch, roll)
    assert np.allclose(z, forward, atol=1e-6) and np.allclose(y, side, atol=1e-6), (z, forward, y, side)
    return yaw, pitch, roll


def edge_frames() -> list[dict]:
    """Per edge (v < w): unit direction t (v -> w), outward normal n and left side l = n x t."""
    V = vertices()
    frames = []
    for v, w in EDGES:
        t = (V[w] - V[v]) / np.linalg.norm(V[w] - V[v])
        n = (V[v] + V[w]) / np.linalg.norm(V[v] + V[w])
        frames.append({"t": t, "n": n, "l": np.cross(n, t), "start": V[v] + SETBACK_NM * t})
    return frames


def _groups() -> dict[str, sc.HelixGroup]:
    groups = {}
    for k, f in enumerate(edge_frames()):
        yaw, pitch, roll = euler_for(f["t"], f["l"])
        p = f["start"] - PITCH / 2 * f["l"]  # helix 2k (right) at grid row 0, helix 2k+1 (left) at row 1
        groups[f"e{k}"] = sc.HelixGroup(grid=sc.square, position=sc.Position3D(*p), yaw=yaw, pitch=pitch,
                                        roll=roll, geometry=GEOMETRY)
    return groups


def ccw_next() -> dict[tuple[int, int], int]:
    """(vertex, edge) -> the next edge counter-clockwise around that vertex, seen from outside."""
    V = vertices()
    out = {}
    for v in range(len(V)):
        n = V[v] / np.linalg.norm(V[v])
        es = [k for k, e in enumerate(EDGES) if v in e]
        ref = None
        ang = {}
        for k in es:
            w = EDGES[k][1] if EDGES[k][0] == v else EDGES[k][0]
            d = V[w] - V[v]
            d -= n * d.dot(n)
            ref = d if ref is None else ref
            ang[k] = math.atan2(np.cross(ref, d).dot(n), ref.dot(d)) % (2 * math.pi)
        order = sorted(es, key=ang.get)
        for i, k in enumerate(order):
            out[(v, k)] = order[(i + 1) % len(order)]
    return out


def axis_point(helix: int, offset: float) -> np.ndarray:
    """Design position (nm) of a helix's axis at an offset."""
    f = edge_frames()[helix // 2]
    side = -PITCH / 2 if helix % 2 == 0 else PITCH / 2
    return f["start"] + side * f["l"] + offset * RISE * f["t"]


def scaffold_walk(mid: dict[int, int]) -> list[tuple[int, bool, int, int] | int]:
    """Scaffold domains (helix, forward, start, end) in 5'->3' order, with an int k = 'vertex hop' marker."""
    nxt = ccw_next()
    path: list = []
    v, k = 0, 0
    while True:
        a, b = EDGES[k]
        if k in TREE:
            path.append((2 * k, True, 0, EDGE_BP) if v == a else (2 * k + 1, False, 0, EDGE_BP))
            v = b if v == a else a
        elif v == a:
            path += [(2 * k, True, 0, mid[k]), (2 * k + 1, False, 0, mid[k])]
        else:
            path += [(2 * k + 1, False, mid[k], EDGE_BP), (2 * k, True, mid[k], EDGE_BP)]
        k = nxt[(v, k)]
        if (v, k) == (0, 0):
            return path
        path.append(-1)  # vertex hop


def _hop_loop(prev: tuple[int, bool, int, int], nxt: tuple[int, bool, int, int]) -> int:
    """Unpaired bases for a hop between helix ends at a vertex, from the gap between their axis ends."""
    h, fwd, s, e = prev
    p = axis_point(h, e if fwd else s)  # 3' end of the domain before the hop
    h, fwd, s, e = nxt
    q = axis_point(h, s if fwd else e)  # 5' end of the domain after it
    gap = np.linalg.norm(p - q)
    return max(1, round((gap - 1.0) / LOOP_NM))


def create_design() -> sc.Design:
    helices = [sc.Helix(grid_position=(0, j), max_offset=EDGE_BP, group=f"e{k}")
               for k in range(len(EDGES)) for j in range(2)]
    design = sc.Design(helices=helices, groups=_groups(), strands=[])
    fwd = {h: h % 2 == 0 for h in design.helices}  # scaffold runs v -> w on the right helix, back on the left

    def full(h: int, h2: int) -> list[int]:
        return crossover_positions(design, h, h2)[f"strand_on_helix_{'forward' if fwd[h] else 'reverse'}"][
            "full_crossover_offsets"]

    mid = {k: pick_near(full(2 * k, 2 * k + 1), EDGE_BP / 2) for k in range(len(EDGES)) if k not in TREE}
    walk = scaffold_walk(mid)
    # Break the circular route in the middle of edge 0's right helix.
    first = walk[0]
    cut = EDGE_BP // 2
    walk = [(first[0], True, cut, EDGE_BP)] + walk[1:] + [-1, (first[0], True, 0, cut)]
    hops = []
    scaffold: list = []
    for i, d in enumerate(walk):
        if d == -1:
            before, after = walk[i - 1], walk[i + 1]
            n = _hop_loop(before, after)
            hops.append((before, after, n))
            scaffold.append(sc.Loopout(length=n))
        else:
            scaffold.append(sc.Domain(helix=d[0], forward=d[1], start=d[2], end=d[3]))
    design.add_strand(sc.Strand(domains=scaffold, is_scaffold=True))

    for h in design.helices:  # one full-length staple per helix, opposite the scaffold
        if fwd[h]:
            design.draw_strand(h, EDGE_BP).move(-EDGE_BP)
        else:
            design.draw_strand(h, 0).move(EDGE_BP)
    # Staple crossovers every two turns, starting two turns in. Between two crossovers the staples close into a
    # 42-nt ring; nicking it mid-domain gives the classic DX staple (10 + 21 + 11 nt, two crossovers). Each end
    # of an edge is a 42-nt U with both ends at the vertex; nicking its domain on the helix the scaffold leaves
    # the vertex by leaves an 11-nt piece there, which the vertex staple carries across to the next edge.
    for k in range(len(EDGES)):
        a, b = 2 * k, 2 * k + 1
        pos = crossover_positions(design, a, b)[f"strand_on_helix_{'reverse' if fwd[a] else 'forward'}"]
        sites = periodic(pos["full_crossover_offsets"], STAPLE_XO_PERIOD, 15, EDGE_BP - 15)
        for o in sites:
            design.add_full_crossover(helix=a, helix2=b, offset=o, forward=not fwd[a])
        for lo, hi in zip(sites, sites[1:]):  # rings between crossovers
            design.add_nick(helix=a, offset=(lo + hi) // 2, forward=not fwd[a])
        # End Us: the scaffold leaves vertex v on helix a (offset 0 end) and vertex w on helix b (offset 84 end).
        design.add_nick(helix=a, offset=sites[0] // 2, forward=not fwd[a])
        design.add_nick(helix=b, offset=(sites[-1] + EDGE_BP) // 2, forward=not fwd[b])

    for before, after, n in hops:  # vertex staples: join the staples either side of each scaffold hop
        _join_vertex(design, before, after, n)

    design.assign_m13_to_scaffold()
    for s in design.strands:
        if not s.is_scaffold and s.dna_sequence and "?" in s.dna_sequence:
            s.set_dna_sequence(s.dna_sequence.replace("?", "T"))
    return design


def _staple_with_end(design: sc.Design, helix: int, offset: int, forward: bool, three_prime: bool) -> sc.Strand | None:
    for s in design.strands:
        if s.is_scaffold:
            continue
        d = s.last_domain() if three_prime else s.first_domain()
        if d.helix == helix and d.forward == forward and (d.offset_3p() if three_prime else d.offset_5p()) == offset:
            return s
    return None


def _join_vertex(design: sc.Design, before: tuple, after: tuple, n: int) -> None:
    """The scaffold leaves `before`'s helix and enters `after`'s at a vertex; the staple runs the other way:
    its 3' end sits on `after`'s helix at the vertex, its 5' end on `before`'s."""
    hb, fb, sb, eb = before
    ha, fa, sa, ea = after
    # Scaffold 3' end of `before` is at eb-1 (forward) or sb (reverse); the staple pairs there, 5' end.
    second = _staple_with_end(design, hb, eb - 1 if fb else sb, not fb, three_prime=False)
    # Scaffold 5' end of `after` is at sa (forward) or ea-1 (reverse); the staple pairs there, 3' end.
    first = _staple_with_end(design, ha, sa if fa else ea - 1, not fa, three_prime=True)
    if first is None or second is None or first is second:
        return
    domains = [_copy(d) for d in first.domains] + [sc.Loopout(length=n)] + [_copy(d) for d in second.domains]
    design.remove_strand(first)
    design.remove_strand(second)
    design.add_strand(sc.Strand(domains=domains))


def _copy(d):
    if isinstance(d, sc.Loopout):
        return sc.Loopout(length=d.length)
    return sc.Domain(helix=d.helix, forward=d.forward, start=d.start, end=d.end,
                     deletions=list(d.deletions), insertions=list(d.insertions))


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "wireframe_tetrahedron.sc"
    d = create_design()
    d.write_scadnano_file(filename=out)
    scaf = next(s for s in d.strands if s.is_scaffold)
    print(f"wrote {out}: scaffold {scaf.dna_length()} nt, {len(d.strands) - 1} staples")
