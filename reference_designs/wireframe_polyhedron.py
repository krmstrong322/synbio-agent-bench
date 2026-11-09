"""Reference wireframe polyhedra (DAEDALUS style, Veneziano et al., Science 2016): cube and octahedron.

Run: uv run python reference_designs/wireframe_polyhedron.py <cube|octahedron|tetrahedron> <out.sc>

A general version of the routing in wireframe_tetrahedron.py (which is kept as the tetrahedron reference and
unchanged). Every edge is a two-helix (DX) bundle of EDGE_BP in its own helix group, oriented so the edge runs
along a polyhedron edge. The scaffold follows a spanning tree, doubling tree edges and running to the middle and
back on non-tree edges, turning counter-clockwise at every vertex; staples pair each edge's helices and join
edges at the vertices. The spanning tree is found by BFS, so this works for any polyhedron whose vertices and
edges match src/synbio_bench/origami/wireframe.polyhedron (same coordinates, so the grader's target lines up).

The delicate Euler-angle maths (euler_for/group_axes) is imported from wireframe_tetrahedron, not re-derived.
"""

from __future__ import annotations

import math
import sys
from collections import deque

import numpy as np
import scadnano as sc
from common import crossover_positions, periodic, pick_near
from wireframe_tetrahedron import euler_for

EDGE_BP = 84  # 8 turns at 10.5 bp/turn
GEOMETRY = sc.Geometry(bases_per_turn=10.5, rise_per_base_pair=0.34, helix_radius=1.0, inter_helix_gap=0.6)
PITCH = GEOMETRY.distance_between_helices()
RISE = GEOMETRY.rise_per_base_pair
STAPLE_XO_PERIOD = 21
LOOP_NM = 0.65

# Vertices (unit) and setback per polyhedron. Vertex sets match wireframe.polyhedron() exactly.
PARAMS: dict[str, dict] = {
    "tetrahedron": {"verts": [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)], "setback_nm": 4.5},
    "cube": {"verts": [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], "setback_nm": 4.5},
    "octahedron": {"verts": [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)], "setback_nm": 4.5},
}


class Polyhedron:
    def __init__(self, kind: str):
        p = PARAMS[kind]
        self.kind = kind
        self.setback = p["setback_nm"]
        edge_nm = EDGE_BP * RISE + 2 * self.setback
        v = np.array(p["verts"], float)
        d = np.linalg.norm(v[:, None] - v[None], axis=2)
        shortest = d[d > 1e-9].min()
        self.V = v * edge_nm / shortest
        d = np.linalg.norm(self.V[:, None] - self.V[None], axis=2)
        length = d[d > 1e-9].min()
        self.edges = [(i, j) for i in range(len(v)) for j in range(i + 1, len(v)) if abs(d[i, j] - length) < 1e-6]
        self.tree = self._spanning_tree()
        self._frames = self._edge_frames()

    def _spanning_tree(self) -> set[int]:
        """Edge indices of a BFS spanning tree from vertex 0."""
        adj: dict[int, list[tuple[int, int]]] = {i: [] for i in range(len(self.V))}
        for k, (a, b) in enumerate(self.edges):
            adj[a].append((b, k))
            adj[b].append((a, k))
        seen, tree, q = {0}, set(), deque([0])
        while q:
            v = q.popleft()
            for w, k in adj[v]:
                if w not in seen:
                    seen.add(w)
                    tree.add(k)
                    q.append(w)
        return tree

    def _edge_frames(self) -> list[dict]:
        frames = []
        for v, w in self.edges:
            t = (self.V[w] - self.V[v]) / np.linalg.norm(self.V[w] - self.V[v])
            n = (self.V[v] + self.V[w]) / np.linalg.norm(self.V[v] + self.V[w])
            frames.append({"t": t, "n": n, "l": np.cross(n, t), "start": self.V[v] + self.setback * t})
        return frames

    def groups(self) -> dict[str, sc.HelixGroup]:
        groups = {}
        for k, f in enumerate(self._frames):
            yaw, pitch, roll = euler_for(f["t"], f["l"])
            p = f["start"] - PITCH / 2 * f["l"]
            groups[f"e{k}"] = sc.HelixGroup(grid=sc.square, position=sc.Position3D(*p), yaw=yaw, pitch=pitch,
                                            roll=roll, geometry=GEOMETRY)
        return groups

    def axis_point(self, helix: int, offset: float) -> np.ndarray:
        f = self._frames[helix // 2]
        side = -PITCH / 2 if helix % 2 == 0 else PITCH / 2
        return f["start"] + side * f["l"] + offset * RISE * f["t"]

    def ccw_next(self) -> dict[tuple[int, int], int]:
        out = {}
        for v in range(len(self.V)):
            n = self.V[v] / np.linalg.norm(self.V[v])
            es = [k for k, e in enumerate(self.edges) if v in e]
            ref, ang = None, {}
            for k in es:
                w = self.edges[k][1] if self.edges[k][0] == v else self.edges[k][0]
                d = self.V[w] - self.V[v]
                d -= n * d.dot(n)
                ref = d if ref is None else ref
                ang[k] = math.atan2(np.cross(ref, d).dot(n), ref.dot(d)) % (2 * math.pi)
            order = sorted(es, key=ang.get)
            for i, k in enumerate(order):
                out[(v, k)] = order[(i + 1) % len(order)]
        return out

    def scaffold_walk(self, mid: dict[int, int]) -> list:
        nxt = self.ccw_next()
        path: list = []
        v, k = 0, 0
        while True:
            a, b = self.edges[k]
            if k in self.tree:
                path.append((2 * k, True, 0, EDGE_BP) if v == a else (2 * k + 1, False, 0, EDGE_BP))
                v = b if v == a else a
            elif v == a:
                path += [(2 * k, True, 0, mid[k]), (2 * k + 1, False, 0, mid[k])]
            else:
                path += [(2 * k + 1, False, mid[k], EDGE_BP), (2 * k, True, mid[k], EDGE_BP)]
            k = nxt[(v, k)]
            if (v, k) == (0, 0):
                return path
            path.append(-1)


def _hop_loop(poly: Polyhedron, prev: tuple, nxt: tuple) -> int:
    h, fwd, s, e = prev
    p = poly.axis_point(h, e if fwd else s)
    h, fwd, s, e = nxt
    q = poly.axis_point(h, s if fwd else e)
    return max(1, round((np.linalg.norm(p - q) - 1.0) / LOOP_NM))


def _copy(d):
    if isinstance(d, sc.Loopout):
        return sc.Loopout(length=d.length)
    return sc.Domain(helix=d.helix, forward=d.forward, start=d.start, end=d.end,
                     deletions=list(d.deletions), insertions=list(d.insertions))


def _staple_with_end(design, helix, offset, forward, three_prime):
    for s in design.strands:
        if s.is_scaffold:
            continue
        d = s.last_domain() if three_prime else s.first_domain()
        if d.helix == helix and d.forward == forward and (d.offset_3p() if three_prime else d.offset_5p()) == offset:
            return s
    return None


def _join_vertex(design, before, after, n):
    hb, fb, sb, eb = before
    ha, fa, sa, ea = after
    second = _staple_with_end(design, hb, eb - 1 if fb else sb, not fb, three_prime=False)
    first = _staple_with_end(design, ha, sa if fa else ea - 1, not fa, three_prime=True)
    if first is None or second is None or first is second:
        return
    domains = [_copy(d) for d in first.domains] + [sc.Loopout(length=n)] + [_copy(d) for d in second.domains]
    design.remove_strand(first)
    design.remove_strand(second)
    design.add_strand(sc.Strand(domains=domains))


def create_design(kind: str) -> sc.Design:
    poly = Polyhedron(kind)
    n_edges = len(poly.edges)
    helices = [sc.Helix(grid_position=(0, j), max_offset=EDGE_BP, group=f"e{k}")
               for k in range(n_edges) for j in range(2)]
    design = sc.Design(helices=helices, groups=poly.groups(), strands=[])
    fwd = {h: h % 2 == 0 for h in design.helices}

    def full(h, h2):
        return crossover_positions(design, h, h2)[f"strand_on_helix_{'forward' if fwd[h] else 'reverse'}"][
            "full_crossover_offsets"]

    mid = {k: pick_near(full(2 * k, 2 * k + 1), EDGE_BP / 2) for k in range(n_edges) if k not in poly.tree}
    walk = poly.scaffold_walk(mid)
    first = walk[0]
    cut = EDGE_BP // 2
    walk = [(first[0], True, cut, EDGE_BP)] + walk[1:] + [-1, (first[0], True, 0, cut)]
    hops, scaffold = [], []
    for i, d in enumerate(walk):
        if d == -1:
            before, after = walk[i - 1], walk[i + 1]
            n = _hop_loop(poly, before, after)
            hops.append((before, after, n))
            scaffold.append(sc.Loopout(length=n))
        else:
            scaffold.append(sc.Domain(helix=d[0], forward=d[1], start=d[2], end=d[3]))
    design.add_strand(sc.Strand(domains=scaffold, is_scaffold=True))

    for h in design.helices:
        if fwd[h]:
            design.draw_strand(h, EDGE_BP).move(-EDGE_BP)
        else:
            design.draw_strand(h, 0).move(EDGE_BP)
    for k in range(n_edges):
        a, b = 2 * k, 2 * k + 1
        pos = crossover_positions(design, a, b)[f"strand_on_helix_{'reverse' if fwd[a] else 'forward'}"]
        sites = periodic(pos["full_crossover_offsets"], STAPLE_XO_PERIOD, 15, EDGE_BP - 15)
        for o in sites:
            design.add_full_crossover(helix=a, helix2=b, offset=o, forward=not fwd[a])
        for lo, hi in zip(sites, sites[1:]):
            design.add_nick(helix=a, offset=(lo + hi) // 2, forward=not fwd[a])
        design.add_nick(helix=a, offset=sites[0] // 2, forward=not fwd[a])
        design.add_nick(helix=b, offset=(sites[-1] + EDGE_BP) // 2, forward=not fwd[b])

    for before, after, n in hops:
        _join_vertex(design, before, after, n)

    design.assign_m13_to_scaffold()
    for s in design.strands:
        if not s.is_scaffold and s.dna_sequence and "?" in s.dna_sequence:
            s.set_dna_sequence(s.dna_sequence.replace("?", "T"))
    return design


if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else "cube"
    out = sys.argv[2] if len(sys.argv) > 2 else f"wireframe_{kind}.sc"
    d = create_design(kind)
    d.write_scadnano_file(filename=out)
    scaf = next(s for s in d.strands if s.is_scaffold)
    print(f"wrote {out}: {kind}, scaffold {scaf.dna_length()} nt, {len(d.strands) - 1} staples")
