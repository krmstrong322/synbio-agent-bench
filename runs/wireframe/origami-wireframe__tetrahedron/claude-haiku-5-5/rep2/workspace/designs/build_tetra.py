import math
import numpy as np
import scadnano as sc

EDGE = 35.0
S_ = EDGE / (2 * math.sqrt(2))
V = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float) * S_
EDGES = [(i, j) for i in range(4) for j in range(i + 1, 4)]  # pair p -> helices 2p (row0), 2p+1 (row1)
L = 84
RISE = 0.34
HALF = L * RISE / 2
SPACING = 2.6
LOOP_NM = 0.7


def group_angles(forward, side):
    f, s = np.asarray(forward, float), np.asarray(side, float)
    yaw, pitch = math.atan2(f[0], f[2]), math.asin(max(-1.0, min(1.0, f[1])))
    x0 = np.array([math.cos(yaw), 0.0, -math.sin(yaw)])
    y0 = np.array([-math.sin(pitch) * math.sin(yaw), math.cos(pitch), -math.sin(pitch) * math.cos(yaw)])
    roll = math.atan2(-s.dot(x0), s.dot(y0))
    return tuple(math.degrees(a) for a in (yaw, pitch, roll))


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


# per edge frame: midpoint M, direction t (i->j), side s = n x t (tangent to circumsphere)
P0, T, SD = [], [], []
for p, (i, j) in enumerate(EDGES):
    M = (V[i] + V[j]) / 2
    t = unit(V[j] - V[i])
    n = unit(M)
    s = unit(np.cross(n, t))
    P0.append(M - HALF * t - 1.3 * s)
    T.append(t)
    SD.append(s)


def pos(h, off):
    p, r = h // 2, h % 2
    return P0[p] + r * SPACING * SD[p] + off * RISE * T[p]


def end_off(h, vertex):
    i, j = EDGES[h // 2]
    return 0 if vertex == i else L


def hop_nm(h1, v1, h2, v2):
    return float(np.linalg.norm(pos(h1, end_off(h1, v1)) - pos(h2, end_off(h2, v2))))


def loop_len(d):
    return max(2, math.ceil(d / LOOP_NM))


# ---------- scaffold route (DFS, no in-pair turns, minimal loop nt) ----------
best = [10**9, None]


def rec(path, cur_v, cost):
    if cost >= best[0]:
        return
    if len(path) == 12:
        best[0], best[1] = cost, list(path)
        return
    used = {x[0]: x for x in path}
    for h in range(12):
        if h in used:
            continue
        p = h // 2
        i, j = EDGES[p]
        partner = h ^ 1
        options = [i, j] if not path else [cur_v]
        for e in options:
            if e not in (i, j):
                continue
            fwd = (e == i)
            if partner in used and used[partner][1] == fwd:
                continue  # the two rows of an edge must run antiparallel
            if path and path[-1][0] // 2 == p:
                continue
            exitv = j if e == i else i
            c = 0
            if path:
                prev = path[-1]
                c = loop_len(hop_nm(prev[0], prev[3], h, e))
            rec(path + [(h, fwd, e, exitv)], exitv, cost + c)


rec([], None, 0)
route = best[1]
print("route loop nt:", best[0])
for (h, fwd, e, x) in route:
    print("helix", h, "pair", h // 2, "fwd" if fwd else "rev", "enter", e, "exit", x)
fwd_of = {h: fwd for (h, fwd, e, x) in route}
loops = {}
for k in range(len(route) - 1):
    a, b = route[k], route[k + 1]
    d = hop_nm(a[0], a[3], b[0], b[2])
    loops[k] = loop_len(d)
    print(f"hop {a[0]}->{b[0]} vertex {a[3]} dist {d:.2f} loop {loops[k]}")

# ---------- design ----------
groups = {}
for p in range(6):
    yaw, pitch, roll = group_angles(T[p], SD[p])
    groups[f"e{p}"] = sc.HelixGroup(grid=sc.square,
                                    position=sc.Position3D(x=P0[p][0], y=P0[p][1], z=P0[p][2]),
                                    yaw=yaw, pitch=pitch, roll=roll,
                                    geometry=sc.Geometry(bases_per_turn=10.5, rise_per_base_pair=0.34,
                                                         helix_radius=1.0, inter_helix_gap=0.6))
helices = [sc.Helix(grid_position=(0, r), group=f"e{p}", max_offset=L) for p in range(6) for r in range(2)]
design = sc.Design(helices=helices, groups=groups, strands=[])

scaf = []
for k, (h, fwd, e, x) in enumerate(route):
    scaf.append(sc.Domain(helix=h, forward=fwd, start=0, end=L))
    if k in loops:
        scaf.append(sc.Loopout(length=loops[k]))
design.add_strand(sc.Strand(domains=scaf, is_scaffold=True))
N = 12 * L + sum(loops.values())
print("scaffold length", N, "paired fraction", round(12 * L / N, 4))


def D(h, fw, a, b):
    return sc.Domain(helix=h, forward=fw, start=a, end=b)


for p in range(6):
    R0, R1 = 2 * p, 2 * p + 1
    f0 = not fwd_of[R0]          # staple direction on row 0
    c1, c2, c3 = (27, 48, 69) if f0 else (22, 43, 64)
    if f0:
        doms = [[D(R0, True, 0, c1), D(R1, False, 0, c1)]]
    else:
        doms = [[D(R1, True, 0, c1), D(R0, False, 0, c1)]]
    for a, b in [(c1, c2), (c2, c3)]:
        m = (a + b) // 2
        if f0:
            doms.append([D(R0, True, m, b), D(R1, False, a, b), D(R0, True, a, m)])
        else:
            doms.append([D(R0, False, a, m), D(R1, True, a, b), D(R0, False, m, b)])
    if f0:
        doms.append([D(R1, False, c3, L), D(R0, True, c3, L)])
    else:
        doms.append([D(R0, False, c3, L), D(R1, True, c3, L)])
    for dl in doms:
        design.add_strand(sc.Strand(domains=dl))

design.assign_m13_to_scaffold()
for s in design.strands:
    if s.dna_sequence and "?" in s.dna_sequence:
        s.set_dna_sequence(s.dna_sequence.replace("?", "T"))

design.write_scadnano_file(filename="designs/tetra.sc")
lens = sorted(set(s.dna_length() for s in design.strands if not s.is_scaffold))
print("staple lengths", lens, "count", sum(1 for s in design.strands if not s.is_scaffold))
