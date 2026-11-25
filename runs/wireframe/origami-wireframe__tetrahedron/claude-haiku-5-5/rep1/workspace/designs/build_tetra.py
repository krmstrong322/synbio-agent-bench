import math
import random
import numpy as np
import scadnano as sc
from collections import defaultdict

# ---------------- parameters ----------------
N = 84                 # bp per helix
RISE = 0.34
HALF = N * RISE / 2.0  # 14.28 nm
SPACING = 2.6          # nm between the two helices of an edge
C = [22, 43, 64]       # crossover boundaries along each edge (offset; valid = 1 mod 21 for this pair)
OUT = "designs/tetra.sc"

# ---------------- geometry ----------------
s = 35.0 / (2 * math.sqrt(2))
VERTS = [s * np.array(p, float) for p in [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]]


def group_angles(forward, side):
    f, sd = np.asarray(forward, float), np.asarray(side, float)
    yaw, pitch = math.atan2(f[0], f[2]), math.asin(max(-1.0, min(1.0, f[1])))
    x0 = np.array([math.cos(yaw), 0.0, -math.sin(yaw)])
    y0 = np.array([-math.sin(pitch) * math.sin(yaw), math.cos(pitch), -math.sin(pitch) * math.cos(yaw)])
    roll = math.atan2(-sd.dot(x0), sd.dot(y0))
    return tuple(math.degrees(a) for a in (yaw, pitch, roll))


edges = [(v, w) for v in range(4) for w in range(v + 1, 4)]
eidx = {e: i for i, e in enumerate(edges)}
geom = {}
for i, (v, w) in enumerate(edges):
    A, B = VERTS[v], VERTS[w]
    M = (A + B) / 2
    t = (B - A) / np.linalg.norm(B - A)
    n = M / np.linalg.norm(M)
    side = np.cross(n, t)
    side = side / np.linalg.norm(side)
    p0 = M - HALF * t - (SPACING / 2) * side
    geom[i] = (p0, t, side)


def axis_point(e, row, o):
    p0, t, side = geom[e]
    return p0 + row * SPACING * side + o * RISE * t


def hid(a, b):
    """helix index and scaffold forward flag for traversal a->b"""
    v, w = min(a, b), max(a, b)
    e = eidx[(v, w)]
    row = 0 if a < b else 1
    return 2 * e + row, (a < b)


def hop_lengths(circ):
    res = []
    for k in range(11):
        a1, b1 = circ[k]
        a2, b2 = circ[k + 1]
        h1, f1 = hid(a1, b1)
        h2, f2 = hid(a2, b2)
        o1 = N - 0.5 if f1 else 0.5
        o2 = 0.5 if f2 else N - 0.5
        d = float(np.linalg.norm(axis_point(h1 // 2, h1 % 2, o1) - axis_point(h2 // 2, h2 % 2, o2)))
        res.append((d, max(3, math.ceil(d / 0.6))))
    return res


# ---------------- Euler circuit over directed K4 (each ordered pair = one helix) ----------------
arcs = [(a, b) for a in range(4) for b in range(4) if a != b]


def euler_random(rng):
    al = arcs[:]
    rng.shuffle(al)
    out = defaultdict(list)
    for (a, b) in al:
        out[a].append((a, b))
    stack = [(0, None)]
    circ = []
    while stack:
        v, e = stack[-1]
        if out[v]:
            arc = out[v].pop()
            stack.append((arc[1], arc))
        else:
            stack.pop()
            if e is not None:
                circ.append(e)
    circ.reverse()
    return circ


rng = random.Random(1)
best = None
for _ in range(400):
    c = euler_random(rng)
    if len(c) != 12:
        continue
    hl = hop_lengths(c)
    score = sum(l for _, l in hl)
    if best is None or score < best[0]:
        best = (score, c, hl)
score, circuit, hl = best
for k in range(11):
    assert circuit[k][1] == circuit[k + 1][0]
LOOPS = [l for _, l in hl]
print("circuit:", circuit)
print("hop distances (nm):", [round(d, 2) for d, _ in hl])
print("loop lengths:", LOOPS, "total", sum(LOOPS))

# ---------------- helices & groups ----------------
groups = {}
for i, (v, w) in enumerate(edges):
    p0, t, side = geom[i]
    yaw, pitch, roll = group_angles(t, side)
    groups[f"e{i}"] = sc.HelixGroup(
        grid=sc.square,
        position=sc.Position3D(x=float(p0[0]), y=float(p0[1]), z=float(p0[2])),
        yaw=yaw, pitch=pitch, roll=roll,
        geometry=sc.Geometry(bases_per_turn=10.5, rise_per_base_pair=0.34,
                             helix_radius=1.0, inter_helix_gap=0.6))

helices = []
for i in range(6):
    for row in range(2):
        helices.append(sc.Helix(grid_position=(0, row), group=f"e{i}", max_offset=N))

# ---------------- scaffold ----------------
dom = []
for k, (a, b) in enumerate(circuit):
    h, f = hid(a, b)
    dom.append(sc.Domain(helix=h, forward=f, start=0, end=N))
    if k < len(circuit) - 1:
        dom.append(sc.Loopout(length=LOOPS[k]))
strands = [sc.Strand(domains=dom, is_scaffold=True)]

# ---------------- staples ----------------
for i in range(6):
    B = 2 * i          # scaffold forward (staple reverse)
    A = 2 * i + 1      # scaffold reverse (staple forward)
    strands.append(sc.Strand(domains=[          # low-end U staple
        sc.Domain(helix=A, forward=True, start=0, end=C[0]),
        sc.Domain(helix=B, forward=False, start=0, end=C[0])]))
    strands.append(sc.Strand(domains=[          # high-end U staple
        sc.Domain(helix=B, forward=False, start=C[-1], end=N),
        sc.Domain(helix=A, forward=True, start=C[-1], end=N)]))
    for k in range(len(C) - 1):                 # rings, nicked mid-B-domain
        c0, c1 = C[k], C[k + 1]
        m = (c0 + c1) // 2
        strands.append(sc.Strand(domains=[
            sc.Domain(helix=B, forward=False, start=c0, end=m),
            sc.Domain(helix=A, forward=True, start=c0, end=c1),
            sc.Domain(helix=B, forward=False, start=m, end=c1)]))

design = sc.Design(helices=helices, groups=groups, strands=strands)
design.assign_m13_to_scaffold()
print("scaffold length:", [st.dna_length() for st in design.strands if st.is_scaffold])
print("staple lengths:", sorted(st.dna_length() for st in design.strands if not st.is_scaffold))
design.write_scadnano_file(filename=OUT)
print("saved", OUT)
