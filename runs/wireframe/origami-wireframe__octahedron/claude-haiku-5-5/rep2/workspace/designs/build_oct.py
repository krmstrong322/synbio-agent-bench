import math, random, sys
import numpy as np
import scadnano as sc

RISE = 0.34
SEP = 2.6
LEN = 84          # bp per helix
HALF = LEN / 2.0
LOOP = 3          # unpaired bases at each vertex hop
C = [10, 31, 52, 73]   # staple crossover offsets (shared by all DX edges)
ITER = int(sys.argv[1]) if len(sys.argv) > 1 else 150000
random.seed(1)

a = 35.0 / math.sqrt(2.0)   # vertex distance from centre -> 35 nm edges
V = [np.array(v, float) for v in [(a, 0, 0), (-a, 0, 0), (0, a, 0), (0, -a, 0), (0, 0, a), (0, 0, -a)]]
edges = []
for i in range(6):
    for j in range(i + 1, 6):
        if np.linalg.norm(V[i] + V[j]) < 1e-6:
            continue  # antipodal vertices are not joined
        edges.append((i, j))
assert len(edges) == 12


def group_angles(forward, side):
    f, s = np.asarray(forward, float), np.asarray(side, float)
    yaw, pitch = math.atan2(f[0], f[2]), math.asin(max(-1.0, min(1.0, f[1])))
    x0 = np.array([math.cos(yaw), 0.0, -math.sin(yaw)])
    y0 = np.array([-math.sin(pitch) * math.sin(yaw), math.cos(pitch), -math.sin(pitch) * math.cos(yaw)])
    roll = math.atan2(-s.dot(x0), s.dot(y0))
    return tuple(math.degrees(x) for x in (yaw, pitch, roll))


geom = sc.Geometry(bases_per_turn=10.5, rise_per_base_pair=RISE, helix_radius=1.0, inter_helix_gap=0.6)

T, S, MID = [], [], []
groups = {}
helices = []
for e, (i, j) in enumerate(edges):
    v, w = V[i], V[j]
    t = (w - v) / np.linalg.norm(w - v)
    M = (v + w) / 2.0
    n = M / np.linalg.norm(M)
    s = np.cross(n, t)
    s = s / np.linalg.norm(s)
    assert abs(np.dot(s, t)) < 1e-9
    T.append(t); S.append(s); MID.append(M)
    yaw, pitch, roll = group_angles(t, s)
    p0 = M - HALF * RISE * t - 0.5 * SEP * s
    g = sc.HelixGroup(grid=sc.square, position=sc.Position3D(*p0.tolist()), yaw=yaw, pitch=pitch,
                      roll=roll, geometry=geom)
    groups[f"e{e}"] = g
    for r in range(2):
        helices.append(sc.Helix(grid_position=(0, r), group=f"e{e}", max_offset=LEN))


def axis(h, o):
    e, r = h // 2, h % 2
    return MID[e] + (r - 0.5) * SEP * S[e] + (o - HALF) * RISE * T[e]


# ---- scaffold routing search ----
def vertex_ends(h, direction):
    """(out_vertex, in_vertex) for helix h given scaffold direction (+1: offset increasing)."""
    i, j = edges[h // 2]
    return (i, j) if direction == 1 else (j, i)


def try_solution(bits):
    dirs = [0] * 24
    for e in range(12):
        d0 = 1 if bits[e] else -1
        dirs[2 * e] = d0
        dirs[2 * e + 1] = -d0
    outs = {x: [] for x in range(6)}
    ins = {x: [] for x in range(6)}
    for h in range(24):
        ov, iv = vertex_ends(h, dirs[h])
        outs[ov].append(h)
        ins[iv].append(h)
    return dirs, outs, ins


best = None
for it in range(ITER):
    bits = [random.random() < 0.5 for _ in range(12)]
    dirs, outs, ins = try_solution(bits)
    succ = [None] * 24
    for x in range(6):
        perm = outs[x][:]
        random.shuffle(perm)
        for h, g in zip(ins[x], perm):
            succ[h] = g
    # single-cycle check
    seen, cur, n = set(), 0, 0
    while cur not in seen:
        seen.add(cur)
        cur = succ[cur]
        n += 1
    if n != 24:
        continue
    cost = 0.0
    maxd = 0.0
    for h in range(24):
        g = succ[h]
        pin = axis(h, LEN if dirs[h] == 1 else 0)
        pout = axis(g, 0 if dirs[g] == 1 else LEN)
        d = float(np.linalg.norm(pin - pout))
        cost += d
        maxd = max(maxd, d)
    if best is None or cost < best[0]:
        best = (cost, maxd, dirs, succ)

cost, maxd, dirs, succ = best
print("best total hop length %.2f nm, max hop %.2f nm" % (cost, maxd))
order = [0]
while True:
    nx = succ[order[-1]]
    if nx == 0:
        break
    order.append(nx)
assert len(order) == 24
print("scaffold order (helix ids):", order)

design_strands = []

# ---- scaffold strand (linear: cycle opened at the hop from order[-1] back to order[0]) ----
dom = []
for k, h in enumerate(order):
    if k > 0:
        dom.append(sc.Loopout(length=LOOP))
    dom.append(sc.Domain(helix=h, forward=(dirs[h] == 1), start=0, end=LEN))
scaffold = sc.Strand(domains=dom, is_scaffold=True)
design_strands.append(scaffold)

# ---- staples: DX edge pairs ----
for e in range(12):
    A, B = 2 * e, 2 * e + 1
    P, Q = (A, B) if dirs[A] == -1 else (B, A)   # P: staples run forward (scaffold reverse)
    C = [27, 48, 69] if P == A else [22, 43, 64]   # valid crossover offsets for this pair's roles
    c0, cm = C[0], C[-1]
    # bottom U
    design_strands.append(sc.Strand(domains=[
        sc.Domain(helix=P, forward=True, start=0, end=c0),
        sc.Domain(helix=Q, forward=False, start=0, end=c0)]))
    # rings between consecutive crossovers, nicked mid-way on P
    for k in range(len(C) - 1):
        lo, hi = C[k], C[k + 1]
        mid = (lo + hi) // 2
        design_strands.append(sc.Strand(domains=[
            sc.Domain(helix=P, forward=True, start=mid, end=hi),
            sc.Domain(helix=Q, forward=False, start=lo, end=hi),
            sc.Domain(helix=P, forward=True, start=lo, end=mid)]))
    # top U
    design_strands.append(sc.Strand(domains=[
        sc.Domain(helix=Q, forward=False, start=cm, end=LEN),
        sc.Domain(helix=P, forward=True, start=cm, end=LEN)]))

design = sc.Design(helices=helices, groups=groups, strands=design_strands)
design.assign_m13_to_scaffold()
for s in design.strands:
    if not s.is_scaffold:
        s.set_dna_sequence(s.dna_sequence.replace("?", "T")) if s.dna_sequence and "?" in s.dna_sequence else None
scaf = [s for s in design.strands if s.is_scaffold][0]
print("scaffold length", scaf.dna_length(), "strands", len(design.strands))
stap = [s.dna_length() for s in design.strands if not s.is_scaffold]
print("staple lengths min/max", min(stap), max(stap), "count", len(stap))
design.write_scadnano_file(filename="designs/octahedron.sc")
print("saved")
