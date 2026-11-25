import itertools, math, random
import numpy as np
import scadnano as sc

R = 17.5          # half edge: vertices at (+-R,+-R,+-R), edge = 35 nm
L_HELIX = 84      # bp per helix
RISE = 0.34
HALF = L_HELIX * RISE / 2.0   # 14.28 nm
ROW = 1.3         # half of 2.6 nm row spacing
END_GAP = R - HALF            # ~3.2 nm from vertex to helix end

verts = [np.array(p, float) for p in itertools.product([-R, R], repeat=3)]
arcs = []
for i in range(8):
    for j in range(8):
        if np.sum(np.abs(verts[i] - verts[j]) > 1e-9) == 1:
            arcs.append((i, j))
assert len(arcs) == 24


def arc_geom(u, v):
    t = (verts[v] - verts[u]) / np.linalg.norm(verts[v] - verts[u])
    mid = (verts[u] + verts[v]) / 2.0
    n = mid / np.linalg.norm(mid)
    s = np.cross(n, t)
    return t, mid, s


def pos_end(arc):   # helix end at head vertex
    t, mid, s = arc_geom(*arc)
    return mid + ROW * s + HALF * t


def pos_start(arc):  # helix end at tail vertex
    t, mid, s = arc_geom(*arc)
    return mid + ROW * s - HALF * t


def hop(x, y):
    return float(np.linalg.norm(pos_end(x) - pos_start(y)))


def euler(rng):
    out = {v: [] for v in range(8)}
    for (u, v) in arcs:
        out[u].append((u, v))
    for v in out:
        rng.shuffle(out[v])
    stack = [(None, 0)]
    path = []
    while stack:
        e, v = stack[-1]
        if out[v]:
            a = out[v].pop()
            stack.append((a, a[1]))
        else:
            path.append(stack.pop())
    seq = [e for e, v in reversed(path) if e is not None]
    assert len(seq) == 24
    return seq


def score(seq):
    h = [hop(seq[i], seq[(i + 1) % 24]) for i in range(24)]
    k = int(np.argmax(h))
    rot = seq[k + 1:] + seq[:k + 1]
    lin = [h[(k + 1 + i) % 24] for i in range(23)]
    return sum(lin), rot, lin


rng = random.Random(1)
best = None
for _ in range(4000):
    seq = euler(rng)
    sc_, rot, lin = score(seq)
    if best is None or sc_ < best[0]:
        best = (sc_, rot, lin)
total, order, hops = best
print("best hop sum", round(total, 2), "max hop", round(max(hops), 2))
print("hops nm", [round(h, 2) for h in hops])

# ---- groups: one per undirected edge ----
def group_angles(forward, side):
    f, s = np.asarray(forward, float), np.asarray(side, float)
    yaw, pitch = math.atan2(f[0], f[2]), math.asin(max(-1.0, min(1.0, f[1])))
    x0 = np.array([math.cos(yaw), 0.0, -math.sin(yaw)])
    y0 = np.array([-math.sin(pitch) * math.sin(yaw), math.cos(pitch), -math.sin(pitch) * math.cos(yaw)])
    roll = math.atan2(-s.dot(x0), s.dot(y0))
    return tuple(math.degrees(a) for a in (yaw, pitch, roll))

geom = sc.Geometry(bases_per_turn=10.5, rise_per_base_pair=RISE, helix_radius=1.0, inter_helix_gap=0.6)
groups = {}
edges = sorted({tuple(sorted(a)) for a in arcs})
helix_of = {}
helices = []
for (a, b) in edges:
    t, mid, s = arc_geom(a, b)           # arc a->b lies on +s side (row 1)
    p0 = mid - HALF * t - ROW * s        # row 0 offset 0 point
    yaw, pitch, roll = group_angles(forward=t, side=s)
    name = f"e{a}_{b}"
    groups[name] = sc.HelixGroup(grid=sc.square, position=sc.Position3D(*p0), yaw=yaw, pitch=pitch,
                                 roll=roll, geometry=geom)
    # row 0 = arc (b,a), row 1 = arc (a,b)
    helix_of[(b, a)] = len(helices)
    helices.append(sc.Helix(max_offset=L_HELIX, grid_position=(0, 0), group=name))
    helix_of[(a, b)] = len(helices)
    helices.append(sc.Helix(max_offset=L_HELIX, grid_position=(0, 1), group=name))

design = sc.Design(helices=helices, groups=groups, strands=[])

# ---- scaffold ----
dom_list = []
loops = []
for i, arc in enumerate(order):
    fw = arc[0] < arc[1]
    dom_list.append(sc.Domain(helix=helix_of[arc], forward=fw, start=0, end=L_HELIX))
    if i < len(order) - 1:
        d = hop(arc, order[i + 1])
        loops.append(max(2, int(round(d / 0.6))))
domains = []
for i, dm in enumerate(dom_list):
    domains.append(dm)
    if i < len(loops):
        domains.append(sc.Loopout(length=loops[i]))
design.add_strand(sc.Strand(domains=domains, is_scaffold=True))
print("loops", loops, "scaffold nt", 24 * L_HELIX + sum(loops))

# ---- staples ----
J1, J2, J3 = 27, 48, 69   # valid full-crossover boundaries (== 6 mod 21)
for (a, b) in edges:
    hP = helix_of[(a, b)]
    hQ = helix_of[(b, a)]
    stp = [
        [sc.Domain(hQ, True, 0, J1), sc.Domain(hP, False, 0, J1)],
        [sc.Domain(hP, False, J3, 84), sc.Domain(hQ, True, J3, 84)],
    ]
    for lo, hi in [(J1, J2), (J2, J3)]:
        mid_ = lo + 11
        stp.append([sc.Domain(hP, False, lo, mid_), sc.Domain(hQ, True, lo, hi), sc.Domain(hP, False, mid_, hi)])
    for doms in stp:
        design.add_strand(sc.Strand(domains=doms))

design.assign_m13_to_scaffold()
for s in design.strands:
    if not s.is_scaffold:
        s.set_dna_sequence(s.dna_sequence.replace("?", "T")) if s.dna_sequence else None
design.write_scadnano_file(filename="designs/cube.sc")
print("strands", len(design.strands), "helices", len(design.helices))
print("staple lengths", sorted({s.dna_length() for s in design.strands if not s.is_scaffold}))
