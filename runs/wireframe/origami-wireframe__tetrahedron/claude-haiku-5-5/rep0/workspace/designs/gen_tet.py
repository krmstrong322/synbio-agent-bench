import math, sys
import numpy as np
import scadnano as sc

EDGE_NM = 35.0
NBP = 84
RISE = 0.34
HALF = (NBP / 2) * RISE          # 14.28 nm
ROW_SP = 2.6
GAP_SIDE = 1.3
C = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [21, 42, 63]
OUT = sys.argv[2] if len(sys.argv) > 2 else "designs/tet.sc"

s = EDGE_NM / (2 * math.sqrt(2))
V = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float) * s
EDGES = [(i, j) for i in range(4) for j in range(i + 1, 4)]
EIDX = {e: k for k, e in enumerate(EDGES)}


def frame(e):
    i, j = EDGES[e]
    a, b = V[i], V[j]
    mid = (a + b) / 2
    t = (b - a) / np.linalg.norm(b - a)
    n = mid / np.linalg.norm(mid)
    side = np.cross(n, t)
    side /= np.linalg.norm(side)
    return mid, t, side


def group_angles(forward, side):
    f, sd = np.asarray(forward, float), np.asarray(side, float)
    yaw, pitch = math.atan2(f[0], f[2]), math.asin(max(-1.0, min(1.0, f[1])))
    x0 = np.array([math.cos(yaw), 0.0, -math.sin(yaw)])
    y0 = np.array([-math.sin(pitch) * math.sin(yaw), math.cos(pitch), -math.sin(pitch) * math.cos(yaw)])
    roll = math.atan2(-sd.dot(x0), sd.dot(y0))
    return tuple(math.degrees(a) for a in (yaw, pitch, roll))


def axis_pt(e, row, s_nm):
    """point on helix axis of edge e, row, at signed position s_nm along the edge (0 = midpoint)"""
    mid, t, side = frame(e)
    return mid + (row - 0.5) * ROW_SP * side + s_nm * t


def vertex_s(e, v):
    i, j = EDGES[e]
    return -HALF if v == i else HALF


# ---------- scaffold route: directed Eulerian circuits of the doubled tetrahedron ----------
ARCS = [(i, j) for i in range(4) for j in range(4) if i != j]


def circuits():
    out = []

    def dfs(path, used):
        if len(path) == len(ARCS):
            if path[-1][1] == path[0][0]:
                out.append(list(path))
            return
        v = path[-1][1]
        for a in ARCS:
            if a not in used and a[0] == v:
                used.add(a)
                path.append(a)
                dfs(path, used)
                path.pop()
                used.remove(a)

    dfs([(0, 1)], {(0, 1)})
    return out


def arc_edge(a):
    return EIDX[tuple(sorted(a))]


def arc_row(a, r0):
    e = arc_edge(a)
    return r0[e] if a[0] < a[1] else 1 - r0[e]


def arc_fwd(a):
    return a[0] < a[1]


def hop_len(A, B, r0):
    v = A[1]
    eA, eB = arc_edge(A), arc_edge(B)
    pA = axis_pt(eA, arc_row(A, r0), vertex_s(eA, v))
    pB = axis_pt(eB, arc_row(B, r0), vertex_s(eB, v))
    return float(np.linalg.norm(pA - pB))


best = None
allc = circuits()
print("circuits:", len(allc))
for circ in allc:
    for mask in range(64):
        r0 = [(mask >> k) & 1 for k in range(6)]
        hops = []
        ok = True
        for k in range(12):
            A, B = circ[k], circ[(k + 1) % 12]
            hl = hop_len(A, B, r0)
            if arc_edge(A) == arc_edge(B):
                hl += 1000.0  # penalty: U-turn on same edge
            hops.append(hl)
        if not ok:
            continue
        m = int(np.argmax(hops))  # this hop is left open (scaffold ends)
        cost = sum(hops) - hops[m]
        if best is None or cost < best[0]:
            best = (cost, circ, r0, m, hops)

cost, circ, r0, m, hops = best
print("best hop-sum (nm)", round(cost, 2), "open hop", m)
print("hops (nm):", [round(h, 2) for h in hops])
# linear order: start after the open hop
order = [circ[(m + 1 + k) % 12] for k in range(12)]
hop_seq = [hops[(m + 1 + k) % 12] for k in range(11)]  # hop between order[k] and order[k+1]

LOOP = []
for h in hop_seq:
    d_eff = max(0.0, h - 0.3)
    LOOP.append(max(2, math.ceil(d_eff / 0.65)))
print("loop lengths:", LOOP, "sum", sum(LOOP))

# ---------- helices and groups ----------
groups = {}
for e in range(6):
    mid, t, side = frame(e)
    p0 = mid - GAP_SIDE * side - HALF * t
    yaw, pitch, roll = group_angles(forward=t, side=side)
    groups[f"g{e}"] = sc.HelixGroup(
        grid=sc.square, position=sc.Position3D(*[float(x) for x in p0]),
        yaw=yaw, pitch=pitch, roll=roll,
        geometry=sc.Geometry(bases_per_turn=10.5, rise_per_base_pair=RISE,
                             helix_radius=1.0, inter_helix_gap=0.6))
helices = []
for e in range(6):
    for row in range(2):
        helices.append(sc.Helix(grid_position=(0, row), group=f"g{e}", max_offset=NBP))
design = sc.Design(helices=helices, groups=groups, strands=[])


def hid(e, row):
    return 2 * e + row


# scaffold direction per helix
scaf_fwd = {}
for a in order:
    scaf_fwd[hid(arc_edge(a), arc_row(a, r0))] = arc_fwd(a)
for e in range(6):
    assert scaf_fwd[hid(e, 0)] != scaf_fwd[hid(e, 1)], "antiparallel violated"

scaf_domains = []
for k, a in enumerate(order):
    h = hid(arc_edge(a), arc_row(a, r0))
    scaf_domains.append(sc.Domain(helix=h, forward=arc_fwd(a), start=0, end=NBP))
    if k < 11:
        scaf_domains.append(sc.Loopout(length=LOOP[k]))
scaf = sc.Strand(domains=scaf_domains, is_scaffold=True)
design.add_strand(scaf)

# ---------- staples: hairpins (one crossover per boundary) ----------
for e in range(6):
    a, b = hid(e, 0), hid(e, 1)
    fa = not scaf_fwd[a]
    # crossover offsets valid for the staple direction on row 0 (from find_crossover_offsets)
    Ce = [27, 48] if fa else [22, 43, 64]
    bounds = [0] + Ce + [NBP]
    segs = [(bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1)]
    for lo, hi in segs:
        hairpin = (fa and hi < NBP) or ((not fa) and lo > 0)
        if hairpin:
            if fa:
                doms = [sc.Domain(helix=a, forward=True, start=lo, end=hi),
                        sc.Domain(helix=b, forward=False, start=lo, end=hi)]
            else:
                doms = [sc.Domain(helix=a, forward=False, start=lo, end=hi),
                        sc.Domain(helix=b, forward=True, start=lo, end=hi)]
            design.add_strand(sc.Strand(domains=doms))
        else:
            design.add_strand(sc.Strand(domains=[sc.Domain(helix=a, forward=fa, start=lo, end=hi)]))
            design.add_strand(sc.Strand(domains=[sc.Domain(helix=b, forward=not fa, start=lo, end=hi)]))

design.assign_m13_to_scaffold()
design.write_scadnano_file(filename=OUT)
L = [s_.dna_length() for s_ in design.strands]
print("scaffold length", scaf.dna_length(), "n strands", len(design.strands))
print("staple lengths min/max", min(x for x, st in zip(L, design.strands) if not st.is_scaffold),
      max(x for x, st in zip(L, design.strands) if not st.is_scaffold))
