import math
import random
import numpy as np
import scadnano as sc

# ---------------- geometry ----------------
A_ = 17.5  # half edge: cube edge = 35 nm
verts = [np.array([x, y, z], float) for x in (-A_, A_) for y in (-A_, A_) for z in (-A_, A_)]
edges = [(i, j) for i in range(8) for j in range(i + 1, 8)
         if abs(np.linalg.norm(verts[i] - verts[j]) - 35.0) < 1e-6]
assert len(edges) == 12

HL = 84                 # bp per helix
RISE = 0.34
HALF = HL * RISE / 2    # 14.28 nm
LAT = 1.3               # half of 2.6 nm helix spacing
CROSS = [22, 43, 64]    # staple crossover boundaries (valid for this lattice, see find_crossover_offsets)


def group_angles(forward, side):
    f, s = np.asarray(forward, float), np.asarray(side, float)
    yaw, pitch = math.atan2(f[0], f[2]), math.asin(max(-1.0, min(1.0, f[1])))
    x0 = np.array([math.cos(yaw), 0.0, -math.sin(yaw)])
    y0 = np.array([-math.sin(pitch) * math.sin(yaw), math.cos(pitch), -math.sin(pitch) * math.cos(yaw)])
    roll = math.atan2(-s.dot(x0), s.dot(y0))
    return tuple(math.degrees(a) for a in (yaw, pitch, roll))


geom = sc.Geometry(bases_per_turn=10.5, rise_per_base_pair=RISE, helix_radius=1.0, inter_helix_gap=0.6)

P0, TT, SS = [], [], []
groups = {}
helices = []
for e, (i, j) in enumerate(edges):
    vi, vj = verts[i], verts[j]
    t = (vj - vi) / 35.0
    m = (vi + vj) / 2
    n = m / np.linalg.norm(m)
    s = np.cross(n, t)
    s = s / np.linalg.norm(s)
    yaw, pitch, roll = group_angles(t, s)
    p0 = m - HALF * t - LAT * s
    P0.append(p0); TT.append(t); SS.append(s)
    groups[f"E{e}"] = sc.HelixGroup(grid=sc.square,
                                    position=sc.Position3D(x=float(p0[0]), y=float(p0[1]), z=float(p0[2])),
                                    yaw=yaw, pitch=pitch, roll=roll, geometry=geom)
    for r in (0, 1):
        helices.append(sc.Helix(max_offset=HL, grid_position=(0, r), group=f"E{e}"))


def axis_point(e, r, off):
    return P0[e] + r * 2 * LAT * SS[e] + off * RISE * TT[e]


# ---------------- scaffold route: Euler circuit of the bidirected cube ----------------
# arc: row0 of edge e walks tail->head (scaffold forward); row1 walks head->tail (reverse)
arcs = []
for e, (i, j) in enumerate(edges):
    arcs.append(dict(e=e, tail=i, head=j, row=0, fwd=True, helix=2 * e))
    arcs.append(dict(e=e, tail=j, head=i, row=1, fwd=False, helix=2 * e + 1))


def entry_off(a):
    return 0 if a['fwd'] else HL - 1


def exit_off(a):
    return HL - 1 if a['fwd'] else 0


def hop_len(a, b):
    p = axis_point(a['e'], a['row'], exit_off(a))
    q = axis_point(b['e'], b['row'], entry_off(b))
    d = float(np.linalg.norm(p - q))
    return max(2, math.ceil(d / 0.6)), d


def random_circuit(rng):
    out = {v: [] for v in range(8)}
    for k, a in enumerate(arcs):
        out[a['tail']].append(k)
    for v in out:
        rng.shuffle(out[v])
    stack = [(0, None)]
    circ = []
    while stack:
        v, a = stack[-1]
        if out[v]:
            k = out[v].pop()
            stack.append((arcs[k]['head'], k))
        else:
            stack.pop()
            if a is not None:
                circ.append(a)
    circ.reverse()
    assert len(circ) == 24
    return circ


def cost(circ):
    return sum(hop_len(arcs[circ[k]], arcs[circ[k + 1]])[0] for k in range(23))


rng = random.Random(1)
best = None
for _ in range(2000):
    c = random_circuit(rng)
    cc = cost(c)
    if best is None or cc < best[0]:
        best = (cc, c)
circ = best[1]
loops = [hop_len(arcs[circ[k]], arcs[circ[k + 1]])[0] for k in range(23)]
print("route:", [(arcs[k]['e'], arcs[k]['row']) for k in circ])
print("loop lengths:", loops, "total unpaired:", sum(loops))

# ---------------- strands ----------------
scaf_domains = []
for k, idx in enumerate(circ):
    a = arcs[idx]
    scaf_domains.append(sc.Domain(helix=a['helix'], forward=a['fwd'], start=0, end=HL))
    if k < 23:
        scaf_domains.append(sc.Loopout(length=loops[k]))
strands = [sc.Strand(domains=scaf_domains, is_scaffold=True)]

for e in range(12):
    B = 2 * e          # row0: staple reverse (walks down)
    A = 2 * e + 1      # row1: staple forward (walks up)
    c1, c2, c3 = CROSS
    strands.append(sc.Strand(domains=[
        sc.Domain(helix=A, forward=True, start=0, end=c1),
        sc.Domain(helix=B, forward=False, start=0, end=c1)]))
    strands.append(sc.Strand(domains=[
        sc.Domain(helix=A, forward=True, start=(c1 + c2) // 2, end=c2),
        sc.Domain(helix=B, forward=False, start=c1, end=c2),
        sc.Domain(helix=A, forward=True, start=c1, end=(c1 + c2) // 2)]))
    strands.append(sc.Strand(domains=[
        sc.Domain(helix=A, forward=True, start=(c2 + c3) // 2, end=c3),
        sc.Domain(helix=B, forward=False, start=c2, end=c3),
        sc.Domain(helix=A, forward=True, start=c2, end=(c2 + c3) // 2)]))
    strands.append(sc.Strand(domains=[
        sc.Domain(helix=B, forward=False, start=c3, end=HL),
        sc.Domain(helix=A, forward=True, start=c3, end=HL)]))

design = sc.Design(helices=helices, groups=groups, strands=strands)
design.assign_m13_to_scaffold()
for s in design.strands:
    if not s.is_scaffold:
        s.set_dna_sequence(s.dna_sequence.replace("?", "T")) if s.dna_sequence else None
design.write_scadnano_file(filename="designs/cube.sc")

scaf = [s for s in design.strands if s.is_scaffold][0]
print("scaffold length", scaf.dna_length())
print("staple lengths", sorted(s.dna_length() for s in design.strands if not s.is_scaffold))
print("n staples", sum(1 for s in design.strands if not s.is_scaffold))
