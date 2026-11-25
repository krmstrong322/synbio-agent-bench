import math, random, json
import numpy as np
import scadnano as sc

AH = 17.5          # cube half-edge: vertices at (+-AH,+-AH,+-AH), edge 35 nm
EDGE = 35.0
N = 84             # bp per helix
RISE = 0.34
GAP = (EDGE - N * RISE) / 2.0   # helix end inset from vertex along edge (~3.22 nm)
HALF = 1.3         # half of 2.6 nm helix spacing

def vert(i):
    return np.array([AH if (i >> k) & 1 else -AH for k in range(3)], float)

EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count('1') == 1]
assert len(EDGES) == 12

def side_options(f):
    out = []
    for k in range(3):
        if abs(f[k]) < 1e-9:
            for sg in (1, -1):
                e = np.zeros(3); e[k] = sg; out.append(e)
    return out

def group_angles(forward, side):
    f, s = np.asarray(forward, float), np.asarray(side, float)
    yaw, pitch = math.atan2(f[0], f[2]), math.asin(max(-1.0, min(1.0, f[1])))
    x0 = np.array([math.cos(yaw), 0.0, -math.sin(yaw)])
    y0 = np.array([-math.sin(pitch) * math.sin(yaw), math.cos(pitch), -math.sin(pitch) * math.cos(yaw)])
    roll = math.atan2(-s.dot(x0), s.dot(y0))
    return tuple(math.degrees(a) for a in (yaw, pitch, roll))

def edge_geom(e, s_idx):
    i, j = EDGES[e]
    P, Q = vert(i), vert(j)
    f = (Q - P) / EDGE
    s = side_options(f)[s_idx]
    return P, Q, f, s

def helix_points(e, r, s_idx):
    """start (offset 0) and far (offset N) axis points of helix row r of edge e."""
    P, Q, f, s = edge_geom(e, s_idx)
    p0 = P + GAP * f + HALF * (2 * r - 1) * s
    return p0, p0 + N * RISE * f

def fwd_of(r, b):
    return r == b

def evaluate(s_idx, bb, perms, want_cycle=True):
    """bb[e] in {0,1}: row r scaffold runs forward (tail->head) iff r==bb[e]."""
    info = {}
    arr_at = {v: [] for v in range(8)}
    dep_at = {v: [] for v in range(8)}
    pts = {}
    for h in range(24):
        e, r = h // 2, h % 2
        i, j = EDGES[e]
        p_start, p_far = helix_points(e, r, s_idx[e])
        if fwd_of(r, bb[e]):
            dep_v, arr_v = i, j
            dep_pt, arr_pt = p_start, p_far
        else:
            dep_v, arr_v = j, i
            dep_pt, arr_pt = p_far, p_start
        arr_at[arr_v].append(h); dep_at[dep_v].append(h)
        pts[(h, 'arr')] = (arr_v, arr_pt)
        pts[(h, 'dep')] = (dep_v, dep_pt)
    nxt = {}
    cost = 0.0
    for v in range(8):
        A, D = arr_at[v], dep_at[v]
        assert len(A) == 3 and len(D) == 3, (v, A, D)
        for k, h in enumerate(A):
            h2 = D[perms[v][k]]
            nxt[h] = h2
            d = np.linalg.norm(pts[(h, 'arr')][1] - pts[(h2, 'dep')][1])
            cost += d
        # clash penalty: all helix end points at vertex v from different helices
        ends = [(h, pts[(h, 'arr')][1]) for h in A] + [(h, pts[(h, 'dep')][1]) for h in D]
        for a in range(6):
            for b in range(a + 1, 6):
                ha, pa = ends[a]; hb, pb = ends[b]
                if ha == hb:
                    continue
                if nxt.get(ha) == hb or nxt.get(hb) == ha:
                    continue
                dd = np.linalg.norm(pa - pb)
                if dd < 3.0:
                    cost += 5.0 * (3.0 - dd)
    # cycle check
    seen = [0]; h = 0; n = 0
    while True:
        h = nxt[h]; n += 1
        if h == 0: break
        if n > 30: break
    single = (n == 24)
    return single, cost, nxt, pts
