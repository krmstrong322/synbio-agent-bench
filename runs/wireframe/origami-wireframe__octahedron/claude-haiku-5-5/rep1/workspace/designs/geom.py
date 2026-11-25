import math
import itertools
import numpy as np
import scadnano as sc

A_VERT = 35.0 / math.sqrt(2)   # vertex distance from centre (edge = 35 nm)
NBP = 84
RISE = 0.34
HALF = NBP * RISE / 2          # 14.28 nm
ROWSEP = 2.6

def group_angles(forward, side):
    f, s = np.asarray(forward, float), np.asarray(side, float)
    yaw, pitch = math.atan2(f[0], f[2]), math.asin(max(-1.0, min(1.0, f[1])))
    x0 = np.array([math.cos(yaw), 0.0, -math.sin(yaw)])
    y0 = np.array([-math.sin(pitch) * math.sin(yaw), math.cos(pitch), -math.sin(pitch) * math.cos(yaw)])
    roll = math.atan2(-s.dot(x0), s.dot(y0))
    return tuple(math.degrees(a) for a in (yaw, pitch, roll))

def build_edges():
    verts = []
    for i in range(3):
        for sgn in (1, -1):
            v = np.zeros(3); v[i] = sgn * A_VERT
            verts.append(v)
    edges = []
    for i, j in itertools.combinations(range(6), 2):
        if np.allclose(verts[i], -verts[j]):
            continue
        edges.append((i, j))
    assert len(edges) == 12
    out = []
    for (i, j) in edges:
        lo, hi = verts[i], verts[j]
        L = np.linalg.norm(hi - lo)
        t = (hi - lo) / L
        M = (lo + hi) / 2
        n = M / np.linalg.norm(M)
        side = np.cross(n, t); side /= np.linalg.norm(side)
        # helices span offsets 1..84 (min_offset=1, max_offset=85); centre them on the edge midpoint
        p0 = M - (HALF + 0.17) * t - (ROWSEP / 2) * side
        out.append(dict(vi=i, vj=j, lo=lo, hi=hi, t=t, side=side, p0=p0))
    return verts, out

def base_pos(edge, row, offset):
    return edge['p0'] + row * ROWSEP * edge['side'] + offset * RISE * edge['t']

def make_groups():
    verts, edges = build_edges()
    geom = sc.Geometry(bases_per_turn=10.5, rise_per_base_pair=0.34, helix_radius=1.0, inter_helix_gap=0.6)
    groups = {}
    for k, e in enumerate(edges):
        yaw, pitch, roll = group_angles(e['t'], e['side'])
        p = e['p0']
        groups[f"e{k}"] = sc.HelixGroup(grid=sc.square, position=sc.Position3D(x=float(p[0]), y=float(p[1]), z=float(p[2])),
                                  yaw=yaw, pitch=pitch, roll=roll, geometry=geom)
    return verts, edges, groups

def make_helices(groups):
    helices = []
    for k in range(12):
        for row in (0, 1):
            helices.append(sc.Helix(min_offset=1, max_offset=NBP + 1, group=f"e{k}", grid_position=(0, row)))
    return helices
