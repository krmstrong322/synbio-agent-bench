import math
import numpy as np
import scadnano as sc

R = 35.0 / math.sqrt(2)  # vertex distance from centre (edge = 35 nm)
VERTS = [np.array(v, float) * R for v in
         [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]]
EDGES = [(i, j) for i in range(6) for j in range(i + 1, 6)
         if {i, j} not in ({0, 1}, {2, 3}, {4, 5})]
NBP = 84
RISE = 0.34
OFF0, OFF1 = 2, 86  # helix domain [2,86) = 84 bp
GEOM = sc.Geometry(bases_per_turn=10.5, rise_per_base_pair=RISE, helix_radius=1.0, inter_helix_gap=0.6)


def group_angles(forward, side):
    f, s = np.asarray(forward, float), np.asarray(side, float)
    yaw, pitch = math.atan2(f[0], f[2]), math.asin(max(-1.0, min(1.0, f[1])))
    x0 = np.array([math.cos(yaw), 0.0, -math.sin(yaw)])
    y0 = np.array([-math.sin(pitch) * math.sin(yaw), math.cos(pitch), -math.sin(pitch) * math.cos(yaw)])
    roll = math.atan2(-s.dot(x0), s.dot(y0))
    return tuple(math.degrees(a) for a in (yaw, pitch, roll))


def edge_frame(e):
    i, j = EDGES[e]
    a, b = VERTS[i], VERTS[j]
    M = (a + b) / 2
    t = (b - a) / np.linalg.norm(b - a)
    n = M / np.linalg.norm(M)
    s = np.cross(n, t)
    s = s / np.linalg.norm(s)
    # row0 axis point at offset 43 is M - 1.3 s
    p0 = M - 1.3 * s - 43 * RISE * t
    return p0, t, s


def make_groups():
    groups = {}
    for e in range(len(EDGES)):
        p0, t, s = edge_frame(e)
        yaw, pitch, roll = group_angles(t, s)
        groups[str(e)] = sc.HelixGroup(grid=sc.square, position=sc.Position3D(*p0.tolist()),
                                  yaw=yaw, pitch=pitch, roll=roll, geometry=GEOM)
    return groups


def make_helices(groups):
    hs = []
    for e in range(len(EDGES)):
        for row in (0, 1):
            hs.append(sc.Helix(max_offset=90, grid_position=(0, row), group=str(e)))
    return hs
