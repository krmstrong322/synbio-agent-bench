import math
import numpy as np
import scadnano as sc

A_ = 17.5
verts = [np.array([x, y, z], float) for x in (-A_, A_) for y in (-A_, A_) for z in (-A_, A_)]
edges = [(i, j) for i in range(8) for j in range(i + 1, 8)
         if abs(np.linalg.norm(verts[i] - verts[j]) - 35.0) < 1e-6]
assert len(edges) == 12


def group_angles(forward, side):
    f, s = np.asarray(forward, float), np.asarray(side, float)
    yaw, pitch = math.atan2(f[0], f[2]), math.asin(max(-1.0, min(1.0, f[1])))
    x0 = np.array([math.cos(yaw), 0.0, -math.sin(yaw)])
    y0 = np.array([-math.sin(pitch) * math.sin(yaw), math.cos(pitch), -math.sin(pitch) * math.cos(yaw)])
    roll = math.atan2(-s.dot(x0), s.dot(y0))
    return tuple(math.degrees(a) for a in (yaw, pitch, roll))


geom = sc.Geometry(bases_per_turn=10.5, rise_per_base_pair=0.34, helix_radius=1.0, inter_helix_gap=0.6)
HL = 84
HALF = HL * 0.34 / 2
LAT = 1.3

if __name__ == "__main__":
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
        g = sc.HelixGroup(grid=sc.square,
                          position=sc.Position3D(x=float(p0[0]), y=float(p0[1]), z=float(p0[2])),
                          yaw=yaw, pitch=pitch, roll=roll, geometry=geom)
        groups[f"E{e}"] = g
        for r in (0, 1):
            helices.append(sc.Helix(max_offset=HL, grid_position=(0, r), group=f"E{e}"))
    design = sc.Design(helices=helices, groups=groups, strands=[])
    design.write_scadnano_file(filename="designs/draft.sc")
    print("ok", len(helices))
