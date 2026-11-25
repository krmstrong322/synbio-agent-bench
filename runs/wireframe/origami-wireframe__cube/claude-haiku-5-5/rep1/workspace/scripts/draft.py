import sys, json
sys.path.insert(0, '/scripts')
from geom import *

cfg = json.load(open('scripts/config.json'))
s_idx, bb = cfg['s_idx'], cfg['bb']

geom = sc.Geometry(bases_per_turn=10.5, rise_per_base_pair=RISE, helix_radius=1.0, inter_helix_gap=0.6)
groups = {}
for e in range(12):
    P, Q, f, s = edge_geom(e, s_idx[e])
    p0, _ = helix_points(e, 0, s_idx[e])
    yaw, pitch, roll = group_angles(f, s)
    groups[f"E{e}"] = sc.HelixGroup(grid=sc.square, position=sc.Position3D(x=float(p0[0]), y=float(p0[1]), z=float(p0[2])),
                              yaw=yaw, pitch=pitch, roll=roll, geometry=geom)
helices = []
for h in range(24):
    e, r = h // 2, h % 2
    helices.append(sc.Helix(grid_position=(0, r), group=f"E{e}", max_offset=N))
design = sc.Design(helices=helices, groups=groups, strands=[])
design.write_scadnano_file(filename='designs/draft.sc')
print('ok', len(design.helices))
