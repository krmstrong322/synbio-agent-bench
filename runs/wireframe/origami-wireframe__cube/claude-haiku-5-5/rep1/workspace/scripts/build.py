import sys, json
sys.path.insert(0, '/scripts')
from geom import *

cfg = json.load(open('scripts/config.json'))
s_idx, bb, perms = cfg['s_idx'], cfg['bb'], cfg['perms']
single, cost, nxt, pts = evaluate(s_idx, bb, perms)
assert single
nxt = {int(k): v for k, v in nxt.items()} if isinstance(list(nxt.keys())[0], str) else nxt

def fwd(h):
    e, r = h // 2, h % 2
    return r == bb[e]

# ---- helix groups / helices ----
geom = sc.Geometry(bases_per_turn=10.5, rise_per_base_pair=RISE, helix_radius=1.0, inter_helix_gap=0.6)
groups = {}
for e in range(12):
    P, Q, f, s = edge_geom(e, s_idx[e])
    p0, _ = helix_points(e, 0, s_idx[e])
    yaw, pitch, roll = group_angles(f, s)
    groups[f"E{e}"] = sc.HelixGroup(grid=sc.square,
                                    position=sc.Position3D(x=float(p0[0]), y=float(p0[1]), z=float(p0[2])),
                                    yaw=yaw, pitch=pitch, roll=roll, geometry=geom)
helices = [sc.Helix(grid_position=(0, h % 2), group=f"E{h // 2}", max_offset=N) for h in range(24)]

# ---- scaffold cycle ----
seq = [0]
while True:
    n_ = nxt[seq[-1]]
    if n_ == 0:
        break
    seq.append(n_)
assert len(seq) == 24
dists = []
for k in range(24):
    a, b_ = seq[k], seq[(k + 1) % 24]
    dists.append(float(np.linalg.norm(pts[(a, 'arr')][1] - pts[(b_, 'dep')][1])))
kcut = max(range(24), key=lambda k: dists[k])
st = (kcut + 1) % 24
seq = seq[st:] + seq[:st]
dists = dists[st:] + dists[:st]   # dists[k] = junction between seq[k] and seq[k+1]; last (23) is the cut
loops = [max(3, int(round(dists[k] / 0.6))) for k in range(23)]

scaf_domains = []
for k, h in enumerate(seq):
    scaf_domains.append(sc.Domain(helix=h, forward=fwd(h), start=0, end=N))
    if k < 23:
        scaf_domains.append(sc.Loopout(length=loops[k]))
scaffold = sc.Strand(domains=scaf_domains, is_scaffold=True)
print('scaffold length (approx):', 24 * N + sum(loops), 'loops', loops)

strands = [scaffold]
# ---- staples: per edge DX ----
for e in range(12):
    b = bb[e]
    w_row = b          # row with scaffold forward (staple reverse)
    u_row = 1 - b      # row with staple forward
    u, w = 2 * e + u_row, 2 * e + w_row
    B = [27, 48, 69] if b == 1 else [22, 43, 64]
    # low U-staple
    strands.append(sc.Strand(domains=[
        sc.Domain(helix=u, forward=True, start=0, end=B[0]),
        sc.Domain(helix=w, forward=False, start=0, end=B[0])]))
    # rings between consecutive crossovers, one nick each (mid-domain on u)
    for i in range(len(B) - 1):
        m = (B[i] + B[i + 1]) // 2
        strands.append(sc.Strand(domains=[
            sc.Domain(helix=u, forward=True, start=m, end=B[i + 1]),
            sc.Domain(helix=w, forward=False, start=B[i], end=B[i + 1]),
            sc.Domain(helix=u, forward=True, start=B[i], end=m)]))
    # high U-staple
    strands.append(sc.Strand(domains=[
        sc.Domain(helix=w, forward=False, start=B[-1], end=N),
        sc.Domain(helix=u, forward=True, start=B[-1], end=N)]))

design = sc.Design(helices=helices, groups=groups, strands=strands)
design.assign_m13_to_scaffold()
for s in design.strands:
    if not s.is_scaffold:
        assert '?' not in s.dna_sequence
print('scaffold nt', design.strands[0].dna_length() if False else [s.dna_length() for s in design.strands if s.is_scaffold])
print('staple lengths', sorted(set(s.dna_length() for s in design.strands if not s.is_scaffold)))
print('n staples', sum(1 for s in design.strands if not s.is_scaffold))
design.write_scadnano_file(filename='designs/cube.sc')
print('saved')
