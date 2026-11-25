import sys, os, random, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import scadnano as sc
from geom import make_groups, make_helices, base_pos, NBP

random.seed(1)
verts, edges, groups = make_groups()
helices = make_helices(groups)

# ---- arcs: helix id = 2k + r ; r=0 scaffold runs lo->hi (offset 1 -> 84), r=1 runs hi->lo
arcs = []
for k, e in enumerate(edges):
    for r in (0, 1):
        if r == 0:
            en_v, en_off, ex_v, ex_off = e['vi'], 1, e['vj'], 84
        else:
            en_v, en_off, ex_v, ex_off = e['vj'], 84, e['vi'], 1
        arcs.append(dict(id=2 * k + r, k=k, r=r, en_v=en_v, en_off=en_off, ex_v=ex_v, ex_off=ex_off,
                         en_pos=base_pos(e, r, en_off), ex_pos=base_pos(e, r, ex_off)))
N = len(arcs)
outs = {v: [a['id'] for a in arcs if a['ex_v'] == v] for v in range(6)}
ins = {v: [a['id'] for a in arcs if a['en_v'] == v] for v in range(6)}
A = {a['id']: a for a in arcs}

def dist(o, i):
    return float(np.linalg.norm(A[o]['ex_pos'] - A[i]['en_pos']))

best = None
for trial in range(300000):
    tmap = {}
    cost = 0.0
    for v in range(6):
        perm = ins[v][:]
        random.shuffle(perm)
        for o, i in zip(outs[v], perm):
            tmap[o] = i
            cost += dist(o, i)
    if best is not None and cost >= best[0]:
        continue
    # single cycle check
    seq = [0]
    cur = tmap[0]
    while cur != 0 and len(seq) <= N:
        seq.append(cur)
        cur = tmap[cur]
    if len(seq) == N and cur == 0:
        best = (cost, tmap, seq)
cost, tmap, seq = best
print("best transition cost (nm):", round(cost, 2), "order:", seq)

# open the circuit at its most expensive transition (scaffold 5'/3' ends there)
trans = [(seq[i], seq[(i + 1) % N]) for i in range(N)]
ci = max(range(N), key=lambda i: dist(*trans[i]))
order = seq[ci + 1:] + seq[:ci + 1]
hops = []
for i in range(N - 1):
    o, nx = order[i], order[i + 1]
    d = dist(o, nx)
    hops.append(max(2, min(12, round(d / 0.6))))
print("hop distances:", [round(dist(order[i], order[i + 1]), 2) for i in range(N - 1)])
print("hop loops:", hops)

# ---- scaffold strand
scaf_domains = []
for idx, h in enumerate(order):
    a = A[h]
    scaf_domains.append(sc.Domain(helix=h, forward=(a['r'] == 0), start=1, end=NBP + 1))
    if idx < N - 1:
        scaf_domains.append(sc.Loopout(length=hops[idx]))
scaffold = sc.Strand(domains=scaf_domains, is_scaffold=True)

strands = [scaffold]
CUTS = [22, 43, 64]
for k in range(12):
    A_, B_ = 2 * k, 2 * k + 1      # A: scaffold forward (staple reverse); B: staple forward
    # lo end U: B fwd [1,22) -> A rev [1,22)
    strands.append(sc.Strand(domains=[sc.Domain(helix=B_, forward=True, start=1, end=22),
                                      sc.Domain(helix=A_, forward=False, start=1, end=22)]))
    # rings
    for (c, d, nick_on_A) in [(22, 43, True), (43, 64, False)]:
        y = c + 10 if nick_on_A else c + 11
        if nick_on_A:
            doms = [sc.Domain(helix=A_, forward=False, start=c, end=y),
                    sc.Domain(helix=B_, forward=True, start=c, end=d),
                    sc.Domain(helix=A_, forward=False, start=y, end=d)]
        else:
            doms = [sc.Domain(helix=B_, forward=True, start=y, end=d),
                    sc.Domain(helix=A_, forward=False, start=c, end=d),
                    sc.Domain(helix=B_, forward=True, start=c, end=y)]
        strands.append(sc.Strand(domains=doms))
    # hi end U: A rev [64,85) -> B fwd [64,85)
    strands.append(sc.Strand(domains=[sc.Domain(helix=A_, forward=False, start=64, end=85),
                                      sc.Domain(helix=B_, forward=True, start=64, end=85)]))

design = sc.Design(helices=helices, groups=groups, strands=strands)
design.assign_m13_to_scaffold()
for s in design.strands:
    if not s.is_scaffold:
        s.set_dna_sequence(s.dna_sequence.replace("?", "T")) if s.dna_sequence and "?" in s.dna_sequence else None
print("scaffold length:", scaffold.dna_length(), " n strands:", len(design.strands))
print("staple lengths:", sorted(set(s.dna_length() for s in design.strands if not s.is_scaffold)))
os.makedirs("designs", exist_ok=True)
design.write_scadnano_file(filename="designs/octahedron.sc")
print("saved")
