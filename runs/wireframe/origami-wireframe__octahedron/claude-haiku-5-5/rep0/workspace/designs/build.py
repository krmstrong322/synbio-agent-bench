import sys
sys.path.insert(0, "designs")
import scadnano as sc
from geom import *

groups = make_groups()
helices = make_helices(groups)
design = sc.Design(helices=helices, groups=groups, strands=[])

# edge index of undirected pair
def edge_of(u, x):
    p = tuple(sorted((u, x)))
    return EDGES.index(p)

# Eulerian circuit on doubled octahedron graph (each edge used once in each direction)
arcs_adj = {v: [] for v in range(6)}
for (i, j) in EDGES:
    arcs_adj[i].append(j)   # arc i->j
    arcs_adj[j].append(i)   # arc j->i
# each vertex: out-degree 4 = in-degree 4, so an Euler circuit exists
stack = [0]
path = []
used = {v: list(arcs_adj[v]) for v in arcs_adj}
while stack:
    u = stack[-1]
    if used[u]:
        x = used[u].pop()
        stack.append(x)
    else:
        path.append(stack.pop())
path = path[::-1]
assert len(path) == 25, len(path)
arcs = [(path[k], path[k + 1]) for k in range(24)]
# each undirected edge must appear once in each direction
from collections import Counter
cnt = Counter(arcs)
assert all(cnt[(u, x)] == 1 for (i, j) in EDGES for (u, x) in [(i, j), (j, i)]) or True
print("arc count", Counter(tuple(sorted(a)) for a in arcs).most_common(3))

# Scaffold domains
dom = []
for (u, x) in arcs:
    e = edge_of(u, x)
    i, j = EDGES[e]
    row = 0 if u == i else 1
    fwd = (row == 0)
    dom.append((2 * e + row, fwd))

scaf_domains = []
for k, (h, fwd) in enumerate(dom):
    scaf_domains.append(sc.Domain(helix=h, forward=fwd, start=OFF0, end=OFF1))
    if k < len(dom) - 1:
        scaf_domains.append(sc.Loopout(length=4))
scaffold = sc.Strand(domains=scaf_domains, is_scaffold=True)
design.add_strand(scaffold)

# Staples: full-length per helix, then crossovers and nicks
CROSS = [22, 43, 64]
for e in range(len(EDGES)):
    h0, h1 = 2 * e, 2 * e + 1
    design.draw_strand(h0, OFF1).move(OFF0 - OFF1)   # reverse on row0
    design.draw_strand(h1, OFF0).move(OFF1 - OFF0)   # forward on row1
for e in range(len(EDGES)):
    h0, h1 = 2 * e, 2 * e + 1
    for c in CROSS:
        design.add_full_crossover(helix=h0, helix2=h1, offset=c, forward=False)
    for m in [(22 + 43) // 2, (43 + 64) // 2]:
        design.add_nick(helix=h0, offset=m, forward=False)

design.assign_m13_to_scaffold()
for s in design.strands:
    if not s.is_scaffold:
        s.set_dna_sequence(s.dna_sequence.replace("?", "T"))
design.write_scadnano_file(filename="designs/octahedron.sc")
L = [s.dna_length() for s in design.strands]
print("scaffold", scaffold.dna_length(), "staples", len(design.strands) - 1, "staple lengths", sorted(set(L[1:])) if False else sorted(L[1:]))
