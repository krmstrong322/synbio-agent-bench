import itertools
import scadnano as sc

# ---- layout: helix index -> honeycomb grid cell (cadnano-style coords) ----
cells = [(h, v) for h in (3, 4, 5) for v in range(4)]
N = 12
# scaffold path (Hamiltonian on the bundle graph); even path index = B (reverse), odd = A (forward)
P = [1, 0, 4, 8, 9, 5, 6, 2, 3, 7, 11, 10]
FWD_HELIX = {0, 2, 5, 7, 8, 10}          # scaffold runs forward on these helices

# ---- turnaround candidate offsets (from find_crossover_offsets, per edge) ----
T_LOW = {
    0: [5, 6, 16, 17, 26, 27, 37, 38, 47, 48, 58, 59, 68, 69, 79, 80, 89, 90],
    2: [9, 10, 19, 20, 30, 31, 40, 41, 51, 52, 61, 62, 72, 73, 82, 83, 93, 94],
    4: [2, 3, 12, 13, 23, 24, 33, 34, 44, 45, 54, 55, 65, 66, 75, 76, 86, 87, 96, 97],
    6: [2, 3, 12, 13, 23, 24, 33, 34, 44, 45, 54, 55, 65, 66, 75, 76, 86, 87, 96, 97],
    8: [9, 10, 19, 20, 30, 31, 40, 41, 51, 52, 61, 62, 72, 73, 82, 83, 93, 94],
    10: [5, 6, 16, 17, 26, 27, 37, 38, 47, 48, 58, 59, 68, 69, 79, 80, 89, 90],
}
T_HIGH = {
    1: [2, 3, 12, 13, 23, 24, 33, 34, 44, 45, 54, 55, 65, 66, 75, 76, 86, 87, 96, 97],
    3: [5, 6, 16, 17, 26, 27, 37, 38, 47, 48, 58, 59, 68, 69, 79, 80, 89, 90],
    5: [5, 6, 16, 17, 26, 27, 37, 38, 47, 48, 58, 59, 68, 69, 79, 80, 89, 90],
    7: [5, 6, 16, 17, 26, 27, 37, 38, 47, 48, 58, 59, 68, 69, 79, 80, 89, 90],
    9: [2, 3, 12, 13, 23, 24, 33, 34, 44, 45, 54, 55, 65, 66, 75, 76, 86, 87, 96, 97],
}
# turn j is between P[j] and P[j+1]; even j low, odd j high
cand = {}
for j in range(11):
    cand[j] = T_LOW[j] if j % 2 == 0 else T_HIGH[j]

# DFS for turn offsets: each helix P[j] (1..10) length = |t_j - t_{j-1}| in [78, 90]
sol = []
def dfs(j, ts):
    if j == 11:
        sol.append(list(ts)); return
    for t in cand[j]:
        if ts:
            L = abs(t - ts[-1])
            if not (78 <= L <= 90):
                continue
        ts.append(t)
        dfs(j + 1, ts)
        ts.pop()
        if len(sol) > 2000:
            return
dfs(0, [])
print("solutions", len(sol))
# pick the one whose domain lengths are closest to 84 (then smallest turn values)
def score(ts):
    return sum(abs(abs(ts[i] - ts[i - 1]) - 84) for i in range(1, len(ts))) + ts[0] * 0.01
best = min(sol, key=score)
t = best  # t[j] for j = 0..10
print("turns", t)

# ---- scaffold domains per helix ----
# helix P[0]: reverse, domain [t0, t0+84); P[j] j=1..10 between t[j-1], t[j]; P[11]: [t10, t10+84)
rng = {}
scaf_dir = {}
for j, h in enumerate(P):
    if j == 0:
        lo, hi = t[0], t[0] + 84
    elif j == 11:
        lo, hi = min(t[10], t[10] + 84), max(t[10], t[10] + 84)
    else:
        lo, hi = min(t[j - 1], t[j]), max(t[j - 1], t[j])
    rng[h] = (lo, hi)
    scaf_dir[h] = (h in FWD_HELIX)
    assert scaf_dir[h] == (j % 2 == 1), (h, j)
print("ranges", {h: (rng[h][0], rng[h][1], rng[h][1] - rng[h][0]) for h in P})

# ---- staple crossover sites: full-crossover phases per edge (staple direction) ----
# edge (a,b): offsets valid; phase lists from find_crossover_offsets
EDGE_SITES = {
    (0, 1): [1, 22, 43, 64, 85],
    (0, 4): [8, 29, 50, 71, 92],
    (8, 9): [1, 22, 43, 64, 85],
    (4, 8): [15, 36, 57, 78, 99],
    (5, 6): [1, 22, 43, 64, 85],
    (5, 9): [8, 29, 50, 71, 92],
    (1, 5): [15, 36, 57, 78, 99],
    (2, 3): [1, 22, 43, 64, 85],
    (2, 6): [8, 29, 50, 71, 92],
    (3, 7): [15, 36, 57, 78, 99],
    (7, 11): [8, 29, 50, 71, 92],
    (10, 11): [1, 22, 43, 64, 85],
    (6, 10): [15, 36, 57, 78, 99],
}
MARGIN = 7
sites = []  # (a, b, o)
for (a, b), offs in EDGE_SITES.items():
    for o in offs:
        lo_a, hi_a = rng[a]
        lo_b, hi_b = rng[b]
        if o - MARGIN >= max(lo_a, lo_b) and o + MARGIN <= min(hi_a, hi_b):
            sites.append((a, b, o))
# check offset collisions per helix
used = {}
for a, b, o in sites:
    for h in (a, b):
        assert (h, o) not in used, ("collision", h, o, used.get((h, o)))
        used[(h, o)] = (a, b)
print("sites", len(sites))

# ---- build staple domains ----
# domain = dict(h, s, e, fwd(staple), nb_lo, nb_hi)
doms = []
dom_at = {}  # (h, start) -> id ; (h, end) -> id
for h in P:
    lo, hi = rng[h]
    bounds = sorted({lo, hi} | {o for (a, b, o) in sites if h in (a, b)})
    for s, e in zip(bounds[:-1], bounds[1:]):
        d = dict(h=h, s=s, e=e, fwd=not scaf_dir[h], nb_lo=None, nb_hi=None, id=len(doms))
        doms.append(d)
        dom_at[(h, s)] = d["id"]
        dom_at[(h, e)] = d["id"]
for a, b, o in sites:
    # domain on a ending at o, domain on b ending at o -> link high ports
    ida = [d["id"] for d in doms if d["h"] == a and d["e"] == o][0]
    idb = [d["id"] for d in doms if d["h"] == b and d["e"] == o][0]
    doms[ida]["nb_hi"] = idb
    doms[idb]["nb_hi"] = ida
    ida = [d["id"] for d in doms if d["h"] == a and d["s"] == o][0]
    idb = [d["id"] for d in doms if d["h"] == b and d["s"] == o][0]
    doms[ida]["nb_lo"] = idb
    doms[idb]["nb_lo"] = ida

# trace chains: 5' end port
visited = set()
chains = []
for d in doms:
    # 5' port: forward staple -> low port; reverse staple -> high port
    five_nb = d["nb_lo"] if d["fwd"] else d["nb_hi"]
    if five_nb is not None:
        continue
    chain = []
    cur = d["id"]
    prev_port = None
    while cur is not None:
        visited.add(cur)
        chain.append(cur)
        dd = doms[cur]
        # 3' port: forward -> high; reverse -> low
        nxt = dd["nb_hi"] if dd["fwd"] else dd["nb_lo"]
        cur = nxt
        if cur in visited:
            raise RuntimeError("loop")
    chains.append(chain)
cyc = [d["id"] for d in doms if d["id"] not in visited]
print("chains", len(chains), "unvisited(cycles)", len(cyc))

def clen(ch):
    return sum(doms[i]["e"] - doms[i]["s"] for i in ch)
for ch in chains:
    L = clen(ch)
    print(L, [(doms[i]["h"], doms[i]["s"], doms[i]["e"]) for i in ch])

import json
json.dump(dict(P=P, t=t, rng={str(k): v for k, v in rng.items()}, sites=sites,
               doms=doms, chains=chains), open("/work/plan.json", "w"))
