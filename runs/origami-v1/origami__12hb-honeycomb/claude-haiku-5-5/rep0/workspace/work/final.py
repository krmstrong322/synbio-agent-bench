import random, math
import scadnano as sc

# ---------------- layout ----------------
cells = [(h, v) for h in (3, 4, 5) for v in range(4)]   # helix k -> cells[k]
P = [1, 0, 4, 8, 9, 5, 6, 2, 3, 7, 11, 10]              # scaffold path
FWD_HELIX = {0, 2, 5, 7, 8, 10}
EDGES = [(0,1),(0,4),(1,5),(2,3),(2,6),(3,7),(4,8),(5,6),(5,9),(6,10),(7,11),(8,9),(10,11)]

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
cand = {j: (T_LOW[j] if j % 2 == 0 else T_HIGH[j]) for j in range(11)}
TURNS = [5, 87, 9, 89, 3, 89, 3, 89, 9, 96, 16]   # chosen (domain lengths 78-90), checked below
for j in range(11):
    assert TURNS[j] in cand[j], j
for j in range(1, 11):
    assert 78 <= abs(TURNS[j] - TURNS[j-1]) <= 90, j

rng, scaf_dir = {}, {}
for j, h in enumerate(P):
    if j == 0:
        rng[h] = (TURNS[0], TURNS[0] + 84)
    elif j == 11:
        rng[h] = (TURNS[10], TURNS[10] + 84)
    else:
        rng[h] = (min(TURNS[j-1], TURNS[j]), max(TURNS[j-1], TURNS[j]))
    scaf_dir[h] = (h in FWD_HELIX)
    assert scaf_dir[h] == (j % 2 == 1)

# full-crossover sites (staple-direction valid), per edge (A-side reverse / B-side forward)
EDGE_SITES = {
    (0, 1): [1, 22, 43, 64, 85], (0, 4): [8, 29, 50, 71, 92],
    (8, 9): [1, 22, 43, 64, 85], (4, 8): [15, 36, 57, 78, 99],
    (5, 6): [1, 22, 43, 64, 85], (5, 9): [8, 29, 50, 71, 92],
    (1, 5): [15, 36, 57, 78, 99], (2, 3): [1, 22, 43, 64, 85],
    (2, 6): [8, 29, 50, 71, 92], (3, 7): [15, 36, 57, 78, 99],
    (7, 11): [8, 29, 50, 71, 92], (10, 11): [1, 22, 43, 64, 85],
    (6, 10): [15, 36, 57, 78, 99],
}
MARGIN = 7
sites = []
for (a, b), offs in EDGE_SITES.items():
    for o in offs:
        if o - MARGIN >= max(rng[a][0], rng[b][0]) and o + MARGIN <= min(rng[a][1], rng[b][1]):
            sites.append((a, b, o))
print("candidate sites", len(sites))


def build(chosen):
    """chosen: list of bools per site. returns (domains, links, chains, cost)."""
    S = [s for s, c in zip(sites, chosen) if c]
    doms = []
    idx_end, idx_start = {}, {}
    for h in P:
        lo, hi = rng[h]
        # domain boundaries at ALL candidate sites (unlinked ones are nicks)
        bnd = sorted({lo, hi} | {o for (a, b, o) in sites if h in (a, b)})
        for s, e in zip(bnd[:-1], bnd[1:]):
            d = dict(h=h, s=s, e=e, fwd=not scaf_dir[h], id=len(doms), hi=None, lo=None)
            doms.append(d)
            idx_end[(h, e)] = d["id"]
            idx_start[(h, s)] = d["id"]
    for a, b, o in S:
        x, y = idx_end[(a, o)], idx_end[(b, o)]
        doms[x]["hi"], doms[y]["hi"] = y, x
        x, y = idx_start[(a, o)], idx_start[(b, o)]
        doms[x]["lo"], doms[y]["lo"] = y, x
    # chains
    seen = set()
    chains = []
    for d in doms:
        five = d["lo"] if d["fwd"] else d["hi"]
        if five is not None:
            continue
        ch, cur = [], d["id"]
        while cur is not None and cur not in seen:
            seen.add(cur)
            ch.append(cur)
            dd = doms[cur]
            cur = dd["hi"] if dd["fwd"] else dd["lo"]
        chains.append(ch)
    cycles = len(doms) - len(seen)
    return doms, S, chains, cycles


def cost_of(chosen):
    doms, S, chains, cycles = build(chosen)
    c = 0.0
    for ch in chains:
        L = sum(doms[i]["e"] - doms[i]["s"] for i in ch)
        c += max(0, 22 - L) * 2 + max(0, L - 50) * 2
    c += cycles * 20
    for d in doms:
        if d["e"] - d["s"] < 7:
            c += 20
    # every edge needs at least one crossover
    for e in EDGES:
        if not any((a, b) == e or (b, a) == e for (a, b, o) in S):
            c += 30
    return c, doms, S, chains


random.seed(23)
best = None
for restart in range(2):
    st = [random.random() < 0.6 for _ in sites]
    cur, *_ = cost_of(st)
    T = 6.0
    for it in range(25000):
        i = random.randrange(len(sites))
        st[i] = not st[i]
        new, *_ = cost_of(st)
        if new <= cur or random.random() < math.exp(-(new - cur) / T):
            cur = new
        else:
            st[i] = not st[i]
        T = max(0.1, T * 0.9998)
        if cur == 0:
            break
    if best is None or cur < best[0]:
        best = (cur, list(st))
    print("restart", restart, "cost", cur)
cost, chosen = best
# tabu search over single flips
random.seed(5)
tabu = {}
cur_st = list(chosen)
cur_c, *_ = cost_of(cur_st)
bestc, best_st = cur_c, list(cur_st)
for step in range(800):
    moves = []
    for i in range(len(sites)):
        cur_st[i] = not cur_st[i]
        c2, *_ = cost_of(cur_st)
        cur_st[i] = not cur_st[i]
        if tabu.get(i, -1) >= step and c2 >= bestc:
            continue
        moves.append((c2 + random.random() * 0.1, c2, i))
    if not moves:
        continue
    _, c2, i = min(moves)
    cur_st[i] = not cur_st[i]
    tabu[i] = step + 4 + random.randrange(4)
    if c2 < bestc:
        bestc, best_st = c2, list(cur_st)
    if bestc == 0:
        break
print("tabu best", bestc)
chosen = best_st
cost = bestc
# steepest descent over single flips
improved = True
while improved:
    improved = False
    for i in range(len(sites)):
        chosen[i] = not chosen[i]
        c2, *_ = cost_of(chosen)
        if c2 < cost:
            cost = c2; improved = True
        else:
            chosen[i] = not chosen[i]
    # pair flips
    for i in range(len(sites)):
        for j in range(i + 1, len(sites)):
            if improved:
                break
            chosen[i] = not chosen[i]; chosen[j] = not chosen[j]
            c2, *_ = cost_of(chosen)
            if c2 < cost:
                cost = c2; improved = True
            else:
                chosen[i] = not chosen[i]; chosen[j] = not chosen[j]
print("after descent", cost)
cost, doms, S, chains = cost_of(chosen)
print("final cost", cost, "n crossover sites", len(S))
for ch in chains:
    L = sum(doms[i]["e"] - doms[i]["s"] for i in ch)
    print(L, [(doms[i]["h"], doms[i]["s"], doms[i]["e"]) for i in ch])

# ---------------- build design ----------------
helices = [sc.Helix(max_offset=100, grid_position=c) for c in cells]
design = sc.Design(helices=helices, strands=[], grid=sc.honeycomb)

# scaffold strand
scaf_domains = []
for j, h in enumerate(P):
    lo, hi = rng[h]
    if scaf_dir[h]:
        scaf_domains.append(sc.Domain(helix=h, forward=True, start=lo, end=hi))
    else:
        scaf_domains.append(sc.Domain(helix=h, forward=False, start=lo, end=hi))
scaffold = sc.Strand(domains=scaf_domains, is_scaffold=True)
design.add_strand(scaffold)

# staple strands from chains
for ch in chains:
    doms_s = []
    for i in ch:
        d = doms[i]
        doms_s.append(sc.Domain(helix=d["h"], forward=d["fwd"], start=d["s"], end=d["e"]))
    design.add_strand(sc.Strand(domains=doms_s))

design.assign_m13_to_scaffold()
for s in design.strands:
    if not s.is_scaffold:
        s.set_dna_sequence(s.dna_sequence.replace("?", "T")) if s.dna_sequence and "?" in s.dna_sequence else None
design.write_scadnano_file(filename="designs/bundle12.sc")

tot = sum(s.dna_length() for s in design.strands if s.is_scaffold)
print("scaffold length", tot)
stap = [s.dna_length() for s in design.strands if not s.is_scaffold]
print("staples", len(stap), "min", min(stap), "max", max(stap))
