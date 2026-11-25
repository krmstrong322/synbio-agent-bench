import random, math
import scadnano as sc

cells = [(2, -1), (2, 0), (2, 1), (2, 2), (3, -1), (3, 0), (3, 1), (3, 2), (4, -1), (4, 0), (4, 1), (4, 2)]
N = 12
path = [1, 0, 4, 8, 9, 5, 6, 2, 3, 7, 11, 10]   # scaffold visiting order (helix indices)
K = len(path)

def par(i):
    h, v = cells[i]
    return (h + v) % 2          # scaffold fwd iff par==0

def adjacent(i, j):
    (h1, v1), (h2, v2) = cells[i], cells[j]
    if v1 == v2 and abs(h1 - h2) == 1:
        return True
    if h1 == h2 and abs(v1 - v2) == 1:
        lo = min(v1, v2)
        return (h1 + lo) % 2 == 1
    return False

def edge_info(i, j):
    (h1, v1), (h2, v2) = cells[i], cells[j]
    if h1 == h2:
        a, b = (i, j) if v1 < v2 else (j, i)
        t = 'V'
        assert par(a) == 1
    else:
        a, b = (i, j) if h1 < h2 else (j, i)
        t = 'A' if par(a) == 1 else 'B'
    return a, b, t

edges = []
for i in range(N):
    for j in range(i + 1, N):
        if adjacent(i, j):
            edges.append(edge_info(i, j))
print("edges", len(edges))

# staple full-crossover residues (mod 21) for edge type; chosen consistent set T1
# (V:6, A:13, B:20 keeps every helix's crossover sites >=7 bp apart)
RES = {'V': 6, 'A': 13, 'B': 20}
# scaffold junction residues (half crossovers at helix ends)
HF = {'V': {5, 6, 16, 17}, 'A': {2, 3, 12, 13}, 'B': {4, 14, 15}}
HR = {'V': {0, 1, 11}, 'A': {7, 8, 18}, 'B': {9, 10, 19, 20}}

# ---- 1. scaffold junction offsets ----
jres = []
for k in range(K - 1):
    X, Y = path[k], path[k + 1]
    a, b, t = edge_info(X, Y)
    if par(X) == 0:   # top junction
        res = HF[t] if X == a else HR[t]
    else:             # bottom junction
        res = HR[t] if X == a else HF[t]
    jres.append(res)

sols = []
def dfs(k, J, acc):
    if k == K - 1:
        sols.append((acc, list(J)))
        return
    for off in range(0, 170):
        if off % 21 not in jres[k]:
            continue
        if k == 0 and off < 84:
            continue
        if k == K - 2 and off < 84:
            continue
        if k >= 1:
            X = path[k]
            L = (J[-1] - off) if par(X) == 1 else (off - J[-1])
            if not (78 <= L <= 90):
                continue
            J.append(off)
            dfs(k + 1, J, acc + abs(L - 84))
            J.pop()
        else:
            J.append(off)
            dfs(k + 1, J, acc)
            J.pop()
dfs(0, [], 0)
sols.sort(key=lambda s: s[0])
cost, J = sols[0]
print("junction cost", cost, "J", J)

lo = {}; hi = {}
for k in range(K):
    h = path[k]
    if k == 0:
        hi[h] = J[0]; lo[h] = J[0] - 84
    elif k == K - 1:
        hi[h] = J[K - 2]; lo[h] = J[K - 2] - 84
    else:
        lo[h] = min(J[k - 1], J[k]); hi[h] = max(J[k - 1], J[k])
for k in range(K):
    h = path[k]
    print("helix", h, "par", par(h), "extent", lo[h], hi[h], "len", hi[h] - lo[h])

# ---- 2. candidate staple crossovers (edge, offset) ----
cand = []
for (a, b, t) in edges:
    r = RES[t]
    cl = [o for o in range(max(lo[a], lo[b]) + 4, min(hi[a], hi[b]) - 3) if o % 21 == r]
    cand.append(cl)
print("cands per edge", [len(c) for c in cand])

# site ownership: (h, p) -> edge index (only one edge can use residue at a helix)
site_edge = {}
for e, (a, b, t) in enumerate(edges):
    for o in cand[e]:
        for h in (a, b):
            assert (h, o) not in site_edge
            site_edge[(h, o)] = e

def segments(xs, nicks):
    cuts = {h: [] for h in range(N)}
    for (e, o) in xs:
        a, b, t = edges[e]
        cuts[a].append(o); cuts[b].append(o)
    for (h, o) in nicks:
        cuts[h].append(o)
    segs = []
    by_end = {}; by_start = {}
    bad = 0
    for h in range(N):
        cs = cuts[h]
        if len(set(cs)) != len(cs):
            bad += 1
        bounds = [lo[h]] + sorted(set(cs)) + [hi[h]]
        for idx in range(len(bounds) - 1):
            s, e_ = bounds[idx], bounds[idx + 1]
            sid = len(segs)
            segs.append((h, s, e_))
            by_end[(h, e_)] = sid
            by_start[(h, s)] = sid
    nb = [[] for _ in segs]
    for (e, o) in xs:
        a, b, t = edges[e]
        nb[by_end[(a, o)]].append(by_end[(b, o)]); nb[by_end[(b, o)]].append(by_end[(a, o)])
        nb[by_start[(a, o)]].append(by_start[(b, o)]); nb[by_start[(b, o)]].append(by_start[(a, o)])
    return segs, nb, bad

def evaluate(xs, nicks):
    segs, nb, bad = segments(xs, nicks)
    pen = 1000 * bad
    for (h, s, e_) in segs:
        L = e_ - s
        if L < 7:
            pen += (7 - L) * 30
    seen = [False] * len(segs)
    comps = []
    for sid in range(len(segs)):
        if seen[sid]:
            continue
        stack = [sid]; seen[sid] = True; comp = []
        while stack:
            x = stack.pop(); comp.append(x)
            for y in nb[x]:
                if not seen[y]:
                    seen[y] = True; stack.append(y)
        comps.append(comp)
    for comp in comps:
        L = sum(segs[x][2] - segs[x][1] for x in comp)
        if all(len(nb[x]) == 2 for x in comp):
            pen += 50
            continue
        if L < 18:
            pen += (18 - L) * 3
        if L > 60:
            pen += (L - 60) * 3
        if len(comp) > 1 and max(segs[x][2] - segs[x][1] for x in comp) < 14:
            pen += 4
        if L < 24 or L > 50:
            pen += 1
    for e, (a, b, t) in enumerate(edges):
        offs = sorted(o for (ee, o) in xs if ee == e)
        if not offs:
            pen += 100; continue
        pts = [max(lo[a], lo[b])] + offs + [min(hi[a], hi[b])]
        gap = max(pts[i + 1] - pts[i] for i in range(len(pts) - 1))
        if gap > 24:
            pen += (gap - 24) * 1.5
    return pen, segs, nb, comps

def run_sa(seed, iters=50000):
    rnd = random.Random(seed)
    all_x = [(e, o) for e in range(len(edges)) for o in cand[e]]
    xs = set(all_x)
    nicks = set()
    cur = evaluate(xs, nicks)[0]
    best = (cur, set(xs), set(nicks))
    sites = list(site_edge.keys())
    for it in range(iters):
        T = 3.0 * (1 - it / iters) + 0.05
        xs2 = set(xs); nk2 = set(nicks)
        if rnd.random() < 0.5:
            e, o = rnd.choice(all_x)
            a, b, t = edges[e]
            if (e, o) in xs2:
                xs2.remove((e, o))
            else:
                xs2.add((e, o))
                nk2.discard((a, o)); nk2.discard((b, o))
        else:
            h = rnd.randrange(N)
            o = rnd.randrange(lo[h] + 7, hi[h] - 6)
            if (h, o) in site_edge and (site_edge[(h, o)], o) in xs2:
                continue
            if (h, o) in nk2:
                nk2.remove((h, o))
            else:
                nk2.add((h, o))
        p2 = evaluate(xs2, nk2)[0]
        if p2 <= cur or rnd.random() < math.exp(-(p2 - cur) / T):
            xs, nicks, cur = xs2, nk2, p2
            if cur < best[0]:
                best = (cur, set(xs), set(nicks))
        if best[0] == 0:
            break
    return best

best = None
for seed in range(8):
    b = run_sa(seed)
    print("seed", seed, "penalty", round(b[0], 2), "xs", len(b[1]), "nicks", len(b[2]))
    if best is None or b[0] < best[0]:
        best = b
    if best[0] == 0:
        break

pen, xs, nicks = best
# break any remaining rings: nick the middle of segments roughly every 40 nt around the ring
for _ in range(8):
    _, segs, nb, comps = evaluate(xs, nicks)
    added = False
    for comp in comps:
        Lcomp = sum(segs[x][2] - segs[x][1] for x in comp)
        if not all(len(nb[x]) == 2 for x in comp) and Lcomp > 56:
            # path too long: nick the >=14 segment nearest the middle
            ends = [x for x in comp if len(nb[x]) < 2]
            seq = [ends[0]]; prev = None; cur = ends[0]
            while True:
                nxt = [y for y in nb[cur] if y != prev and y not in seq]
                if not nxt:
                    break
                prev = cur; cur = nxt[0]; seq.append(cur)
            cum = 0; best_x = None; best_d = 1e9
            for x in seq:
                h, s, e_ = segs[x]
                L = e_ - s
                mid = cum + L / 2
                if L >= 14 and abs(mid - Lcomp / 2) < best_d:
                    best_d = abs(mid - Lcomp / 2); best_x = x
                cum += L
            if best_x is not None:
                h, s, e_ = segs[best_x]
                nicks.add((h, s + (e_ - s) // 2))
                added = True
            continue
        if not all(len(nb[x]) == 2 for x in comp):
            continue
        order = [comp[0]]; prev = None; cur = comp[0]
        while True:
            nxt = [y for y in nb[cur] if y != prev]
            if not nxt or nxt[0] == comp[0]:
                break
            prev = cur; cur = nxt[0]; order.append(cur)
            if len(order) > len(comp):
                break
        total = 0
        for x in order:
            h, s, e_ = segs[x]
            L = e_ - s
            total += L
            if total >= 40 and L >= 14:
                nicks.add((h, s + L // 2))
                total = 0
                added = True
        if not any(segs[x][2] - segs[x][1] >= 14 for x in order):
            # no room for a nick: drop one crossover on this ring instead
            bnd = set()
            for x in comp:
                h, s, e_ = segs[x]
                bnd.add((h, s)); bnd.add((h, e_))
            for (e, o) in sorted(xs):
                a, b, t = edges[e]
                if (a, o) in bnd and (b, o) in bnd:
                    xs.discard((e, o))
                    added = True
                    break
    if not added:
        break
pen, segs, nb, comps = evaluate(xs, nicks)
print("final penalty", pen, "xs", len(xs), "nicks", len(nicks))
for comp in comps:
    L = sum(segs[x][2] - segs[x][1] for x in comp)
    cyc = all(len(nb[x]) == 2 for x in comp)
    print("comp len", L, "nseg", len(comp), "cycle" if cyc else "")

# ---- 3. build design ----
helices = [sc.Helix(max_offset=max(hi.values()) + 1, grid_position=cells[i]) for i in range(N)]
design = sc.Design(helices=helices, strands=[], grid=sc.honeycomb)

scaf_domains = []
for k in range(K):
    h = path[k]
    scaf_domains.append(sc.Domain(helix=h, forward=(par(h) == 0), start=lo[h], end=hi[h]))
design.add_strand(sc.Strand(domains=scaf_domains, is_scaffold=True))

for comp in comps:
    if len(comp) == 1:
        order = comp
    else:
        start = None
        for x in comp:
            if len(nb[x]) >= 2:
                continue
            h, s, e_ = segs[x]
            fwd = par(h) == 1
            xb = set()
            for (ee, oo) in xs:
                xb.add((edges[ee][0], oo)); xb.add((edges[ee][1], oo))
            has_low = (h, s) in xb
            has_high = (h, e_) in xb
            five_free = (not has_low) if fwd else (not has_high)
            if five_free:
                start = x
                break
        if start is None:
            for x in comp:
                if len(nb[x]) < 2:
                    print("endpoint", segs[x], "nb", [segs[y] for y in nb[x]], "par", par(segs[x][0]))
            print("comp", [segs[x] for x in comp])
        assert start is not None, "no 5' end found"
        order = [start]; prev = None; cur = start
        while True:
            nxt = [y for y in nb[cur] if y != prev and y not in order]
            if not nxt:
                break
            prev = cur; cur = nxt[0]; order.append(cur)
        assert len(order) == len(comp)
    doms = []
    for x in order:
        h, s, e_ = segs[x]
        doms.append(sc.Domain(helix=h, forward=(par(h) == 1), start=s, end=e_))
    design.add_strand(sc.Strand(domains=doms))

design.assign_m13_to_scaffold()

tot = 0
for st in design.strands:
    if st.is_scaffold:
        print("scaffold length", st.dna_length())
    else:
        L = st.dna_length(); tot += L
        print("staple", L, [(d.helix, d.start, d.end, "F" if d.forward else "R") for d in st.domains])
print("staple total", tot)
design.write_scadnano_file(filename="designs/bundle12.sc")
print("saved")
