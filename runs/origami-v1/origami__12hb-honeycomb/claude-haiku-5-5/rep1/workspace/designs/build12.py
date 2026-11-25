import random
import scadnano as sc

# Honeycomb cell positions (cadnano coords). Path order used for scaffold: helix k = index k.
POS = [(0,2),(0,1),(1,1),(2,1),(2,2),(1,2),(1,3),(0,3),(0,4),(1,4),(2,4),(2,3)]
NH = 12
# Turn boundary between helix k and k+1 (scaffold turn). Valid residues checked with find_crossover_offsets.
V = {0:105, 1:18, 2:99, 3:11, 4:91, 5:11, 6:91, 7:11, 8:98, 9:18, 10:105}
LO, HI = {}, {}
for k in range(NH):
    if k == 0:
        LO[k], HI[k] = V[0] - 84, V[0]
    elif k == NH - 1:
        LO[k], HI[k] = V[10] - 84, V[10]
    else:
        a, b = V[k-1], V[k]
        LO[k], HI[k] = min(a, b), max(a, b)

def sfwd(k):  # scaffold direction
    return k % 2 == 0

def stfwd(k):  # staple direction (antiparallel to scaffold)
    return not sfwd(k)

EDGES = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),(10,11),(0,5),(6,11)]

# Full crossover offsets from find_crossover_offsets: key (a, b, fwd of strand on a)
R = lambda *xs: list(xs)
FULLDATA = {
 (0,1,True): [1,22,43,64], (0,1,False): [6,17,27,38,48,59,69,80],
 (1,0,True): [6,17,27,38,48,59,69,80], (1,0,False): [1,22,43,64],
 (1,2,True): [3,13,24,34,45,55,66,76], (1,2,False): [8,29,50,71],
 (2,1,True): [8,29,50,71], (2,1,False): [3,13,24,34,45,55,66,76],
 (0,5,True): [15,36,57,78], (0,5,False): [10,20,31,41,52,62,73,83],
 (5,0,True): [10,20,31,41,52,62,73,83], (5,0,False): [15,36,57,78],
 (2,3,True): [15,36,57,78], (2,3,False): [10,20,31,41,52,62,73,83],
 (3,2,True): [10,20,31,41,52,62,73,83], (3,2,False): [15,36,57,78],
 (3,4,True): [6,17,27,38,48,59,69,80], (3,4,False): [1,22,43,64],
 (4,3,True): [1,22,43,64], (4,3,False): [6,17,27,38,48,59,69,80],
 (4,5,True): [8,29,50,71], (4,5,False): [3,13,24,34,45,55,66,76],
 (5,4,True): [3,13,24,34,45,55,66,76], (5,4,False): [8,29,50,71],
 (5,6,True): [6,17,27,38,48,59,69,80], (5,6,False): [1,22,43,64],
 (6,5,True): [1,22,43,64], (6,5,False): [6,17,27,38,48,59,69,80],
 (6,7,True): [8,29,50,71], (6,7,False): [3,13,24,34,45,55,66,76],
 (7,6,True): [3,13,24,34,45,55,66,76], (7,6,False): [8,29,50,71],
 (7,8,True): [6,17,27,38,48,59,69,80], (7,8,False): [1,22,43,64],
 (8,7,True): [1,22,43,64], (8,7,False): [6,17,27,38,48,59,69,80],
 (8,9,True): [15,36,57,78], (8,9,False): [10,20,31,41,52,62,73,83],
 (9,8,True): [10,20,31,41,52,62,73,83], (9,8,False): [15,36,57,78],
 (9,10,True): [3,13,24,34,45,55,66,76], (9,10,False): [8,29,50,71],
 (10,9,True): [8,29,50,71], (10,9,False): [3,13,24,34,45,55,66,76],
 (10,11,True): [1,22,43,64], (10,11,False): [6,17,27,38,48,59,69,80],
 (11,10,True): [6,17,27,38,48,59,69,80], (11,10,False): [1,22,43,64],
 (6,11,True): [15,36,57,78], (6,11,False): [10,20,31,41,52,62,73,83],
 (11,6,True): [10,20,31,41,52,62,73,83], (11,6,False): [15,36,57,78],
}
RESID = {k: set(x % 21 for x in v) for k, v in FULLDATA.items()}


def cands(a, b):
    res = RESID[(a, b, stfwd(a))]
    lo = max(LO[a], LO[b]) + 7
    hi = min(HI[a], HI[b]) - 7
    return [o for o in range(lo, hi + 1) if o % 21 in res]


def build(cuts, nicks):
    """Returns (strands, unvisited) where strands are lists of (helix, forward, lo, hi)."""
    bounds = {}
    for h in range(NH):
        cs = sorted(set(cuts[h].keys()) | nicks[h])
        bounds[h] = [LO[h]] + cs + [HI[h]]
    J = {}
    for h in range(NH):
        for o, b in cuts[h].items():
            i = bounds[h].index(o)
            j = bounds[b].index(o)
            J[(h, i-1, 'U')] = (b, j-1, 'U'); J[(b, j-1, 'U')] = (h, i-1, 'U')
            J[(h, i, 'L')] = (b, j, 'L'); J[(b, j, 'L')] = (h, i, 'L')
    visited = set()
    strands = []
    for h in range(NH):
        for s in range(len(bounds[h]) - 1):
            for e in ('L', 'U'):
                if (h, s) in visited:
                    break
                if (h, s, e) in J:
                    continue
                doms = []
                cur = (h, s, e)
                while True:
                    hh, ss, ee = cur
                    if (hh, ss) in visited:
                        return None, None
                    visited.add((hh, ss))
                    lo_, hi_ = bounds[hh][ss], bounds[hh][ss+1]
                    ex = 'U' if ee == 'L' else 'L'
                    doms.append((hh, ee == 'L', lo_, hi_))
                    nxt = J.get((hh, ss, ex))
                    if nxt is None:
                        break
                    cur = nxt
                strands.append(doms)
    unvis = []
    for h in range(NH):
        for s in range(len(bounds[h]) - 1):
            if (h, s) not in visited:
                unvis.append((h, bounds[h][s], bounds[h][s+1]))
    return strands, unvis


rng_global = random.Random(3)


def drop_cross(cuts, h, o):
    b = cuts[h].pop(o)
    cuts[b].pop(o, None)


def solve(cuts):
    """Remove crossovers until every staple is 18-60 nt and there are no closed loops (no nicks used)."""
    nicks = {h: set() for h in range(NH)}
    for it in range(200):
        strands, unvis = build(cuts, nicks)
        if strands is None:
            return None
        if unvis:
            h, lo_, hi_ = max(unvis, key=lambda t: t[2] - t[1])
            if hi_ - lo_ >= 14:
                nicks[h].add((lo_ + hi_) // 2)
                continue
            done = False
            for (h, lo_, hi_) in unvis:
                for o in (lo_, hi_):
                    if o in cuts[h]:
                        drop_cross(cuts, h, o); done = True; break
                if done:
                    break
            if not done:
                return None
            continue
        changed = False
        for st in strands:
            L = sum(hi_ - lo_ for (_, _, lo_, hi_) in st)
            if L < 18:
                return None
            if L > 60:
                # candidate joins = crossovers between consecutive domains
                cum = 0
                best = None
                for i, (h1, f1, lo1, hi1) in enumerate(st):
                    cum += hi1 - lo1
                    if i + 1 < len(st):
                        h2 = st[i+1][0]
                        o = hi1 if f1 else lo1
                        if o in cuts[h1] and cuts[h1][o] == h2:
                            score = abs(cum - L / 2)
                            if best is None or score < best[0]:
                                best = (score, h1, o)
                if best is None:
                    d = max(st, key=lambda x: x[3] - x[2])
                    if d[3] - d[2] < 14:
                        return None
                    nicks[d[0]].add((d[2] + d[3]) // 2)
                    changed = True
                    break
                drop_cross(cuts, best[1], best[2])
                changed = True
                break
        if not changed:
            return strands, nicks
    return None


def attempt_ls(rng, iters=4000):
    """Local search over crossover subsets: no two cuts on a helix closer than 7 bp, >=2 crossovers per edge."""
    C = {e: cands(*e) for e in EDGES}
    S = {e: set(rng.sample(C[e], k=rng.randint(0, len(C[e])))) for e in EDGES}

    def cost():
        cc = 0
        per = {h: [] for h in range(NH)}
        for e in EDGES:
            for o in S[e]:
                per[e[0]].append(o); per[e[1]].append(o)
        for h in range(NH):
            items = []
            for e in EDGES:
                if h in e:
                    other = e[1] if e[0] == h else e[0]
                    for o in S[e]:
                        items.append((o, other))
            items.sort()
            for i in range(len(items) - 1):
                if items[i+1][0] - items[i][0] < 7:
                    cc += 10
                if items[i+1][1] == items[i][1]:
                    cc += 10
        for e in EDGES:
            cc += 6 * max(0, 2 - len(S[e]))
        return cc

    cur = cost()
    for it in range(iters):
        if cur == 0:
            break
        e = rng.choice(EDGES)
        if not C[e]:
            continue
        o = rng.choice(C[e])
        if o in S[e]:
            S[e].discard(o)
        else:
            S[e].add(o)
        new = cost()
        if new <= cur or rng.random() < 0.02:
            cur = new
        else:
            if o in S[e]:
                S[e].discard(o)
            else:
                S[e].add(o)
    if cur != 0:
        return None
    cuts = {h: {} for h in range(NH)}
    for e in EDGES:
        for o in S[e]:
            cuts[e[0]][o] = e[1]
            cuts[e[1]][o] = e[0]
    return cuts


def attempt(rng, p):
    cuts = {h: {} for h in range(NH)}
    allc = []
    for (a, b) in EDGES:
        for o in cands(a, b):
            allc.append((o, a, b))
    allc.sort(key=lambda t: (t[0] + rng.uniform(-3, 3), rng.random()))
    cnt = {e: 0 for e in EDGES}
    for o, a, b in allc:
        if rng.random() > p:
            continue
        if any(abs(o - x) < 7 for x in cuts[a]) or any(abs(o - x) < 7 for x in cuts[b]):
            continue
        cuts[a][o] = b
        cuts[b][o] = a
        cnt[(a, b)] += 1
    return cuts


SPACING = 18
MIN_X = 1


def attempt3(rng):
    """Crossovers on each helix spaced >=20 bp; nick midway between consecutive crossovers.
    Every staple then is a two-domain U (one crossover), so there are no closed loops."""
    allc = []
    for (a, b) in EDGES:
        for o in cands(a, b):
            allc.append((o, a, b))
    allc.sort(key=lambda t: (rng.random()))
    X = {h: [] for h in range(NH)}
    cnt = {e: 0 for e in EDGES}
    chosen = []
    for o, a, b in allc:
        if any(abs(o - x) < SPACING for x in X[a]) or any(abs(o - x) < SPACING for x in X[b]):
            continue
        X[a].append(o); X[b].append(o)
        chosen.append((o, a, b))
        cnt[(a, b)] += 1
    if min(cnt.values()) < MIN_X:
        return None
    cuts = {h: {} for h in range(NH)}
    for o, a, b in chosen:
        cuts[a][o] = b
        cuts[b][o] = a
    nicks = {h: set() for h in range(NH)}
    for h in range(NH):
        if not X[h]:
            return None
        xs = sorted(X[h])
        for x1, x2 in zip(xs, xs[1:]):
            nicks[h].add((x1 + x2) // 2)
        if xs[0] - LO[h] < 9 or HI[h] - xs[-1] < 9:
            return None
        if xs[0] - LO[h] > 36:
            nicks[h].add((LO[h] + xs[0]) // 2)
        if HI[h] - xs[-1] > 36:
            nicks[h].add((xs[-1] + HI[h]) // 2)
    return cuts, nicks, cnt


def main():
    rng = random.Random(1)
    for trial in range(20000):
        r = attempt3(rng)
        if r is None:
            continue
        cuts, nicks, cnt = r
        strands, unvis = build(cuts, nicks)
        if strands is None or unvis:
            continue
        L = [sum(b - a for (_, _, a, b) in st) for st in strands]
        if min(L) < 18 or max(L) > 60:
            continue
        print("accepted trial", trial, "staples", len(strands), "min", min(L), "max", max(L))
        sol = (cuts, (strands, nicks))
        break
    else:
        print("no solution")
        return
    emit(sol)


def main_old():
    rng = random.Random(1)
    sol = None
    for trial in range(60):
        p = 1.0
        cuts = attempt_ls(rng)
        if cuts is None:
            continue
        r = solve(cuts)
        if r is None:
            continue
        cnt = {e: sum(1 for o in cuts[e[0]] if cuts[e[0]][o] == e[1]) for e in EDGES}
        if min(cnt.values()) >= 2:
            sol = (cuts, r)
            print("found at trial", trial, "p", round(p, 2))
            break
    if sol is None:
        print("no solution")
        return
    cuts, (strands, nicks) = sol

    # orient each staple so that its domains on even helices run opposite to the scaffold (forward=False)
    fixed = []
    for st in strands:
        even_fwd = [f for (h, f, _, _) in st if h % 2 == 0]
        if even_fwd and even_fwd[0]:
            st = [(h, not f, lo_, hi_) for (h, f, lo_, hi_) in reversed(st)]
        fixed.append(st)

    helices = [sc.Helix(max_offset=110, grid_position=POS[h]) for h in range(NH)]
    scaf_domains = [sc.Domain(helix=k, forward=sfwd(k), start=LO[k], end=HI[k]) for k in range(NH)]
    strand_objs = [sc.Strand(domains=scaf_domains, is_scaffold=True)]
    for st in fixed:
        strand_objs.append(sc.Strand(domains=[sc.Domain(helix=h, forward=f, start=lo_, end=hi_)
                                              for (h, f, lo_, hi_) in st]))
    design = sc.Design(helices=helices, strands=strand_objs, grid=sc.honeycomb)
    design.assign_m13_to_scaffold()
    design.write_scadnano_file(filename="designs/bundle12.sc")

    print("scaffold length", sum(HI[k]-LO[k] for k in range(NH)))
    print("helix extents", {k: (LO[k], HI[k], HI[k]-LO[k]) for k in range(NH)})
    lens = []
    for st in fixed:
        L = sum(hi_-lo_ for (_, _, lo_, hi_) in st)
        lens.append(L)
    print("n staples", len(fixed), "min", min(lens), "max", max(lens))
    print("crossovers per edge", {e: sum(1 for o in cuts[e[0]] if cuts[e[0]][o] == e[1]) for e in EDGES})
    for s in design.strands:
        print("scaffold" if s.is_scaffold else "staple", s.dna_length(),
              [(d.helix, d.start, d.end, "F" if d.forward else "R") for d in s.domains])


def emit(sol):
    cuts, (strands, nicks) = sol
    fixed = []
    for st in strands:
        even_fwd = [f for (h, f, _, _) in st if h % 2 == 0]
        if even_fwd and even_fwd[0]:
            st = [(h, not f, lo_, hi_) for (h, f, lo_, hi_) in reversed(st)]
        fixed.append(st)
    helices = [sc.Helix(max_offset=110, grid_position=POS[h]) for h in range(NH)]
    scaf_domains = [sc.Domain(helix=k, forward=sfwd(k), start=LO[k], end=HI[k]) for k in range(NH)]
    strand_objs = [sc.Strand(domains=scaf_domains, is_scaffold=True)]
    for st in fixed:
        strand_objs.append(sc.Strand(domains=[sc.Domain(helix=h, forward=f, start=lo_, end=hi_)
                                              for (h, f, lo_, hi_) in st]))
    design = sc.Design(helices=helices, strands=strand_objs, grid=sc.honeycomb)
    design.assign_m13_to_scaffold()
    design.write_scadnano_file(filename="designs/bundle12.sc")
    print("scaffold length", sum(HI[k] - LO[k] for k in range(NH)))
    print("crossovers per edge", {e: sum(1 for o in cuts[e[0]] if cuts[e[0]][o] == e[1]) for e in EDGES})
    for s in design.strands:
        print("scaffold" if s.is_scaffold else "staple", s.dna_length(),
              [(d.helix, d.start, d.end, "F" if d.forward else "R") for d in s.domains])


main()
