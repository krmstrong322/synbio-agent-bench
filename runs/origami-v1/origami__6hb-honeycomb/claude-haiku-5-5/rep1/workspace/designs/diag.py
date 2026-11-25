import itertools
exec(open("designs/search2.py").read().split("sols = []")[0])
best = []
for choice in itertools.product(*[[(r, s) for r in RES[p] for s in (1, 2)] for p in PAIRS]):
    B = [set() for _ in range(6)]
    X = {}
    for pair, (r, s) in zip(PAIRS, choice):
        pts = crossover_points(pair, r, s)
        X[pair] = pts
        for p in pts:
            B[pair[0]].add(p)
            B[pair[1]].add(p)
    bad = 0
    Ls = []
    for pair in PAIRS:
        a, b = pair
        for p in X[pair]:
            pa = max([q for q in B[a] | {S[a]} if q < p])
            pb = max([q for q in B[b] | {S[b]} if q < p])
            L = (p - pa) + (p - pb)
            Ls.append(L)
            if not 18 <= L <= 60:
                bad += 1
    ends = [E[h] - (max(B[h]) if B[h] else S[h]) for h in range(6)]
    bad_end = sum(1 for e in ends if not 18 <= e <= 60)
    best.append((bad + bad_end, bad, bad_end, choice, sorted(Ls)[:6], ends))
best.sort(key=lambda t: t[0])
for t in best[:5]:
    print(t)
